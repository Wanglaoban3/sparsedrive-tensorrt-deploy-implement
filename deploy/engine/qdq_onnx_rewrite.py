"""Restructure the ModelOpt torch-exported QDQ ONNX into standard ORT-style QDQ.

The torch-side export stores every Q/DQ's scale/zero_point as in-graph Constant
nodes (695 of them, 220 duplicated) and quantizes weight initializers inside the
graph (float weight -> QuantizeLinear -> DequantizeLinear per site). Standard
ONNX-quantized models (onnxruntime quantize_static / modelopt.onnx.quantization)
instead carry weights as int8 initializers feeding DequantizeLinear, with
scale/zp as shared initializers.

This pass is representation-only:
  1. Weight fold: for QuantizeLinear(float-initializer, scale, zp), apply the
     ONNX QuantizeLinear spec offline (saturate(round(x/scale) + zp), round
     half to even — identical to np.round and to runtime rounding) and store
     the result as an int8 initializer feeding the same DequantizeLinear.
  2. Constant -> initializer: every Constant node's value tensor becomes an
     initializer under the same name; byte-identical tensors share one
     initializer (dedup), all references rewritten.
  3. Prune initializers no longer referenced by any node.

Verification (no GPU, no custom-op build needed):
  - every folded weight chain is replayed as a minimal Q->DQ vs int8->DQ pair
    in onnxruntime on the graph's real scale/zp tensors and compared bitwise;
  - every surviving Q/DQ's scale/zp initializer is checked byte-equal to the
    Constant tensor it replaced;
  - the full model still contains the sparsedrivev2 custom op, so full-model
    ORT loading is not possible without the plugin library (unchanged by this
    pass); structural parity is covered by the two checks above.

Usage:
  python deploy/qdq_onnx_rewrite.py IN.onnx [OUT.onnx]

OUT defaults to IN (validated before overwrite; the original is kept beside it
as <name>_pre_rewrite.onnx). Writes <out>_rewrite_report.json and a summary.
"""
import argparse
import collections
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


def const_values(graph):
    """tensor_name -> np.ndarray for all Constant nodes."""
    vals = {}
    for n in graph.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.name == "value" and a.HasField("t"):
                vals[n.output[0]] = numpy_helper.to_array(a.t)
    return vals


def get_axis(node):
    for a in node.attribute:
        if a.name == "axis":
            return a.i
    return None


def fold_weight_quant(graph, inits, cvals):
    """QuantizeLinear(initializer) -> int8 initializer + re-point DQ consumers.

    A Q is folded only if its data input is an initializer and every consumer
    of its output is a DequantizeLinear. Returns fold records for verification.
    """
    consumers = collections.defaultdict(list)
    for n in graph.node:
        for i in n.input:
            consumers[i].append(n)

    folds = []
    for n in graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        wname = n.input[0]
        if wname not in inits or any(c.op_type != "DequantizeLinear"
                                     for c in consumers[n.output[0]]):
            continue
        scale = cvals.get(n.input[1]) if len(n.input) > 1 else None
        if scale is None and len(n.input) > 1 and n.input[1] in inits:
            scale = numpy_helper.to_array(inits[n.input[1]])
        if scale is None:
            continue
        zp = None
        if len(n.input) > 2:
            zp = cvals.get(n.input[2])
            if zp is None and n.input[2] in inits:
                zp = numpy_helper.to_array(inits[n.input[2]])
        dtype = np.int8 if zp is None else zp.dtype.type
        w = numpy_helper.to_array(inits[wname]).astype(np.float32)
        scale_b = scale.astype(np.float32)
        zp_b = zp
        # per-axis quantization: Q/DQ carry an `axis` attribute; the 1-D scale
        # must be reshaped to broadcast along that axis like the runtime does
        ax1 = get_axis(n)
        if ax1 is not None and scale_b.ndim == 1 and w.ndim > 1:
            assert scale_b.shape[0] == w.shape[ax1], \
                f"per-axis scale {scale_b.shape} vs axis {ax1} dim {w.shape}"
            shp = [1] * w.ndim
            shp[ax1] = scale_b.shape[0]
            scale_b = scale_b.reshape(shp)
            if zp_b is not None and zp_b.ndim == 1:
                zp_b = zp_b.reshape(shp)
        # ONNX spec: y = saturate(round(x / scale) + zero_point); round = half-to-even
        wq = np.clip(np.round(w / scale_b) + (0 if zp_b is None else zp_b),
                     np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
        q_init_name = n.output[0] + "_int8"
        q_init = numpy_helper.from_array(wq, q_init_name)
        graph.initializer.append(q_init)
        inits[q_init_name] = q_init
        dq = consumers[n.output[0]][0]
        folds.append(dict(
            name=n.name, w=wname, w_arr=w,
            s1=n.input[1] if len(n.input) > 1 else None,
            z1=n.input[2] if len(n.input) > 2 else None,
            ax1=get_axis(n), s2=dq.input[1],
            z2=dq.input[2] if len(dq.input) > 2 else None,
            ax2=get_axis(dq), wq=q_init_name))
        for c in consumers[n.output[0]]:
            c.input[0] = q_init_name
        graph.node.remove(n)
    return folds


def constants_to_initializers(graph, inits):
    """Move Constant node values into initializers, dedup by content bytes."""
    by_bytes = {}
    repl = {}
    n_const = 0
    for n in list(graph.node):
        if n.op_type != "Constant":
            continue
        t = None
        for a in n.attribute:
            if a.name == "value" and a.HasField("t"):
                t = a.t
        if t is None:
            continue  # non-tensor constant stays a node
        key = t.SerializeToString()
        if key in by_bytes:
            repl[n.output[0]] = by_bytes[key]
        else:
            init = t
            init.name = n.output[0]
            graph.initializer.append(init)
            inits[init.name] = init
            by_bytes[key] = init.name
            repl[n.output[0]] = init.name
        graph.node.remove(n)
        n_const += 1
    for n in graph.node:
        for i, nm in enumerate(n.input):
            if nm in repl:
                n.input[i] = repl[nm]
    return n_const, len(by_bytes), repl


def prune_initializers(graph, inits):
    referenced = set()
    for n in graph.node:
        referenced.update(n.input)
    referenced.update(o.name for o in graph.output)
    referenced.update(i.name for i in graph.input)
    dead = [nm for nm in inits if nm not in referenced]
    for nm in dead:
        del inits[nm]
    return len(dead)


def sort_graph_topo(graph):
    """Strict topological re-sort (node removal/rewriting can expose, or an
    export can ship, out-of-order nodes; ORT tolerates it, the checker does
    not — and a sorted graph is strictly better for downstream tooling)."""
    produced = {i.name for i in graph.initializer}
    produced.update(i.name for i in graph.input)
    remaining = list(graph.node)
    ordered = []
    while remaining:
        nxt = []
        progressed = False
        for n in remaining:
            if all(inp == "" or inp in produced for inp in n.input):
                ordered.append(n)
                produced.update(n.output)
                progressed = True
            else:
                nxt.append(n)
        remaining = nxt
        if not progressed:
            missing = sorted({i for n in remaining for i in n.input
                              if i not in produced})
            raise SystemExit("topo sort stuck, unresolved tensors: %s"
                             % missing[:8])
    del graph.node[:]
    graph.node.extend(ordered)


def graph_stats(graph):
    hist = collections.Counter(n.op_type for n in graph.node)
    init_types = collections.Counter(
        onnx.TensorProto.DataType.Name(i.data_type) for i in graph.initializer)
    return dict(nodes=len(graph.node), initializers=len(graph.initializer),
                constant_nodes=hist.get("Constant", 0),
                quantize_linear=hist.get("QuantizeLinear", 0),
                dequantize_linear=hist.get("DequantizeLinear", 0),
                initializer_dtypes=dict(init_types))


def verify_scale_relocation(graph, inits, cvals, repl, orig_init_names):
    """Every surviving Q/DQ scale/zp must be byte-equal to its source: either
    the Constant tensor it replaced, or an initializer the model already had."""
    rev = {new: orig for orig, new in repl.items()}
    checked = checked_preexisting = 0
    for n in graph.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for nm in n.input[1:]:
            if nm not in inits:
                continue
            if nm in orig_init_names:
                checked_preexisting += 1
                continue
            arr = numpy_helper.to_array(inits[nm])
            orig = rev.get(nm, nm)
            assert orig in cvals, f"scale/zp {nm} not traceable to a Constant"
            ref = cvals[orig]
            assert arr.dtype == ref.dtype and arr.tobytes() == ref.tobytes(), \
                f"scale/zp content mismatch for {nm}"
            checked += 1
    return checked, checked_preexisting


def verify_folds_ort(folds, inits, workdir):
    """Replay each folded chain as a minimal model in ORT:
        W --QuantizeLinear(s1,z1,ax1)--> DequantizeLinear(s2,z2,ax2) --> out_orig
        WQ(int8) ----------------------> DequantizeLinear(s2,z2,ax2) --> out_new
    and require bitwise equality. This exercises ORT's own rounding/axis
    handling as the referee for the offline quantization."""
    import onnxruntime as ort

    def arr(name):
        if name is None:
            return None
        if name in inits:
            return numpy_helper.to_array(inits[name])
        return None

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    max_diff = 0.0
    for k, f in enumerate(folds):
        w = f["w_arr"]
        s1, z1, s2, z2 = arr(f["s1"]), arr(f["z1"]), arr(f["s2"]), arr(f["z2"])
        wq = numpy_helper.to_array(inits[f["wq"]])
        q_inputs = ["W", "S1"] + (["Z1"] if z1 is not None else [])
        d_inputs = ["q", "S2"] + (["Z2"] if z2 is not None else [])
        d_inputs2 = ["WQ", "S2"] + (["Z2"] if z2 is not None else [])
        nodes = [
            helper.make_node("QuantizeLinear", q_inputs, ["q"]),
            helper.make_node("DequantizeLinear", d_inputs, ["o1"]),
            helper.make_node("DequantizeLinear", d_inputs2, ["o2"]),
        ]
        inits_mini = [
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(wq, "WQ"),
            numpy_helper.from_array(s1.astype(np.float32), "S1"),
            numpy_helper.from_array(s2.astype(np.float32), "S2"),
        ] + ([numpy_helper.from_array(z1, "Z1")] if z1 is not None else []) \
          + ([numpy_helper.from_array(z2, "Z2")] if z2 is not None else [])
        ins = [helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, list(w.shape))]
        outs = [helper.make_tensor_value_info("o1", onnx.TensorProto.FLOAT, list(w.shape)),
                helper.make_tensor_value_info("o2", onnx.TensorProto.FLOAT, list(w.shape))]
        graph = helper.make_graph(nodes, "chain", ins, outs, inits_mini)
        if f["ax1"] is not None:
            graph.node[0].attribute.append(helper.make_attribute("axis", f["ax1"]))
        if f["ax2"] is not None:
            for ni in (1, 2):
                graph.node[ni].attribute.append(helper.make_attribute("axis", f["ax2"]))
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        model.ir_version = 10
        path = os.path.join(workdir, f"_chain_{k}.onnx.tmp")
        onnx.save(model, path)
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        o1, o2 = sess.run(None, {"W": w})
        os.remove(path)
        d = float(np.max(np.abs(o1 - o2)))
        max_diff = max(max_diff, d)
        assert np.array_equal(o1, o2), f"chain {f['name']}: bitwise mismatch (max {d})"
    return dict(mode="qdq_chain_ort_replay", n_chains=len(folds),
                bitwise_equal=True, max_abs_diff=max_diff)


def check_dfa_boundary(m):
    """Guarantee every SparseDrive::DeformableAggregation input edge ends in
    a FLOAT tensor: a Q->DQ pair (fake-quant) is kept as-is (the DQ output is
    float and matches QAT values exactly), while a bare QuantizeLinear output
    (int8) feeding the plugin gets a DequantizeLinear appended with the same
    scale/zp.  Returns a report dict."""
    g = m.graph
    dfa_nodes = [n for n in g.node
                 if n.op_type == "DeformableAggregation"
                 or "Deformable" in n.op_type]
    if not dfa_nodes:
        return dict(dfa_nodes=0, int8_edges_fixed=0, pruned=0)

    prod = _producers(g)
    int8_fixed = 0
    added = []
    for n in dfa_nodes:
        for idx, inp in enumerate(n.input):
            q = prod.get(inp)
            if q is None or q.op_type != "QuantizeLinear":
                continue
            # reuse an existing DQ on the same Q output if present
            dq = next((nd for nd in g.node
                       if nd.op_type == "DequantizeLinear"
                       and nd.input[0] == inp), None)
            if dq is None:
                dq = helper.make_node(
                    "DequantizeLinear",
                    [inp] + list(q.input[1:]),
                    [f"{inp}_fp_dq"],
                    name=f"dfa_boundary_dq_{len(added)}")
                added.append(dq)
                g.node.append(dq)
            n.input[idx] = dq.output[0]
            int8_fixed += 1

    # prune Q/DQ nodes left with no consumers (fixpoint)
    pruned = 0
    while True:
        used = {x for nd in g.node for x in nd.input}
        dead = [nd for nd in g.node
                if nd.op_type in ("QuantizeLinear", "DequantizeLinear")
                and not any(o in used for o in nd.output)]
        if not dead:
            break
        for nd in dead:
            g.node.remove(nd)
        pruned += len(dead)
    return dict(dfa_nodes=len(dfa_nodes), int8_edges_fixed=int8_fixed,
                pruned=pruned)


def _producers(g):
    out = {}
    for n in g.node:
        for o in n.output:
            out[o] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    args = ap.parse_args()
    dst = args.dst or args.src
    out_dir = os.path.dirname(os.path.abspath(dst)) or "."

    m = onnx.load(args.src)
    g = m.graph
    before = graph_stats(g)

    inits = {i.name: i for i in g.initializer}
    orig_init_names = set(inits)
    cvals = const_values(g)
    folds = fold_weight_quant(g, inits, cvals)
    n_const, n_unique, repl = constants_to_initializers(g, inits)
    n_pruned = prune_initializers(g, inits)
    after = graph_stats(g)
    n_scale_checked, n_preexisting = verify_scale_relocation(
        g, inits, cvals, repl, orig_init_names)
    print(f"folded weight Q nodes : {len(folds)}")
    print(f"Constant -> initializer: {n_const} nodes ({n_unique} unique after dedup)")
    print(f"pruned dead initializers: {n_pruned}")
    print(f"scale/zp relocation checks: {n_scale_checked} byte-equal "
          f"(+{n_preexisting} already-initializer inputs)")
    print("before:", json.dumps(before))
    print("after :", json.dumps(after))

    dfa_report = check_dfa_boundary(m)
    print("dfa boundary:", json.dumps(dfa_report))
    sort_graph_topo(g)
    onnx.checker.check_model(m)

    tmp = os.path.join(out_dir, "_rewriting.tmp.onnx")
    onnx.save(m, tmp)

    parity = verify_folds_ort(folds, inits, out_dir)
    print("parity:", json.dumps(parity))

    if os.path.abspath(dst) == os.path.abspath(args.src):
        backup = dst.rsplit(".onnx", 1)[0] + "_pre_rewrite.onnx"
        if os.path.exists(dst):
            os.replace(dst, backup)
    os.replace(tmp, dst)

    report = dict(src=os.path.basename(args.src), dst=os.path.basename(dst),
                  folded_weight_q=len(folds), constants_folded=n_const,
                  constants_unique_after_dedup=n_unique,
                  initializers_pruned=n_pruned,
                  scale_zp_relocation_checks=n_scale_checked,
                  scale_zp_preexisting_initializers=n_preexisting,
                  before=before, after=after, parity=parity)
    rp = os.path.join(out_dir, os.path.basename(dst).rsplit(".onnx", 1)[0]
                      + "_rewrite_report.json")
    json.dump(report, open(rp, "w"), indent=2)
    print("saved ->", dst)
    print("report ->", rp)


if __name__ == "__main__":
    main()
