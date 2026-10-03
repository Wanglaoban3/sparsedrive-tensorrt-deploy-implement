# -*- coding: utf-8 -*-
"""make dump onnx v2: Identity nodes after every exposed DFA IO tensor
(forces TRT to materialize them as network outputs without internalizing)."""
import io
import sys

import onnx
from onnx import helper, shape_inference, TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = W + r"\v5_P1h.onnx"
DST = W + r"\v5_dump.onnx"

m = onnx.load(SRC)
mi = shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(mi.graph.value_info) + list(mi.graph.output)}
g = m.graph

dfas = [n for n in g.node if n.op_type == "DeformableAggregation"]
print("DFA nodes:", len(dfas))

# collect (tensor, suffix) pairs
pairs = []
feat_done = False
for k, n in enumerate(dfas):
    if not feat_done:
        pairs.append((n.input[0], "dfa_feat"))
        feat_done = True
    pairs.append((n.input[3], f"dfa{k}_loc"))
    pairs.append((n.input[4], f"dfa{k}_log"))
    pairs.append((n.output[0], f"dfa{k}_out"))

# insert Identity: tensor -> tensor.__dump
prod_pos = {}
for i, n in enumerate(g.node):
    for o in n.output:
        prod_pos[o] = i

new_nodes_by_pos = {}
vi_list = list(vi.items())
info = dict(vi_list)

nl = []
inserted = set()
for i, node in enumerate(list(g.node)):
    nl.append(node)
    for tname, suffix in pairs:
        if prod_pos.get(tname) == i and tname not in inserted:
            out_name = tname + ".__dump"
            nl.append(helper.make_node("Identity", [tname], [out_name],
                                       name="dump_id_" + suffix))
            inserted.add(tname)
            # register as graph output
            v = info.get(tname)
            if v is not None:
                vo = helper.make_tensor_value_info(out_name,
                                                   v.type.tensor_type.elem_type,
                                                   [d.dim_value for d in
                                                    v.type.tensor_type.shape.dim])
                g.output.append(vo)
            else:
                print("!! no shape for", tname)

del g.node[:]
g.node.extend(nl)
onnx.save(m, DST)
print("identities:", len(pairs), "-> saved", DST)
