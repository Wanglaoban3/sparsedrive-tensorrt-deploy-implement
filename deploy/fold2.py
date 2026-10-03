"""自研全图 shape 传播 + Shape->常量折叠 + onnxsim 收尾.

背景: TRT 8.6.12 backport 对 (i64 shape-tensor 链 + QDQ + --int8) 会
segfault/assert. 静态图里这些链可全折叠, 但 onnx 官方 shape inference 在
SparseDrive 自定义节点后断链且多轮不收敛 -> 自写传播绕过.

用法: python deploy/fold2.py <in.onnx> <out.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

DT = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
      TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
      TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8,
      TensorProto.BOOL: np.bool_, TensorProto.DOUBLE: np.float64}
NP_DT = {v: k for k, v in DT.items()}

VALUE_ELEMS = 200000  # 只对小张量求值 (形状辅助图), 大特征张量只求 shape


class G:
    def __init__(self, path):
        self.m = onnx.load(path)
        self.g = self.m.graph
        self.shape = {}   # name -> tuple
        self.dtype = {}   # name -> TensorProto
        self.value = {}   # name -> np.ndarray (仅小张量)
        self.attrs = {}   # node -> {attr: val}
        self.itotal = 0
        # 导出器 trace 时留下的具体维度 = 真值神谕
        self.vi_shape = {}
        for vi in (list(self.g.input) + list(self.g.output)
                   + list(self.g.value_info)):
            tt = vi.type.tensor_type
            if tt.shape.dim and all(d.HasField("dim_value")
                                    for d in tt.shape.dim):
                self.vi_shape[vi.name] = tuple(d.dim_value
                                               for d in tt.shape.dim)

    # ---------- env ----------
    def seed(self):
        for init in self.g.initializer:
            arr = numpy_helper.to_array(init)
            self.shape[init.name] = tuple(arr.shape)
            self.dtype[init.name] = init.data_type
            if arr.size <= VALUE_ELEMS:
                self.value[init.name] = arr
            self.itotal += 1
        for v in self.g.input:
            tt = v.type.tensor_type
            dims = tuple(d.dim_value for d in tt.shape.dim)
            assert all(isinstance(d, int) and d > 0 for d in dims), v.name
            self.shape[v.name] = dims
            self.dtype[v.name] = tt.elem_type
        # vi 神谕直接播种: 导出器记录的具体维度优先于推断
        for name, dims in self.vi_shape.items():
            self.shape[name] = dims

    def attr(self, n, name, default=None):
        for a in n.attribute:
            if a.name == name:
                return onnx.helper.get_attribute_value(a)
        return default

    def ints(self, n, name, default):
        v = self.attr(n, name)
        if v is None:
            return default
        return list(v) if isinstance(v, (list, tuple)) else [v]

    def val(self, name):
        return self.value.get(name)

    def put(self, n, idx, shape, dtype=None, value=None):
        out = n.output[idx]
        changed = False
        if out in self.vi_shape:
            # 神谕锁: 导出器 trace 的维度是真相, 推断无权覆盖
            if self.shape.get(out) != self.vi_shape[out]:
                self.shape[out] = self.vi_shape[out]
                changed = True
            shape = self.vi_shape[out]
        else:
            if self.shape.get(out) != tuple(int(d) for d in shape):
                self.shape[out] = tuple(int(d) for d in shape)
                changed = True
            else:
                self.shape[out] = tuple(int(d) for d in shape)
        self.dtype[out] = dtype if dtype is not None else self.dtype.get(n.input[0], TensorProto.FLOAT) if n.input else TensorProto.FLOAT
        if value is not None and value.size <= VALUE_ELEMS:
            self.value[out] = value
            self.dtype[out] = NP_DT.get(value.dtype.type, self.dtype[out])
        return changed

    def known(self, n):
        return all(i in self.shape for i in n.input if i)

    # ---------- broadcast helpers ----------
    def bcast(self, shapes):
        ss = [list(s) for s in shapes if s]
        if not ss:
            return ()
        L = max(len(s) for s in ss)
        r = [1] * L
        for s in ss:
            for i, d in enumerate(reversed(s)):
                r[L - 1 - i] = max(r[L - 1 - i], d)
        return tuple(r)

    def unsq(self, shape, axes):
        s = list(shape)
        for a in sorted(axes):
            s.insert(a if a >= 0 else a + len(s) + 1, 1)
        return tuple(s)

    def sq(self, shape, axes):
        return tuple(d for i, d in enumerate(shape) if i not in
                     {a % len(shape) for a in axes})


# ---------- value ops (小张量) ----------
def v_unary(g, n, fn):
    x = g.val(n.input[0])
    if x is None:
        return None
    return fn(x)


def v_gather(g, n):
    d, i = g.val(n.input[0]), g.val(n.input[1])
    if d is None or i is None:
        return None
    axis = g.attr(n, "axis", 0)
    return np.take(d, i.astype(np.int64), axis=axis)


def v_slice(g, n):
    d = g.val(n.input[0])
    st, en = g.val(n.input[1]), g.val(n.input[2])
    ax = g.val(n.input[3]) if len(n.input) > 3 else None
    sp = g.val(n.input[4]) if len(n.input) > 4 else None
    if any(v is None for v in (d, st, en)):
        return None
    axes = ax.astype(int).tolist() if ax is not None else list(range(len(st)))
    steps = sp.astype(int).tolist() if sp is not None else [1] * len(axes)
    s = [slice(None)] * d.ndim
    for k, a in enumerate(axes):
        s[a] = slice(int(st[k]), int(en[k]), int(steps[k]))
    return d[tuple(s)]


def v_concat(g, n):
    xs = [g.val(i) for i in n.input]
    if any(x is None for x in xs):
        return None
    # onnxsim 折掉 Unsqueeze(标量) 后会有 0-d 输入, numpy 不允许 concatenate
    xs = [np.atleast_1d(x) for x in xs]
    return np.concatenate(xs, axis=g.attr(n, "axis", 0))


def v_constant(g, n):
    for name in ("value", "value_int", "value_ints",
                 "value_float", "value_floats"):
        v = g.attr(n, name)
        if v is not None:
            if name == "value":
                return numpy_helper.to_array(v)
            if name == "value_int":
                return np.asarray(v, np.int64)
            if name == "value_ints":
                return np.asarray(list(v), np.int64)
            if name == "value_float":
                return np.asarray(v, np.float32)
            return np.asarray(list(v), np.float32)
    raise KeyError(f"Constant without value attr: {n.name}")


def v_constantofshape(g, n):
    t = g.val(n.input[0]) if n.input else None
    if t is None:
        return None
    a = g.attr(n, "value")
    fill = numpy_helper.to_array(a) if a is not None \
        else np.asarray(0, np.float32)
    return np.full([int(v) for v in np.atleast_1d(t)], fill, fill.dtype)


VALUE_FNS = {
    "Constant": v_constant,
    "ConstantOfShape": v_constantofshape,
    "Shape": lambda g, n: np.asarray(g.shape.get(n.input[0], []), np.int64),
    "Gather": v_gather,
    "Concat": v_concat,
    "Squeeze": lambda g, n: (lambda x: x.reshape(g.sq(x.shape, g.ints(n, "axes", list(range(x.ndim))))) if x is not None else None)(g.val(n.input[0])),
    "Unsqueeze": lambda g, n: (lambda x: x.reshape(g.unsq(x.shape, g.ints(n, "axes", []))) if x is not None else None)(g.val(n.input[0])),
    "Slice": v_slice,
    "Cast": lambda g, n: (lambda x: x.astype(DT[g.attr(n, "to")]) if x is not None else None)(g.val(n.input[0])),
    "Reshape": lambda g, n: (lambda x, t: x.reshape(
        [x.shape[i] if t[i] == 0 else (x.size // -np.prod([t[j] for j in range(len(t)) if t[j] == -1]) if t[i] == -1 else t[i]) for i in range(len(t))]) if x is not None and t is not None else None)(g.val(n.input[0]), g.val(n.input[1])),
    "Add": lambda g, n: v_unary(g, n, lambda x: x + g.val(n.input[1])),
    "Sub": lambda g, n: v_unary(g, n, lambda x: x - g.val(n.input[1])),
    "Mul": lambda g, n: v_unary(g, n, lambda x: x * g.val(n.input[1])),
    "Div": lambda g, n: v_unary(g, n, lambda x: x // np.maximum(g.val(n.input[1]), 1) if x.dtype.kind in "iu" else x / g.val(n.input[1])),
    "Range": lambda g, n: (lambda a, b, c: np.arange(int(a), int(b), int(c), dtype=np.int64) if a is not None and b is not None and c is not None else None)(g.val(n.input[0]), g.val(n.input[1]), g.val(n.input[2])),
    "Equal": lambda g, n: v_unary(g, n, lambda x: (x == g.val(n.input[1]))),
    "Less": lambda g, n: v_unary(g, n, lambda x: (x < g.val(n.input[1]))),
    "Greater": lambda g, n: v_unary(g, n, lambda x: (x > g.val(n.input[1]))),
    "LessOrEqual": lambda g, n: v_unary(g, n, lambda x: (x <= g.val(n.input[1]))),
    "GreaterOrEqual": lambda g, n: v_unary(g, n, lambda x: (x >= g.val(n.input[1]))),
    "Where": lambda g, n: (lambda c, a, b: np.where(c, a, b) if c is not None and a is not None and b is not None else None)(g.val(n.input[0]), g.val(n.input[1]), g.val(n.input[2])),
    "Min": lambda g, n: v_unary(g, n, lambda x: np.minimum(x, g.val(n.input[1]))),
    "Max": lambda g, n: v_unary(g, n, lambda x: np.maximum(x, g.val(n.input[1]))),
    "Tile": lambda g, n: (lambda x, r: np.tile(x, r.astype(int)) if x is not None and r is not None else None)(g.val(n.input[0]), g.val(n.input[1])),
    "Expand": lambda g, n: (lambda x, t: np.broadcast_to(
        x, [max(d, int(tv)) for d, tv in zip(x.shape, t)] if len(t) == x.ndim else np.broadcast_to(x, [int(v) for v in t]).shape).copy() if x is not None and t is not None else None)(g.val(n.input[0]), g.val(n.input[1])),
    "Transpose": lambda g, n: (lambda x: np.transpose(x, g.ints(n, "perm", list(range(x.ndim))[::-1])) if x is not None else None)(g.val(n.input[0])),
    "Sqrt": lambda g, n: v_unary(g, n, np.sqrt),
    "Abs": lambda g, n: v_unary(g, n, np.abs),
    "ReduceMax": lambda g, n: (lambda x: x.max(axis=tuple(g.ints(n, "axes", list(range(x.ndim)))), keepdims=g.attr(n, "keepdims", 1)) if x is not None else None)(g.val(n.input[0])),
    "ReduceSum": lambda g, n: (lambda x: x.sum(axis=tuple((g.val(n.input[1]) if len(n.input) > 1 else g.ints(n, "axes", list(range(x.ndim))))), keepdims=g.attr(n, "keepdims", 1)) if x is not None else None)(g.val(n.input[0])),
    "Clip": lambda g, n: (lambda x, a, b: np.clip(x, a, b) if x is not None else None)(g.val(n.input[0]), g.val(n.input[1]) if len(n.input) > 1 else None, g.val(n.input[2]) if len(n.input) > 2 else None),
}


# ---------- shape ops ----------
def sh_broadcast(g, n):
    # 严格模式: 任一输入形状未知必须返回 None 重试, 绝不部分猜测
    if any(i and i not in g.shape for i in n.input):
        return None
    return g.bcast([g.shape[i] for i in n.input if i])


def sh_matmul(g, n):
    a, b = g.shape[n.input[0]], g.shape[n.input[1]]
    out_last = (a[-1], b[-1]) if len(b) == 2 else (a[-1], b[-1])
    core = (a[-2], b[-1])
    if len(a) == 2 and len(b) == 2:
        return core
    ba = a[:-2] if len(a) > 2 else ()
    bb = b[:-2] if len(b) > 2 else ()
    return g.bcast([ba, bb]) + core


def sh_reduce(g, n):
    x = g.shape[n.input[0]]
    axes = g.ints(n, "axes", None)
    if axes is None and len(n.input) > 1:
        v = g.val(n.input[1])
        axes = v.astype(int).tolist() if v is not None else None
    keep = g.attr(n, "keepdims", 1)
    if axes is None:
        axes = list(range(len(x)))
    axes = {a % len(x) for a in axes}
    if keep:
        return tuple(1 if i in axes else d for i, d in enumerate(x))
    return tuple(d for i, d in enumerate(x) if i not in axes)


def sh_reshape(g, n):
    t = g.val(n.input[1])
    if t is None:
        return None
    t = [int(v) for v in np.atleast_1d(t)]
    total = int(np.prod(g.shape[n.input[0]]))
    if -1 in t:
        known = int(np.prod([d for d in t if d != -1])) or 1
        t = [total // known if d == -1 else d for d in t]
    t = [g.shape[n.input[0]][i] if d == 0 else d for i, d in enumerate(t)]
    return tuple(t)


def sh_topk(g, n):
    x = g.shape[n.input[0]]
    k = None
    if len(n.input) > 1 and g.val(n.input[1]) is not None:
        k = int(np.atleast_1d(g.val(n.input[1]))[0])
    if k is None:
        k = g.attr(n, "k", 1)
    ax = g.attr(n, "axis", -1) % len(x)
    s = tuple(k if i == ax else d for i, d in enumerate(x))
    return [s, s]


def sh_slice(g, n):
    d = g.shape[n.input[0]]
    st, en = g.val(n.input[1]), g.val(n.input[2])
    if st is None or en is None:
        return None
    axv = g.val(n.input[3]) if len(n.input) > 3 else None
    spv = g.val(n.input[4]) if len(n.input) > 4 else None
    axes = [int(v) for v in np.atleast_1d(axv)] if axv is not None \
        else list(range(len(np.atleast_1d(st))))
    steps = [int(v) for v in np.atleast_1d(spv)] if spv is not None \
        else [1] * len(axes)
    out = list(d)
    for k, a in enumerate(axes):
        a %= len(d)
        step = steps[k] or 1
        s, e = int(np.atleast_1d(st)[k]), int(np.atleast_1d(en)[k])
        if s < 0:
            s += d[a]
        if e < 0:
            e += d[a]
        if step > 0:
            s, e = min(max(s, 0), d[a]), min(max(e, 0), d[a])
        else:
            s, e = min(max(s, -1), d[a] - 1), min(max(e, -1), d[a] - 1)
        out[a] = max(len(range(s, e, step)), 0)
    return tuple(out)


def sh_conv(g, n):
    x, w = g.shape[n.input[0]], g.shape[n.input[1]]
    stride = g.ints(n, "strides", [1, 1])
    pads = g.ints(n, "pads", [0, 0, 0, 0])
    dil = g.ints(n, "dilations", [1, 1])
    h = (x[2] + pads[0] + pads[2] - dil[0] * (w[2] - 1) - 1) // stride[0] + 1
    w_ = (x[3] + pads[1] + pads[3] - dil[1] * (w[3] - 1) - 1) // stride[1] + 1
    return (x[0], w[0], h, w_)


def sh_pool(g, n):
    x = g.shape[n.input[0]]
    ker = g.ints(n, "kernel_shape", [1, 1])
    stride = g.ints(n, "strides", [1, 1])
    pads = g.ints(n, "pads", [0, 0, 0, 0])
    dil = g.ints(n, "dilations", [1, 1])
    h = (x[2] + pads[0] + pads[2] - dil[0] * (ker[0] - 1) - 1) // stride[0] + 1
    w_ = (x[3] + pads[1] + pads[3] - dil[1] * (ker[1] - 1) - 1) // stride[1] + 1
    return (x[0], x[1], h, w_)


def sh_globalpool(g, n):
    x = g.shape[n.input[0]]
    return (x[0], x[1], 1, 1)


def sh_pad(g, n):
    x = g.shape[n.input[0]]
    p = g.val(n.input[1]) if len(n.input) > 1 else None
    if p is None:
        p = g.ints(n, "pads", None)
        if p is None:
            return x
        p = np.asarray(p)
    p = p.astype(int).flatten()
    # ONNX pads 布局: [begin_0..begin_{r-1}, end_0..end_{r-1}]
    r = len(x)
    if len(p) != 2 * r:
        return None
    return tuple(x[i] + int(p[i]) + int(p[i + r]) for i in range(r))


def sh_dfa(g, n):
    f, w = g.shape[n.input[0]], g.shape[n.input[3]]
    return (f[0], w[1], f[2])


def sh_resize(g, n):
    x = g.shape[n.input[0]]
    sizes = g.val(n.input[3]) if len(n.input) > 3 else None
    if sizes is not None:
        return tuple(int(v) for v in sizes)
    scales = g.val(n.input[2]) if len(n.input) > 2 else None
    if scales is not None:
        return tuple(int(d * s) for d, s in zip(x, scales))
    return None


def sh_split(g, n):
    x = g.shape[n.input[0]]
    ax = (g.attr(n, "axis", 0) + len(x)) % len(x)
    sv = g.val(n.input[1]) if len(n.input) > 1 else None
    if sv is None:
        sv = g.attr(n, "split")
    if sv is not None:
        sizes = [int(v) for v in sv]
    else:
        k = g.attr(n, "num_outputs", len(n.output))
        sizes = [x[ax] // k] * k
    outs, off = [], 0
    for s in sizes:
        outs.append(tuple(s if i == ax else d for i, d in enumerate(x)))
        off += s
    return outs


SHAPE_FNS = {
    "Add": sh_broadcast, "Sub": sh_broadcast, "Mul": sh_broadcast,
    "Div": sh_broadcast, "Pow": sh_broadcast, "Min": sh_broadcast,
    "Max": sh_broadcast, "Equal": sh_broadcast, "Less": sh_broadcast,
    "Greater": sh_broadcast, "LessOrEqual": sh_broadcast, "GreaterOrEqual": sh_broadcast, "Xor": sh_broadcast, "Not": lambda g, n: g.shape[n.input[0]], "IsNaN": lambda g, n: g.shape[n.input[0]], "IsInf": lambda g, n: g.shape[n.input[0]], "Where": sh_broadcast, "Or": sh_broadcast,
    "And": sh_broadcast,
    "Relu": lambda g, n: g.shape[n.input[0]],
    "Sigmoid": lambda g, n: g.shape[n.input[0]],
    "Exp": lambda g, n: g.shape[n.input[0]],
    "Sqrt": lambda g, n: g.shape[n.input[0]],
    "Abs": lambda g, n: g.shape[n.input[0]],
    "Erf": lambda g, n: g.shape[n.input[0]],
    "Tanh": lambda g, n: g.shape[n.input[0]],
    "Neg": lambda g, n: g.shape[n.input[0]],
    "Clip": lambda g, n: g.shape[n.input[0]],
    "Identity": lambda g, n: g.shape[n.input[0]],
    "Cast": lambda g, n: g.shape[n.input[0]],
    "QuantizeLinear": lambda g, n: g.shape[n.input[0]],
    "DequantizeLinear": lambda g, n: g.shape[n.input[0]],
    "BatchNormalization": lambda g, n: g.shape[n.input[0]],
    "Softmax": lambda g, n: g.shape[n.input[0]],
    "Dropout": lambda g, n: g.shape[n.input[0]],
    "MatMul": sh_matmul,
    "Gemm": lambda g, n: (g.shape[n.input[0]][0], g.shape[n.input[1]][1]),
    "ReduceMean": sh_reduce, "ReduceMax": sh_reduce, "ReduceMin": sh_reduce,
    "ReduceSum": sh_reduce, "ReduceProd": sh_reduce,
    "Unsqueeze": lambda g, n: g.unsq(g.shape[n.input[0]], g.ints(n, "axes", (g.val(n.input[1]).astype(int).tolist() if len(n.input) > 1 and g.val(n.input[1]) is not None else []))),
    "Squeeze": lambda g, n: g.sq(g.shape[n.input[0]], g.ints(n, "axes", None) if g.attr(n, "axes") is not None else (g.val(n.input[1]).astype(int).tolist() if len(n.input) > 1 and g.val(n.input[1]) is not None else list(range(len(g.shape[n.input[0]]))))),
    "Flatten": lambda g, n: (lambda x, a: (int(np.prod(x[:a])), int(np.prod(x[a:]))))(g.shape[n.input[0]], g.attr(n, "axis", 1)),
    "Transpose": lambda g, n: tuple(g.shape[n.input[0]][i] for i in g.ints(n, "perm", list(range(len(g.shape[n.input[0]])))[::-1])),
    "Reshape": sh_reshape,
    "Concat": lambda g, n: (lambda ss, ax: tuple(
        sum(s[i] for s in ss) if i == ax else ss[0][i]
        for i in range(len(ss[0]))))(
        [g.shape[i] for i in n.input],
        (g.attr(n, "axis", 0) + len(g.shape[n.input[0]]))
        % len(g.shape[n.input[0]])),
    "Gather": lambda g, n: (lambda d, i, ax: d[:ax] + tuple(i) + d[ax + 1:] if i else d[:ax] + d[ax + 1:])(g.shape[n.input[0]], g.shape[n.input[1]], g.attr(n, "axis", 0)),
    "Slice": sh_slice,
    "Shape": lambda g, n: (len(g.shape[n.input[0]]),),
    "Range": lambda g, n: None,
    "ConstantOfShape": lambda g, n: None,
    "Expand": lambda g, n: (lambda x, t: tuple(
        max(d, int(tv)) for d, tv in zip(x, t)) if t is not None and len(t) == len(x) else (
        tuple(int(v) for v in t) if t is not None else None))(
        g.shape[n.input[0]], g.val(n.input[1])),
    "Tile": lambda g, n: (lambda x, r: tuple(
        int(d) * int(rv) for d, rv in zip(x, r))
        if r is not None and len(r) == len(x) else None)(
        g.shape[n.input[0]], g.val(n.input[1])),
    "Conv": sh_conv,
    "ConvTranspose": sh_conv,
    "MaxPool": sh_pool, "AveragePool": sh_pool,
    "GlobalAveragePool": sh_globalpool, "GlobalMaxPool": sh_globalpool,
    "LpPool": sh_pool,
    "LayerNormalization": lambda g, n: g.shape[n.input[0]],
    "InstanceNormalization": lambda g, n: g.shape[n.input[0]],
    "LRN": lambda g, n: g.shape[n.input[0]],
    "CumSum": lambda g, n: g.shape[n.input[0]],
    "Reciprocal": lambda g, n: g.shape[n.input[0]],
    "Floor": lambda g, n: g.shape[n.input[0]],
    "Ceil": lambda g, n: g.shape[n.input[0]],
    "Round": lambda g, n: g.shape[n.input[0]],
    "Sign": lambda g, n: g.shape[n.input[0]],
    "Mod": sh_broadcast,
    "Pad": sh_pad,
    "Gelu": lambda g, n: g.shape[n.input[0]],
    "TopK": sh_topk,
    "ScatterND": lambda g, n: g.shape[n.input[0]],
    "Split": sh_split,
    "Resize": sh_resize,
    "SparseDrive::DeformableAggregation": sh_dfa,
}


def value_shape_fallback(g, n):
    """值已知时, 输出 shape = value.shape (Slice/Range/Expand/Tile/Resize/...)."""
    fn = VALUE_FNS.get(n.op_type)
    if fn is None:
        return None
    try:
        v = fn(g, n)
    except Exception:
        return None
    return v


def value_pass(g):
    changed = False
    for n in g.g.node:
        if len(n.output) != 1 or n.op_type == "Constant":
            continue
        if n.output[0] not in g.shape:
            continue
        op = (n.domain + "::" if n.domain else "") + n.op_type
        try:
            if n.op_type == "Shape":
                if n.input[0] in g.shape:
                    v = np.asarray(g.shape[n.input[0]], np.int64)
                    old = g.value.get(n.output[0])
                    if old is None or not np.array_equal(old, v):
                        changed = True
                    g.value[n.output[0]] = v
            elif op in VALUE_FNS and all(
                    g.value.get(i) is not None for i in n.input if i):
                vv = VALUE_FNS[op](g, n)
                if vv is not None and vv.size <= VALUE_ELEMS:
                    old = g.value.get(n.output[0])
                    if old is None or not np.array_equal(old, vv):
                        changed = True
                    g.value[n.output[0]] = vv
        except Exception:
            pass
    return changed


def shape_pass(g):
    changed = False
    for n in g.g.node:
        op = (n.domain + "::" if n.domain else "") + n.op_type

        if n.op_type == "Constant":
            if n.output[0] in g.shape:
                continue  # 纯属性决定, 无部分输入问题
            v = VALUE_FNS["Constant"](g, n)
            g.shape[n.output[0]] = tuple(v.shape)
            g.dtype[n.output[0]] = NP_DT.get(v.dtype.type, TensorProto.FLOAT)
            if v.size <= VALUE_ELEMS:
                g.value[n.output[0]] = v
            changed = True
            continue

        if not g.known(n):
            continue

        fn = SHAPE_FNS.get(op) or SHAPE_FNS.get(n.op_type)
        shapes = None
        try:
            if fn is not None:
                shapes = fn(g, n)
        except Exception:
            shapes = None

        if shapes is None and len(n.output) == 1:
            v = value_shape_fallback(g, n)
            if v is not None:
                dt = g.dtype.get(n.input[0], TensorProto.FLOAT)
                if n.op_type in ("Shape", "Range", "ConstantOfShape"):
                    dt = TensorProto.INT64
                elif n.op_type in ("Equal", "Less", "Greater", "Not", "And", "Or"):
                    dt = TensorProto.BOOL
                changed = g.put(n, 0, tuple(v.shape), dtype=dt, value=v) \
                    or changed
                continue

        if shapes is not None:
            if not (isinstance(shapes, (list, tuple)) and shapes
                    and isinstance(shapes[0], (list, tuple))):
                shapes = [shapes]
            for j, sh in enumerate(shapes):
                if j >= len(n.output) or sh is None:
                    continue
                dt = g.dtype.get(n.input[0], TensorProto.FLOAT)
                if n.op_type in ("Shape", "Range", "ConstantOfShape"):
                    dt = TensorProto.INT64
                elif n.op_type in ("Equal", "Less", "Greater", "Not", "And", "Or"):
                    dt = TensorProto.BOOL
                elif n.op_type == "TopK" and j == 1:
                    dt = TensorProto.INT64
                changed = g.put(n, j, sh, dtype=dt) or changed
    return changed


def propagate(g):
    for _ in range(150):
        c1 = shape_pass(g)
        c2 = value_pass(g)
        if not (c1 or c2):
            break
    # 报告
    miss = []
    first_blocker = None
    for n in g.g.node:
        if all(o in g.shape for o in n.output if o):
            continue
        op = (n.domain + "::" if n.domain else "") + n.op_type
        for o in n.output:
            if o and o not in g.shape:
                miss.append((op, o))
        if first_blocker is None and g.known(n):
            first_blocker = (op, n.name, list(n.input))
    if first_blocker:
        print("    first blocker (inputs known, output unknown):", first_blocker)
    # vi 神谕: 传播结果与导出器 trace 记录的具体维度逐一比对
    conf = []
    for name, dims in g.vi_shape.items():
        got = g.shape.get(name)
        if got is not None and tuple(got) != tuple(dims):
            conf.append((name, tuple(dims), tuple(got)))
    g.vi_conflicts = conf
    if conf:
        print(f"    VI-ORACLE CONFLICTS: {len(conf)}")
        for c in conf[:10]:
            print("       vi=%s propagated=%s  %s" % (c[1], c[2], c[0]))
    return miss


def fold_shapes(g):
    consumers = {}
    for n in g.g.node:
        for i, inp in enumerate(n.input):
            consumers.setdefault(inp, []).append((n, i))
    replaced = 0
    for n in list(g.g.node):
        if n.op_type != "Shape" or n.domain:
            continue
        dims = g.shape.get(n.input[0])
        if dims is None:
            continue
        cname = n.output[0] + "_c"
        arr = np.asarray(dims, np.int64)
        g.g.initializer.append(numpy_helper.from_array(arr, name=cname))
        for cn, ci in consumers.get(n.output[0], []):
            cn.input[ci] = cname
        g.shape[n.output[0]] = (len(dims),)
        g.dtype[n.output[0]] = TensorProto.INT64
        g.value[n.output[0]] = arr
        replaced += 1
    used = set()
    for n in g.g.node:
        used.update(n.input)
    used |= {o.name for o in g.g.output}
    for n in list(g.g.node):
        if n.op_type == "Shape" and not any(o in used for o in n.output):
            g.g.node.remove(n)
    return replaced


def main():
    src, dst = sys.argv[1], sys.argv[2]
    g = G(src)
    g.seed()
    print(f"seed: {len(g.shape)} known tensors")

    for it in range(6):
        miss = propagate(g)
        n_shape = sum(1 for n in g.g.node if n.op_type == "Shape")
        print(f"[prop iter{it}] missing outputs: {len(miss)} | Shape nodes: {n_shape}")
        if miss:
            from collections import Counter
            c = Counter(op for op, _ in miss)
            print("   ", c.most_common(8), "| e.g.", miss[0])
        if getattr(g, "vi_conflicts", None):
            print("VI_ORACLE_FAIL: propagated shapes contradict exporter vi")
            sys.exit(3)
        if n_shape:
            r = fold_shapes(g)
            print(f"    folded {r} Shape->const")
        if not miss and not n_shape:
            break

    # onnxsim 收尾
    from onnxsim import simplify
    ms, ok = simplify(g.m)
    print("onnxsim:", ok, "| nodes", len(g.m.graph.node), "->", len(ms.graph.node),
          "| Shape", sum(1 for n in g.m.graph.node if n.op_type == "Shape"),
          "->", sum(1 for n in ms.graph.node if n.op_type == "Shape"))
    onnx.save(ms if ok else g.m, dst)
    print("FOLD2_DONE")


if __name__ == "__main__":
    main()
