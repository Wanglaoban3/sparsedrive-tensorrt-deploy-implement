# -*- coding: utf-8 -*-
"""Show fpn_convs.0/conv/Conv wiring in sp_backbone (P2h) vs v5_P1h."""
import io
import sys

import onnx
from onnx import numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")


def show(fn, nodename):
    m = onnx.load(W + "\\" + fn)
    g = m.graph
    inits = {x.name: x for x in g.initializer}
    prod = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            prod[o] = i
    print(f"== {fn} ==")
    for n in g.node:
        if n.name == nodename:
            for x in n.input:
                if x in inits:
                    a = numpy_helper.to_array(inits[x])
                    print(f"   init {x} {a.dtype}{a.shape}")
                else:
                    p = prod.get(x)
                    pn = g.node[p] if p is not None else None
                    print(f"   in  {x}\n       <- [{pn.op_type}] "
                          f"{pn.name if pn is not None else '?'}")
            print(f"   out {list(n.output)}")
            # one step deeper on the first input
            x0 = n.input[0]
            p = prod.get(x0)
            if p is not None:
                pn = g.node[p]
                for xx in pn.input:
                    if xx in inits:
                        a = numpy_helper.to_array(inits[xx])
                        print(f"     {pn.name} init {xx} {a.dtype}{a.shape}")
                    else:
                        pp = prod.get(xx)
                        ppn = g.node[pp] if pp is not None else None
                        print(f"     {pn.name} in {xx} <- "
                              f"[{ppn.op_type if ppn else '?'}] "
                              f"{ppn.name if ppn is not None else ''}")


show("sp_backbone.onnx", "/img_neck/fpn_convs.0/conv/Conv")
show("v5_P1h.onnx", "/img_neck/fpn_convs.0/conv/Conv")
