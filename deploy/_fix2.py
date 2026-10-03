# -*- coding: utf-8 -*-
p = r"<REPO>\deploy\test_dfa_ab.cu"
s = open(p, encoding="utf-8").read()
old = """        kv8::dfa_gather_v8(d_out[i], d_feat[i], d_ws, ws_size, bs, num_feat,
                           C, cams, S, P, G, 0);"""
new = """        kv8::dfa_gather_v8(d_out[i], d_feat[i], d_ws, ws_size, bs, A,
                           num_feat, C, cams, S, P, G, 0);"""
n = s.count(old)
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("replaced", n)
