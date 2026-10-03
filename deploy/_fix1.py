# -*- coding: utf-8 -*-
p = r"<REPO>\deploy\test_dfa_ab.cu"
s = open(p, encoding="utf-8").read()
n = s.count("bs, 6, num_feat")
s = s.replace("bs, 6, num_feat", "bs, cams, num_feat")
open(p, "w", encoding="utf-8").write(s)
print("replaced", n)
