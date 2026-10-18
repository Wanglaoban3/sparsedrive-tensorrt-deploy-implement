# -*- coding: utf-8 -*-
"""Split the 81-frame mini dumps into scene1 (0..39) / scene2 (40..80) subset
dirs (renamed out_00..) so eval_t6_mini_map.py can evaluate them separately.
"""
import io, os, shutil, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np

EV = r"H:\projects\sparsedrive-tensorrt-deploy-implement\work_dirs\sparsedrive_small_stage2\evaldata"
SPLITS = {"s1": range(0, 40), "s2": range(40, 81)}
SRCS = {"eng": (os.path.join(EV, "mini_eng_v8"), "outv8_%02d"),
        "b2": (os.path.join(EV, "mini_b2"), "out_%02d"),
        "sp32": (os.path.join(EV, "mini_sp32"), "out_%02d")}

tokens = np.load(os.path.join(EV, "mini_b2", "mini_meta.npz"),
                 allow_pickle=True)["tokens"]
assert len(tokens) == 81

for sname, rng in SPLITS.items():
    idx = list(rng)
    tk = [str(tokens[i]) for i in idx]
    for tag, (src, pref) in SRCS.items():
        dst = os.path.join(EV, "mini_%s_%s" % (tag, sname))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        for j, i in enumerate(idx):
            d = os.path.join(dst, "out_%02d" % j)
            shutil.copytree(os.path.join(src, pref % i), d)
        np.savez(os.path.join(dst, "mini_meta.npz"), tokens=np.array(tk))
        print("built", dst, len(idx), "frames")
print("SPLIT_DONE")
