import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np

ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
EV = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
ENG, B2, SP32 = EV + r"\mini_eng_v8", EV + r"\mini_b2", EV + r"\mini_sp32"

def load(d, k, n):
    p = os.path.join(d, ("outv8_%02d" if d is ENG else "out_%02d") % k, n + ".bin")
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

def top(d, k, topk=10):
    cls = load(d, k, "map_cls").reshape(100, 3)
    pts = load(d, k, "map_pts").reshape(100, 20, 2)
    s = 1 / (1 + np.exp(-cls.max(axis=1)))
    idx = np.argsort(-s, kind="stable")[:topk]
    return s[idx], pts[idx]

print("matched-pair chamfer distance of engine top-10 vs B2a (and B2a vs FP32)")
print("drift signature: linear growth = systematic bias; sqrt growth = random walk")
print("%4s  %28s  %28s" % ("frame", "eng|B2: mean/med/max >1m>2m", "B2|FP32: mean/med/max >1m>2m"))
for k in list(range(0, 12)) + [16, 20, 25, 30, 35, 39, 40, 41, 45, 50, 60, 70, 80]:
    line = "%4d " % k
    for da, db in ((ENG, B2), (B2, SP32)):
        sa, pa = top(da, k)
        sb, pb = top(db, k)
        d = np.linalg.norm(pa[:, None, :, :] - pb[None, :, :, :], axis=-1).mean(-1).min(1)
        line += "  %.3f/%.3f/%.2f %d|%d" % (d.mean(), np.median(d), d.max(),
                                            int((d > 1).sum()), int((d > 2).sum()))
    print(line)
