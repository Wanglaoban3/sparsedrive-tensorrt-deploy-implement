import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np

ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
EV = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
ENG, B2, SP32 = EV + r"\mini_eng_v8", EV + r"\mini_b2", EV + r"\mini_sp32"

def load(d, k, n):
    p = os.path.join(d, "outv8_%02d" % k if d is ENG else "out_%02d" % k, n + ".bin")
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

def top_pred(d, k, topk=10):
    cls = load(d, k, "map_cls").reshape(100, 3)
    pts = load(d, k, "map_pts").reshape(100, 20, 2)
    s = 1 / (1 + np.exp(-cls.max(axis=1)))
    idx = np.argsort(-s, kind="stable")[:topk]
    return s[idx], pts[idx], cls[idx].argmax(1)

def chamfer(a, b):
    d = np.linalg.norm(a[:, None, :, :] - b[None, :, :, :], axis=-1).mean(-1)
    return d.min(1)   # for each a: distance to nearest b

print("order-invariant damage curve: engine top-10 vs B2a (and B2a vs FP32 ctrl)")
print("%4s  %22s  %22s" % ("frame", "eng|B2 twin<=1m / dscore", "B2|FP32 twin<=1m / dscore"))
for k in list(range(0, 8)) + [20, 30, 39, 40, 41, 50, 60, 80]:
    line = "%4d " % k
    for da, db in ((ENG, B2), (B2, SP32)):
        sa, pa, ca = top_pred(da, k)
        sb, pb, cb = top_pred(db, k)
        d = chamfer(pa, pb)
        twin = (d <= 1.0)
        # score of best-matching b for each a
        dmat = np.linalg.norm(pa[:, None, :, :] - pb[None, :, :, :], axis=-1).mean(-1)
        j = dmat.argmin(1)
        ds = np.abs(sa - sb[j])
        line += "  %2d/10 %.3f" % (int(twin.sum()), float(ds[twin].mean()) if twin.any() else float("nan"))
    print(line)
