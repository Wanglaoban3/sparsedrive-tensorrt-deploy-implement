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

K = 0
for K in (0, 1):
    se, pe = top(ENG, K)
    sb, pb = top(B2, K)
    print("=== frame %d ===" % K)
    print("score eng|b2:", " ".join("%.3f/%.3f" % (a, b) for a, b in zip(se[:5], sb[:5])))
    # greedy match eng->b2 within 3m
    d = np.linalg.norm(pe[:, None] - pb[None], axis=-1).mean(-1)
    for i in range(5):
        j = int(d[i].argmin())
        vec = (pe[i] - pb[j]).mean(0)
        print("eng#%d -> b2#%d  d=%.3f  meanvec=(%.3f,%.3f)" % (i, j, d[i, j], vec[0], vec[1]))
    print("eng#0 pts:", np.round(pe[0][::4], 2).tolist())
    print("b2 #0 pts:", np.round(pb[int(d[0].argmin())][::4], 2).tolist())

# flat cls diff frame 0
ce = load(ENG, 0, "map_cls").reshape(100, 3)
cb = load(B2, 0, "map_cls").reshape(100, 3)
print("map_cls f0: max|d|=%.4f  mean|d|=%.4f" % (np.abs(ce - cb).max(), np.abs(ce - cb).mean()))
pe = load(ENG, 0, "map_pts").reshape(100, 20, 2)
pb = load(B2, 0, "map_pts").reshape(100, 20, 2)
# same-slot (no matching) per-anchor point distance distribution
dd = np.linalg.norm(pe - pb, axis=-1).mean(-1)
print("map_pts f0 same-slot: med=%.3f mean=%.3f  frac>0.5m=%.2f" % (np.median(dd), dd.mean(), (dd > 0.5).mean()))
# anchor bank + features f0
for n in ("next_map_anchor", "next_map_conf", "map_instance_feature"):
    a, b = load(ENG, 0, n), load(B2, 0, n)
    if a is not None and b is not None:
        print("%s f0: shape %s/%s rel_l2=%.4g" % (n, a.shape, b.shape,
              np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)))
