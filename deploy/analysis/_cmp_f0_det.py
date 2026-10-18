import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np

ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
EV = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
ENG, B2, SP32 = EV + r"\mini_eng_v8", EV + r"\mini_b2", EV + r"\mini_sp32"

def load(d, k, n):
    p = os.path.join(d, ("outv8_%02d" if d is ENG else "out_%02d") % k, n + ".bin")
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

print("files in eng f0:", sorted(os.listdir(os.path.join(ENG, "outv8_00"))))
print()
# det f0: top-10 by cls, bbox 10-dim
ce, cb = load(ENG, 0, "det_cls"), load(B2, 0, "det_cls")
be, bb = load(ENG, 0, "det_bbox"), load(B2, 0, "det_bbox")
print("det_cls f0 shape", ce.shape, cb.shape, "rel_l2=%.4g max|d|=%.4g" % (
    np.linalg.norm(ce - cb) / np.linalg.norm(cb), np.abs(ce - cb).max()))
print("det_bbox f0 rel_l2=%.4g max|d|=%.4g" % (
    np.linalg.norm(be - bb) / np.linalg.norm(bb), np.abs(be - bb).max()))
# per-anchor det bbox abs diff (900,10)
dbe = be.reshape(-1, be.shape[-1]) if be.ndim > 1 else be
dbb = bb.reshape(-1, bb.shape[-1]) if bb.ndim > 1 else bb
d = np.abs(dbe - dbb)
print("det_bbox per-col max|d|:", np.round(d.max(0), 3).tolist())
qe, qb = load(ENG, 0, "det_quality"), load(B2, 0, "det_quality")
print("det_quality f0 rel_l2=%.4g" % (np.linalg.norm(qe - qb) / np.linalg.norm(qb)))
# same for frame 5 (loop already active)
ce5, cb5 = load(ENG, 5, "det_cls"), load(B2, 5, "det_cls")
be5, bb5 = load(ENG, 5, "det_bbox"), load(B2, 5, "det_bbox")
print("det_cls f5 rel_l2=%.4g  det_bbox f5 rel_l2=%.4g" % (
    np.linalg.norm(ce5 - cb5) / np.linalg.norm(cb5),
    np.linalg.norm(be5 - bb5) / np.linalg.norm(bb5)))
