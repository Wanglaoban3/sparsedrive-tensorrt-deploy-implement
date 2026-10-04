# gen synthetic NV12 frames + manifest for filesrc smoke (WSL-side run)
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "/root/spbuild/nv12_syn"
w, h, n = 1600, 900, 4
os.makedirs(out, exist_ok=True)
lines = ['{"w":%d,"h":%d}' % (w, h)]
for f in range(n):
    d = os.path.join(out, "frame_%02d" % f)
    os.makedirs(d, exist_ok=True)
    cams = []
    for c in range(6):
        y = ((f * 16 + c * 40) % 256)
        uv = ((f * 51 + c * 7) % 256)
        yp = os.path.join(d, "cam_%d.nv12" % c)
        with open(yp, "wb") as fp:
            fp.write(bytes([y]) * (w * h))
            fp.write(bytes([uv]) * (w * h // 2))
        cams.append("frame_%02d/cam_%d.nv12" % (f, c))
    lines.append(json.dumps({
        "frame": f, "scene": 0, "ts_ns": 1533151603547590000 + f * 500000000,
        "cams": cams}))
with open(os.path.join(out, "manifest.jsonl"), "w") as fp:
    fp.write("\n".join(lines) + "\n")
print("wrote", n, "frames ->", out)
