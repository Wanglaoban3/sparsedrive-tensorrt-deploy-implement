import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "work_dirs", "preproc_ref", "m3")


def pctl(v, qs=(0.5, 0.9, 0.99)):
    s = np.sort(np.asarray(v, dtype=float))
    return {q: float(s[min(int(round(q * (len(s) - 1))), len(s) - 1)])
            for q in qs}, float(s.mean())


fp = os.path.join(D, "frame_log.tsv")
if os.path.exists(fp):
    rows = np.genfromtxt(fp, delimiter="\t", names=True)
    seq = np.atleast_1d(rows["seq"])
    cols = {c: np.atleast_1d(rows[c]) for c in
            ("pre_ms", "infer_ms", "post_ms", "gpu_ms", "svc_ms", "acq_ms")}
    ready = np.atleast_1d(rows["ready_ns"])
    n = len(seq)
    span = (ready[-1] - ready[0]) / 1e9
    print("frames=%d ready_span=%.1fs fps(node-ready)=%.2f"
          % (n, span, (n - 1) / span if span > 0 else 0))
    for name in ("pre_ms", "infer_ms", "post_ms", "gpu_ms", "svc_ms",
                 "acq_ms"):
        v = cols[name]
        m = {}
        for tag, mask in [("all", np.ones(n, bool)),
                          ("steady(seq>=10)", seq >= 10)]:
            p, mean = pctl(v[mask])
            m[tag] = "p50=%6.2f p90=%6.2f p99=%6.2f mean=%6.2f" % (
                p[0.5], p[0.9], p[0.99], mean)
        print("%-8s all: %s | steady: %s" % (name, m["all"],
                                             m["steady(seq>=10)"]))
else:
    print("no frame_log.tsv at", fp)

tp = os.path.join(D, "tegra.log")
if os.path.exists(tp):
    gru, freq, ram, ramt = [], [], [], []
    pat_g = re.compile(r"GR3D_FREQ (\d+)%(?:@(\d+))?")
    pat_g2 = re.compile(r"GR3D_FREQ (\d+)%@\[")
    pat_r = re.compile(r"RAM (\d+)/(\d+)MB")
    with open(tp, errors="ignore") as f:
        for ln in f:
            mm = pat_g.search(ln)
            if mm:
                gru.append(int(mm.group(1)))
                if mm.group(2):
                    freq.append(int(mm.group(2)))
            else:
                m2 = pat_g2.search(ln)
                if m2:
                    gru.append(int(m2.group(1)))
            mr = pat_r.search(ln)
            if mr:
                ram.append(int(mr.group(1)))
                ramt.append(int(mr.group(2)))
    if gru:
        a = np.array(gru, dtype=float)
        print("GR3D util: samples=%d mean=%.1f%% p50=%.0f%% p90=%.0f%% "
              "max=%.0f%%" % (len(a), a.mean(), np.percentile(a, 50),
                              np.percentile(a, 90), a.max()))
        if freq:
            fa = np.array(freq, dtype=float)
            print("GR3D freq MHz: p50=%.0f max=%.0f" % (np.percentile(fa, 50),
                                                        fa.max()))
    if ram:
        ra = np.array(ram, dtype=float)
        print("RAM used MB: start=%d end=%d min=%d max=%d / %d"
              % (ra[0], ra[-1], ra.min(), ra.max(),
                 ramt[0] if ramt else -1))
else:
    print("no tegra.log at", tp)
