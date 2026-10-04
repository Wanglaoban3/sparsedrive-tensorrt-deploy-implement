# -*- coding: utf-8 -*-
"""节点实际 proj/tmat/dt (probe in_XX.bin) vs 离线文件值."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VM = os.path.join(ROOT, "work_dirs", "preproc_ref", "vecmeta")
np.set_printoptions(precision=3, suppress=True, linewidth=200)

for k in (0, 1):
    raw = np.fromfile(os.path.join(VM, "node_in_%02d.bin" % k), np.float32)
    proj_n = raw[:96].reshape(6, 4, 4)
    tmat_n = raw[96:112].reshape(4, 4)
    dt_n = raw[112]
    proj_f = np.fromfile(os.path.join(VM, "in_%02d_projection_mat.bin" % k),
                         np.float32).reshape(6, 4, 4)
    tmat_f = np.fromfile(os.path.join(VM, "in_%02d_instance_t_matrix.bin" % k),
                         np.float32).reshape(4, 4)
    dt_f = np.fromfile(os.path.join(VM, "in_%02d_time_interval.bin" % k),
                       np.float32)[0]
    print("== frame %d" % k)
    print("  proj |diff|max = %.6g  (cam0 row1 node=%s file=%s)" %
          (np.abs(proj_n - proj_f).max(), proj_n[0, 1], proj_f[0, 1]))
    print("  tmat |diff|max = %.6g" % np.abs(tmat_n - tmat_f).max())
    print("  dt node=%.7f file=%.7f" % (dt_n, dt_f))
print("PROBE_CMP_DONE")
