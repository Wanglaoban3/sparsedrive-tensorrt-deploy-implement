# -*- coding: utf-8 -*-
"""fetch e_T12 mini outputs (det_cls/det_quality/det_bbox x81) to local
mini_eng/out7_%02d for eval_t12_mini.py."""
import os
import sys

import paramiko

L = r"<REPO>"
DST = os.path.join(L, "work_dirs", "sparsedrive_small_stage2", "evaldata",
                   "mini_eng")
NAMES = ["det_cls", "det_quality", "det_bbox"]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
base = "/opt/m0/trt-dev/vec/mini"
n = 0
for k in range(81):
    d = f"out7_{k:02d}"
    od = os.path.join(DST, d)
    os.makedirs(od, exist_ok=True)
    for nm in NAMES:
        rp = f"{base}/{d}/{nm}.bin"
        try:
            sftp.get(rp, os.path.join(od, nm + ".bin"))
            n += 1
        except IOError:
            print(f"MISSING {rp}")
            sys.exit(1)
sftp.close()
cli.close()
print(f"fetched {n} bins ({n//3}/3 frames)")
