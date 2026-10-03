# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path

W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2\outs_replay")
for f in sorted(W.iterdir()):
    print(f.name, hashlib.md5(f.read_bytes()).hexdigest()[:16],
          int(f.stat().st_mtime))
