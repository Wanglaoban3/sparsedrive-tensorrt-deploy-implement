# -*- coding: utf-8 -*-
"""push mod_p1h/mod_p2h/modbig_p1h/modbig_p2h to the board and launch the
trtexec A/B under nohup (module-level iteration per user directive)."""
import io
import os
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
L = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
BD = "/opt/m0/trt-dev/mods"

RUN_SH = """#!/bin/bash
cd {bd}
rm -f DONE
for f in mod_p1h mod_p2h modbig_p1h modbig_p2h; do
  echo "===== $f ====="
  /usr/src/tensorrt/bin/trtexec --onnx=$f.onnx --fp16 \
      --plugins=/usr/local/lib/libdfaplug_v3.so \
      --dumpProfile --iterations=200 > $f.log 2>&1
  echo "rc=$? for $f"
done
touch DONE
echo ALLDONE
""".format(bd=BD)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cli.exec_command(f"mkdir -p {BD}")[1].channel.recv_exit_status()
sftp = cli.open_sftp()
local_sh = os.path.join(os.path.dirname(L), "run_mods.sh")
with open(local_sh, "w", newline="\n") as f:
    f.write(RUN_SH)
for nm in ("mod_p1h.onnx", "mod_p2h.onnx",
           "modbig_p1h.onnx", "modbig_p2h.onnx"):
    dst = f"{BD}/{nm}"
    sftp.put(os.path.join(L, nm), dst)
    print("pushed", dst, os.path.getsize(os.path.join(L, nm)) // 1024, "KB")
sftp.put(local_sh, f"{BD}/run_mods.sh")
sftp.close()
cmd = (f"sed -i 's/\\r//' {BD}/run_mods.sh && "
       f"chmod +x {BD}/run_mods.sh && "
       f"cd {BD} && nohup bash run_mods.sh > mods_run.log 2>&1 & echo launched")
_, out, err = cli.exec_command(cmd, timeout=30)
print(out.read().decode("utf-8", "replace"))
print(err.read().decode("utf-8", "replace"))
cli.close()
