# -*- coding: utf-8 -*-
"""poll mini_pipeline_v8 progress; when MINI_V8_DONE, fetch outv8_* to
work_dirs/sparsedrive_small_stage2/evaldata/mini_eng_v8 and report."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
MARK = "/opt/m0/trt-dev/repro/MINI_V8_DONE"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


deadline = time.time() + int(sys.argv[1]) * 60 if len(sys.argv) > 1 else \
    time.time() + 240 * 60
while time.time() < deadline:
    st = run(f"ls {MARK} 2>/dev/null; tail -2 "
             "/opt/m0/trt-dev/repro/mini_pipeline_v8.log 2>/dev/null")
    if "MINI_V8_DONE" in st:
        print("marker found:\n" + st)
        break
    if "FAIL" in st:
        print("PIPELINE FAILED:\n" + st)
        print(run("tail -20 /opt/m0/trt-dev/repro/mini_v8_*.log 2>/dev/null "
                  "| tail -40"))
        sys.exit(2)
    time.sleep(120)
else:
    print("TIMEOUT waiting for marker; last state:\n" + st)
    sys.exit(3)

print(run("grep -c '^done' /opt/m0/trt-dev/repro/mini_pipeline_v8.log"))
print(run("du -sh /opt/m0/trt-dev/vec/mini/outv8_00"))
cli.close()
print("POLL_DONE")
