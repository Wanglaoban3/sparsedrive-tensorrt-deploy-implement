# -*- coding: utf-8 -*-
"""SSA7: wmma 外科级语义验证 (往返 + 单位阵). Marker: SSA7_DONE."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
D = r"<REPO>\deploy"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


s = cli.open_sftp()
s.put(D + r"\selftest_fa3.cu", "/opt/m0/trt-dev/src/selftest_fa3.cu")
s.close()
print("put selftest_fa3 ok")

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa7.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa7_run.log
echo "=== build selftest_fa3 ===" > $LOG
cd src && nvcc -O3 -arch=sm_87 -o /usr/local/bin/selftest_fa3 \
  selftest_fa3.cu > ../st3_build.log 2>&1
rc=$?
cd ..
echo "build rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 st3_build.log >> $LOG
  echo "SSA7_DONE" >> /opt/m0/trt-dev/mods/SSA7_DONE; exit 1; fi
/usr/local/bin/selftest_fa3 >> $LOG 2>&1
echo "SSA7_DONE" >> /opt/m0/trt-dev/mods/SSA7_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa7.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA7_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa7.sh "
              "> ssa7.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa7 job launched")
