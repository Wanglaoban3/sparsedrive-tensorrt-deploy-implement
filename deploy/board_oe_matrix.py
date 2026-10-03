# -*- coding: utf-8 -*-
"""E1/E2/E3: onnx2engine2-built modbig_p2h engines, one flag varied.
E1 = --fp16 --int8 --ws-mb 2048  (e_T12 config; expect slow FN if repro)
E2 = --fp16 --ws-mb 2048         (isolates kINT8)
E3 = --fp16 --int8 --ws-mb 8192  (isolates workspace)"""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run("pkill -f 'trtexec.*modbig_p2h' ; sleep 2; "
          "ps aux | grep trtexec | grep -v grep | wc -l"))
SH = r"""cat > /opt/m0/trt-dev/mods/run_oe.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev/mods
P=/usr/local/lib/libdfaplug_v3.so
T=/usr/src/tensorrt/bin/trtexec
for v in "oe_a --fp16 --int8 --ws-mb 2048" \
         "oe_b --fp16 --ws-mb 2048" \
         "oe_c --fp16 --int8 --ws-mb 8192"; do
  set -- $v
  n=$1; shift
  echo "=== $n start $(date +%T)" >> oe_run.log
  /usr/local/bin/onnx2engine2 modbig_p2h.onnx $n.engine $* \
    --plugins $P > ${n}_build.log 2>&1
  echo "$n build rc=$?" >> oe_run.log
  $T --loadEngine=$n.engine --plugins=$P --dumpProfile \
    --iterations=100 > ${n}_prof.log 2>&1
  echo "$n prof rc=$?" >> oe_run.log
done
touch OE_DONE
EOF
chmod +x /opt/m0/trt-dev/mods/run_oe.sh"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/OE_DONE /opt/m0/trt-dev/mods/oe_run.log")
run("cd /opt/m0/trt-dev/mods && setsid nohup ./run_oe.sh > oe.out 2>&1 "
    "< /dev/null & echo GO", t=10)
print(run("sleep 5; cat /opt/m0/trt-dev/mods/oe.out; "
          "cat /opt/m0/trt-dev/mods/oe_run.log 2>/dev/null"))
cli.close()
