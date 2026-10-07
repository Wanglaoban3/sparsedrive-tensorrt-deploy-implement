# -*- coding: utf-8 -*-
"""M9a Task 2 冒烟: 板上编译 sp_trigger + 前台跑两轮 (SIGTERM 优雅 /
kill -9 续写) + 断言. 前提: 5fps 常驻链 ACTIVE (信箱有活发布者)."""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS")
import paramiko  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(ROOT, "deploy", "prepost")
BD = "/opt/m0/trt-dev/prepost"
W = "/tmp/m9a_t2"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=180):
    _, o, e = cli.exec_command(cmd, timeout=t)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return o.channel.recv_exit_status(), out, err


print(run("mkdir -p %s/rules %s/out" % (W, W))[1])
sftp = cli.open_sftp()
for f in ["sp_trigger.cpp", "sp_result.h", "postproc.h", "sp_rule.h",
          "sp_ruleload.h", "sp_rule_template.c"]:
    sftp.put(os.path.join(PRE, f), "%s/%s" % (BD, f))
sftp.close()
rc, o, e = run("cd %s && g++ -O2 -std=c++14 -Wall -Wextra -pthread "
               "sp_trigger.cpp -o /usr/local/bin/sp_trigger -lrt -ldl"
               % BD)
print("build rc=%d" % rc)
if e.strip():
    print(e[-1500:])
assert rc == 0, "sp_trigger build failed"
run("g++ -shared -fPIC -I%s %s/sp_rule_template.c -o %s/rules/"
    "demo_speed_high.so && printf '[demo_speed_high]\\nspeed_high=-1.0\\n' "
    "> %s/thr.conf" % (BD, BD, W, W))

# ---- run 1: SIGTERM 优雅退出 + ctx 数 == 处理帧数 ----
run("rm -rf %s/out; mkdir -p %s/out" % (W, W))
rc, o, e = run("cd %s && SP_TRIG_RING=m3 SP_TRIG_HZ=10 "
               "SP_TRIG_MANIFEST=/opt/m0/trt-dev/nv12_r0/manifest.jsonl "
               "SP_TRIG_RULES_DIR=%s/rules SP_TRIG_THR=%s/thr.conf "
               "SP_TRIG_OUT=%s/out SP_TRIG_CTX_DUMP=1 "
               "timeout -s TERM 30 /usr/local/bin/sp_trigger" % (W, W, W, W),
               t=90)
print("run1 rc=%d (124=timeout graceful)" % rc)
print(e[-800:])
run("sleep 1")
_, o, _ = run("ls %s/out/ctx 2>/dev/null | wc -l; wc -l < %s/out/events."
              "jsonl 2>/dev/null || echo 0" % (W, W))
n_ctx, n_ev = [x.strip() for x in o.split()]
print("run1: ctx=%s events=%s" % (n_ctx, n_ev))
assert int(n_ctx) > 20, "ctx dump too few"

# ---- run 2: kill -9 后 events.jsonl 续写 (append, 行数单调) ----
_, o, _ = run("wc -l < %s/out/events.jsonl" % W)
before = int(o.strip() or 0)
p = run("(setsid nohup bash -c '{ env SP_TRIG_RING=m3 SP_TRIG_HZ=10 "
        "SP_TRIG_MANIFEST=/opt/m0/trt-dev/nv12_r0/manifest.jsonl "
        "SP_TRIG_RULES_DIR=%s/rules SP_TRIG_THR=%s/thr.conf "
        "SP_TRIG_OUT=%s/out SP_TRIG_CTX_DUMP=0 /usr/local/bin/sp_trigger; } "
        "> %s/run2.log 2>&1 < /dev/null' > /dev/null 2>&1 < /dev/null &); "
        "echo GO" % (W, W, W, W))[1]
time.sleep(6)
run("pkill -9 -x sp_trigger")
time.sleep(2)
run("(setsid nohup bash -c '{ env SP_TRIG_RING=m3 SP_TRIG_HZ=10 "
    "SP_TRIG_MANIFEST=/opt/m0/trt-dev/nv12_r0/manifest.jsonl "
    "SP_TRIG_RULES_DIR=%s/rules SP_TRIG_THR=%s/thr.conf "
    "SP_TRIG_OUT=%s/out SP_TRIG_CTX_DUMP=0 /usr/local/bin/sp_trigger; } "
    ">> %s/run2.log 2>&1 < /dev/null' > /dev/null 2>&1 < /dev/null &); "
    "echo GO" % (W, W, W, W))
time.sleep(10)
run("pkill -TERM -x sp_trigger")
time.sleep(3)
run("pkill -9 -x sp_trigger; true")
_, o, _ = run("wc -l < %s/out/events.jsonl" % W)
after = int(o.strip() or 0)
print("run2: events %d -> %d (grew=%s)" % (before, after, after > before))
_, o, _ = run("grep -c 'trigger: stop' %s/run2.log" % W)
print("graceful stop lines:", o.strip())
print("SMOKE_%s" % ("PASS" if after > before else "FAIL"))
cli.close()
