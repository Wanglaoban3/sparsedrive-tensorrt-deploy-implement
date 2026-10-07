# -*- coding: utf-8 -*-
"""M9a §7.5 一致性门禁驱动: 板上 C 触发器 (ctx_dump+raw events) vs
numpy 参考实现 (_m9a_replay) 全等比对.
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python deploy\\_m9a_parity.py [minutes=3]
留档: work_dirs/preproc_ref/m9a_parity/ (ctx 夹具不拉, report + events)"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
assert os.environ.get("BOARD_HOST") and os.environ.get("BOARD_PASS")
import paramiko  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m9a_replay as rp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(ROOT, "deploy", "prepost")
BD = "/opt/m0/trt-dev/prepost"
W = "/tmp/m9a_parity"
ARC = os.path.join(ROOT, "work_dirs", "preproc_ref", "m9a_parity")
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
THR = os.path.join(ROOT, "deploy", "systemd", "thr.conf")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=240):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.channel.recv_exit_status(), \
        o.read().decode("utf-8", "replace"), \
        e.read().decode("utf-8", "replace")


print("== 板上: 编译规则 + 触发器, ctx_dump 模式跑 %.0fmin ==" % MIN)
run("rm -rf %s && mkdir -p %s/rules %s/out" % (W, W, W))
sftp = cli.open_sftp()
for f in ["sp_trigger.cpp", "sp_result.h", "postproc.h", "sp_rule.h",
          "sp_rule_util.h", "sp_ruleload.h", "sp_egoring.h", "sp_quota.h"]:
    sftp.put(os.path.join(PRE, f), "%s/%s" % (BD, f))
import glob  # noqa: E402
for f in glob.glob(os.path.join(PRE, "rules", "*.c")):
    sftp.put(f, "%s/rules/%s" % (W, os.path.basename(f)))
sftp.put(THR, "%s/thr.conf" % W)
sftp.close()
rc, o, e = run("cd %s/rules && for f in *.c; do g++ -shared -fPIC -I%s $f "
               "-o ${f%%.c}.so || exit 1; done" % (W, BD))
assert rc == 0, "rule compile failed:\n" + e[-1200:]
rc, o, e = run("cd %s && g++ -O2 -std=c++14 -Wall -Wextra -pthread "
               "sp_trigger.cpp -o /usr/local/bin/sp_trigger.parity "
               "-lrt -ldl" % BD)
assert rc == 0, "trigger build failed:\n" + e[-1200:]
rc, o, e = run("cd %s && SP_TRIG_RING=m3 SP_TRIG_HZ=10 SP_TRIG_CTX_DUMP=1 "
               "SP_TRIG_RULES_DIR=%s/rules SP_TRIG_THR=%s/thr.conf "
               "SP_TRIG_OUT=%s/out timeout -s TERM %d /usr/local/bin/"
               "sp_trigger.parity" % (W, W, W, W, int(MIN * 60 + 10)),
               t=int(MIN * 60 + 60))
run("rm -f /usr/local/bin/sp_trigger.parity")
print(e[-700:])

# 拉回 raw events + thr + (ctx 数量统计)
os.makedirs(ARC, exist_ok=True)
sftp = cli.open_sftp()
sftp.get(W + "/out/events_raw.jsonl", os.path.join(ARC, "events_raw.jsonl"))
sftp.get(W + "/thr.conf", os.path.join(ARC, "thr.conf"))
n_ctx_board, _ = run("ls %s/out/ctx | wc -l" % W)[1:]
sftp.close()
cli.close()
print("board ctx files:", n_ctx_board.strip())

# 本地回放: 需要同款 ctx 夹具 → 直接从板上拉 (数量少, 3min ≈ 900 个小文件)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
import shutil  # noqa: E402
ctx_dir = os.path.join(ARC, "ctx")
shutil.rmtree(ctx_dir, ignore_errors=True)  # 清跨代残留 (文件名会复用)
os.makedirs(ctx_dir, exist_ok=True)
n = 0
for fn in sftp.listdir(W + "/out/ctx"):
    sftp.get("%s/out/ctx/%s" % (W, fn), os.path.join(ctx_dir, fn))
    n += 1
sftp.close()
cli.close()
print("fetched ctx:", n)

# 回放 + 比对
replayed = rp.replay_dir(ctx_dir, os.path.join(ARC, "thr.conf"))
raw = [json.loads(l) for l in open(os.path.join(ARC, "events_raw.jsonl"),
                                   encoding="utf-8") if l.strip()]
board = [(r["seq"], r["event"], r["strength"]) for r in raw]

set_r = {(s, e) for s, e, _ in replayed}
set_b = {(s, e) for s, e, _ in board}
only_r = sorted(set_r - set_b)
only_b = sorted(set_b - set_r)
strg_bad = []
brd = {(s, e): g for s, e, g in board}
for s, e, g in replayed:
    if (s, e) in set_b and abs(brd[(s, e)] - g) > 1e-3:
        strg_bad.append((s, e, g, brd[(s, e)]))

ok = not only_r and not only_b and not strg_bad
lines = ["# M9a §7.5 一致性门禁 (%s)" % time.strftime("%Y-%m-%d %H:%M")]
lines.append("")
lines.append("- ctx 夹具: %d tick; 参考实现事件: %d; 板上原始事件: %d"
             % (n, len(replayed), len(board)))
lines.append("- 事件集合 (seq,event): 仅参考=%s 仅板上=%s"
             % (only_r[:10] or "无", only_b[:10] or "无"))
lines.append("- strength |Δ|>1e-3: %s" % (strg_bad[:10] or "无"))
lines.append("")
lines.append("- **判定: %s**" % ("PASS (集合全等 + 强度容差内)"
                              if ok else "FAIL"))
report = "\n".join(lines) + "\n"
open(os.path.join(ARC, "report.md"), "w", encoding="utf-8").write(report)
print(report)
print("PARITY_%s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
