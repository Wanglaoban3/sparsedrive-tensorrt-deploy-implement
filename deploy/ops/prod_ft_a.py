# -*- coding: utf-8 -*-
"""M-PROD Phase A 故障注入验收 (spec §5 表, 4 场景): 每场景独立进程/独立恢复.
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_ft_a.py [all|1|2|3|4]
前置: _prod_install.py 已安装且健康. FT 中途改 /etc/sp/m3.env 的场景结束后
一律恢复原 env + reset-failed + restart + 等 NOMINAL."""
import os
import sys
import time

import paramiko

INST = "m3"
NODE_U = "sp-modelnode@%s" % INST
FS_U = "sp-filesrc@%s" % INST
ENV_R = "/etc/sp/%s.env" % INST
NODE_BIN = "/usr/local/bin/sp_modelnode"
LOG = "/var/log/sp/node-%s.log" % INST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_L = os.path.join(ROOT, "deploy", "systemd", "%s.env" % INST)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    try:
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
    except Exception:
        return "[channel-timeout] " + cmd
    return out + err


def v3():
    """读信箱快照: dict(status,age,lage,last_valid) 或 None.
    age = 发布时冻结的处理时延; lage = 读时活年龄 (死信箱上持续增长)."""
    out = run("/usr/local/bin/sp_resultmon sp_result_%s --wait-ms 800 "
              "--duration-ms 600 --quiet" % INST, t=20)
    for ln in out.splitlines():
        if "v3 status=" not in ln:
            continue
        d = {}
        for tok in ln.replace("v3 ", "").split():
            if "=" in tok:
                kk, vv = tok.split("=", 1)
                if kk in ("age", "lage") and vv.endswith("ms"):
                    vv = vv[:-2]
                d[kk] = vv
        return d
    return None


def units():
    o = run("systemctl is-active %s %s; systemctl is-failed %s %s"
            % (FS_U, NODE_U, FS_U, NODE_U)).split()
    return {"fs_act": o[0], "node_act": o[1],
            "fs_fail": o[2], "node_fail": o[3]}


def wait_nominal(deadline_s, min_seq=None):
    """恢复判据 = 连续两次探测 seq 递进 且 末次 lage<1500ms.
    死节点的冻结信箱 status 恒 NOMINAL / age 恒 ~200ms / seq 恒定,
    单读判"恢复"会撞上旧进程死前多发几帧的冻结值 (FT1 假阳性教训);
    seq 两次递进必来自活发布者."""
    t0 = time.time()
    prev = None
    last = None
    while time.time() - t0 < deadline_s:
        s = v3()
        if s and s.get("status") == "NOMINAL":
            sq = int(s.get("last_valid", 0))
            lg = int(s.get("lage", 99999))
            if (prev is not None and sq > prev and lg < 1500
                    and (min_seq is None or sq > min_seq)):
                return True, s, time.time() - t0
            prev = sq
        last = s
        time.sleep(0.3)
    return False, last, deadline_s


def push_env(text):
    sftp = cli.open_sftp()
    with sftp.open(ENV_R, "w") as f:
        f.write(text.replace("\r\n", "\n"))
    sftp.close()
    run("sed -i 's/\\r$//' %s" % ENV_R)


def restore(who):
    print("  [%s] 恢复现场..." % who)
    push_env(open(ENV_L, "rb").read().decode("utf-8"))
    run("systemctl reset-failed %s %s; systemctl restart %s %s; true"
        % (FS_U, NODE_U, FS_U, NODE_U))
    ok, s, dt = wait_nominal(90)
    print("  [%s] 恢复 %s (%.1fs)" % (who, "OK" if ok else "FAIL", dt))
    return ok


RESULTS = []


def report(idx, name, ok, evidence):
    RESULTS.append((idx, name, ok))
    print("FT%d %s: %s" % (idx, name, "PASS" if ok else "FAIL"))
    for ln in evidence:
        print("    " + ln)


# ---- 前置: 系统健康 ----
u = units()
print("pre-check units:", u)
ok, s, _ = wait_nominal(30)
if not ok:
    raise SystemExit("前置失败: 信箱无 NOMINAL, 先跑 _prod_install.py")
print("pre-check mailbox: NOMINAL (last_valid=%s)" % s.get("last_valid"))
run("systemctl reset-failed %s %s" % (FS_U, NODE_U))

# ================= FT1: kill -9 modelnode =================
def ft1():
    # 判据: kill 后先看到 seq 冻结 (死信箱, lage 增长), 再看到 seq 重新
    # 递进 + lage<1500 (新进程在发布). 恢复时间 = 冻结结束时刻.
    pre_seq = 0
    for _ in range(2):
        s = v3()
        if s:
            pre_seq = max(pre_seq, int(s.get("last_valid", 0)))
        time.sleep(0.2)
    run("kill -9 $(systemctl show sp-modelnode@%s -p MainPID --value); true"
        % INST)
    t0 = time.time()
    prev = None
    frozen = 0
    frozen_seq = None
    rec_seq = None
    rec_t = None
    rec_lage = None
    while time.time() - t0 < 15:
        s = v3()
        if s:
            sq = int(s.get("last_valid", 0))
            lg = int(s.get("lage", 99999))
            if rec_t is None:
                if prev is None or sq == prev:
                    frozen += 1
                    frozen_seq = sq
                elif sq > prev and lg < 1500:
                    rec_seq, rec_t, rec_lage = sq, time.time() - t0, lg
            prev = sq
        time.sleep(0.3)
    ev = ["pre_seq=%d 冻结: %d 次探测 seq=%s" % (pre_seq, frozen, frozen_seq),
          "恢复: seq=%s lage=%sms t=%.1fs" % (rec_seq, rec_lage,
                                              rec_t if rec_t else -1)]
    ok = rec_t is not None and rec_t < 15 and frozen >= 1
    report(1, "kill -9 -> 信箱冻结可见 -> 15s 内新进程发布恢复", ok, ev)


# ================= FT2: watchdog stall (exit 13) =================
def ft2():
    env = open(ENV_L, "rb").read().decode("utf-8")
    push_env(env + "SP_WD_TEST_STALL=2\n")
    run("truncate -s 0 %s; systemctl restart %s" % (LOG, NODE_U))
    t0 = time.time()
    fatal = False
    fline = ""
    # FATAL 行格式 = "FATAL code=13 stage=<名>" (stage 打名字如 hd, 非数字)
    while time.time() - t0 < 60:
        out = run("grep -m1 'FATAL code=13' %s || true" % LOG)
        if "FATAL code=13" in out:
            fatal = True
            fline = out.strip().splitlines()[0]
            break
        time.sleep(1.5)
    st13 = run("journalctl -u %s -n 200 --no-pager | grep -c 'status=13' "
               "|| true" % NODE_U).strip()
    ev = ["FATAL in node log: %s" % (fline or "<none>"),
          "journalctl 'status=13' hits: %s (%.0fs)" % (st13,
                                                       time.time() - t0)]
    ok = fatal and st13.isdigit() and int(st13) >= 1
    report(2, "SP_WD_TEST_STALL=2 -> exit 13 -> systemd 拉起恢复", ok, ev)
    push_env(env)  # 立刻还原, 避免重启循环再吃旧 env
    run("systemctl reset-failed %s %s; systemctl restart %s"
        % (FS_U, NODE_U, NODE_U))
    ok2, s, dt = wait_nominal(90)
    ev2 = ["post-restore NOMINAL=%s (%.1fs, last_valid=%s)"
           % (ok2, dt, (s or {}).get("last_valid"))]
    print("FT2 恢复: %s" % ("PASS" if ok2 else "FAIL"))
    for ln in ev2:
        print("    " + ln)
    RESULTS.append((2, "恢复", ok2))


# ================= FT3: manifest 路径错 -> filesrc exit 10 熔断 =================
def ft3():
    env = open(ENV_L, "rb").read().decode("utf-8")
    bad = env.replace("/nv12_r0/manifest.jsonl",
                      "/nv12_r0/manifest_MISSING.jsonl")
    push_env(bad)
    run("truncate -s 0 /var/log/sp/filesrc-%s.log; systemctl restart %s"
        % (INST, FS_U))
    t0 = time.time()
    failed_at = None
    while time.time() - t0 < 90:
        if units()["fs_fail"] == "failed":
            failed_at = time.time() - t0
            break
        time.sleep(1)
    # 退出行在 journal 是 "status=10/n/a"; FATAL 行在 filesrc 自己的 log
    n10 = run("journalctl -u %s -n 100 --no-pager | grep -c 'status=10' "
              "|| true" % FS_U).strip()
    fline = run("grep -m1 'FATAL code=10' /var/log/sp/filesrc-%s.log || true"
                % INST).strip()
    node_state = units()["node_act"]
    ev = ["filesrc failed at %.1fs (5 熔断: ~5x(退出+RestartSec1s))"
          % (failed_at if failed_at else -1),
          "filesrc FATAL line: %s" % (fline or "<none>"),
          "journal 'status=10' hits: %s" % n10,
          "node state while filesrc failed: %s" % node_state]
    ok = (failed_at is not None and "FATAL code=10" in fline
          and n10.isdigit() and int(n10) >= 1)
    report(3, "manifest 错 -> filesrc exit10 -> StartLimit 熔断 failed",
           ok, ev)
    push_env(env)
    # filesrc 失效窗里 node 会 exit20 循环重启并把自己也熔断, 必须
    # 两个都 reset-failed, 否则 node 的 restart 被 StartLimit 拒绝
    run("systemctl reset-failed %s %s; systemctl restart %s %s"
        % (FS_U, NODE_U, FS_U, NODE_U))
    ok2, s, dt = wait_nominal(120)
    print("FT3 恢复: %s (%.1fs)" % ("PASS" if ok2 else "FAIL", dt))
    RESULTS.append((3, "恢复", ok2))


# ================= FT4: 连续 kill 6 次 -> StartLimit 熔断 =================
def ft4():
    run("systemctl reset-failed %s; timeout 60 systemctl restart %s; true"
        % (NODE_U, NODE_U), t=90)
    wait_nominal(60)
    kills = 0
    t0 = time.time()
    failed = False
    while kills < 8 and time.time() - t0 < 150:
        pid = run("systemctl show sp-modelnode@%s -p MainPID --value"
                  % INST).strip()
        if not pid.isdigit() or pid == "0":
            if units()["node_fail"] == "failed":
                failed = True
                break
            time.sleep(0.6)
            continue
        run("kill -9 %s; true" % pid)
        kills += 1
        time.sleep(1.2)
        if units()["node_fail"] == "failed":
            failed = True
            break
    time.sleep(5)
    still = units()
    ev = ["kills=%d (%.0fs), node failed=%s, 恢复期后 state: act=%s fail=%s"
          % (kills, time.time() - t0, failed, still["node_act"],
             still["node_fail"]),
          "pid after cooldown: %r"
          % run("pgrep -x sp_modelnode || echo none").strip()]
    ok = failed and kills >= 5 and still["node_fail"] == "failed"
    report(4, "连续 kill -> StartLimit 熔断保持 failed", ok, ev)
    run("systemctl reset-failed %s; timeout 60 systemctl restart %s; true"
        % (NODE_U, NODE_U), t=90)
    ok2, s, dt = wait_nominal(120)
    print("FT4 恢复: %s (%.1fs)" % ("PASS" if ok2 else "FAIL", dt))
    RESULTS.append((4, "恢复", ok2))


which = sys.argv[1] if len(sys.argv) > 1 else "all"
fns = {"1": ft1, "2": ft2, "3": ft3, "4": ft4}
if which == "all":
    for f in (ft1, ft2, ft3, ft4):
        f()
else:
    fns[which]()

print("\n==== FT SUMMARY ====")
bad = 0
for idx, nm, ok in RESULTS:
    print("FT%d %-24s %s" % (idx, nm, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
cli.close()
raise SystemExit(1 if bad else 0)
