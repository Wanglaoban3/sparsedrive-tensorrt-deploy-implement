# -*- coding: utf-8 -*-
"""M-PROD Phase B 故障注入验收 (spec §9 B 出口条件).
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_ft_b.py [all|1|2|3]

FTB1 篡改金标 (det_cls 4B NaN) → restart → SELFTEST_FAIL 心跳可见
     (status/last_valid=0), 节点存活 (阻止 ACTIVE 语义) → 还原金标 → NOMINAL
FTB2 SP_INJECT_NAN=6 单发 (SP_FPS=1 拉长 DEGRADED_RESET 可见窗) →
     RESET 日志 + nan_hits 累计 ≥1 + 信箱捕到 DEGRADED_RESET → 恢复 NOMINAL
FTB3 SP_INJECT_NAN=0 PERIOD=1 连发 → 复位 >5/60s → LATCH → exit 14 →
     systemd 拉起→自检过→再 latch → 120s×5 StartLimit 熔断 failed → 还原
每用例独立恢复现场 (env 原样 + reset-failed 双单元 + NOMINAL)."""
import os
import sys
import time

import paramiko

INST = "m3"
NODE_U = "sp-filesrc@%s" % INST
NODE_U2 = "sp-modelnode@%s" % INST
ENV_R = "/etc/sp/%s.env" % INST
LOG = "/var/log/sp/node-%s.log" % INST
GOLDEN = "/opt/m0/trt-dev/golden/%s" % INST
GOLDEN_LOCAL = os.path.join("work_dirs", "golden_cal")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    try:
        return o.read().decode("utf-8", "replace") + \
            e.read().decode("utf-8", "replace")
    except Exception:
        return "[channel-timeout] " + cmd


def v3():
    out = run("/usr/local/bin/sp_resultmon sp_result_%s --wait-ms 500 "
              "--duration-ms 400 --quiet" % INST, t=20)
    for ln in out.splitlines():
        if "v3 status=" in ln:
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
            % (NODE_U, NODE_U2, NODE_U, NODE_U2)).split()
    return {"fs_act": o[0], "node_act": o[1],
            "fs_fail": o[2], "node_fail": o[3]}


def wait_nominal(deadline_s):
    t0 = time.time()
    prev = None
    last = None
    while time.time() - t0 < deadline_s:
        s = v3()
        if s and s.get("status") == "NOMINAL":
            sq = int(s.get("last_valid", 0))
            lg = int(s.get("lage", 99999))
            if prev is not None and sq > prev and lg < 1500:
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
    push_env(open("deploy/systemd/%s.env" % INST, "rb").read().decode("utf-8"))
    run("systemctl reset-failed %s %s; systemctl restart %s %s; true"
        % (NODE_U, NODE_U2, NODE_U, NODE_U2))
    ok, s, dt = wait_nominal(120)
    print("  [%s] 恢复 %s (%.1fs)" % (who, "OK" if ok else "FAIL", dt))
    return ok


RESULTS = []


def report(idx, name, ok, ev):
    RESULTS.append((idx, name, ok))
    print("FTB%d %s: %s" % (idx, name, "PASS" if ok else "FAIL"))
    for ln in ev:
        print("    " + ln)


# ---- 前置 ----
u = units()
ok, s, _ = wait_nominal(30)
if not ok:
    raise SystemExit("前置失败: 无 NOMINAL, 先 _prod_install.py")
print("pre-check: NOMINAL (last_valid=%s)" % s.get("last_valid"))
ENV_BASE = open("deploy/systemd/%s.env" % INST, "rb").read().decode("utf-8")
run("systemctl reset-failed %s %s" % (NODE_U, NODE_U2))


# ==== FTB1: 篡改金标 → SELFTEST_FAIL 阻止 ACTIVE ====
def ftb1():
    p = "%s/frame0/det_cls.bin" % GOLDEN
    run("chmod 644 %s; printf '\\x00\\x00\\xc0\\x7f' | dd of=%s bs=1 seek=96 "
        "conv=notrunc 2>/dev/null; chmod 444 %s" % (p, p, p))
    run("truncate -s 0 %s; systemctl restart %s" % (LOG, NODE_U2))
    t0 = time.time()
    s1 = s2 = None
    while time.time() - t0 < 60:
        s = v3()
        if s and s.get("status") == "SELFTEST_FAIL":
            s1 = s
            time.sleep(2)
            s2 = v3()
            break
        time.sleep(1.5)
    u2 = units()
    st_line = run("grep -c 'SELFTEST FAIL' %s || true" % LOG).strip()
    ev = ["probe1=%s" % (s1 or {}),
          "probe2=%s" % (s2 or {}),
          "node active=%s (存活语义), 'SELFTEST FAIL' log=%s"
          % (u2["node_act"], st_line)]
    ok = (s1 is not None and s2 is not None
          and s1.get("last_valid") == "0" and s2.get("last_valid") == "0"
          and u2["node_act"] == "active")
    report(1, "篡改金标 → SELFTEST_FAIL 心跳, 不发有效结果, 节点存活", ok, ev)
    # 还原金标 + 恢复
    sftp = cli.open_sftp()
    sftp.put(os.path.join(GOLDEN_LOCAL, "run0_det_cls.bin"), p)
    sftp.close()
    run("chmod 444 %s" % p)
    ok2 = restore("FTB1")
    RESULTS.append((1, "还原金标恢复", ok2))


# ==== FTB2: SP_INJECT_NAN 单发 → RESET + 计数 + 恢复 ====
def ftb2():
    # SP_FPS=1: 帧周期 1s → DEGRADED_RESET 在信箱上可见 ~2s (帧 n 与脏帧
    # n+1), 探测窗足够
    bad = ENV_BASE.replace("SP_FPS=5", "SP_FPS=1") + "SP_INJECT_NAN=6\n"
    push_env(bad)
    run("truncate -s 0 %s; systemctl restart %s %s" % (LOG, NODE_U, NODE_U2))
    t0 = time.time()
    deg = None
    while time.time() - t0 < 60 and deg is None:
        s = v3()
        if s and s.get("status") == "DEGRADED_RESET":
            deg = s
        time.sleep(0.8)
    nreset = run("grep -c 'safety: RESET' %s || true" % LOG).strip()
    # 等回 NOMINAL
    ok2, s, dt = wait_nominal(90)
    # 实测机制: 注入的 NaN 状态被 mp 引擎 fp16 链在状态边界吸收成
    # 大有限值 → 探测走 state-absmax 发散通道 (div), 输出保持有限
    # (nan 不动). 探测帧信箱 div/nan 任一 ≥1 即证明"注入→探测→复位"
    # 链路成立. 计数是进程累计: SP_FPS=1 下 81 帧清单 ~81s 枯竭 →
    # node exit20 重启清零, 故只看探测帧+RESET 日志+恢复, 不看恢复帧计数.
    d_nan = (deg or {}).get("nan", "0")
    d_div = (deg or {}).get("div", "0")
    hit = ((str(d_nan).isdigit() and int(d_nan) >= 1)
           or (str(d_div).isdigit() and int(d_div) >= 1))
    ev = ["DEGRADED_RESET probe=%s" % (deg or "<missed>"),
          "safety: RESET log hits=%s" % nreset,
          "恢复 NOMINAL=%s (%.1fs), 探测帧计数 nan=%s div=%s"
          % (ok2, dt, d_nan, d_div)]
    ok = (deg is not None and hit and int(nreset or 0) >= 1 and ok2)
    report(2, "SP_INJECT_NAN → 探测→复位→DEGRADED_RESET→恢复有限值", ok, ev)
    ok3 = restore("FTB2")
    RESULTS.append((2, "恢复", ok3))


# ==== FTB3: 连发 NaN → LATCH → exit14 → 熔断 ====
def ftb3():
    bad = ENV_BASE + "SP_INJECT_NAN=0\nSP_INJECT_NAN_PERIOD=1\n"
    push_env(bad)
    run("systemctl reset-failed %s; truncate -s 0 %s; systemctl restart %s"
        % (NODE_U2, LOG, NODE_U2))
    t0 = time.time()
    latch = False
    ex14 = False
    while time.time() - t0 < 150:
        if not latch and "safety: LATCH" in run("grep -m1 'safety: LATCH' "
                                                "%s || true" % LOG):
            latch = True
        if not ex14 and "code=14" in run("grep -m1 'code=14' %s || true"
                                         % LOG):
            ex14 = True
        if latch and ex14 and units()["node_fail"] == "failed":
            break
        time.sleep(1.5)
    j14 = run("journalctl -u %s -n 300 --no-pager | grep -c 'status=14' "
              "|| true" % NODE_U2).strip()
    u2 = units()
    ev = ["LATCH log=%s, exit14 log=%s, journal status=14 hits=%s"
          % (latch, ex14, j14),
          "熔断后 node: act=%s fail=%s (不无限重启)" % (u2["node_act"],
                                                      u2["node_fail"])]
    ok = latch and ex14 and u2["node_fail"] == "failed"
    report(3, "连发 NaN → LATCH → exit14 → StartLimit 熔断保持", ok, ev)
    ok2 = restore("FTB3")
    RESULTS.append((3, "恢复", ok2))


which = sys.argv[1] if len(sys.argv) > 1 else "all"
fns = {"1": ftb1, "2": ftb2, "3": ftb3}
if which == "all":
    for f in (ftb1, ftb2, ftb3):
        f()
else:
    fns[which]()

print("\n==== FTB SUMMARY ====")
bad = 0
for idx, nm, ok in RESULTS:
    print("FTB%d %-30s %s" % (idx, nm, "PASS" if ok else "FAIL"))
    bad += 0 if ok else 1
cli.close()
raise SystemExit(1 if bad else 0)
