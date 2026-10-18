# -*- coding: utf-8 -*-
"""M9a FT 轮 (4 用例, 前提: _prod_install.py 已装, trigger 单元 ACTIVE):
FT1 crash:     kill -9 trigger → systemd 拉起 → events.jsonl 续写 + 关键链无感
FT2 drop-in:   新规则 .so + SIGHUP → 热加载; 坏 ABI .so → REJECT + 宿主不倒
FT3 零扰动:    trigger 启停全程 filesrc 节拍 / 信箱 forced=0 / 零 FATAL
FT4 轮转:      events.jsonl >10MB → 5 份轮转不写满
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_ft_m9a.py [all|1|2|3|4]"""
import os
import re
import sys
import time

import paramiko

INST = "m3"
TRIG_U = "sp-trigger@%s" % INST
TRIG_BIN = "/usr/local/bin/sp_trigger"
TRIG_OUT = "/var/lib/sp/trigger"
RULES_DIR = "/usr/local/share/sp/rules"
BD = "/opt/m0/trt-dev/ftm9a"
PREPOST_BD = "/opt/m0/trt-dev/prepost"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, o, e = cli.exec_command(cmd, timeout=t)
    try:
        return o.read().decode("utf-8", "replace") + \
            e.read().decode("utf-8", "replace")
    except Exception:
        return "[channel-timeout] " + cmd


RESULTS = []


def report(idx, name, ok, ev):
    RESULTS.append((idx, name, ok))
    print("FT-M9a-%d %s: %s" % (idx, name, "PASS" if ok else "FAIL"))
    for ln in ev:
        print("    " + ln)


def trig_pid():
    o = run("systemctl show %s -p MainPID --value" % TRIG_U).strip()
    return o.strip()


def wait_trig_active(budget=30):
    t0 = time.time()
    while time.time() - t0 < budget:
        if "active" in run("systemctl is-active %s" % TRIG_U):
            return True
        time.sleep(1)
    return False


def prep():
    run("mkdir -p %s" % BD)
    run("timeout 60 systemctl reset-failed %s; true" % TRIG_U)


# ---- FT1: crash → 拉起 → 续写 ----
def ft1():
    ev = []
    run("timeout 60 systemctl restart %s" % TRIG_U)
    if not wait_trig_active():
        report(1, "crash-restart", False, ["trigger not active"])
        return False
    # 确定性事件源: 临时注入每帧必发规则 (新进程配额状态清零 → 首帧即发),
    # 否则 mini 回环 + 配额冷却下 17s 窗口可能零事件 (断言变赌数据)
    run("cp %s/sp_rule_template.c %s/sp_rule_ft1.c && "
        "sed -i 's/demo_speed_high/demo_ft1/g' %s/sp_rule_ft1.c && "
        "g++ -O2 -shared -fPIC -I%s %s/sp_rule_ft1.c -o %s/sp_rule_ft1.so"
        % (PREPOST_BD, PREPOST_BD, PREPOST_BD, PREPOST_BD, PREPOST_BD,
           RULES_DIR))
    run("cp /etc/sp/thr.conf %s/thr.bak && printf '[demo_ft1]\\n"
        "speed_high=-1.0\\n' >> /etc/sp/thr.conf" % BD)
    run("kill -HUP %s" % trig_pid())
    time.sleep(4)
    o = run("wc -l < %s/events.jsonl 2>/dev/null || echo 0"
            % TRIG_OUT).strip()
    n0 = int(o or 0)
    pid = trig_pid()
    run("kill -9 %s" % pid)
    time.sleep(4)  # Restart=on-failure 1s + attach
    ok_active = wait_trig_active(20)
    pid2 = trig_pid()
    time.sleep(8)
    o = run("wc -l < %s/events.jsonl 2>/dev/null || echo 0"
            % TRIG_OUT).strip()
    n1 = int(o or 0)
    lage = run("/usr/local/bin/sp_resultmon sp_result_%s --wait-ms 1500 "
               "--duration-ms 800 --quiet" % INST, t=20)
    lagems = re.search(r"lage=(\d+)ms", lage)
    lage_ok = lagems and int(lagems.group(1)) < 3000
    ev.append("pid %s -> %s (new=%s)" % (pid, pid2, pid != pid2))
    ev.append("events.jsonl %d -> %d (续写=%s)" % (n0, n1, n1 > n0))
    ev.append("mailbox lage=%s (关键链无感=%s)" %
              (lagems.group(1) + "ms" if lagems else "?", bool(lage_ok)))
    # 清场: 还原 thr.conf + 摘掉 demo 规则
    run("cp %s/thr.bak /etc/sp/thr.conf && rm -f %s/sp_rule_ft1.so"
        % (BD, RULES_DIR))
    run("kill -HUP %s" % trig_pid())
    ok = ok_active and pid != pid2 and n1 > n0 and lage_ok
    report(1, "crash-restart", ok, ev)
    return ok


# ---- FT2: drop-in 热加载 + 坏 ABI 拒载 ----
BAD_SRC = ('#include "sp_rule.h"\n'
           'static int ev0(const sp_frame_ctx*, sp_rule_event*) { return 0; }\n'
           'extern "C" const sp_rule_desc* sp_rule_query(void) {\n'
           '  static const sp_rule_desc d = {999, "bad_abi", "1.0.0", 0, '
           'ev0, 0};\n  return &d;\n}\n')


def ft2():
    ev = []
    run("timeout 60 systemctl restart %s" % TRIG_U)
    wait_trig_active()
    time.sleep(3)
    # 基线 = 服务本次启动的权威行 (grep 'rescan' 会被 FT1 清场残留行误导)
    o = run("grep -a 'trigger: [0-9]* rules' /var/log/sp/trigger-%s.log | "
            "tail -1" % INST)
    n0 = int(re.search(r"(\d+) rules", o).group(1)) if "rules" in o else 15
    # drop-in: 模板换名 (demo_dropin, 阈值段不存在 → 内建默认 12 m/s)
    run("cp %s/sp_rule_template.c %s/sp_rule_dropin_demo.c" %
        (PREPOST_BD, PREPOST_BD))
    run("sed -i 's/demo_speed_high/demo_dropin/g' %s/sp_rule_dropin_demo.c"
        % PREPOST_BD)
    run("g++ -O2 -shared -fPIC -I%s %s/sp_rule_dropin_demo.c -o %s/"
        "sp_rule_dropin_demo.so" % (PREPOST_BD, PREPOST_BD, RULES_DIR))
    pid = trig_pid()
    run("kill -HUP %s" % pid)
    time.sleep(3)
    o = run("grep -a 'rescan' /var/log/sp/trigger-%s.log | tail -1" % INST)
    m = re.search(r"rescan -> (\d+) rules", o)
    n1 = int(m.group(1)) if m else -1
    loaded = run("grep -ac 'loaded demo_dropin' /var/log/sp/trigger-%s.log"
                 % INST).strip()
    ok1 = n1 == n0 + 1 and int(loaded or 0) >= 1
    ev.append("drop-in: rules %d -> %d, demo_dropin loaded=%s" %
              (n0, n1, int(loaded or 0) >= 1))
    # 坏 ABI: 拒载 + 宿主不倒
    run("cat > %s/bad_abi.c <<'EOF'\n%sEOF" % (PREPOST_BD, BAD_SRC))
    run("g++ -shared -fPIC -I%s %s/bad_abi.c -o %s/bad_abi.so"
        % (PREPOST_BD, PREPOST_BD, RULES_DIR))
    run("kill -HUP %s" % pid)
    time.sleep(3)
    rej = run("grep -a 'REJECT %s/bad_abi.so' /var/log/sp/trigger-%s.log | "
              "tail -1" % (RULES_DIR, INST)).strip()
    o = run("grep -a 'rescan' /var/log/sp/trigger-%s.log | tail -1" % INST)
    m = re.search(r"rescan -> (\d+) rules", o)
    n2 = int(m.group(1)) if m else -1
    still = "active" in run("systemctl is-active %s" % TRIG_U)
    ok2 = "abi mismatch" in rej and n2 == n1 and still
    ev.append("bad-abi: REJECT 行=%s, rules 仍=%d, 宿主存活=%s" %
              ("abi mismatch" in rej, n2, still))
    run("rm -f %s/bad_abi.so %s/sp_rule_dropin_demo.so" % (RULES_DIR,
                                                           RULES_DIR))
    run("kill -HUP %s" % pid)
    ok = ok1 and ok2
    report(2, "drop-in+bad-abi", ok, ev)
    return ok


# ---- FT3: trigger 启停对关键链零扰动 ----
def ft3():
    ev = []

    def window():
        a = run("grep -a 'filesrc: seq=' /var/log/sp/filesrc-%s.log | tail -1"
                % INST).strip()
        forced_a = re.search(r"forced=(\d+)", a)
        time.sleep(15)
        b = run("grep -a 'filesrc: seq=' /var/log/sp/filesrc-%s.log | tail -1"
                % INST).strip()
        forced_b = re.search(r"forced=(\d+)", b)
        seqa = int(re.search(r"seq=(\d+)", a).group(1))
        seqb = int(re.search(r"seq=(\d+)", b).group(1))
        return seqb - seqa, int(forced_b.group(1)) - int(forced_a.group(1))

    run("timeout 60 systemctl restart %s" % TRIG_U)
    wait_trig_active()
    s1, f1 = window()  # trigger ACTIVE
    run("timeout 60 systemctl stop %s" % TRIG_U)
    s2, f2 = window()  # trigger STOPPED
    run("timeout 60 systemctl restart %s" % TRIG_U)
    wait_trig_active()
    s3, f3 = window()  # 再 ACTIVE
    fatal = run("tail -50 /var/log/sp/filesrc-%s.log | grep -c 'claim "
                "timeout' || true" % INST).strip()
    ok = (f1 == 0 and f2 == 0 and f3 == 0 and s1 > 20 and s2 > 20
          and s3 > 20 and int(fatal or 0) == 0)
    ev.append("forced 增量: on=%d off=%d on=%d (全 0)" % (f1, f2, f3))
    ev.append("15s seq 前进: %d/%d/%d (节拍不受扰)" % (s1, s2, s3))
    ev.append("filesrc claim timeout=%s" % fatal)
    report(3, "zero-disturbance", ok, ev)
    return ok


# ---- FT4: 事件流轮转 ----
def ft4():
    ev = []
    run("rm -rf %s/rot && mkdir -p %s/rot" % (BD, BD))
    # 预填 11MB 事件 (轮转阈值 10MB)
    run("dd if=/dev/zero of=%s/rot/events.jsonl bs=1M count=11 2>/dev/null; "
        "printf 'x\\n' >> %s/rot/events.jsonl" % (BD, BD))
    run("cd %s && SP_TRIG_RING=m3 SP_TRIG_HZ=10 SP_TRIG_RULES_DIR=%s "
        "SP_TRIG_THR=/etc/sp/thr.conf SP_TRIG_OUT=%s/rot "
        "timeout -s TERM 12 %s" % (BD, RULES_DIR, BD, TRIG_BIN), t=60)
    o = run("ls -la %s/rot/ | tail -8" % BD).strip()
    ev.append(o)
    rotated = ("events.jsonl.1" in o)
    ok = rotated
    report(4, "jsonl rotation", ok, ev)
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    prep()
    fns = {"1": ft1, "2": ft2, "3": ft3, "4": ft4}
    for ch in ("1234" if which == "all" else which):
        fns[ch]()
    print("\n==== FT-M9a SUMMARY ====")
    all_ok = True
    for idx, nm, r in RESULTS:
        print("FT-M9a-%d %-18s %s" % (idx, nm, "PASS" if r else "FAIL"))
        all_ok = all_ok and r
    cli.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
