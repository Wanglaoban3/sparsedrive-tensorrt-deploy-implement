# -*- coding: utf-8 -*-
"""M10 数据面 FT 轮 (plan Task 5, 4 用例, 全部独立进程跑, 不经 systemd):
FT1 wrap-lease: 卡死消费者被定向强抢 (forced>=1, 发布不断) + 活消费者不误抢
FT2 skip-lag:   SIGSTOP 跨场景边界 → 跳序 rc0 + scene 复位行; 对照严格模式 FATAL 12
FT3 watchdog:   STALL(返回型)→ABANDON+弃帧继续 rc0; HANG(不返回)→exit13 tier2
FT4 晚接入:     --wait-cons 2 首消费者即开拍 (starting anyway), 第二消费者
                中途入列不扰动发布
用法: set BOARD_HOST=..&& set BOARD_PASS=..&& python _prod_ft_m10.py [all|1|2|3|4]
前置: M10 二进制已装 /usr/local/bin (board_m1 --stage build); 单元会被 stop,
结束统一恢复 (reset-failed + restart + 等 NOMINAL)."""
import os
import re
import sys
import time

import paramiko

INST = "m3"
NODE_U = "sp-modelnode@%s" % INST
FS_U = "sp-filesrc@%s" % INST
MAN = "/opt/m0/trt-dev/nv12_r0/manifest.jsonl"
NROOT = "/opt/m0/trt-dev/nv12_r0"
BB = "/opt/m0/trt-dev/models/e_bb2.engine"
HD = "/opt/m0/trt-dev/models/e_hd.engine"
MP = "/opt/m0/trt-dev/models/e_mp.engine"
PLUG = "/usr/local/lib/libdfaplug_v8.so"
BD = "/opt/m0/trt-dev/ftm10"
NODE_BIN = "/usr/local/bin/sp_modelnode"
FS_BIN = "/usr/local/bin/sp_filesrc"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=90):
    _, o, e = cli.exec_command(cmd, timeout=t)
    try:
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
    except Exception:
        return "[channel-timeout] " + cmd
    return out + err


def clean_ring(ring):
    run("pkill -9 -f 'sp_filesr[c] %s'; pkill -9 -f 'sp_modelnod[e] %s'; "
        "pkill -9 -f 'sp_su[b] %s'; sleep 0.5; rm -rf /dev/shm/sp_%s*"
        % (ring, ring, ring, ring))


def launch(cmd, log):
    # 重定向只由这里做一次: 包大括号组, 组内可含 '; echo rc=$? > rcfile'
    # (若把 '> log' 直接拼在链尾, 重定向会落到 echo 头上 → echo 启动时
    #  截断 node 日志、rc 写进日志而非 rcfile, FT2 首跑实测踩中)
    run("(setsid nohup bash -c '{ %s; } > %s 2>&1 < /dev/null' > /dev/null "
        "2>&1 < /dev/null &); echo GO" % (cmd, log))


def node_cmd(ring, outdir, extra, rcfile):
    # rc 条件化落盘 ($? 紧跟命令, AGENTS.md 板端坑); 无自帶重定向
    return ("%s %s %s %s %s %s --hd %s --mp %s --warmup 2 --frames 81 "
            "--no-dump %s; echo rc=$? > %s"
            % (NODE_BIN, ring, BB, PLUG, MAN, outdir, HD, MP, extra, rcfile))


def fs_cmd(ring, fps, extra):
    return "%s %s 1600 900 %d %s %s --fresh --wait-cons 1 %s" % (
        FS_BIN, ring, fps, MAN, NROOT, extra)


def wait_done(rcfile, budget_s):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        o = run("cat %s 2>/dev/null" % rcfile).strip()
        if o.startswith("rc="):
            return int(o[3:])
        time.sleep(2)
    return None


def wait_line(pattern, log, budget_s):
    t0 = time.time()
    while time.time() - t0 < budget_s:
        o = run("grep -c '%s' %s 2>/dev/null" % (pattern, log)).strip()
        try:
            if int(o.splitlines()[-1]) > 0:
                return True
        except (ValueError, IndexError):
            pass
        time.sleep(1)
    return False


def node_pid(ring):
    # comm 精确匹配: -f 会先命中 wrapper bash (命令行同样含 node 字样,
    # STOP 停 bash 而子进程 node 照跑, FT2 二跑实测) — 只有二进制 comm
    # 是 sp_modelnode; FT 期间单元已停, 全板仅一个 node 在跑
    del ring
    o = run("pgrep -x sp_modelnode | head -1").strip()
    return o.splitlines()[-1] if o else ""


RESULTS = []


def report(idx, name, ok, evidence):
    RESULTS.append((idx, name, ok))
    print("FT-M10-%d %s: %s" % (idx, name, "PASS" if ok else "FAIL"))
    for ln in evidence:
        print("    " + ln)


def prep():
    print("prep: stop units + clean")
    run("timeout 60 systemctl stop %s %s" % (NODE_U, FS_U))
    run("pkill -9 -f 'sp_filesr[c]'; pkill -9 -f 'sp_modelnod[e]'; true")
    run("mkdir -p %s" % BD)
    time.sleep(1)


# ================= FT1: wrap-lease =================
def ft1():
    ring = "ft1"
    clean_ring(ring)
    ev = []
    # -- 卡死持有者: 强抢 + 发布不断 --
    fslog = "%s/fs1.log" % BD
    launch("SP_BUS_WRAP_LEASE_MS=300 " + fs_cmd(ring, 30, "--loop"), fslog)
    launch("%s %s 0 --hold-ms 60000 --no-cuda" % ("/usr/local/bin/sp_sub",
                                                  ring), "%s/sub1.log" % BD)
    time.sleep(20)
    o = run("grep 'filesrc: seq=' %s | tail -1" % fslog)
    m = re.search(r"seq=(\d+) n=(\d+) read\+copy=\d+ms forced=(\d+)", o)
    seq1 = int(m.group(1)) if m else 0
    forced1 = int(m.group(3)) if m else -1
    ev.append("stuck-holder: seq=%d forced=%d (want seq>=300 forced>=1)" %
              (seq1, forced1))
    ok1 = seq1 >= 300 and forced1 >= 1
    clean_ring(ring)
    # -- 对照: 活消费者不误抢 --
    fslog2 = "%s/fs1b.log" % BD
    launch("SP_BUS_WRAP_LEASE_MS=300 " + fs_cmd(ring, 30, "--loop"), fslog2)
    launch("%s %s 0 --no-cuda" % ("/usr/local/bin/sp_sub", ring),
           "%s/sub1b.log" % BD)
    time.sleep(20)
    o = run("grep 'filesrc: seq=' %s | tail -1" % fslog2)
    m = re.search(r"forced=(\d+)", o)
    forced2 = int(m.group(1)) if m else -1
    ev.append("live-holder: forced=%d (want 0)" % forced2)
    ok2 = forced2 == 0
    clean_ring(ring)
    report(1, "wrap-lease", ok1 and ok2, ev)
    return ok1 and ok2


# ================= FT2: skip-lag 跨场景边界 =================
def ft2():
    ring = "ft2"
    ev = []
    ok_all = True
    for tag, extra in (("skip", "--skip-lag"), ("strict", "")):
        clean_ring(ring)
        outd = "%s/out_%s" % (BD, tag)
        nlog = "%s/node_%s.log" % (BD, tag)
        rcfile = "%s/rc_%s" % (BD, tag)
        run("rm -rf %s; rm -f %s %s" % (outd, nlog, rcfile))
        launch(fs_cmd(ring, 5, ""), "%s/fs_%s.log" % (BD, tag))
        launch(node_cmd(ring, outd, extra, rcfile), nlog)
        # 等 "40/81 submitted" → node 刚处理完 k=39 (场景边界 k=40 之前)
        if not wait_line("40/81 submitted", nlog, 60):
            ev.append("%s: never reached 40/81" % tag)
            ok_all = False
            clean_ring(ring)
            continue
        pid = node_pid(ring)
        if not pid:
            ev.append("%s: node pid not found" % tag)
            ok_all = False
            clean_ring(ring)
            continue
        # STOP 后验 /proc state=='T' 确认真停住 (pid 打错 = 白停)
        o = run("kill -STOP %s; sleep 0.4; state=$(cut -d' ' -f3 "
                "/proc/%s/stat); sleep 6; kill -CONT %s; echo ST=$state"
                % (pid, pid, pid), t=30)
        ev.append("%s: stopped state=%s (want T)" %
                  (tag, "T" if "ST=T" in o else "?" + o.strip()[-40:]))
        rc = wait_done(rcfile, 180)
        # mini manifest scene id 从 0 起 (boston=0, queenstown=1)
        o = run("grep -aE 'SKIP lag|scene 0 -> 1|FATAL' %s" % nlog).strip()
        m = re.search(r"skipped=(\d+)", o)
        skipped = int(m.group(1)) if m else 0
        # 终审 I1 后 SKIP 行必须唯一 (seq_base 回滚, 不再逐帧重入 skip 分支)
        n_skip = o.count("SKIP lag")
        if tag == "skip":
            ok = (rc == 0 and skipped >= 20 and n_skip == 1
                  and "scene 0 -> 1" in o and "FATAL" not in o)
            ev.append("skip: rc=%s skipped=%d skip_lines=%d scene_reset=%s "
                      "fatal=%s" % (rc, skipped, n_skip, "scene 0 -> 1" in o,
                                    "FATAL" in o))
        else:
            ok = rc == 12 and "FATAL code=12" in o
            ev.append("strict: rc=%s fatal12=%s (对照)" %
                      (rc, "FATAL code=12" in o))
        ok_all = ok_all and ok
        clean_ring(ring)
    report(2, "skip-lag boundary", ok_all, ev)
    return ok_all


# ================= FT3: watchdog 分级 =================
def ft3():
    ring = "ft3"
    ev = []
    ok_all = True
    # -- a) STALL 返回型 → tier1 弃帧继续 --
    clean_ring(ring)
    outd = "%s/out_3a" % BD
    nlog = "%s/node_3a.log" % BD
    rcfile = "%s/rc_3a" % BD
    run("rm -rf %s; rm -f %s %s" % (outd, nlog, rcfile))
    launch(fs_cmd(ring, 5, ""), "%s/fs_3a.log" % BD)
    launch("SP_WD_TEST_STALL=0 " + node_cmd(ring, outd, "--skip-lag",
                                            rcfile), nlog)
    rc = wait_done(rcfile, 240)
    o = run("grep -E 'ABANDON req|frame abandoned|FATAL' %s" % nlog).strip()
    ok = (rc == 0 and "ABANDON req" in o and "frame abandoned k=5 at=pre"
          in o and "FATAL" not in o)
    ev.append("stall: rc=%s abandon=%s fatal=%s" %
              (rc, "frame abandoned" in o, "FATAL" in o))
    ok_all = ok_all and ok
    clean_ring(ring)
    # -- b) HANG 不返回 → tier2 exit13 --
    clean_ring(ring)
    outd = "%s/out_3b" % BD
    nlog = "%s/node_3b.log" % BD
    rcfile = "%s/rc_3b" % BD
    run("rm -rf %s; rm -f %s %s" % (outd, nlog, rcfile))
    launch(fs_cmd(ring, 5, ""), "%s/fs_3b.log" % BD)
    launch("SP_WD_TEST_HANG=4 " + node_cmd(ring, outd, "--skip-lag",
                                           rcfile), nlog)
    rc = wait_done(rcfile, 120)
    o = run("grep -E 'ABANDON req|FATAL' %s" % nlog).strip()
    ok = (rc == 13 and "FATAL code=13" in o and "watchdog-tier2" in o
          and "ABANDON req" in o)
    ev.append("hang: rc=%s tier2=%s abandon_first=%s" %
              (rc, "watchdog-tier2" in o, "ABANDON req" in o))
    ok_all = ok_all and ok
    clean_ring(ring)
    report(3, "watchdog tiered", ok_all, ev)
    return ok_all


# ================= FT4: --wait-cons 首消费者即开拍 =================
def ft4():
    ring = "ft4"
    clean_ring(ring)
    ev = []
    fslog = "%s/fs4.log" % BD
    launch(fs_cmd(ring, 5, "--wait-cons 2"), fslog)
    time.sleep(2)
    started_early = "start publishing" in run("cat %s" % fslog)
    ev.append("zero-consumer waits=%s (want True)" % (not started_early))
    outd = "%s/out_4" % BD
    nlog = "%s/node_4.log" % BD
    launch(node_cmd(ring, outd, "--skip-lag", "%s/rc_4" % BD), nlog)
    ok1 = wait_line("starting anyway", fslog, 60)
    ev.append("starting-anyway line=%s (want True)" % ok1)
    time.sleep(8)
    launch("%s %s 0 --no-cuda" % ("/usr/local/bin/sp_sub", ring),
           "%s/sub4.log" % BD)
    o1 = run("grep 'filesrc: seq=' %s | tail -1" % fslog)
    time.sleep(12)
    o2 = run("grep -E 'filesrc: seq=|claim timeout' %s | tail -5" % fslog)
    m1 = re.search(r"seq=(\d+)", o1)
    m2 = re.search(r"seq=(\d+)", o2.splitlines()[-1] if o2 else "")
    advanced = (m1 and m2 and int(m2.group(1)) > int(m1.group(1)))
    no_timeout = "claim timeout" not in o2
    ev.append("late-join: seq advanced=%s no_claim_timeout=%s "
              "(want True/True)" % (advanced, no_timeout))
    clean_ring(ring)
    ok = (not started_early) and ok1 and advanced and no_timeout
    report(4, "wait-cons late join", ok, ev)
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    prep()
    fns = {"1": ft1, "2": ft2, "3": ft3, "4": ft4}
    if which == "all":
        for k in ("1", "2", "3", "4"):
            fns[k]()
    else:
        for ch in which:
            fns[ch]()
    # 恢复现场: reset-failed + restart 双单元 + 等 NOMINAL
    print("\nrestore: reset-failed + restart units")
    run("systemctl reset-failed %s %s; systemctl restart %s %s; true"
        % (FS_U, NODE_U, FS_U, NODE_U))
    t0 = time.time()
    ok = False
    while time.time() - t0 < 120:
        o = run("/usr/local/bin/sp_resultmon sp_result_%s --wait-ms 800 "
                "--duration-ms 600 --quiet" % INST, t=20)
        if "v3 status=NOMINAL" in o:
            ok = True
            break
        time.sleep(2)
    print("restore mailbox NOMINAL: %s" % ok)
    print("\n==== FT SUMMARY ====")
    all_ok = ok
    for idx, nm, r in RESULTS:
        print("FT-M10-%d %-22s %s" % (idx, nm, "PASS" if r else "FAIL"))
        all_ok = all_ok and r
    cli.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
