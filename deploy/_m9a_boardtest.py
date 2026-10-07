"""_m9a_boardtest.py — push given test .cpp + m9a/sp sources, compile on
board, run, print output. Usage:

  set BOARD_HOST=...&& set BOARD_PASS=...&& python deploy\\_m9a_boardtest.py test_rule_load [args...]

Deps pushed: sp_rule.h sp_ruleload.h sp_rule_util.h sp_quota.h sp_egoring.h
sp_result.h postproc.h + rules/*.c (rule fixtures compile at test runtime).
Binaries go to /usr/local/bin (/opt/m0 is noexec). Exit code = board-side
test's exit code.
"""
import os
import sys
import glob

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPOST = os.path.join(ROOT, "deploy", "prepost")
BDIR = "/opt/m0/trt-dev/prepost"
BIN = "/usr/local/bin"
DEPS = ["sp_rule.h", "sp_ruleload.h", "sp_rule_util.h", "sp_quota.h",
        "sp_egoring.h", "sp_result.h", "postproc.h", "sp_rule_template.c"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    args = " ".join(sys.argv[2:])
    host = os.environ["BOARD_HOST"]
    pw = os.environ["BOARD_PASS"]
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, username="root", password=pw, timeout=15)

    def run(cmd, t=180):
        _, o, e = cli.exec_command(cmd, timeout=t)
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
        rc = o.channel.recv_exit_status()
        return rc, out, err

    sftp = cli.open_sftp()
    for f in DEPS + [name + ".cpp"]:
        sftp.put(os.path.join(PREPOST, f), BDIR + "/" + f)
    run("mkdir -p %s/rules" % BDIR)
    for f in glob.glob(os.path.join(PREPOST, "rules", "*.c")):
        sftp.put(f, BDIR + "/rules/" + os.path.basename(f))
    sftp.close()
    build = ("cd %s && g++ -O2 -std=c++14 -Wall -Wextra -pthread %s.cpp "
             "-o %s -lrt -ldl && cp -f %s %s/%s"
             % (BDIR, name, name, name, BIN, name))
    rc, out, err = run(build)
    if rc != 0:
        print("[build rc=%d]\n%s\n%s" % (rc, out[-3000:], err[-3000:]))
        cli.close()
        return 3
    rc, out, err = run("stdbuf -oL %s/%s %s; echo RC=$?" % (BIN, name, args))
    print(out)
    if err.strip():
        print("[stderr]", err[-2500:])
    # shell 层 RC=$? 能区分信号死亡 (139=SIGSEGV); paramiko 对信号死亡给 -1
    rcline = [ln for ln in out.splitlines() if ln.startswith("RC=")]
    brc = -2
    if rcline:
        try:
            brc = int(rcline[-1][3:])
        except ValueError:
            pass
    print("[board rc=%d]" % brc)
    cli.close()
    return brc if brc != -2 else rc


if __name__ == "__main__":
    sys.exit(main())
