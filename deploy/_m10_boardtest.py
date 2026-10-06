"""_m10_boardtest.py — push given test .cpp + current bus/watch sources,
compile on board, run, print output. Usage:

  set BOARD_HOST=...&& set BOARD_PASS=...&& python deploy\\_m10_boardtest.py test_bus_lease [args...]

Binaries are copied to /usr/local/bin (/opt/m0 is noexec). Exit code = the
board-side test's exit code.
"""
import os
import sys

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPOST = os.path.join(ROOT, "deploy", "prepost")
BDIR = "/opt/m0/trt-dev/prepost"
BIN = "/usr/local/bin"


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

    def run(cmd, t=120):
        _, o, e = cli.exec_command(cmd, timeout=t)
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
        rc = o.channel.recv_exit_status()
        return rc, out, err

    sftp = cli.open_sftp()
    # 测试程序 + 它依赖的现役源码 (始终推当前版本, 不吃板上陈旧拷贝)
    for f in ["sp_bus.h", "sp_bus.cpp", "sp_watch.h", name + ".cpp"]:
        sftp.put(os.path.join(PREPOST, f), BDIR + "/" + f)
    sftp.close()
    build = ("cd %s && g++ -O2 -std=c++14 -Wall -Wextra -pthread %s.cpp "
             "sp_bus.cpp -o %s -lrt && cp -f %s %s/%s"
             % (BDIR, name, name, name, BIN, name))
    rc, out, err = run(build, t=180)
    if rc != 0:
        print("[build rc=%d]\n%s\n%s" % (rc, out[-3000:], err[-3000:]))
        cli.close()
        return 3
    rc, out, err = run("stdbuf -oL %s/%s %s; echo RC=$?" % (BIN, name, args),
                       t=180)
    print(out)
    if err.strip():
        print("[stderr]", err[-2000:])
    # shell 层 RC=$? 能区分信号死亡 (139=SIGSEGV 等); paramiko channel 对
    # 信号死亡给不出 exit status (-1)
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
