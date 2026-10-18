# -*- coding: utf-8 -*-
"""M10 install 收口断言: ExecStart/EnvironmentFiles/生效参数/自检/FATAL 时效."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import paramiko  # noqa: E402

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root",
            password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


pid = run("systemctl show sp-modelnode@m3 -p MainPID --value").strip()
print("node MainPID:", pid)
print("== ExecStart ==")
print(run("systemctl show sp-modelnode@m3 -p ExecStart --value"
          ).strip()[:400])
print("== EnvironmentFiles ==")
print(run("systemctl show sp-modelnode@m3 -p EnvironmentFiles --value"
          ).strip())
print("== /proc cmdline (生效参数) ==")
print(run("tr '\\0' ' ' < /proc/%s/cmdline; echo" % pid).strip())
print("== 环境变量生效 ==")
print(run("tr '\\0' '\\n' < /proc/%s/environ | grep -E "
          "'SP_BUS_WRAP_LEASE_MS|SP_PLUGIN'" % pid).strip())
print("== 本次启动自检/健康 (按启动时间过滤) ==")
print(run("grep -aE 'SELFTEST|selftest' /var/log/sp/node-m3.log | tail -3"
          ).strip())
print("== FATAL 时效 (最后一条的时间戳 vs 启动时刻) ==")
print(run("grep -a 'FATAL' /var/log/sp/node-m3.log | tail -2 | "
          "cut -c1-80").strip())
print(run("systemctl show sp-modelnode@m3 -p ActiveEnterTimestamp "
          "--value").strip())
print("== 信箱活性双探 (seq 递进) ==")
a = run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 1500 "
        "--duration-ms 800 --quiet", t=20)
b = run("/usr/local/bin/sp_resultmon sp_result_m3 --wait-ms 1500 "
        "--duration-ms 800 --quiet", t=20)
import re  # noqa: E402
ma = re.search(r"last_valid=(\d+)", a)
mb = re.search(r"last_valid=(\d+)", b)
print("probe1:", ma.group(1) if ma else "?",
      "probe2:", mb.group(1) if mb else "?",
      "-> advancing:", bool(ma and mb and int(mb.group(1)) >
                            int(ma.group(1))))
cli.close()
