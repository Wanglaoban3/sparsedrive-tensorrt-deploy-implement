# -*- coding: utf-8 -*-
"""M-PROD Phase C2 8h 浸泡 (spec §7 C2).
用法:
  python _prod_soak.py start [hours=8]   重启 systemd 双单元 → 板端采样器
                                         (60s×N sp_status --csv 行) 启动,
                                         tegrastats 存在则并行录
  python _prod_soak.py poll              只查进度 (SOAK_DONE / 行数)
  python _prod_soak.py report            拉样 + 生成 report.md + 验收线判定
验收线: node RSS 斜率 <1MB/h (稳态=去首 10min); 稳态 forced_recycles 增量=0;
        journal 零 exit 13/14; 信箱 age p99 <400ms (2×200ms 帧周期).
板端坑: heredoc 生成脚本 (CRLF 会静默死); setsid 括号孤儿启动."""
import datetime
import io
import os
import sys
import time

import numpy as np
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "/opt/m0/trt-dev/soak_out"
SH = OUT + "/soak.sh"
SAMPLES = OUT + "/samples.csv"
DONE = OUT + "/SOAK_DONE"
INST = "m3"

SOAK_SH = """#!/bin/bash
# M-PROD C2 soak sampler: 每 60s 一行 sp_status --csv (字段序 = 工具定义),
# tegrastats 存在则并行录. 参数: $1=采样次数
OUT=__OUT__
N=${1:-480}
echo "ts,latest_seq,published,pub_idx,slots_held,ring_depth,forced,nc,\\
mbox_status,age_ms,lage_ms,writes,last_valid,resets60,nan,div,\\
node_pid,node_rss_kb,node_up_s,fs_pid,fs_rss_kb,fs_up_s,gpu_temp_c,\\
cpu_temp_c,sm_clock_mhz" > $OUT/samples.csv
for i in $(seq 1 $N); do
  sleep 60
  S=$(/usr/local/bin/sp_status __INST__ --csv)
  echo "$(date +%s),$S" >> $OUT/samples.csv
done
touch $OUT/SOAK_DONE
"""


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["BOARD_HOST"], username="root",
                password=os.environ["BOARD_PASS"], timeout=15)
    return cli


def run(cli, cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")


def cmd_start():
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    n = int(hours * 60)
    cli = connect()
    # 常驻形态重启 (fresh env, 计数清零), 等 NOMINAL
    run(cli, "systemctl reset-failed sp-filesrc@%s sp-modelnode@%s; "
             "systemctl restart sp-filesrc@%s sp-modelnode@%s"
        % (INST, INST, INST, INST))
    print("等待 NOMINAL ...")
    t0 = time.time()
    while time.time() - t0 < 90:
        if "NOMINAL" in run(cli, "/usr/local/bin/sp_resultmon sp_result_%s "
                                "--wait-ms 500 --duration-ms 400 --quiet"
                                % INST):
            break
        time.sleep(2)
    else:
        raise SystemExit("启动后 90s 无 NOMINAL, 放弃")
    # heredoc 生成采样脚本 (CRLF 防疫: 板上生成)
    run(cli, "rm -rf %s; mkdir -p %s" % (OUT, OUT))
    sh = SOAK_SH.replace("__OUT__", OUT).replace("__INST__", INST)
    sftp = cli.open_sftp()
    with sftp.open(SH, "w") as f:
        f.write(sh)
    sftp.close()
    run(cli, "sed -i 's/\\r$//' %s && chmod +x %s" % (SH, SH))
    # tegrastats 并行录 (不存在优雅跳过)
    has_tegra = run(cli, "command -v tegrastats || echo NO").strip()
    if "NO" not in has_tegra:
        run(cli, "(setsid nohup tegrastats --interval 60000 > %s/tegrastats.log "
                 "2>&1 < /dev/null &); echo TEGRA_GO" % OUT)
        print("tegrastats 录制已启动")
    else:
        print("tegrastats 不存在, 跳过 (spec 允许)")
    # 括号孤儿启动采样器
    o = run(cli, "(setsid nohup bash %s %d > %s/soak_run.log 2>&1 "
                 "< /dev/null &); echo GO" % (SH, n, OUT))
    print("soak 采样器已启动: %d 样 x 60s = %.1fh" % (n, n / 60.0), o.strip())
    print("SOAK_STARTED hours=%g done_marker=%s" % (hours, DONE))
    cli.close()


def cmd_poll():
    cli = connect()
    n = run(cli, "grep -c '' %s 2>/dev/null || echo 0" % SAMPLES).strip()
    done = run(cli, "ls %s 2>/dev/null || echo NO" % DONE).strip()
    tail = run(cli, "tail -1 %s 2>/dev/null" % SAMPLES).strip()
    print("samples_rows=%s done=%s" % (n, "YES" if "SOAK" in done else "no"))
    print("tail:", tail[:200])
    cli.close()


def cmd_report():
    stamp = datetime.date.today().isoformat()
    dst = os.path.join(ROOT, "work_dirs", "soak_%s" % stamp)
    os.makedirs(dst, exist_ok=True)
    cli = connect()
    # journal 退出码统计 (本单元本次浸泡窗口)
    jn = run(cli, "journalctl -u sp-modelnode@%s --since '-10h' --no-pager | "
                  "grep -cE 'status=13|status=14' || true" % INST).strip()
    jf = run(cli, "journalctl -u sp-filesrc@%s --since '-10h' --no-pager | "
                  "grep -cE 'status=13|status=14' || true" % INST).strip()
    sftp = cli.open_sftp()
    sftp.get(SAMPLES, os.path.join(dst, "samples.csv"))
    try:
        sftp.get(OUT + "/tegrastats.log",
                 os.path.join(dst, "tegrastats.log"))
    except IOError:
        print("(无 tegrastats.log)")
    sftp.close()
    cli.close()

    raw = open(os.path.join(dst, "samples.csv"), encoding="utf-8").read()\
        .splitlines()
    hdr = raw[0].split(",")
    rows = [ln.split(",") for ln in raw[1:] if ln.strip()]
    print("samples=%d journal: node 13/14=%s filesrc 13/14=%s"
          % (len(rows), jn, jf))
    col = {nm: i for i, nm in enumerate(hdr)}
    node_rss = np.array([float(r[col["node_rss_kb"]]) for r in rows])
    node_up = np.array([float(r[col["node_up_s"]]) for r in rows])
    age = np.array([float(r[col["age_ms"]]) for r in rows])
    forced = np.array([float(r[col["forced"]]) for r in rows])
    status = [r[col["mbox_status"]] for r in rows]
    lage = np.array([float(r[col["lage_ms"]]) for r in rows])

    # 稳态 = 去首 10min (6 样); node uptime 也应覆盖全程 (无重启)
    steady = node_up >= 600
    n_nominal = sum(1 for s in status if s == "NOMINAL")
    # RSS 斜率: kB/h, 对稳态段线性回归
    if steady.sum() >= 10:
        z = np.polyfit(node_up[steady] / 3600.0, node_rss[steady], 1)
        slope_mb_h = z[0] / 1024.0
    else:
        slope_mb_h = float("nan")
    p99_age = float(np.percentile(age[steady] if steady.sum() else age, 99))
    forced_delta = float(forced[steady][-1] - forced[steady][0]) \
        if steady.sum() >= 2 else float("nan")
    n_degraded = len(rows) - n_nominal

    lines = []
    lines.append("# 8h 浸泡报告 (M-PROD Phase C2, %s)" % stamp)
    lines.append("")
    lines.append("- 样本: %d 行 x 60s; node uptime %.2f->%.2f h"
                 % (len(rows), node_up[0] / 3600, node_up[-1] / 3600))
    lines.append("- 信箱状态: NOMINAL %d/%d (非 NOMINAL %d)"
                 % (n_nominal, len(rows), n_degraded))
    lines.append("- lage p99 = %.0f ms (存活判活年龄, 参考)" % np.percentile(
        lage[steady] if steady.sum() else lage, 99))
    lines.append("")
    lines.append("## 验收线 (spec §7 C2)")
    lines.append("")
    lines.append("| 线 | 实测 | 判定 |")
    lines.append("|---|---|---|")
    lines.append("| node RSS 斜率 <1MB/h | %+.3f MB/h | %s |"
                 % (slope_mb_h, "PASS" if abs(slope_mb_h) < 1 else "FAIL"))
    lines.append("| 稳态 forced_recycles 增量=0 | %g | %s |"
                 % (forced_delta,
                    "PASS" if forced_delta == 0 else "FAIL"))
    lines.append("| 零 exit 13/14 | node=%s filesrc=%s | %s |"
                 % (jn, jf, "PASS" if jn.strip() == "0" and jf.strip() == "0"
                    else "FAIL"))
    lines.append("| 信箱 age p99 <400ms | %.0f ms | %s |"
                 % (p99_age, "PASS" if p99_age < 400 else "FAIL"))
    lines.append("")
    lines.append("## 原始观测")
    lines.append("")
    lines.append("- node RSS 首/尾: %.0f / %.0f kB (稳态段回归)" % (
        node_rss[steady][0] if steady.sum() else node_rss[0],
        node_rss[steady][-1] if steady.sum() else node_rss[-1]))
    lines.append("- gpu temp 范围: %s-%s C" % (
        min(r[col["gpu_temp_c"]] for r in rows),
        max(r[col["gpu_temp_c"]] for r in rows)))
    lines.append("- samples.csv / tegrastats.log 同目录留档")
    rpt = os.path.join(dst, "report.md")
    open(rpt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nreport ->", rpt)
    ok = (abs(slope_mb_h) < 1 and forced_delta == 0
          and jn.strip() == "0" and jf.strip() == "0" and p99_age < 400)
    print("SOAK_REPORT_%s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "start"
    {"start": cmd_start, "poll": cmd_poll, "report": cmd_report}[c]()
