# -*- coding: utf-8 -*-
"""spdeploy — SparseDrive 板端部署统一入口.

用法: python deploy/spdeploy.py <verb> [args...]
  args 原样透传目标脚本; BOARD_HOST/BOARD_PASS 只从环境读取(运行时注入),
  本工具不落盘、不打印、不拼接进任何日志.

verbs:
  install       ops/prod_install.py     安装/健康断言 (systemd 单元+规则编译)
  uninstall     ops/prod_install.py     卸载 (--uninstall)
  status        (内置)                  只读全景: 信箱 + 三单元 active 态
  gate          ops/mprod_gate.py       精度门禁 (det/map/EPA/L2/col)
  eval          ops/m10_eval_run.py     一键评估 run (gate 全链)
  golden        ops/prod_golden.py      自检金标重标 (--runs 10 --plugin vN)
  ft            ops/prod_ft_m9a.py      故障注入 (--suite a|b|m9a|m10 选路)
  soak          ops/prod_soak.py        8h 浸泡 (满负荷验收 4 判据)
  calib         ops/m9a_thr_calib.py    trigger THR 分位数标定
  parity        ops/m9a_parity.py       trigger 规则 parity 对拍
  boardtest     ops/m9a_boardtest.py    板上板测 (--m10 切 m10_boardtest)
  run           ops/m5_rerun.py         闭环复测 legacy|split [fetch]
  build-plugin  ops/board_mapfix.py     插件/引擎工具链 (默认 tool 段)

薄壳: scripts/sp-<verb>.sh (selftest->golden, build-plugin 同名), 用法
  BOARD_HOST=<ip> BOARD_PASS=<pass> scripts/sp-install.sh [args...]

退出码: 2 用法/缺凭据; 3 status 连接失败; 其余 = 目标脚本原样退出码.
"""
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

DEPLOY = os.path.dirname(os.path.abspath(__file__))

# verb -> (目标脚本, 追加参数); 路径相对 deploy/
VERBS = {
    "install":      ("ops/prod_install.py", []),
    "uninstall":    ("ops/prod_install.py", ["--uninstall"]),
    "gate":         ("ops/mprod_gate.py", []),
    "eval":         ("ops/m10_eval_run.py", []),
    "golden":       ("ops/prod_golden.py", []),
    "ft":           ("ops/prod_ft_m9a.py", []),
    "soak":         ("ops/prod_soak.py", []),
    "calib":        ("ops/m9a_thr_calib.py", []),
    "parity":       ("ops/m9a_parity.py", []),
    "boardtest":    ("ops/m9a_boardtest.py", []),
    "run":          ("ops/m5_rerun.py", []),
    "build-plugin": ("ops/board_mapfix.py", ["tool"]),
}

USAGE = (__doc__ or "").strip()


def _usage_exit(code):
    print(USAGE)
    sys.exit(code)


def _require_creds():
    missing = [k for k in ("BOARD_HOST", "BOARD_PASS") if not os.environ.get(k)]
    if missing:
        print("缺少环境变量: %s (凭据只运行时注入, 不落盘)" % ", ".join(missing))
        _usage_exit(2)


def _pick_ft(argv):
    suite = "m9a"
    for i, a in enumerate(argv):
        if a == "--suite" and i + 1 < len(argv):
            suite = argv[i + 1]
        elif a.startswith("--suite="):
            suite = a.split("=", 1)[1]
    return {
        "a": "ops/prod_ft_a.py",
        "b": "ops/prod_ft_b.py",
        "m9a": "ops/prod_ft_m9a.py",
        "m10": "ops/prod_ft_m10.py",
    }.get(suite, "ops/prod_ft_m9a.py")


def _pick_boardtest(argv):
    if "--m10" in argv:
        return "ops/m10_boardtest.py"
    return "ops/m9a_boardtest.py"


def cmd_status():
    """只读全景: 信箱一行 + 三单元 active 态. 不写板子任何状态."""
    import paramiko
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(hostname=os.environ["BOARD_HOST"],
                    password=os.environ["BOARD_PASS"],
                    username="root", timeout=10)
    except Exception as e:
        print("status: 连接失败 %s: %s" % (os.environ["BOARD_HOST"], e),
              file=sys.stderr)
        sys.exit(3)
    cmds = [
        "/usr/local/bin/sp_status sp_result_m3",
        "systemctl is-active sp-modelnode@m3 sp-filesrc@m3 sp-trigger@m3",
    ]
    rc = 0
    try:
        for c in cmds:
            _, out, err = cli.exec_command(c, timeout=30)
            o = out.read().decode("utf-8", "replace")
            e = err.read().decode("utf-8", "replace")
            r = out.channel.recv_exit_status()
            sys.stdout.write(o)
            if e:
                sys.stderr.write(e)
            if r != 0:
                rc = rc or r
    finally:
        cli.close()
    sys.exit(rc)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _usage_exit(0)
    verb, rest = argv[0], argv[1:]
    if verb == "status":
        _require_creds()
        cmd_status()
    if verb not in VERBS:
        print("未知 verb: %s" % verb)
        _usage_exit(2)
    _require_creds()
    if verb == "ft":
        target = _pick_ft(rest)
    elif verb == "boardtest":
        target = _pick_boardtest(rest)
    else:
        target = VERBS[verb][0]
    script = os.path.join(DEPLOY, target.replace("/", os.sep))
    if not os.path.exists(script):
        print("目标脚本不存在: %s" % script)
        sys.exit(2)
    extra = VERBS[verb][1] if verb in VERBS else []
    sys.exit(subprocess.call([sys.executable, script] + extra + rest))


if __name__ == "__main__":
    main()
