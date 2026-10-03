# -*- coding: utf-8 -*-
"""clean up board: keep only e_T6 + libdfaplug_v3 + v5_P1h + mini data"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)

cmds = [
    # 保留: e_T6.engine, v5_P1h.onnx, libdfaplug_v3.so, mini data, core sources
    # 删除: 所有旧实验引擎
    "rm -f /opt/m0/trt-dev/models/e_T3.engine "
    "/opt/m0/trt-dev/models/e_T4.engine "
    "/opt/m0/trt-dev/models/e_T5.engine "
    "/opt/m0/trt-dev/models/e_T7.engine "
    "/opt/m0/trt-dev/models/e_T8.engine "
    "/opt/m0/trt-dev/models/e_T9.engine "
    "/opt/m0/trt-dev/models/e_T10.engine "
    "/opt/m0/trt-dev/models/e_T11.engine "
    "/opt/m0/trt-dev/models/e_dump.engine "
    "/opt/m0/trt-dev/models/e_fp16v2.engine "
    "/opt/m0/trt-dev/models/e_int8_first.engine",
    # 删除: 所有旧实验 ONNX（保留 v5_P1h）
    "cd /opt/m0/trt-dev/models && for f in *.onnx; do "
    "case \"$f\" in v5_P1h.onnx) ;; *) rm -f \"$f\";; esac; done",
    # 删除: 旧插件 .so（保留 v3）
    "rm -f /usr/local/lib/libdfaplug_v2.so "
    "/usr/local/lib/libdfaplug_v4.so "
    "/usr/local/lib/libdfaplug_v5.so "
    "/usr/local/lib/libdfaplug_v6.so "
    "/usr/local/lib/libdfaplug_v7.so",
    # 删除: dump / repro 大文件
    "rm -rf /opt/m0/trt-dev/repro/dump_out "
    "/opt/m0/trt-dev/repro/t3_out /opt/m0/trt-dev/repro/t3x_out "
    "/opt/m0/trt-dev/repro/t6_ref /opt/m0/trt-dev/repro/t8_out "
    "/opt/m0/trt-dev/repro/t9_out /opt/m0/trt-dev/repro/strong_out "
    "/opt/m0/trt-dev/repro/fp16x_out "
    "/opt/m0/trt-dev/t3_out /opt/m0/trt-dev/t3x_out "
    "/opt/m0/trt-dev/fp16x_out",
    # 删除: vec 里非 mini 数据
    "rm -rf /opt/m0/trt-dev/vec/inputs_npzx /opt/m0/trt-dev/vec/ref_npzx "
    "/opt/m0/trt-dev/vec/inputs_chain2_mtq /opt/m0/trt-dev/vec/ref_chain2_mtq",
    # 删除: dfa_real dump 数据
    "rm -rf /opt/m0/trt-dev/dfa_real",
    # 清理: smoke 文件 + 旧 log
    "rm -f /opt/m0/trt-dev/smoke17.* /opt/m0/trt-dev/repro/R*.engine "
    "/opt/m0/trt-dev/repro/*.engine",
    # 显示结果
    "echo === DISK ===; df -h /opt/m0 | tail -1; "
    "echo === MODELS ===; ls -lh /opt/m0/trt-dev/models/; "
    "echo === PLUGINS ===; ls -lh /usr/local/lib/libdfaplug*.so; "
    "echo === VEC ===; du -sh /opt/m0/trt-dev/vec/*/ 2>/dev/null",
]

for cmd in cmds:
    _, out, _ = cli.exec_command(cmd, timeout=60)
    r = out.read().decode("utf-8", "replace").strip()
    if r:
        print(r)

cli.close()
print("CLEANUP_DONE")
