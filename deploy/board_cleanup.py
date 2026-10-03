# -*- coding: utf-8 -*-
"""cleanup board: delete cut probes, list remaining key artifacts"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cmd = (
    "cd /opt/m0/trt-dev/models && rm -f v3_cut*.onnx e_cut*.engine "
    "e_f*.engine e_n*.engine e_opt*.engine e_notf32.engine e_nolt.engine "
    "e_v3i.engine e_v3i_i8.engine e_v3h2.engine e_v3h5.engine "
    "e_v3S1.engine e_v3S2.engine e_v3S3.engine e_v3S4.engine e_dfa1.engine "
    "e_v3_fold5_fp32.engine && "
    "df -h /opt/m0 | tail -1 && echo === && ls -la e_T3.engine e_fp16.engine "
    "e_int8_first.engine sparsedrive_int8_v3_fold5.onnx v3_T3.onnx v3_T2.onnx "
    "v3_V6.onnx v3_DFA2.onnx 2>/dev/null"
)
_, out, _ = cli.exec_command(cmd, timeout=60)
print(out.read().decode("utf-8", "replace"))
cli.close()
