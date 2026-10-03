# -*- coding: utf-8 -*-
import os
import paramiko
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
_, out, _ = cli.exec_command(
    "rm -rf /opt/m0/trt-dev/vec/dfa_unit /opt/m0/trt-dev/vec/dump_first "
    "/opt/m0/trt-dev/vec/inputs_chain2 /opt/m0/trt-dev/vec/inputs_chain2e "
    "/opt/m0/trt-dev/vec/inputs_first /opt/m0/trt-dev/vec/inputs_temporal "
    "/opt/m0/trt-dev/vec/ref_chain2 /opt/m0/trt-dev/vec/ref_first "
    "/opt/m0/trt-dev/vec/ref_temporal; "
    "du -sh /opt/m0/trt-dev/vec/*/ 2>/dev/null; "
    "df -h /opt/m0 | tail -1", timeout=60)
print(out.read().decode())
cli.close()
