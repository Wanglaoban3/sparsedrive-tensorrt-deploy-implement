# -*- coding: utf-8 -*-
"""push repro/mini_pipeline_t12.sh to board, strip CR, verify"""
import os
import paramiko

L = r"<REPO>"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
sftp.put(L + r"\repro\mini_pipeline_t12.sh",
         "/opt/m0/trt-dev/repro/mini_pipeline_t12.sh")
sftp.close()
_, out, _ = cli.exec_command(
    "sed -i 's/\\r//' /opt/m0/trt-dev/repro/mini_pipeline_t12.sh && "
    "bash -n /opt/m0/trt-dev/repro/mini_pipeline_t12.sh && "
    "echo '--- verify ---' && "
    "grep -E 'ENGINE=|PLUGIN=|OUTD=' "
    "/opt/m0/trt-dev/repro/mini_pipeline_t12.sh", timeout=30)
print(out.read().decode("utf-8", "replace"))
cli.close()
