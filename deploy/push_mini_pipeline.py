# -*- coding: utf-8 -*-
"""push repro/mini_pipeline.sh to board, strip CR, verify"""
import os
import paramiko

L = r"<REPO>"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
sftp.put(L + r"\repro\mini_pipeline.sh", "/opt/m0/trt-dev/repro/mini_pipeline.sh")
sftp.close()
_, out, _ = cli.exec_command(
    "sed -i 's/\\r//' /opt/m0/trt-dev/repro/mini_pipeline.sh && "
    "bash -n /opt/m0/trt-dev/repro/mini_pipeline.sh && "
    "echo '--- verify ---' && "
    "grep -E 'ENGINE=|PLUGIN=' /opt/m0/trt-dev/repro/mini_pipeline.sh", timeout=30)
print(out.read().decode("utf-8", "replace"))
cli.close()
