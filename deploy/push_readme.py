# -*- coding: utf-8 -*-
"""push README.md to board"""
import os
import paramiko

L = r"<REPO>"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
sftp.put(L + r"\repro\write_readme.sh", "/opt/m0/trt-dev/repro/write_readme.sh")
sftp.close()
_, out, _ = cli.exec_command(
    "sed -i 's/\\r//' /opt/m0/trt-dev/repro/write_readme.sh && "
    "bash /opt/m0/trt-dev/repro/write_readme.sh && "
    "echo '--- verify ---' && head -5 /opt/m0/README.md", timeout=30)
print(out.read().decode("utf-8", "replace"))
cli.close()
