# -*- coding: utf-8 -*-
"""fetch a remote log to a local path via sftp. args: remote local"""
import os
import sys

import paramiko

remote, local = sys.argv[1], sys.argv[2]
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
sftp.get(remote, local)
sftp.close()
cli.close()
print(f"fetched {remote} -> {local} ({os.path.getsize(local)} B)")
