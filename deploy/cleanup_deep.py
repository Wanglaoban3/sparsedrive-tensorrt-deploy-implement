# -*- coding: utf-8 -*-
"""thorough board cleanup: keep only production files"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)

cmds = [
    # repro: 只留 t6 profile
    "cd /opt/m0/trt-dev/repro && "
    "ls | grep -v 't6_profile\\|t6_prof' | xargs rm -f",
    # src: 只留 v3 插件 + 核心工具
    "cd /opt/m0/trt-dev/src && "
    "ls | grep -v 'dfaplug_v3\\|onnx2engine\\|run_engine' | xargs rm -f",
    # root: 删杂文件
    "cd /opt/m0/trt-dev && "
    "rm -f verbose_build.log build1.log build2.log build3.log build4.log "
    "build5.log build6.log build7.log eng1.log eng2.log eng3.log eng4.log "
    "eng5.log chain_vec_mtq.log e_fp16_temporal.log e_fp16_first.log "
    "e_fp16dec.log e_fp16dec_run.log calib_first.cache smoke17.onnx "
    "smoke17.engine e_int8_first.log "
    "NvCaffeParser.h NvInfer.h NvInferConsistency.h NvInferConsistencyImpl.h "
    "NvInferImpl.h NvInferLegacyDims.h NvInferPlugin.h NvInferPluginUtils.h "
    "NvInferRuntime.h NvInferRuntimeBase.h NvInferRuntimeCommon.h "
    "NvInferSafeRuntime.h NvInferRuntimePlugin.h NvOnnxConfig.h "
    "NvOnnxParser.h NvUffParser.h NvUtils.h",
    # /tmp 测试二进制
    "rm -f /tmp/test_dfa* /tmp/test_v8 /tmp/o2e* /tmp/eps*",
    # 插件: 留 v3 和 v1(原始版以防回退)
    # (已清理过，确认无多余)
    # dfaplug_v3.cu 需要的 include 头文件不能删——恢复到 include/ 下
    "ls /opt/m0/trt-dev/include/ | head -3",
    # 最终状态
    "echo === FINAL ===; "
    "du -sh /opt/m0/trt-dev/; "
    "find /opt/m0/trt-dev -type f | wc -l; "
    "ls -lh /opt/m0/trt-dev/models/ /opt/m0/trt-dev/src/ "
    "/opt/m0/trt-dev/repro/ /opt/m0/trt-dev/vec/; "
    "df -h /opt/m0 | tail -1",
]

for cmd in cmds:
    _, out, _ = cli.exec_command(cmd, timeout=60)
    r = out.read().decode("utf-8", "replace").strip()
    if r:
        print(r)
cli.close()
print("CLEANUP_DONE")
