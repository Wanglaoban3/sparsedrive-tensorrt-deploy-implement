# `deploy/` — 部署工程（七层地图）

上游 SparseDrive checkpoint 变成板上交付物的全部工程都在这层。
完整故事（21 轮优化、归因、坑）见
[docs/OPTIMIZATION_SUMMARY.md](../docs/OPTIMIZATION_SUMMARY.md)；
全链路复刻步骤见 [docs/FULL_CHAIN_REPRO.md](../docs/FULL_CHAIN_REPRO.md)。

## 目录树

```
deploy/
├── spdeploy.py         # 统一 CLI：python deploy/spdeploy.py <verb> [args...]
├── board/              # board_m1.py — paramiko 板端驱动库（launch/push/fetch，
│                       #   括号孤儿启动、绝对路径 marker；ops 全家按路径调用）
├── board_src/          # 推到板上编译的源码单元（板上编译，产物进 /usr/local）
│   ├── prepost/        #   节点/环/信箱/规则/触发器/测试（39 文件，文件名是文档锚）
│   ├── plugin/         #   dfaplug_v3/v8/v11.cu + test_dfa_v11.cu（回退链+v11 配套）
│   └── runner/         #   run_engine.cpp / run_engine2.cpp / run_engines.cpp /
│                       #   onnx2engine.cpp
├── engine/             # 导出与量化链：export_*.py、qat/ptq/flashmha/
│                       #   group_sensitivity、ssa_surgery2.py、_gen_f32_names.py、
│                       #   ONNX_QUANT_STRATEGY.md
├── eval/               # 评估：eval_t6_mini{,_v8,_map}.py、eval_mp_mini.py、
│                       #   eval_nuscenes.py、map_gt.py、infer_*.py、
│                       #   compare_m2/m3.py、run_eval_*_val.bat
├── ops/                # 量产运维（spdeploy verb 的落点）：prod_install/golden/
│                       #   ft_*/soak、m10_*、m9a_*、mprod_gate/mkeval/mkmap、
│                       #   pd_step1-4、m5_rerun、board_mapfix、v11_gate/bench/
│                       #   reginfo、preproc_ref、ref_check
├── analysis/           # 文档点名留档的分析/取证工具（_cmp_*、_probe_*、
│                       #   _inv_verify、_fa_part.cu、selftest_fa*.cu）
├── systemd/            # systemd 单元模板（sp-filesrc@/sp-modelnode@/sp-trigger@）
└── artifacts/          # 评估/剖面证据 JSON + PTQ_SENSITIVITY_REPORT.md
```

## spdeploy verb 表（13 个）

| verb | 落点 | 说明 |
|---|---|---|
| `install` | ops/prod_install.py | 安装/健康断言（systemd 单元+规则编译） |
| `uninstall` | ops/prod_install.py | 卸载（--uninstall） |
| `status` | （内置 paramiko 只读） | 信箱全景 + 三单元 active 态 |
| `gate` | ops/mprod_gate.py | 精度门禁（det/map/EPA/L2/col） |
| `eval` | ops/m10_eval_run.py | 一键评估 run |
| `golden` | ops/prod_golden.py | 自检金标重标 |
| `ft` | ops/prod_ft_m9a.py | 故障注入（`--suite a\|b\|m9a\|m10` 选路，旗标为 dispatcher 所有不透传；**裸 ft = m9a 全 4 案例，对在线单元连杀/STOP**） |
| `soak` | ops/prod_soak.py | 8h 浸泡验收 |
| `calib` | ops/m9a_thr_calib.py | trigger THR 分位数标定 |
| `parity` | ops/m9a_parity.py | trigger 规则 parity 对拍 |
| `boardtest` | ops/m9a_boardtest.py | 板上板测（`--m10` 切 m10 版，旗标不透传） |
| `run` | ops/m5_rerun.py | 闭环复测 legacy\|split [fetch]（仅 CLI，无薄壳） |
| `build-plugin` | ops/board_mapfix.py | 插件/引擎工具链（默认 tool 段） |

薄壳：`scripts/sp-<verb>.sh`（12 个，`sp-selftest.sh`→golden；
`BOARD_HOST=... BOARD_PASS=...` 前缀注入，壳内断言缺失即拒）。

## 加新脚本前先读这个

1. **一次性脚本不入库**：探针/对拍/排查脚本用完即删；结论写进
   docs（OPTIMIZATION_SUMMARY 对应轮次）。历史上 245 个临时脚本
   已清洗过一次，别再攒。
2. **运维脚本必须可被 spdeploy 调**：进 `ops/` 的脚本要么挂进
   `spdeploy.py` 的 VERBS 表，要么是某 verb 的子步骤（pd_step* 形态）；
   去掉 `_` 前缀，名字即文档名。
3. **文档引用必须过 ref_check**：任何 docs/AGENTS 点名的文件名必须
   真实存在（`python deploy/ops/ref_check.py` 零悬空）；删文件前先把
   叙事点名的名字加进 ref_check 的 ALLOW（历史叙事白名单）。
