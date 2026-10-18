# 仓库重整设计（spec）

- 日期：2026-10-18
- 状态：待用户审阅（brainstorming 产出，批准后进 writing-plans）
- 决策人：用户（4 项方向决策已通过问答确认，全部采纳推荐项）
- 范围：仓库宿主侧（deploy/、根目录文档、入口形态）。**板上产物与常驻态零触碰**。

## 0. 背景与动机

项目经 21 轮优化迭代后，`deploy/` 积累了 ~370 个 tracked 文件：约 95 个
`board_*` 一次性 paramiko 驱动、`ssa1-8`/`_m6b_*`/`_m7a_*`/`make_*`/`trace_*`/
`probe_*` 等历史会话脚本，以及 7 篇根目录/部署文档中已过时的第八轮单引擎
叙事。用户明确否定"几十个 py 命令文件"的形态，要求：

1. 明确命名的 sh 入口去做明确的事；
2. 整理项目架构；
3. README 亮点前置，不必遵循旧架构。

上一轮（2610b3d）已清完 210 个 untracked 垃圾；本轮处理 tracked 历史脚本、
目录分层、统一入口与文档。所有被删脚本均可从 git 历史找回，无数据丢失。

## 1. 已锁定的四项决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 入口形态 | 统一 `deploy/spdeploy.py` CLI（dispatch 转发，不重写逻辑）+ `scripts/sp-*.sh` 薄壳入口 |
| 2 | README | 中文主 README，亮点前置，顶部一行英文 tagline；`README_CN.md` 删除并入 |
| 3 | ~200 个 tracked 历史脚本 | 直接删（git 历史留档），不做 archive 目录 |
| 4 | 架构深度 | 清洗 + `deploy/` 内部分层 + 统一入口；不动 `tools/`、`projects/`、`repro/`（上游/板端形态已对） |

## 2. 目标布局

```
├── README.md               # 中文重写：亮点首屏 + 指标 + 架构 + 快速上手 + 文档地图
├── AGENTS.md               # agent 速查（gitignored），全量路径同步，不入库
├── docs/                   # 6 篇 + superpowers 归档；OPTIMIZATION_SUMMARY 历史轮次不改
├── scripts/                # 上游 5 个 sh 不动 + 新增 12 个 sp-*.sh 薄壳
├── repro/                  # 板端链脚本（mini_pipeline*.sh 等 7 个），不动
├── tools/, projects/       # 上游 SparseDrive，不动
└── deploy/
    ├── spdeploy.py         # 【新增】统一 CLI dispatcher（argparse 子命令 → ops/ 脚本转发）
    ├── README.md           # 重写：deploy/ 目录地图 + 常用命令
    ├── board/              # board_m1.py（paramiko 驱动库，被 ops 全家 import）
    ├── board_src/          # 板上编译单元（push 到板子编译的东西）
    │   ├── prepost/        #   ← deploy/prepost/* 原结构平移（节点/环/信箱/规则/触发器/测试）
    │   ├── plugin/         #   ← dfaplug_v3/v8/v11.cu + test_dfa_v11.cu（回退链 + v11 配套测试）
    │   └── runner/         #   ← run_engine.cpp / run_engine2.cpp / run_engines.cpp / onnx2engine.cpp
    ├── engine/             # 导出与量化链：export_*.py、qat/ptq/flashmha/group_sensitivity/
    │                       #   task_sentinel/render_ptq_report/smoke_test/test_flashmha_qkv、
    │                       #   ssa_surgery2.py、_gen_f32_names.py、ONNX_QUANT_STRATEGY.md
    ├── eval/               # 评估：eval_t6_mini{,_v8,_map}.py、eval_mp_mini.py、eval_nuscenes.py、
    │                       #   eval_sp32_mini_det.py、infer_sp32_mini{,_mp_mini}.py、
    │                       #   infer_b2_int8fp16_mini.py、map_gt.py、_mk_evaldir.py、
    │                       #   _mk_scene_split.py、_seed_f0.py、_seg_feed.py、
    │                       #   compare_m2/m3.py、run_eval_*_val.bat
    ├── ops/                # 量产运维（spdispatch 的落点）：board_m1.py 之外的全部活跃驱动——
    │                       #   prod_install/golden/ft_*/soak、m10_*、m9a_*（含 probe_ctx/man/
    │                       #   replay/parity/smoke/thr_calib/unwrap_check/boardtest）、
    │                       #   mprod_gate/mkeval/mkmap、pd_step1-4（gate 链）、m5_rerun、
    │                       #   board_mapfix、v11_gate/bench/reginfo、preproc_ref、ref_check
    ├── analysis/           # 文档点名留档的分析/取证工具：_cmp_drift_curve、_cmp_f0_detail、
    │                       #   _cmp_f0_det、_cmp_order_invariant、_probe_dumpmeta、
    │                       #   _probe_decode_check、_probe_m4_report、_inv_verify、
    │                       #   _fa_part.cu、selftest_fa{,2,3,4}.cu
    ├── systemd/            # 7 文件不动（路径由 prod_install 引用，同步其内部相对路径）
    └── artifacts/          # 评估证据 JSON + PTQ_SENSITIVITY_REPORT.md + onnx2engine_board.cpp，不动
```

命名规则：move 进 `ops/`、`eval/` 的脚本**去掉 `_` 前缀**（如
`_prod_install.py → ops/prod_install.py`、`_m10_eval_run.py → ops/m10_eval_run.py`）；
`board_src/prepost/` 内部文件名全部不动（板上编译脚本/文档引用以这些名为锚）。

## 3. Phase A：大清洗（删除规则与名单）

### 3.1 判定器（对每个"默认删"候选执行）

1. 全仓 tracked 文本（docs/、AGENTS.md、deploy/ 保留者、scripts/、repro/、
   README）grep 该文件名：
   - 被 `docs/*.md` 或 AGENTS.md 引用 → 改判保留，落 `deploy/analysis/`；
   - 被保留 `.py` import → 改判保留，跟随 importer 所在目录；
   - 两者皆无 → 删。
2. 删除一律 `git rm`（历史即归档，不搬运到 archive 目录）。

### 3.2 默认删名单（按模式，例外单列）

| 模式/家族 | 数量（约） | 说明 |
|---|---|---|
| `board_*.py` | ~95 | 一次性 paramiko 驱动（含 `board_ssa*`）；**例外保留** `board_m1.py`（驱动库）、`board_mapfix.py`（mapfix 会话一键） |
| `_pull_ssa{1,4,5,6}.py` + `_ssa_recon{1..4}.py` + `ssa_surgery.py` + `_ss_part.cu` + `_dump_ssa.py` | 11 | ssa 手术系列非 board 部分（board_ssa* 已计入上行）；**例外保留** `ssa_surgery2.py`、`_fa_part.cu`（AGENTS.md 点名留档，归 analysis/） |
| `_m6*/_m7*/_m8*` | 13 | 已定案实验的 gate/取证（dual/M8 DMA 负收益留档在文档） |
| `make_*.py`（dump_onnx{,2,3}/modq/p1h/tool_patch/v5_p2h/v5p1h/v5t3/v6p1h/v6t3） | 11 | 历史图手术 |
| `trace_*.py` | 7 | 层级 trace 取证 |
| `probe_closures/fa_io/mul2/mul2b/zero_dims` + `_probe3/4/build/check/cmp/diag/img/infer/refs/shapes/split` | 16 | probe 取证；**例外保留** `_probe_dumpmeta/_probe_decode_check/_probe_m4_report`（上轮入库名单） |
| `check_*.py`（6 个）+ `build_only_test.py` | 7 | 历史语义/一致性检查 |
| `cleanup_*`（3）+ `_backup_and_clean/_organize_sp/_rename_sp/_doc_scan/_du_deep/_du_workdirs/find_tools/census_det` | 11 | 历次清理自身的工具 |
| `_cmp_detout/fold5/mapcls/outs`、`_diff_*`（4）、`_chk_*`（2）、`_diag3`、`_mk_boundary`、`_mk_eval_sp`、`_final_check`、`_hash_replay`、`_replay_check`、`_runner_err`、`_io_raw`、`_fpn_wiring`、`_fix1/2`、`_ego_dep`、`_preo_shape`、`_bx` | 24 | 历史对拍/修图/诊断脚本 |
| `fetch_out7`、`push_mini_pipeline*.py`、`push_readme.py` | 4 | 被 board_mapfix/run_engine 链替代 |
| `convert_engine_dump{,2}`、`cmp_native_vs_onnx`、`extract_dfa_consts/region`、`fix_mixed_types`、`fold2`、`reformat_rows`、`strip_casts`、`split_analysis/engine/head`、`dfastub`、`sim_dfa_v8`、`verify_inputs`、`verify_p2h_stub`、`list_*`(3)、`module_report`、`analyze_spans`、`profile_buckets`、`_prof_*`(5)、`_rd_metrics` | 28 | 历史分析/变换 |
| `dfaplug_v9/v10.cu`、`dfaplug_dump.cu`、`dfa_timer.cu`、`test_dfa_{ab,kernels,pure,real,real2}.cu` | 9 | 废弃插件版本与非交付测试；**例外保留** `test_dfa_v11.cu`（交付插件配套） |
| `eval_t3_mini/eval_t12_mini/eval_t6_mini_sp.py` | 3 | 废弃引擎代次的评估 |
| `export_v6.py`、`onnx2engine3.cpp`、`qdq_onnx_rewrite.py`、`requant_fpn.py`、`regen_mini_meta.py`、`prep_mini_inputs.py`、`preprocess_dump{,2}.py`、`_fa_draft.cu`、`_gen_m9a_rules.py` | 10 | 边界案例默认删（判定器兜底；`_gen_m9a_rules.py` 为规则生成器，14 个规则 .c 已是库内实体，改规则直接改 .c） |
| 根目录 `README_CN.md` | 1 | 并入新主 README |
| 根目录 `TECH_DETAILS.md` / `TECH_DETAILS_CN.md` | 2 | 实施时比对新 README/docs 覆盖度：独有内容并入 `docs/tech_details.md`（中文一份），否则删 |

合计约 252 个删除（含例外判定后的浮动）。**上轮已确认无 tracked `.log`。**

### 3.3 明确保留（不被误删的锚）

`board_m1.py`、`board_mapfix.py`、prepost/ 全部、dfaplug_v3/v8/v11.cu、
`test_dfa_v11.cu`、run_engine{,2}/run_engines/onnx2engine.cpp、export_v5/
export_mp_onnx/export_fp32_onnx/export_quant_onnx、量化工具链 8 个（qat/
ptq_sensitivity/flashmha_qkv/group_sensitivity/task_sentinel/render_ptq_report/
smoke_test/test_flashmha_qkv——263a1eb 误删恢复件，幸存脚本 import 它们）、
`ssa_surgery2.py`、`_fa_part.cu`、`selftest_fa{,2,3,4}.cu`、`_probe_dumpmeta/
_probe_decode_check/_probe_m4_report`、eval 全家（见 §2）、infer 三件、
ops 活跃全家（见 §2）、analysis 名单（见 §2）、systemd/、artifacts/、
deploy/ONNX_QUANT_STRATEGY.md。

## 4. Phase B：deploy/ 分层（move 映射）

按 §2 布局执行 `git mv`（保历史）。要点：

- `deploy/prepost/` → `deploy/board_src/prepost/` 整目录平移，内部零改名
  （rules/、测试 cpp、probe_vmm{,2}.cu、_mk_syn_nv12.py 随目录走）；
- 插件/runner 单文件按 §2 归位；
- 去 `_` 前缀改名与 move 同一 commit（如 `deploy/_prod_install.py →
  deploy/ops/prod_install.py`）；
- move 后统一修正**代码内部**引用：
  - `import board_m1`（ops 全家）→ `from board import board_m1` 或
    `sys.path` 处理（实施时按现有 import 风格选最小改动方案）；
  - `deploy/prepost/` 硬编码路径（prod_install 的 push 清单、systemd 相对
    路径、m9a/m10/mprod/pd/v11 各驱动的源路径常量）；
  - `_m10_eval_run.py` 对 `_mprod_mkeval.py`、`eval_t6_mini_v8.py` 等的
    相对路径。
- 文档引用不在本 Phase 改（Phase E 统一做）。

## 5. Phase C：spdeploy CLI + scripts 薄壳

### 5.1 `deploy/spdeploy.py`

- argparse 子命令 dispatch；每个 verb = `subprocess.run([sys.executable,
  <ops 目标脚本>, ...原生参数])`，**转发不重写**；退出码透传。
- 启动即断言 `BOARD_HOST`/`BOARD_PASS` 环境变量存在（缺失打印用法并退 2）；
  **绝不读取/落盘任何凭据文件**。
- 无板依赖的 verb（eval）同样走 env 断言之外的可先 `--help`。

| verb | 目标（move 后路径） | 说明 |
|---|---|---|
| `install` | `ops/prod_install.py` | 装/重装三单元 + 规则编译 + 健康断言（`--uninstall` 透传） |
| `uninstall` | `ops/prod_install.py --uninstall` | |
| `status` | `board/board_m1.py` status 路径（或板上 `sp_status` 拉回） | 只读全景 |
| `restore` | `ops/prod_install.py --restore`（现有回退形态；若无则该 verb 缺省指向 install --help 并在计划期核定） | 回退 v8 形态 |
| `gate` | `ops/mprod_gate.py` | 闭环精度门禁 run |
| `eval` | `ops/m10_eval_run.py <tag>` | 单 run 全指标评估 |
| `golden` | `ops/prod_golden.py` | 金标生成/自检（`--plugin` 透传） |
| `ft` | `ops/prod_ft_{a,b,m9a,m10}.py`（`--suite a|b|m9a|m10` 选路） | 故障注入 |
| `soak` | `ops/prod_soak.py` | 浸泡 |
| `calib` | `ops/m9a_thr_calib.py` | THR 分位数标定 |
| `parity` | `ops/m9a_parity.py` | 规则 parity 门禁 |
| `boardtest` | `ops/m9a_boardtest.py` / `ops/m10_boardtest.py`（`--m9a/--m10`） | 板级冒烟 |
| `run` | `ops/m5_rerun.py legacy|split [graph|nograph] [fetch]` | 81 帧闭环复测 |
| `build-plugin` | `ops/board_mapfix.py tool` | 板上模块级编译插件 |

注：`restore` 与各 ft suite 的确切旗标在实施 Phase C 时以目标脚本
argparse 为准核定（dispatch 只转发，不改语义）。

### 5.2 `scripts/sp-*.sh`（12 个）

`sp-install.sh`、`sp-uninstall.sh`、`sp-status.sh`、`sp-gate.sh`、
`sp-eval.sh`、`sp-ft.sh`、`sp-soak.sh`、`sp-calib.sh`、`sp-parity.sh`、
`sp-selftest.sh`（→ golden）、`sp-boardtest.sh`、`sp-build-plugin.sh`。

统一形态（无 CRLF，LF 提交）：

```bash
#!/usr/bin/env bash
# sp-<verb>.sh — <一句话>. 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-<verb>.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py <verb> "$@"
```

与上游 `scripts/{create_data,kmeans,test,train,visualize}.sh` 共存，
`sp-` 前缀防混淆。薄壳只覆盖高频 verb；`run`（闭环复测）、`restore` 等
低频操作直接 `python deploy/spdeploy.py <verb> ...` 调用，不配 sh。

## 6. Phase D：README

- `README.md` 中文重写，顶部一行英文 tagline（如
  `English: see docs/ — SparseDrive end-to-end TensorRT deployment on Orin X, 82.6→39.1 ms closed-loop with production hardening.`）。
- 章节顺序（亮点前置）：
  1. **首屏亮点**：三引擎闭环链 39.11 ms（82.6 ms INT8 起点 → 2.11×）、
     det mAP 0.4157 / NDS 0.4724 / map mAP 0.7462（FP32 0.7512）、
     量化仅 -0.4 pt（PTQ MTQ + 敏感层保护）、量产硬化（watchdog/systemd/
     selftest/遥测/权限）、14 规则事件触发器（零关键链改动）、
     21 轮优化日志（负结果同权重留档）；
  2. 指标总表（延迟/三指标/EPA/L2/col，标注 mini 口径与基线对照）；
  3. 架构图（bb2+hd+mp 三引擎数据流、v11 DFA 插件、三 systemd 单元与信箱拓扑；
     复用 `resources/overview.png` 等）；
  4. 快速上手（clone → env → `scripts/sp-install.sh` → `sp-status.sh` →
     `sp-gate.sh`；引擎/数据获取指向 docs/FULL_CHAIN_REPRO.md）；
  5. 交付物清单（e_bb2/e_hd/e_mp/libdfaplug_v11.so/run_engine2/sp_modelnode/
     sp_trigger/14 rules .so + md5 指纹口径）；
  6. 仓库地图（scripts/repro/deploy 各层职责一句话）；
  7. 文档地图（6 篇 docs 各一句话）；
  8. 板子环境与约束（JetPack 5.1.5/TRT 8.6.1.2/sm_87、noexec、不许动环境）；
  9. 致谢（SparseDrive、nuScenes）。
- `deploy/README.md` 重写为新目录地图 + spdeploy verb 表。
- `README_CN.md` 删除；TECH_DETAILS 双语处置按 §3.2 判定。

## 7. Phase E：引用同步与验证

1. **AGENTS.md**（gitignored，不入库）：全量路径同步（prepost→board_src/
   prepost、`_prod_install.py`→`ops/prod_install.py` 等），工具点名清单逐项过。
2. **docs 当前有效引用**：`quick_start.md`、`FULL_CHAIN_REPRO.md` 当前命令段、
   `M9A_PLUGIN_GUIDE.md`、`PREPOST_OPS_DESIGN.md`、`BOARD_ENV.md` 批量替换。
   **`OPTIMIZATION_SUMMARY.md` 历史轮次文本不改**（历史快照），仅在文件头部
   插入"路径迁移表"（旧路径 → 新路径对照）。
3. **`ops/ref_check.py` 升级**：对 `docs/superpowers/`（历史 specs/plans）与
   OPTIMIZATION_SUMMARY 历史轮次设 allowlist/章节豁免；重跑**零悬空**。
4. **编译/导入冒烟**：全部保留 `.py` 过 `py_compile`；关键 import 链冒烟
   （export_v5→量化工具链、spdeploy dispatch、ops 脚本 `--help`）。
5. **CLI 冒烟**：`spdeploy.py <verb> --help` 全 verb 通过。
6. **实板只读冒烟**（可选，运行时注入凭据）：`sp-status.sh` 拉回一行全景；
   不 install、不 restart、不碰常驻态。
7. **push**：双向探测（直连 github.com:443 / 代理 socks5://127.0.0.1:10808）
   → 推送 → `ls-remote` 验证远端指针。

## 8. 实施顺序与 commit 切分

| Commit | 内容 | 门禁 |
|---|---|---|
| 1 | Phase A 大清洗（~250 删） | `_ref_check`（旧逻辑）+ 全仓 grep 无残留 import |
| 2 | Phase B 分层 move + 代码内引用修正 | py_compile + import 冒烟全绿 |
| 3 | Phase C spdeploy.py + 12 薄壳 | verb --help 冒烟全绿 |
| 4 | Phase D README 重写 + deploy/README + TECH_DETAILS 处置 | 内容评审（用户可读性） |
| 5 | Phase E 文档/AGENTS 同步 + ref_check 升级 | 零悬空 + push 验证 |

每个 commit 独立可回退；Phase A/B 之间强依赖（先删后移，避免移动将死文件），
C/D/E 顺序可并行度低但按序提交。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| move 破坏 import（board_m1 被 ops 全家 import） | board/ 包化 + 一次改全；py_compile + import 冒烟兜底 |
| `_prod_install.py` push 清单漏改 → 下次板上安装失败 | Phase B 内对其源路径常量做显式 review + 板上路径断言逻辑不变；实板 install 不在本轮强制执行（常驻态在跑），差异以代码评审+路径常量核对覆盖 |
| 文档悬空引用 | ref_check 升级 + 零悬空门禁（§7.3） |
| Windows 侧无 sed/grep | 批量替换用临时 python 脚本（多行 python -c 禁令照旧） |
| 误删仍被引用文件 | §3.1 判定器 + 上轮 30 个入库名单逐项核对 |
| 历史文档被改写失真 | OPTIMIZATION_SUMMARY/superpowers 归档不改写，只加迁移表 |

## 10. 非目标

- 不动 `tools/`、`projects/`、`repro/` 内容（repro 板端链脚本形态已符合
  "明确 sh 做明确事"）；
- 不改板端任何二进制 ABI、shm 布局、金标、常驻态（三单元 active 照旧）；
- 不做 30fps drop 语义实验、M9b sp_dla_runner（远期备选）；
- 不引入新依赖（spdeploy 仅标准库）；
- 凭据永不落盘（维持运行时注入纪律）。
