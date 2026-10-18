# 仓库重整实施计划（Phase A-E）

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按已批 spec 删除 ~250 个 tracked 历史脚本、`deploy/` 分七层、新增 `spdeploy` CLI + 12 个 `scripts/sp-*.sh` 薄壳、重写中文 README、全量引用同步后 push。

**Architecture:** 五个独立 commit 顺序推进：大清洗 → 分层 move（含路径常量修正）→ CLI+薄壳 → README → 文档同步。不引入新依赖（spdeploy 仅标准库+paramiko）；板上产物/常驻态零触碰。

**Tech Stack:** Windows cmd 宿主 + 临时 python 脚本（禁多行 `python -c`）；git 在 `C:\Users\wangyi\mingw64\bin\git.exe`；验证用 py_compile / findstr / WSL `bash -n`。

**Spec:** `docs/superpowers/specs/2026-10-18-repo-restructure-design.md`（本计划从 spec 论证；spec §3.1 判定器与 §5.1 verb 表在计划里有精化裁定，见 Global Constraints GC-4/GC-5 与 Task 3）。

## Global Constraints

- **GC-1 环境坑（每条命令前过一遍）**：无 ls/grep/sed/cat；多行 `python -c` 静默失败（一律 Write 临时 .py 再 `python 文件.py`，用完删）；路径 raw string 或正斜杠；`timeout /t` 死；git 用全路径 `C:\Users\wangyi\mingw64\bin\git.exe`；cmd 父提交语法用 `~` 不用 `^`。
- **GC-2 板子零触碰**：本计划唯一允许的板交互 = Task 5 可选只读 status 冒烟。不 install、不 restart、不编译引擎、不碰金标/常驻态。
- **GC-3 凭据纪律**：`BOARD_HOST`/`BOARD_PASS` 只运行时注入；spdeploy 与 sh 只读 `os.environ`，任何情况下不写凭据到文件/日志/git。
- **GC-4 判定器三原则（精化 spec §3.1）**：①AGENTS.md 点名 → 硬保留；②docs/*.md 中命令/路径形态引用（`deploy/xxx.py`）→ 保留；③docs 纯叙事点名（OPTIMIZATION_SUMMARY 历史轮、FULL_CHAIN_REPRO 叙事段的 `board_bb2.py`/`_m8_probe.py` 等）→ 可删，名字进 ref_check 的 `ALLOW`。`deploy/README.md` 的引用**不算**（Task 4 重写它）。
- **GC-5 verb 终表（精化 spec §5.1）**：13 个 verb——`install/uninstall/status/gate/eval/golden/ft/soak/calib/parity/boardtest/run/build-plugin`。spec 预留的 `restore` **砍掉**（`_prod_install.py` 只有 `--uninstall/--no-start`，无 restore 形态；回退流程走 AGENTS.md 已文档化的 install+golden 舞步）。
- **GC-6 行尾**：新建 `.sh` 必须 LF（CRLF 会坏 bash）；提交前用 `git ls-files --eol scripts/` 验。
- **GC-7 不动区**：`tools/`、`projects/`、`repro/`、`deploy/systemd/`、`deploy/artifacts/`、`model_anchor_config/`、`resources/` 内容零改动（已核实 tools/projects 无 deploy 反向引用）。
- **GC-8 历史文档不改写**：`docs/OPTIMIZATION_SUMMARY.md` 轮次正文、`docs/superpowers/**` 归档保持原文；OPTIMIZATION_SUMMARY 仅在 Task 5 头部插入迁移表。

## Review Focus

1. **凭据落盘**——任何人跑 `sp-status.sh` 后，凭据不得出现在任何新文件/日志里。测试：Task 3 步骤 grep spdeploy.py + scripts/*.sh 无 `BOARD_PASS` 写操作、仅 `os.environ` 读。
2. **删除破坏保留件**——保留的 py 引用了已删模块。测试：Task 1 步骤"import 残留 grep"零命中 + Task 2 py_compile 全绿。
3. **`_prod_install.py` push 清单悬空**——下次实板安装会 push 失败。测试：Task 2 步骤断言 `PP` 常量指向存在的 `deploy/board_src/prepost` 且 push 清单逐项 `os.path.isfile` 通过。
4. **ref_check ALLOW 滥用**——ALLOW 把本该保留的文件也豁免了。测试：Task 1 步骤断言 ALLOW 每项都属于已删名单、且不在 AGENTS.md 提及。
5. **sh 薄壳 CRLF**——Windows 写的 .sh 落库带 `\r` 直接坏。测试：Task 3 步骤 `git ls-files --eol scripts/sp-*.sh` 全部 `w/lf`。

---

### Task 1: Phase A 大清洗（commit 1）

**Files:**
- Delete: ~250 个 tracked 文件（名单 = spec §3.2 家族模式 − GC-4/下述保级修正）
- Modify: `deploy/_ref_check.py`（加 ALLOW 豁免机制）
- Temp: `_tmp_del_list.py`（生成删除清单，用完删）

**Interfaces:**
- Produces: 干净的 `deploy/` 平铺层（保留件原地名，Task 2 才 move）；`_ref_check.py` 支持 `ALLOW` 集合 + 跳过 `docs/superpowers/`，输出 `NO_DANGLING_REFS`。
- **对 spec §7.3 的顺序裁定**：ref_check 的 ALLOW 机制从 Phase E **提前到本 Task**——Phase A 的 commit 门禁就是 ref_check 零悬空，机制必须先在；Task 5 只做 docs 同步后的复跑。
- 保级修正（相对 spec §3.2，已按 GC-4 裁定）：`qdq_onnx_rewrite.py`、`split_engine.py`、`split_head.py`、`requant_fpn.py`（FULL_CHAIN_REPRO 命令/重建链点名 → Task 2 归 engine/）、`prep_mini_inputs.py`（BOARD_ENV 点名 → eval/）**从删除名单移出**；`split_analysis.py` 仍删。

- [ ] **Step 1: 写删除清单生成器并核对**

Write `_tmp_del_list.py`：内嵌 15 个家族 glob/前缀模式 + 例外集合（`board_m1.py`、`board_mapfix.py`、`ssa_surgery2.py`、`_fa_part.cu`、`selftest_fa{,2,3,4}.cu`、`_probe_dumpmeta.py`、`_probe_decode_check.py`、`_probe_m4_report.py`、`test_dfa_v11.cu`、GC-4 五个保级件）+ 边界删件（`export_v6.py`、`onnx2engine3.cpp`、`regen_mini_meta.py`、`preprocess_dump.py`、`preprocess_dump2.py`、`_fa_draft.cu`、`_gen_m9a_rules.py`、`_m7a_gate.py`、`_m7a_bit2.py`、`_m7a_cmp.py`、`board_bb2.py`、`_m8_probe.py`、`_diag3.py`、`_mk_boundary.py`、`_mk_eval_sp.py`、`cmp_native_vs_onnx.py`）+ 根目录 `README_CN.md`、`TECH_DETAILS.md`、`TECH_DETAILS_CN.md`。对每个候选执行 GC-4 三原则自动裁决（AGENTS.md/docs 命令形态命中→移入保级打印警告），输出最终清单到 stdout。

Run: `python _tmp_del_list.py`
Expected: 清单 240-260 个；**人工逐行核对**与 spec §3.2 家族行数一致（board_* ~95、make_* 11、trace_* 7、_m6/7/8 13、ssa 11、probe 16、check 7、cleanup 11、cmp/diff 24、fetch/push 4、convert 28、dfa 9、eval 3、边界 ~10、根 3）；五个保级件不在清单。

- [ ] **Step 2: 执行删除**

`_tmp_del_list.py` 末尾加 `git rm` 调用（`subprocess.check_output([GIT,"rm","-q",f] for f in LIST)`）。运行后 `git status --porcelain | findstr /c:"D " | find /c /v ""` 计数核对 == 清单长度。

- [ ] **Step 3: import 残留 grep**

Write `_tmp_imp_check.py`：对保留的全部 `deploy/**/*.py` 扫 `import <已删模块名>` / `from <已删模块名>`（模块名 = 已删 .py 的 stem 集合）。
Expected: 零命中（ops 对历史件全是 subprocess 路径调用，已核实）。

- [ ] **Step 4: ref_check 加 ALLOW 机制**

Modify `deploy/_ref_check.py`：文件头加 `ALLOW = set()`（空集起步）+ docs 收集时跳过路径含 `superpowers` 的 .md；缺域名进 `missing` 前先 `if m in ALLOW: continue`。ALLOW 注释注明"历史叙事点名、文件已删（git 历史可查）"。

- [ ] **Step 5: 生成 ALLOW 并复跑**

Run: `python deploy/_ref_check.py`
Expected: 首跑报 DANGLING（OPTIMIZATION_SUMMARY/FULL_CHAIN_REPRO 叙事点名的已删件）。把每个缺域名逐一核对：属于已删名单且不在 AGENTS.md → 填入 ALLOW；**若出现保留件名 → 说明 Step 1 误删，`git checkout HEAD -- <file>` 找回并回查清单**。复跑到 `NO_DANGLING_REFS`。

- [ ] **Step 6: Commit**

```bash
C:\Users\wangyi\mingw64\bin\git.exe add -A
C:\Users\wangyi\mingw64\bin\git.exe commit -m "repo: drop ~250 tracked one-off historical scripts (git history keeps them); ref_check gains ALLOW for narrative-only mentions"
```
删除 `_tmp_del_list.py`、`_tmp_imp_check.py` 后再 commit（临时件不入库）。

---

### Task 2: Phase B 分层 move + 路径常量（commit 2）

**Files:**
- Move: 按 spec §2 布局（映射见 Step 1）
- Modify: 保留件中硬编码 `deploy` 路径常量的文件（Step 3 判定）
- Modify: `deploy/_ref_check.py` 的 walk 无需改（按文件名全仓匹配，move 不影响）

**Interfaces:**
- Consumes: Task 1 的干净平铺层。
- Produces: 最终目录布局（spec §2）；ops 脚本的路径常量全部指向新位置；`deploy/board/board_m1.py` 可被 `python deploy/board/board_m1.py` 直接调用（无包化 import 依赖——已核实 ops 全是 subprocess 路径调用）。- [ ] **Step 1: 执行 move（git mv 保历史）**

Write `_tmp_move.py` 内嵌映射并逐条 `git mv`（目标目录先 `os.makedirs`）：

```
deploy/board/board_m1.py                       ← deploy/board_m1.py
deploy/board_src/prepost/                      ← deploy/prepost/*（整目录，内部零改名）
deploy/board_src/plugin/dfaplug_v3.cu|v8.cu|v11.cu|test_dfa_v11.cu
deploy/board_src/runner/run_engine.cpp|run_engine2.cpp|run_engines.cpp|onnx2engine.cpp
deploy/engine/export_v5.py|export_mp_onnx.py|export_fp32_onnx.py|export_quant_onnx.py
deploy/engine/{qat,ptq_sensitivity,flashmha_qkv,group_sensitivity,task_sentinel,render_ptq_report,smoke_test,test_flashmha_qkv,qdq_onnx_rewrite,split_engine,split_head,requant_fpn,ssa_surgery2,_gen_f32_names}.py
deploy/engine/ONNX_QUANT_STRATEGY.md
deploy/eval/{eval_t6_mini,eval_t6_mini_v8,eval_t6_mini_map,eval_mp_mini,eval_nuscenes,eval_sp32_mini_det,infer_sp32_mini,infer_sp32_mp_mini,infer_b2_int8fp16_mini,map_gt,compare_m2,compare_m3,prep_mini_inputs}.py
deploy/eval/{_mk_evaldir,_mk_scene_split,_seed_f0,_seg_feed}.py
deploy/eval/run_eval_fp32_val.bat|run_eval_int8_val.bat
deploy/ops/  ← _prod_{install,golden,ft_a,ft_b,ft_m9a,ft_m10,soak}.py → prod_{install,golden,ft_a,ft_b,ft_m9a,ft_m10,soak}.py（一律去 _ 前缀，其余文件名不变）
deploy/ops/  ← _m10_{boardtest,eval_run,install_assert}.py → m10_*.py；_m9a_{boardtest,parity,replay,smoke,thr_calib,unwrap_check,probe_ctx,probe_man,probe_replay}.py → m9a_*.py
deploy/ops/  ← _mprod_{gate,mkeval,mkmap}.py → mprod_*.py；_pd_step{1_build,2_golden,2b_resume,3_verify,3b_rebuild,4_det}.py → pd_step*.py
deploy/ops/  ← _m5_rerun.py→m5_rerun.py、board_mapfix.py、_v11_{gate,bench,reginfo}.py→v11_*.py、preproc_ref.py、_ref_check.py→ref_check.py
deploy/analysis/ ← _cmp_{drift_curve,f0_detail,f0_det,order_invariant}.py、_probe_{dumpmeta,decode_check,m4_report}.py、_inv_verify.py、_fa_part.cu、selftest_fa{,2,3,4}.cu
deploy/README.md 不动位置（Task 4 重写内容）
```

Run: `python _tmp_move.py`，然后 `C:\Users\wangyi\mingw64\bin\git.exe status --porcelain | find /c /v ""` 确认 R/D/A 计数合理；`dir /b deploy` 核对七层目录齐全。
 Expected: `deploy/` 根下无散落 .py（只剩 spdeploy.py 之后新增、README.md、ONNX_QUANT_STRATEGY.md 已入 engine/）。

- [ ] **Step 2: 路径常量修正（三类判定）**

Write `_tmp_pathfix.py`：对保留 py 扫 `os.path.join(...)` 与字符串里的 `deploy/` 引用，按三类打印处置建议——(a) `join(os.path.dirname(__file__),...)` 同目录引用 → 无需改；(b) `join(ROOT,"deploy","<旧平铺名>")` → 改为新路径（映射同 Step 1 + 去 `_` 前缀）；(c) `join(ROOT,"deploy","prepost",...)` → `board_src/prepost`。人工按打印结果逐文件 Edit。

必改点名（已核实）：`ops/m10_eval_run.py`（引 `_mprod_mkeval/eval_t6_mini_v8/eval_t6_mini_map/eval_mp_mini` 5 处）、`ops/pd_step1_build.py` + `ops/pd_step3b_rebuild.py`（引 `board_m1.py` → `board/board_m1.py`）、`ops/prod_install.py`（`PP = join(ROOT,"deploy","prepost")` → `board_src/prepost`；push 清单同）、`ops/m9a_parity.py`（2 处 prepost 路径）、`ops/m9a_thr_calib.py`（3 处）、`ops/board_mapfix.py`（2 处）、`ops/prod_ft_a.py`、`ops/m10_boardtest.py`、`ops/m9a_boardtest.py`、`ops/m9a_smoke.py`、`ops/v11_gate.py`、`ops/v11_bench.py`、`ops/v11_reginfo.py`、`ops/pd_step2_golden.py`、`ops/pd_step4_det.py`、`eval/_seg_feed.py`。其余 deploy-join 多为 `artifacts`（不动区），按打印结果确认无需改。

- [ ] **Step 3: 编译/安装面断言**

Write `_tmp_verify_b.py`：①`py_compile` 遍历全部 `deploy/**/*.py`（零 SyntaxError）；②静态断言 `ops/prod_install.py` 的 push 清单：`import` 该文件会执行顶层代码（它是脚本非库）→ 改为源码级检查：正则抽出 `PP` 常量与 push 清单里的相对路径，逐项 `os.path.isfile(join(ROOT,"deploy",rel))` 全 True。
Run: `python _tmp_verify_b.py`，Expected: `VERIFY_B_PASS`。

- [ ] **Step 4: docs 命令形态引用暂挂说明**

此时 `deploy/xxx.py` 形态的 docs 引用必然悬空——`_ref_check.py` 按文件名匹配（不含路径）**不受 move 影响**，预期仍 `NO_DANGLING_REFS`。
Run: `python deploy/ops/ref_check.py`，Expected: `NO_DANGLING_REFS`（ALLOW 已含 Task 1 名单；保留件 move 后文件名未变）。

- [ ] **Step 5: Commit**

```bash
C:\Users\wangyi\mingw64\bin\git.exe add -A
C:\Users\wangyi\mingw64\bin\git.exe commit -m "repo: layer deploy/ into board/board_src/{prepost,plugin,runner}/engine/eval/ops/analysis; strip _ prefixes in ops/eval; fix path constants"
```
删除 `_tmp_move.py`、`_tmp_pathfix.py`、`_tmp_verify_b.py` 后再 commit。

---

### Task 3: Phase C spdeploy CLI + scripts 薄壳（commit 3）

**Files:**
- Create: `deploy/spdeploy.py`
- Create: `scripts/sp-{install,uninstall,status,gate,eval,ft,soak,calib,parity,selftest,boardtest,build-plugin}.sh`（12 个）

**Interfaces:**
- Consumes: Task 2 的 ops/ 脚本路径（下表右列）。
- Produces: `python deploy/spdeploy.py <verb> [args...]`（args 原样透传目标脚本）；12 个 sh 一一对应（`run`/`uninstall` 外的低频 verb 无 sh，见 GC-5 与 spec §5.2）。

- [ ] **Step 1: 写 `deploy/spdeploy.py`**

规格（实现者写 body，以下为钉死的接口）：

```python
# dispatch 表（值 = move 后路径，相对 deploy/）
VERBS = {
    "install":      ("ops/prod_install.py", []),
    "uninstall":    ("ops/prod_install.py", ["--uninstall"]),
    "gate":         ("ops/mprod_gate.py", []),
    "eval":         ("ops/m10_eval_run.py", []),
    "golden":       ("ops/prod_golden.py", []),
    "ft":           ("ops/prod_ft_m9a.py", []),   # --suite a|b|m9a|m10 选路，见下
    "soak":         ("ops/prod_soak.py", []),
    "calib":        ("ops/m9a_thr_calib.py", []),
    "parity":       ("ops/m9a_parity.py", []),
    "boardtest":    ("ops/m9a_boardtest.py", []), # --m10 切 m10_boardtest.py
    "run":          ("ops/m5_rerun.py", []),
    "build-plugin": ("ops/board_mapfix.py", ["tool"]),
}
# status: 内置实现（不走 dispatch）
```

行为钉死：①启动即断言 `os.environ` 有 `BOARD_HOST`/`BOARD_PASS`，缺则打印用法 + `sys.exit(2)`（`status`/`--help` 之外所有路径）；②dispatch verb = `subprocess.call([sys.executable, <目标>, *extra, *argv])`，退出码原样 return；③`ft --suite a|b|m10` 选 `ops/prod_ft_{a,b,m10}.py`（默认 m9a）；`boardtest --m10` 选 `ops/m10_boardtest.py`（默认 m9a）——旗标本身继续透传给目标脚本；④`status`：内置 ~25 行 paramiko 只读 exec（`connect(hostname=env, password=env, timeout=10)`），依次跑 `/usr/local/bin/sp_status sp_result_m3`、`systemctl is-active sp-modelnode@m3 sp-filesrc@m3 sp-trigger@m3`，stdout 原样回显，连接失败打印 stderr 退 3；⑤无 argparse 子命令或 `-h` → 打印 verb 表退出（`--help` 不要求凭据）。

- [ ] **Step 2: 冒烟 spdeploy**

```bash
python deploy/spdeploy.py --help
set BOGUS=1&& python deploy/spdeploy.py status
python deploy/spdeploy.py install --help
python deploy/spdeploy.py run --help
```
Expected: 第 1 条打印 verb 表；第 2 条**在无 BOARD_HOST 的前提下**退 2 并打印用法提示（若本机恰有 BOARD_HOST 则改用 `set BOARD_HOST=` 清空测）；第 3/4 条透传目标脚本 --help（会因缺凭据先退 2——预期行为，能证明 dispatch 路径解析正确即可）。

- [ ] **Step 3: 写 12 个薄壳**

统一模板（verb 替换 `<verb>`，`sp-selftest.sh` 用 `golden`，`sp-build-plugin.sh` 用 `build-plugin`；GC-6 LF 行尾）：

```bash
#!/usr/bin/env bash
# sp-<verb>.sh — <一句话说明>. 用法: BOARD_HOST=... BOARD_PASS=... scripts/sp-<verb>.sh [args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BOARD_HOST:?export BOARD_HOST=<board ip> first}"
: "${BOARD_PASS:?export BOARD_PASS=<board password> first}"
exec python deploy/spdeploy.py <verb> "$@"
```

- [ ] **Step 4: 验证薄壳**

```bash
wsl bash -n /mnt/h/projects/sparsedrive-tensorrt-deploy-implement/scripts/sp-install.sh   （12 个逐个）
C:\Users\wangyi\mingw64\bin\git.exe ls-files --eol scripts/
findstr /n "BOARD_PASS" deploy\spdeploy.py scripts\sp-install.sh
```
Expected: `bash -n` 零输出（语法 OK，不执行）；`--eol` 12 个全 `w/lf`（无 `w/crlf`）；findstr 命中行**逐行过目**——只允许 `os.environ`/`${BOARD_PASS:?}` 形态的读取与断言，零写入/落盘/拼接进日志（Review Focus 1 的测试落点）。

- [ ] **Step 5: Commit**

```bash
C:\Users\wangyi\mingw64\bin\git.exe add deploy/spdeploy.py scripts/
C:\Users\wangyi\mingw64\bin\git.exe commit -m "feat: spdeploy CLI dispatcher (13 verbs) + scripts/sp-*.sh thin entries"
```

---

### Task 4: Phase D README 重写（commit 4）

**Files:**
- Modify: `README.md`（中文重写）
- Create: `docs/onnx_export_surgery.md`（← TECH_DETAILS_CN.md 内容迁入）
- Modify: `deploy/README.md`（重写为目录地图 + verb 表）
- Delete: `README_CN.md`、`TECH_DETAILS.md`、`TECH_DETAILS_CN.md`

**Interfaces:**
- Produces: README 章节骨架 = spec §6；数字口径如下表（写错即 reviewer 拒）。

| 指标 | 值 | 口径 |
|---|---|---|
| 闭环链延迟 | 39.11 ms（起点 82.6 ms，2.11×） | bb2+hd 拆分链 + v11 插件，v8 口径 39.86 |
| det / NDS | 0.4157 / 0.4724 | mini 81 帧，v8 对照 0.4160/0.4729 |
| map mAP | 0.7462 | FP32 PyTorch 0.7512 |
| 量化损伤 | map -0.4 pt（B2a 0.7473） | 量化数学 vs 引擎编译损伤已分离定案 |
| EPA(car/ped) / L2 / col | 0.6039/0.5080 / 0.7398 / 0.161% | M-PROD Phase A 闭环 |
| 事件触发器 | 14 规则 .so，CPU 10Hz，零关键链改动 | M9a parity 951 tick 集合全等 |
| 优化日志 | 21 轮，负结果同权重留档 | OPTIMIZATION_SUMMARY |

- [ ] **Step 1: 写主 README.md**

按 spec §6 九节顺序写中文内容：亮点首屏（上表前三行）→ 指标总表（全表）→ 架构（bb2+hd+mp 三引擎数据流 + v11 插件 + 三 systemd 单元/信箱拓扑；引用 `resources/overview.png`）→ 快速上手（clone→`pip install -r requirement.txt` 上游环境→`scripts/sp-install.sh`→`sp-status.sh`→`sp-gate.sh`；引擎构建/数据准备指向 `docs/FULL_CHAIN_REPRO.md`）→ 交付物清单（e_bb2/e_hd/e_mp/libdfaplug_v11.so/run_engine2/sp_modelnode/sp_trigger/14 rules + md5 指纹口径）→ 仓库地图（scripts/repro/deploy 各一句）→ 文档地图（6 篇各一句 + onnx_export_surgery）→ 板子环境与约束（JetPack 5.1.5/TRT 8.6.1.2/sm_87、/opt/m0 noexec、禁动环境）→ 致谢（SparseDrive、nuScenes）。顶部一行英文 tagline：`English: see docs/ — end-to-end SparseDrive on NVIDIA Orin X, 82.6→39.1 ms closed-loop with production hardening.`

- [ ] **Step 2: 迁移 TECH_DETAILS**

`git mv TECH_DETAILS_CN.md docs/onnx_export_surgery.md` 后编辑：标题改"ONNX 导出手术：SparseDrive 静态图化与算子级优化"、文首加一句指向（导出侧细节；TRT 侧见 OPTIMIZATION_SUMMARY）。`git rm TECH_DETAILS.md README_CN.md`。

- [ ] **Step 3: 重写 deploy/README.md**

内容：七层目录树（spec §2）+ spdeploy verb 表（GC-5 的 13 行）+ "加新脚本前先读这个"三条纪律（一次性脚本不入库/ops 脚本必须可被 spdeploy 调/文档引用必须过 ref_check）。

- [ ] **Step 4: Commit**

```bash
C:\Users\wangyi\mingw64\bin\git.exe add -A
C:\Users\wangyi\mingw64\bin\git.exe commit -m "docs: rewrite README (CN, highlights first); fold TECH_DETAILS into docs/onnx_export_surgery.md; deploy/README becomes layout map"
```
Run: `python deploy/ops/ref_check.py`，Expected: 仍 `NO_DANGLING_REFS`（README 新引用的都是存在文件；TECH_DETAILS 旧名如有 docs 残留引用会在此暴露→回 Step 2 检查漏删引用）。

---

### Task 5: Phase E 文档同步 + push（commit 5）

**Files:**
- Modify: `AGENTS.md`（gitignored，不入库）、`docs/FULL_CHAIN_REPRO.md`、`docs/quick_start.md`、`docs/M9A_PLUGIN_GUIDE.md`、`docs/PREPOST_OPS_DESIGN.md`、`docs/BOARD_ENV.md`、`docs/OPTIMIZATION_SUMMARY.md`（仅头部迁移表）
- Modify: `deploy/ops/ref_check.py`（如需补 ALLOW）

**Interfaces:**
- Consumes: Task 2 的最终路径映射。
- Produces: 全仓引用一致；远端 push 完成。

- [ ] **Step 1: 批量路径替换（临时脚本）**

Write `_tmp_docsync.py`：内置 old→new 替换对（`deploy/prepost/` → `deploy/board_src/prepost/`；`deploy/board_m1.py` → `deploy/board/board_m1.py`；`deploy/dfaplug_v11.cu` → `deploy/board_src/plugin/dfaplug_v11.cu`（v3/v8 同）；`deploy/run_engine2.cpp` 等 → `deploy/board_src/runner/`；`deploy/_prod_install.py` → `deploy/ops/prod_install.py`（全部 `_` 前缀件按 Task 2 映射）；`deploy/eval_t6_mini*.py`/`eval_mp_mini.py` 等 → `deploy/eval/`）。作用于 AGENTS.md + 上述 5 篇 docs；**跳过** OPTIMIZATION_SUMMARY 正文与 docs/superpowers/**。
Run 后人工抽查 AGENTS.md 三处（`_m5_rerun`→`m5_rerun`、prepost、prod_install）与 FULL_CHAIN_REPRO 的 `python deploy/qdq_onnx_rewrite.py` 命令行已变 `deploy/engine/`。

- [ ] **Step 2: OPTIMIZATION_SUMMARY 头部迁移表**

文件头部（标题后）插入一节"路径迁移表（2026-10-18 重整）"：两列表格 old→new（按 Task 2 映射的目录级规则 + 去 `_` 前缀规则各一行通配说明），正文零改动。

- [ ] **Step 3: 终验**

```bash
python deploy/ops/ref_check.py
python deploy/spdeploy.py --help
python deploy/spdeploy.py gate --help   （若本机有凭据则允许退 2，路径解析正确即可）
```
Write `_tmp_final.py`：py_compile 全量 + `scripts/*.sh` 逐个 `wsl bash -n` + GC-6 eol 检查。
Expected: `NO_DANGLING_REFS`、verb 表正常、`FINAL_PASS`。（可选，凭据在手时）`python deploy/spdeploy.py status` 拉回一行全景——只读，不碰常驻态。

- [ ] **Step 4: Commit + push**

```bash
C:\Users\wangyi\mingw64\bin\git.exe add docs/
C:\Users\wangyi\mingw64\bin\git.exe commit -m "docs: sync effective docs + AGENTS to new layout; migration table atop OPTIMIZATION_SUMMARY"
```
push 按口诀：先双向探测（`curl -sI https://github.com` 直连 + `curl -x socks5h://127.0.0.1:10808 http://example.com` 代理），活的那条推；推完 `git ls-remote origin -h refs/heads/main` 验指针 == 本地 HEAD。删除全部 `_tmp_*.py` 后再提交（确认 `git status` 干净）。
