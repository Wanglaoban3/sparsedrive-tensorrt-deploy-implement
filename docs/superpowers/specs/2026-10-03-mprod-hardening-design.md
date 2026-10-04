# M-PROD 量产化加固设计（spec）

日期：2026-10-03
状态：待用户评审（brainstorming 架构路径产物）
前置：M1–M8 全部定案（见 OPTIMIZATION_SUMMARY 第 1–14 轮），交付形态 =
`sp_filesrc → sp_bus(mapped-shm 环) → sp_modelnode(bb2+hd+mp) →
sp_result v2 信箱 → sp_resultmon`，闭环 22.48fps，e2e@5fps 48.35ms。

---

## 1. 背景与目标

当前系统是"故障行为未定义"的 A 样：算力吃满（22.48 vs 22.7fps 天花板）、
数据完整性原语齐全（心跳+lease 强制回收、CRC、checksum、版本 guard），
但**任何进程级故障的后续行为没有定义**——TRT hang 无超时、无监督重启、
信箱静默（fail-silent）、运行时无数值异常探测、无时长验证。

本设计把系统升级为"故障时行为有定义"的量产形态，四个阶段：

| 阶段 | 内容 | 解决的问题 |
|---|---|---|
| A | ①watchdog ②systemd 监督重启 ③fail-visible 降级帧 | 挂死无人知、崩了不自愈、下游不知道数据旧了 |
| B | ④运行时发散探测+自动复位 ⑤开机自检（容差制） | 带病输出、静默数值退化 |
| C | ⑥在线遥测 ⑦8h 浸泡 | 慢了不知道为什么、"没泄漏"只是推断 |
| D | ⑧权限收紧+日志轮转 | 最小安全卫生 |

## 2. 非目标

- **M9 多模型共置调度不做**（已记 AGENTS.md 代办），但 ② 的 unit 模板 +
  EnvironmentFile 形状按多模型多实例可扩展设计（加模型=加 env 文件+unit
  实例，不改架构）。
- 真实相机 / NvBufSurface 不做（M8 定案的平台限制不变）。
- DLA 不做（此前实测负收益）。
- **不整引擎重编、不碰 TRT/CUDA/JetPack/系统包**：全部改动在节点侧
  C++/Python + systemd 配置 + 编排脚本。
- SIGSTOP 类进程冻结不覆盖（连 watchdog 线程一起冻结；真实故障面是
  hang 不是 STOP，见 §10 取舍）。

## 3. 已定决策（用户批准，不再重议）

1. 单 spec、四阶段按序交付，每阶段板端验证通过后进下一阶段。
2. 授权写入面扩展：`/etc/systemd/system/sp-*.service`、`/etc/tmpfiles.d/`、
   `/etc/logrotate.d/`、`/etc/sp/`、`/run/sp/`、`/var/log/sp/`、
   `/var/lib/sp/`、金标 `/opt/m0/trt-dev/golden/`（其余约束不变：
   不动 TRT/CUDA/JetPack/系统包）。
3. 开机自检失败**阻止 ACTIVE**：节点存活、心跳正常、不发布有效结果，
   等人工（`SP_SELFTEST_OVERRIDE=1` 开发放行）。
4. 安全卫生做文件权限收紧，维持 root，不动用户体系。

## 4. 系统总览（目标态）

```
systemd
 ├── sp-filesrc@m3.service      Restart=on-failure, StartLimitBurst=5
 │     sp_filesrc --loop --wait-cons 1   [claim 连续超时→exit 20]
 │        │ (mapped-shm ring, 心跳/lease 强制回收 —— 既有)
 │        ▼
 └── sp-modelnode@m3.service    Requires/After filesrc
       sp_modelnode --selftest [自检→ACTIVE] [watchdog 线程] [发散探测器]
          │  帧循环: pre→bb2→hd→mp→post, 每阶段 touch watchdog
          ▼
       sp_result v3 信箱 (status/age/fail-visible 语义) ──▶ sp_resultmon
                                                        sp_status
金标: /opt/m0/trt-dev/golden/<ring>/frame0.npz + meta.json(容差+指纹)
日志: /var/log/sp/*.log (logrotate)   运行态: /run/sp/ (tmpfiles.d)
```

## 5. Phase A —— 可用性主轴

### A1 节点内 watchdog（新文件 `deploy/prepost/sp_watch.h` + 模型节点集成）

- 主循环每**完成**一个阶段调用 `watch_touch(stage_id)`：更新原子
  `beat_ms`（CLOCK_MONOTONIC）与 `cur_stage`。**只覆盖处理阶段
  PRE/BB2/HD/MP/POST**；ACQUIRE 明确不纳入——输入饥饿属于发布端故障域
  （filesrc exit 20 升级 + systemd 编排负责），消费端等待是正确行为，
  不能被 watchdog 打成自己的超时。
- 阶段枚举：`PRE=0, BB2=1, HD=2, MP=3, POST=4, ACQUIRE=5`。
- watchdog 线程每 20ms 轮询：`now - beat_ms > max(3×p50[stage], 500ms)`
  → 触发。p50 用各阶段最近 64 次耗时的滚动中位数（warmup 段自然填充）。
  `SP_WD_MS=<ms>` 强制覆盖（测试用）。
- 触发动作：打印 `FATAL code=13 stage=<sid> stalled_ms=<n> seq=<seq>` →
  `fflush` → **`_exit(13)`**。绝不调 CUDA cleanup / 引擎析构（CUDA 出错
  context 沾毒时清理路径会死锁——项目实测坑）。
- 测试钩子 `SP_WD_TEST_STALL=<sid>`：编译进二进制，默认关闭；命中时该
  阶段在帧 5 处 `sleep 5s`（越过 warmup），仅供故障注入。

### A2 退出码与日志契约

| 进程 | 码 | 含义 |
|---|---|---|
| modelnode | 0 | SIGTERM 正常退出 |
| | 10 | manifest/配置错误 |
| | 11 | 引擎/插件加载失败 |
| | 12 | CUDA 错误（现有错误路径收编） |
| | 13 | watchdog 超时 |
| | 14 | 发散 latch 退出（§B1） |
| | 15 | 自检失败（§B2） |
| filesrc | 0 | 正常退出 |
| | 10 | manifest/数据错误 |
| | 20 | claim 连续超时 ≥30 次，主动升级退出 |
| | 21 | manifest 耗尽（非 --loop 模式） |

所有异常退出前必须打印一行可解析的
`FATAL code=<ec> stage=<sid> seq=<seq> detail=<一句话>`。
日志落 `/var/log/sp/filesrc-<ring>.log`、`node-<ring>.log`（unit
`StandardOutput=append:`，不依赖 journald；轮转见 Phase D）。

### A3 systemd 单元（`deploy/systemd/` + 安装脚本 `deploy/_prod_install.py`）

`sp-filesrc@.service`：

```ini
[Unit]
Description=SparseDrive replay publisher (%i)

[Service]
Type=simple
EnvironmentFile=/etc/sp/%i.env
ExecStart=/usr/local/bin/sp_filesrc ${SP_RING} ${SP_W} ${SP_H} ${SP_FPS} \
  ${SP_MANIFEST} ${SP_ROOT} --loop --wait-cons 1
Restart=on-failure
RestartSec=1s
StartLimitIntervalSec=120
StartLimitBurst=5
StandardOutput=append:/var/log/sp/filesrc-%i.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
```

`sp-modelnode@.service`：

```ini
[Unit]
Description=SparseDrive perception node (%i)
Requires=sp-filesrc@%i.service
After=sp-filesrc@%i.service

[Service]
Type=simple
EnvironmentFile=/etc/sp/%i.env
ExecStart=/usr/local/bin/sp_modelnode ${SP_RING} ${SP_MODELS_DIR}/e_bb2.engine \
  ${SP_PLUGIN} ${SP_MANIFEST} ${SP_OUT} \
  --hd ${SP_MODELS_DIR}/e_hd.engine --mp ${SP_MODELS_DIR}/e_mp.engine \
  ${SP_NODE_ARGS}
Restart=on-failure
RestartSec=1s
StartLimitIntervalSec=120
StartLimitBurst=5
StandardOutput=append:/var/log/sp/node-%i.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
```

`/etc/sp/m3.env`（实例配置，M9 多模型的扩展点）：

```ini
SP_RING=m3
SP_W=1600
SP_H=900
SP_FPS=5
SP_MANIFEST=/opt/m0/trt-dev/nv12_r0/manifest.jsonl
SP_ROOT=/opt/m0/trt-dev/nv12_r0
SP_MODELS_DIR=/opt/m0/trt-dev/models
SP_PLUGIN=/usr/local/lib/libdfaplug_v8.so
SP_OUT=/var/lib/sp/m3
SP_NODE_ARGS=--warmup 2 --graph --selftest
```

- `/etc/tmpfiles.d/sp.conf`：`d /run/sp 0750 root root -`（tmpfs 开机自建，
  降级帧 marker/UDS 落这里）。
- 安装脚本 `deploy/_prod_install.py`（paramiko）：push unit/env/tmpfiles/
  logrotate → `daemon-reload` → `enable --now` 两单元 → 健康断言（信箱
  60s 内出现 NOMINAL）。带 `--uninstall`（stop/disable/删文件），全程可逆。
- **unit 不带 `--fresh`（关键决策）**：filesrc 改为 attach-or-create
  （几何不符才重建）。若重启时 `--fresh` 重建环，而 node 持有的还是旧
  shm 对象的映射（unlink 后仍有效但永远静默）→ 节点饿死且无人察觉。
  环持久化后：重启只续发，seq 单调不回绕，node 的映射始终有效。
- **恢复编排（设计的核心舞步，依赖既有 M1.5 机制）**：node 崩 → 它握着
  的 ≤1 个槽引用由 lease 过期强制回收（心跳停 → `forced_recycles`）→
  filesrc **不需要任何升级动作继续发布**（环深 4 槽轮转）→ systemd 1s
  后拉起 node（引擎反序列化 3–10s）→ node 重新 attach 到同一个环继续
  消费，seq 无缝。**exit 20 降级为兜底**：仅在回收也推进不了时
  （异常 lease 配置等）连续 30 次 claim 超时才触发。两边 StartLimit
  独立熔断（120s 内 5 次 → failed，人工介入，不无限重启掩盖真故障）。
- **启动竞态**：Type=simple 下 unit"已启动"≠环已建好；node attach 失败
  不再立即退出，改为重试 ≤10s（500ms 间隔），覆盖 filesrc 打开清单
  建环的窗口，避免一次无谓的 systemd 重启churn。

### A4 信箱 v3 —— fail-visible 语义（`sp_result.h` 版本 2→3）

`kVer = 3`；`ResultMsg` 在 `reserved2` 之后**尾部追加**（既有字段偏移
不动；旧读端按 `header_size` 自证拒绝或忽略尾部，不崩）：

```cpp
// ---- v3 追加段 (M-PROD fail-visible) ----
uint8_t  status;        // 0=NOMINAL 1=DEGRADED_RESET 2=DEGRADED_LATCH 3=SELFTEST_FAIL
uint8_t  reason;        // 0=none 1=nan 2=divergence 3=reset_count 4=selftest 5=stage_fail
uint8_t  wd_stage;      // 最后完成阶段 id (诊断)
uint8_t  reserved3[5];
uint32_t last_valid_seq;  // status!=0 时, 有效结果所属帧 seq
uint16_t frame_age_ms;    // 结果相对输入帧年龄; LATCH/重启间隙为真实旧化
uint16_t resets_60s;      // 60s 窗口复位次数 (健康遥测)
uint32_t nan_hits;        // 累计 NaN/Inf 命中
uint32_t div_hits;        // 累计发散命中
```

- 语义：**下游要么拿到新结果（NOMINAL），要么明确知道拿到的是
  `frame_age_ms` 前的旧结果（LATCH/重启间隙），要么明确知道节点不可用
  （SELFTEST_FAIL 消息 1Hz 心跳发布、det/map 清零、`last_valid_seq=0`/
  `age=0` 表示"自本进程启动从未有有效结果"）**。不再有静默。
- 兼容性：`shm_open(O_CREAT)` 复用旧 v2 信箱时 `ftruncate` 扩到 v3 尺寸；
  旧读端映射更小的 `sizeof(Slot)`（按其编译期常量），写端写尾部不破坏
  旧读端映射 —— 前向安全。
- `sp_resultmon` 同步升级：打印 `STATUS=<s> AGE=<ms> RESETS=<n>` 行。

### Phase A 故障注入验收（`deploy/_prod_ft_a.py`，每场景独立进程）

| # | 注入 | 断言 |
|---|---|---|
| 1 | `kill -9` modelnode | 信箱在 15s 内恢复 NOMINAL（新 seq 前进）；期间信箱最后一条 age_ms 单调增长 |
| 2 | `SP_WD_TEST_STALL=2`（hd 段睡 5s） | node 以 13 退出；日志有 `FATAL code=13 stage=2`；systemd 拉起后恢复 NOMINAL |
| 3 | manifest 路径改错 → filesrc exit 10 | node 因 Requires 依赖失败不启动（或 attach 重试后退出 10）；filesrc 单元 failed、StartLimit 熔断，不无限重启 |
| 4 | 连续 kill modelnode 6 次 | 第 6 次 systemd 保持 failed（StartLimit 生效） |

## 6. Phase B —— 安全运行时

### B1 运行时异常探测器（`deploy/prepost/sp_safety.cu` + 节点集成）

检查点与处置（全部 env 可调 `SP_DIV_*`）：

| 检查 | 输入 | 判定 | 成本 |
|---|---|---|---|
| NaN/Inf 扫描 | det_cls/det_reg/motion_reg/plan_reg/plan_status（宿主侧已有 D2H） | 任一非有限值 | µs 级/帧 |
| 检出合理性 | det[] | BEV 界 \|x\|,\|y\|>100m；尺寸 w,l,h∈(0,20m] 外 | µs 级/帧 |
| 检出数漂移 | n_det | 滚动 100 帧 z-score>6 | O(1) |
| plan 合理性 | final_plan/plan_reg | 曲率/加速度超界（|a|>15m/s²、|κ|>1.0） | µs 级/帧 |
| 状态向量发散 | hd 8 状态 + mp 9 状态 | 设备侧 absmax kernel → D2H 几十字节；范数比 > 模板值 10× | 预算 ≤0.5ms/帧：状态总字节 ≤6MB 全查；超出则大状态每 10 帧抽 1 |

**处置状态机（防掩盖真故障，不做无限复位循环）**：

```
单帧异常 ─▶ 计数(nan_hits/div_hits) + 模板复位该帧状态(复用 k=0/40 机制)
         + 本帧信箱置 status=DEGRADED_RESET(结果基于复位态, 标志位明确)
60s 窗内复位 >5 次 ─▶ LATCH: 持续发布 last_valid + frame_age_ms 真实旧化
                   + status=DEGRADED_LATCH ─▶ 2s 宽限 ─▶ exit 14
systemd 重启 ─▶ 自检 ─▶ 正常 / 120s 内再 latch ─▶ StartLimit 熔断 ─▶ failed(人工)
```

测试钩子 `SP_INJECT_NAN=<n>`：第 n 帧往 mp 状态注入 NaN，断言
`nan_hits+1`、信箱 DEGRADED_RESET、状态复位后输出恢复有限值。

### B2 开机自检（`deploy/_prod_golden.py` + node `--selftest`）

- **金标生成**（`gen` 模式）：板端跑帧 0 → 小输出张量（det_cls/det_reg/
  motion_cls/plan_cls/final_plan 等，<5MB）存
  `/opt/m0/trt-dev/golden/<ring>/frame0.npz` + `meta.json`：
  `{engine_md5, plugin_md5, tol:{per-tensor abs}}`。
- **容差校准**（`--calibrate`）：同引擎同输入连跑 10 次，逐张量取最大
  绝对偏差 ×3（下限 1e-3，f16 噪声底）。**依据 M7 定案：跨 run 非 bit
  确定（det_cls ~1e-2 抖动），严禁 hash 比对**。
- **节点启动流**（`SP_NODE_ARGS=--selftest`，unit 默认）：加载引擎 →
  校验金标指纹（engine/plugin md5 不符 → 直接 SELFTEST_FAIL，防金标
  过期）→ 跑帧 0 → 容差比对 → 过：置 ACTIVE 正常运行；不过：**节点
  存活、watchdog 照常、1Hz 发布 status=SELFTEST_FAIL 心跳（det/map 清
  零）**，不发布有效结果；`SP_SELFTEST_OVERRIDE=1` 时置 DEGRADED 继续
  （开发用）。

## 7. Phase C —— 可观测

### C1 在线遥测

- `frame_log.tsv` 扩列：`gpu_temp_c, cpu_temp_c, sm_clock_mhz`（1Hz 采样
  /sys thermal zone + clock；读不到写 `n/a`，不阻塞主循环）。
- 节点 512 帧滑动窗口，每 1s 落一行：
  `perf t=<s> pre_p50=.. bb2_p50=.. hd_p50=.. mp_p50=.. pre_p99=.. ...`。
- 新工具 `deploy/prepost/sp_status.cpp` → `/usr/local/bin/sp_status`：
  一行概览 ring 满度/forced_recycles/信箱 status+age/writes/节点 RSS/
  uptime；`--json` 供脚本。
- 遥测持久化 `/var/lib/sp/<ring>/telemetry.jsonl`（轮转同 Phase D）。
- 不建网络遥测（板子离线），本地落盘回捞。

### C2 8h 浸泡（`deploy/_prod_soak.py`）

- 起 systemd 单元（`--loop` 81 帧 manifest）运行 8h；每 60s 采样：两进程
  VmRSS（/proc/<pid>/status）、信箱 age/writes、forced_recycles、温度/
  时钟、`tegrastats` 若存在则并行录（不存在优雅跳过）。
- 结束生成 `work_dirs/soak_<date>/{report.md, samples.csv}`：RSS 线性
  回归斜率、p99 漂移、异常计数汇总。
- **验收线：RSS 斜率 <1MB/h；稳态 forced_recycles=0（前 10min 瞬态除外）；
  零 exit 13/14；信箱 age p99 < 2×帧周期。**
- 已知局限（记录不回避）：iGPU unified memory 下 RSS 只部分反映设备侧
  分配；设备池一次性静态分配 + 主循环无逐帧 malloc，泄漏面在宿主侧，
  RSS 是正确观测量。

## 8. Phase D —— 卫生

| 对象 | 权限 |
|---|---|
| /dev/shm/sp_* (ring) | 0640 root（创建处 fchmod） |
| /dev/shm/sp_res_* (信箱) | 0640 root |
| /run/sp/ | 0750（tmpfiles.d）；其内 UDS 0640、marker 0640 |
| /opt/m0/trt-dev/golden/ | 0444 只读 |
| /var/log/sp/*.log | 0640 + logrotate（`/etc/logrotate.d/sp`：100MB/5 份/rotate 后压缩；镜像若无 logrotate → 节点 open 时 >100MB 截断重开兜底） |

- UDS 路径 `/tmp/sp_dma_<ring>.sock` → `/run/sp/sp_dma_<ring>.sock`
  （`dma_sock_path` 改一处 + install 脚本建目录）。
- 维持 root 运行（§3 决策 4），不做用户体系改动。

## 9. 验证总闸（每阶段出口条件）

| 阶段 | 出口条件 |
|---|---|
| A | 故障注入 4 用例全过；81 帧闭环 mAP/EPA 复跑不劣化（防 watchdog/信箱改动碰坏主链）；pipe 吞吐变化 ≤2% |
| B | `SP_INJECT_NAN` 用例过（探测→复位→标志→恢复）；自检通过路径→ACTIVE；篡改金标→SELFTEST_FAIL；mAP/EPA 复跑不劣化 |
| C | 遥测列与离线 frame_log 交叉一致；8h 浸泡报告达 §7 验收线 |
| D | 权限断言（ls -l 清单）；logrotate 强制触发一次成功；UDS 迁移后编译过、base 门禁复跑过 |

每阶段独立 commit + 推送（socks5 代理，流程既有）。

## 10. 取舍记录

| 取舍 | 理由 |
|---|---|
| watchdog 放节点内线程，不用 systemd WatchdogSec/sd_notify | 后者需链接 libsystemd 且不报"卡在哪个阶段"；节点内线程零依赖、信息量大；两者都不覆盖 SIGSTOP（接受，真实故障面是 hang） |
| 触发后 `_exit` 不做清理 | CUDA context 沾毒时清理路径死锁（项目实测坑）；状态都在 shm，进程消失即一致 |
| LATCH 后 exit 14 重启而非永久降级运行 | 3–10s 重启窗口换"绝不带病持续输出"；反复 latch 由 StartLimit 熔断转人工 |
| 信箱 v3 尾部追加而非新文件 | 消费端迁移成本最低，旧读端前向安全 |
| 自检用容差不用 hash | M7 定案跨 run 非 bit 确定（det_cls ~1e-2），hash 必假警报 |
| 测试钩子（SP_WD_TEST_STALL/SP_INJECT_NAN）编译进二进制默认关 | 量产惯例是不带；本平台是验证平台，保留钩子使故障注入可自动化，文档标注 |
| 发散探测不每帧全量拷状态 D2H | 状态缓冲可达几十 MB，D2H 是抢回来的带宽；设备侧 absmax kernel + 预算规则 |
| unit 不带 `--fresh`，filesrc attach-or-create | `--fresh` 重建环会产生"旧映射僵尸环"（unlink 后 node 侧映射仍有效但永久静默），环持久化才使重启舞步收敛 |
| watchdog 不覆盖 ACQUIRE | 输入饥饿是发布端故障域；消费端等待是正确行为，打成自己的超时会引发 node 无谓重启循环 |

## 11. 交付物清单

**修改**：`sp_modelnode.cpp`（watchdog 集成/探测器/v3/自检/遥测钩子/退出码/
attach 重试）、`sp_filesrc.cpp`（attach-or-create、exit 20 升级）、
`sp_result.h`（v3）、`sp_resultmon.cpp`（v3 解码）、`sp_bus.h/.cpp`
（fchmod、几何校验复用打开）、`sp_dmapool.cpp`（UDS 路径）、
`board_m1.py`（新源文件入构建）。

**新增**：`prepost/sp_watch.h`、`prepost/sp_safety.cu`、`prepost/sp_status.cpp`、
`systemd/sp-filesrc@.service`、`systemd/sp-modelnode@.service`、
`systemd/sp.conf`（tmpfiles）、`systemd/sp-logrotate`、`_prod_install.py`、
`_prod_golden.py`、`_prod_ft_a.py`（A/B 注入共用框架）、`_prod_soak.py`、
`_mprod_gate.py`（mAP/EPA 复跑驱动，复用 `_m5_rerun` 模式）。

**文档**：OPTIMIZATION_SUMMARY 第十五轮（A/B 各一小节）、
FULL_CHAIN_REPRO §9 追加量产化运行形态、AGENTS.md 交付组合更新。

## 12. 实施顺序

1. Phase A：sp_watch + 退出码 → v3 信箱 + resultmon → unit 文件 + 安装
   脚本 → 板端部署 → 注入 4 用例 → mAP 复跑 → commit/push。
2. Phase B：sp_safety.cu 探测器 → 状态机 → 金标工具 + 自检 → 注入用例 +
   自检双路径 → mAP 复跑 → commit/push。
3. Phase C：遥测 → sp_status → 8h 浸泡（挂后台跑）→ 报告 → commit/push。
4. Phase D：权限/轮转/UDS 迁移 → 断言 → commit/push。

风险提示：引擎反序列化时间波动（3–10s）决定重启窗口宽度，age_ms 语义
已覆盖，不另做预连接；systemd 需 ≥230（${VAR} 展开），JetPack 5 = 245 ✓；
tegrastats 可能缺席 → 遥测降级为 n/a，不断主流程。
