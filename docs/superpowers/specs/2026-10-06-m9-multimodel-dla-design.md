# M9 多模型共置设计 spec —— 小 CNN 二级模型 · cuDLA standalone 路径

日期：2026-10-06
状态：设计已用户确认（四项决策见 §3），待评审后转实施计划
前置依赖：M10（数据面三件套）先行交付

## 1. 背景与硬约束

- **M7 定律**：单 iGPU 调度不创造 SM-秒；无 QoS 时间片让关键链段膨胀
  1.6-2.2×（实测 hd 19.9→32.7ms）。每帧 SM 总工作量 ~44.3ms（v11 链）
  是吞吐天花板（~23fps）。
- **30fps 相机满负荷 = slack 归零**：关键链饱和时 SM 维度上多模型与满负荷
  对立，必须靠"别的硬件"或"可控牺牲"解决，不靠调度魔法。
- **本板系统冻结**：JetPack 5.1 / CUDA 11.4 / TRT 8.6.1.2。**MPS 要
  CUDA 12.5+（JetPack 6）才支持 Orin**（NVIDIA 论坛确认），当前不可用；
  系统升级不在本设计依赖路径上（若未来随整车升级解锁，重模型可走 MPS
  分区，见 §9 开放项）。
- **板端坑约束**：GPU 正在读的 shm 槽被覆写 = nvgpu 通道 fault 硬崩；
  CUDA context 一旦出错沾毒（同进程全体 CUDA API 连环炸）→ 故障域设计
  必须让二级模型的死法不触碰关键链的 context。
- DLA 实测（2026-10-04）：DCN 类算子 DLA 不支持且回退反而慢（backbone
  上 DLA 慢 4×）；QDQ 网络 + DLA 必须配 `--int8`。板在 NVDLA0/1 双核
  （tegrastats 可见）。

## 2. 目标 / 非目标

目标：
1. 关键链（bb2+hd+mp 感知规划链）在多模型共置 + 30fps 满负荷下**零回归、
   不挨饿**（帧龄 p99 <400ms 契约不变）。
2. 二级模型（红绿灯/路牌/地面标识等 5-10Hz 小 CNN）以**独立进程 + 独立
   systemd 单元**接入：崩溃自愈、降级联动、上线只加配置不改关键链代码。
3. 传输层语义升级为**采样（last-is-best）**：消费者取最新帧、永不阻塞
   发布者、落后即跳序。

非目标（明确剔除，YAGNI）：
- GPU 时间片窗口调度 / 预算发牌（GXF 式原语）——只有重模型上 SM 才需要；
  本设计只预留遥测接口位（per-model budget 字段）。
- MPS 分区（系统不满足）。
- 权重跨进程共享、TRT 引擎多模型单进程联邦（故障域合并，不做）。

## 3. 已确认的设计决策（2026-10-06 用户拍板）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 二级模型形态 | 小 CNN 为主（5-10Hz 分类/小检测） | 纯 conv 可上 DLA，硬件隔离零 SM 争抢 |
| 宿主形态 | **通用 runner**（一个进程跑任意 DLA engine，env 指定路径/输入输出/节拍） | 新模型上线 = 加 env + 单元，不改码 |
| 节拍语义 | **定时采样**（自持定时器，env 配 5-10Hz，到点取当前最新帧） | 与 last-is-best 一致，负载可预测，不积压 |
| 降级联动编排者 | **独立进程 sp_orchestrator** | 职责单一（只读健康 + systemd 开关），死了无级联 |

## 4. 架构

```
                         systemd
  sp-filesrc@m3 ──发布──> sp_ring(m3, 4槽, mapped-shm)
        │                      │ in-order/latest 两种消费语义
        │                      ├── sp_modelnode@m3   (关键链 GPU SM 独占, 不变)
        │                      ├── sp_dla@tl        (二级 A: cuDLA standalone, NVDLA0)
        │                      └── sp_dla@ts        (二级 B: cuDLA standalone, NVDLA1)
  sp_result_m3 信箱 <──发布── sp_modelnode
  sp_result_tl/ts 信箱 <────── sp_dla runner × N
        │
  sp_orchestrator (新, 只读: 信箱 lage/p99/div + telemetry)
        └── 超阈 → systemctl stop sp_dla@* ；恢复 → start
```

- **关键链**：零改动 GPU 形态；仅经 M10 获得生产跳序（--skip-lag）。
- **sp_dla_runner（通用宿主）**：acquire 环上最新 seq（sampling，短持锁
  ~ms 把 NV12 吸进私有缓冲立即还槽）→ 前处理 → cuDLA standalone enqueue
  （NvSci 路径，无 CUDA context）→ 后处理 → 发 `sp_result_<model>` 小信
  箱（模型名/延迟/置信度/seq+ts_capture）。模型由 env 定义（engine 路径、
  输入输出张量名、节拍 Hz、DLA core 号）。
- **sp_orchestrator**：无 GPU、不碰数据面。输入 = 关键链信箱健康字段
  （lage/p99/div/resets60）+ telemetry.jsonl；输出 = systemctl 开二级单
  元。状态机：NOMINAL→(关键链 p99 连续超阈 N 秒)→DEGRADED(停二级)→(恢
  复 M 秒)→重开二级；所有迁移写 syslog + 自己的小信箱（fail-visible）。

## 5. 数据面语义（依赖 M10 先行）

1. **last-is-best 采样**（DDS KEEP_LAST(1) / AUTOSAR ara::com 同名语义）：
   二级 runner 取 `最新 seq` 跑，不等 done+1；落后即跳，永不积压。
2. **wrap-lease（M10 ③）**：租约 = k×ring_depth×帧周期（4 槽@30fps =
   133ms/圈，k 默认 2）；发布端 claim 等待上限一圈，到点对**该槽持有者**
   定向强抢评估（心跳过期 + 绕回超时双条件，防误杀 fence 在途读者）。
3. **filesrc 多消费者语义（M10 ④）**：`--wait-cons` 不再要求等满 N 个，
   后到消费者 resync 三件套入列。
4. **跨模型帧配对**：所有信箱消息携带 seq + manifest ts_capture；融合层
   按 seq join（不假设两家看过同一帧），时间对齐以采集纪元 ts 为准
   （frame_age 墙钟口径只用于活性，不做配对键——8 年偏差坑已录）。

## 6. 故障域矩阵

| 故障 | 影响 | 恢复 |
|---|---|---|
| 二级 runner 崩溃（CUDA 无关，如段错误） | 自己单元死；其槽引用由 wrap-lease 定向回收 | systemd 1s 重启 + resync |
| DLA 会话/NvSci 错误 | 仅该 runner（无 CUDA context 可毒化——standalone 的第二收益） | 同上；连续失败 → orchestrator 熔断该模型 |
| 关键链卡死 | 关键链自愈机制不变（watchdog 13/systemd） | 6.1s 热恢复（已实测）；期间二级模型采样到旧帧，靠 lage 可见 |
| 二级模型把关键链拖慢（EMC/热） | 关键链 p99 上升 | orchestrator 停二级 → 关键链回住（降级联动闭环） |
| orchestrator 死 | 只损失降级联动，数据面无感 | systemd 拉起 |
| filesrc 死 | 全体消费者源枯竭 exit 20（现有语义） | systemd 拉起 + 全员 resync（现状） |

## 7. 验收与测试

门禁（全部在 30fps 满负荷源 + 共置双开条件下）：
1. 关键链精度：闭环 det/map/EPA/L2 在共置启停全周期不回归（mprodf 或
   后续 tag，对比 mprodd/mprodc 带）。
2. 关键链 SLA：帧龄 p99 <400ms、watchdog 零误杀、跳序次数记账可见。
3. 二级 SLA：每模型处理延迟 < 2×采样周期；信箱 lage 正常。
4. 降级联动 FT：注关键链慢化 → 二级被停 → 关键链 p99 回落 → 自动重开；
   注二级死循环 → wrap-lease 回收其槽 → filesrc 节拍不受扰。
5. 组合 soak：满负荷 + 双二级模型 + 故障注入脚本 8h，RSS/温度/零 13/14。

验收方法学：调度非确定 ⇒ 张量对拍仅在关键链内部自洽口径使用；跨模型/
跨 run 一律端到端指标 + SLA（延续 M7a"原模式重跑对照"纪律）。

## 8. 分期交付

| 期 | 内容 | 门禁 |
|---|---|---|
| **M10**（前置） | --skip-lag 生产跳序 · watchdog 分级弃帧（CPU 停顿限时排干→弃帧，排干失败才 exit13；连续超时接 LATCH 报警）· wrap-lease · filesrc 多消费者语义 | FT 轮 + mprode 精度门禁 |
| **M9a** | sp_dla_runner 通用宿主 + 示范模型（真实小 CNN 或 mock 分类头）走通 cuDLA standalone 全链 | 单模型 FT（kill/慢化/错引擎注入）+ 关键链零回归 |
| **M9b** | sp_orchestrator 降级联动 + per-model 遥测（sp_status 扩列）+ 组合 soak/FT | §7 全套 |

## 9. 风险与开放项

- **DLA 上模型的可编译性风险**：候选模型必须先过准入评审（纯 conv？
  int8 精度过门？DLA 延迟预算内？），不可编译的回退 = 不上（不进 SM）。
- **cuDLA standalone 与 TRT 的边界**：runner 用 TRT 编 DLA engine 还是
  cuDLA 原生 API？——M9a 第一个 spike：TRT `--useDLACore` 产物能否被
  standalone 方式加载（NVIDIA-AI-IOT/cuDLA-samples 先例是 cuDLA 原生）；
  若 TRT-DLA 引擎必须 CUDA context，则 runner 退"独立进程 + CUDA
  context"形态（隔离仍在进程边界，放弃免 context 收益）。
- **EMC 观测缺口**：tegrastats 有 EMC 频率无 util%；M9b 遥测补 EMC
  利用率采集（SM 未满但全变慢 = EMC 饱和的特征签名）。
- 重模型将来上 SM 时：窗口调度（GXF 式）或 JetPack 6 MPS——另行立项，
  本架构的 orchestrator/遥测/采样语义均为其前置。
- nvpmodel 功耗模式选择（30fps 满负荷该跑哪个模式）随满负荷 soak 数据
  定，非本 spec 范围。

## 10. 参考

- NVIDIA-AI-IOT/cuDLA-samples：cuDLA standalone（NvSci、免 CUDA
  context）与 GPU 任务真并发的官方先例（YOLOv5 on Orin DLA）。
- NVIDIA GXF（Isaac ROS）：scheduling terms（Periodic /
  MessageAvailable / ExecutionClock）= 预算/窗口原语的正式化；本设计
  剔除窗口调度但遥测接口按其字段预留。
- TimeGraph（Kato et al., USENIX）：GPU 执行时间预算/预留的学术原型。
- Eclipse iceoryx（大陆集团，车规量产）：wait-free chunk-loan，读端永不
  阻塞写端——本设计传输语义的量产参照。
- DDS QoS / AUTOSAR ara::com：KEEP_LAST(1) "last is best" 采样语义的
  标准出处。
- LMAX Disruptor：gating sequence（发布端最多领先最慢消费者一圈）=
  wrap-lease 的理论出处。
- NVIDIA 论坛（2024-10）：MPS 自 CUDA 12.5 起支持 Jetson——JetPack 6
  解锁路径的依据。
- arXiv 2025（Profiling Concurrent Vision Inference on Jetson；
  Optimizing Concurrent DNN Training/Inference on Jetsons）：多模型
  并发画像与时间片现状的第三方实测。
