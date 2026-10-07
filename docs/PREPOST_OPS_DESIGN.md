# Orin 侧前后处理加速算子库 — 设计稿(评审中,未实现)

> 状态:**已评审通过,实现中**(2026-10-03 按评审意见更新)。
> 已确认的前提:双链路形态(本设计只做"链路一",链路二离线精度口径不动)、
> 采集接口直接发布 NV12、~~单引擎 e_T6(不拆分)~~。
> **2026-10-04 更新**:感知交付改为拆分链 `e_bb2+e_hd`(run_engine2 D2D,
> map mAP 0.7481 vs 单引擎 0.6471,第十一轮终审)+ 新增 MP 二级引擎
> `e_mp`(8.74ms,指标≈FP32,first 双图方案废弃)。里程碑重编号:
> **M5=拆分链接入, M6=MP 接入, M7=重叠优化**,原 dmabuf/VIC 里程碑顺延
> 为 M8/M9;新增 6.1 节设计定稿(用户批准,精度优先)。
> 已确认的 IPC 语义:图像在 CPU 读取解码;订阅方通过映射+注册零拷贝只读
> 访问,不可写;引用计数回收锁(消费方未处理完,发布线程不得回收槽位);
> 6 路相机 = 6 个订阅点指针,注册后 CUDA 算子写入一块连续缓冲直绑引擎;
> 去畸变走 LUT(字典)预计算,nuScenes 数据已去畸变,全流程需以测试数据
> 验证数值恒等;前处理数值差异按"影响微乎其微"对待。
> 评审批准的增量:大项 = 源适配器接口 + dmabuf 采集源、model_node 双缓冲
> 流水线、fence 回收;小项 = CUDA graph 实测、latest-wins 结果信箱 +
> 分级时间戳、ring 深度 4、init 预热、源状态位;回放形式定为 **NV12 图像
> 帧**(不引入视频流),回放时长加长(见"长回放数据"节)。

## 1. 目标与边界

在 Orin 上构建完整、参数化的前处理 + 后处理 CUDA/C++ 算子库,组成"实时接口"
链路:本地文件模拟相机输出 → 固定内存零拷贝发布 → 模型节点订阅 → 前处理 →
~~单引擎推理(e_T6,backbone+head 同引擎;P4 已实测拆分更慢,不做拆分)~~
**2026-10-04 起:拆分链 e_bb2→e_hd + MP 二级引擎 e_mp(见 6.1 节;P4"拆分
更慢"被第十一轮 mAP 归因推翻——0.87ms 代价换 +10.1pt map)** →
后处理 → 检测结果发布。

**边界**:
- 链路二(服务器预处理数据 + `run_engines`,只测 engine 精度/性能)保持完全不动;
- 不改模型/训练侧任何东西;板子 TRT/CUDA/JetPack 环境不动(只新增部署产物)。

## 2. 已钉死的契约(来自现有代码,实现必须逐位对齐)

| 项 | 契约 |
|---|---|
| 引擎输入 | `img [1,6,3,256,704]`、`projection_mat [1,6,4,4]`、时序状态(prev_det_feat/anchor/conf/id、map_feat/anchor/conf、time_interval) |
| 引擎输出 | det_cls/quality/bbox + `next_det_*`;map_cls `[1,100,3]`、map_pts `[1,100,40]`(lidar 系绝对坐标,20 点折线)+ `next_map_*` |
| 测试时几何 | resize=0.44 → 704×396 → crop y∈[140,396) → 704×256;`cam_intrinsic[:3,:3]×=0.44`;crop 平移并入 lidar2img |
| 归一化 | mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],RGB 顺序 |
| 性能口径 | 引擎 38.99ms(25.6FPS);链路一目标:前处理+后处理+IPC 总开销 ≤ 引擎耗时,端到端 ≥25FPS。**M4 实测达成**: 闭环含前处理+IPC+19 路 D2H 为 27.5fps(graph, 不限速), ≥25fps 目标 ✓ |

## 3. 进程拓扑

### 方案 A(推荐):两进程最小闭环

```
proc 1: capture_pub                     proc 2: model_node
┌────────────────────────┐             ┌──────────────────────────────────┐
│ CPU 读本地 JPG/RAW      │  shm ring   │ 订阅帧 = 6 个 NV12 指针 + 元数据   │
│ → 解码 NV12 1600×900×6 │ ──────────> │ ↓ 只读映射注册(6 个订阅点)         │
│ 写入槽位,等全消费方     │  只读映射    │ ↓ 前处理算子链(参数化)             │
│ 释放后才能回收槽位       │             │ LUT undistort+resize+crop+norm    │
│            ↑           │             │ → 一块连续 device 缓冲(直绑引擎)    │
│ 结果 topic <────────────┼── shm ──────┤ ↓ 单引擎 e_T6 (TRT + DFA 插件)     │
│ JSON 旁路 / 可视化      │             │ ↓ 后处理算子链(阈值参数化)          │
│                        │             │ det/map decode + 时序状态回写       │
│                        │             │ 发布 topic "results"              │
└────────────────────────┘             └──────────────────────────────────┘
```

- **节点总线**:自研轻量跨进程 shm pub/sub(固定内存环形缓冲,topic 寻址,
  单写多读)。总线本身跨进程,后续拆更多节点只改配置不改库。
- 备选 B(四进程全拆分):叙事最强但联调成本最高,首期不做;
- 备选 C(单进程线程化):不满足"跨进程发布"的字面要求,不采纳。

### IPC 内存契约(零拷贝 + 只读 + 回收锁)

- **发布侧**:CPU 解码后的 NV12 直接写入 shm 槽位(捕获进程是唯一写者)。
- **订阅侧 = 注册机制**:消费进程 `mmap(PROT_READ)` 只读映射槽位;GPU 侧把
  映射区间 **`cudaHostRegisterMapped`** 注册,再用
  **`cudaHostGetDevicePointer` 取设备指针**交给 kernel(Orin 统一内存)。
  **板端实测红线**:用 `cudaHostRegisterDefault` + 直接解引用 host 指针 =
  GPU 页表没有该映射 → illegal memory access,nvgpu 拆通道、进程崩;
  设备指针与 host 指针**不同值**,kernel 一律走设备指针。注册后 GPU 直读
  带宽实测 12.96 GB/s(12.9MB 帧仅 ~1ms)。6 路相机 = 6 个订阅点/6 个
  只读指针,零拷贝交给前处理 kernel。消费进程对图像内存没有任何写权限。
  另外:GPU 正在读的槽位若被发布端覆写(race)同样炸通道 —— 回收锁+fence
  在本板是硬性要求,不是锦上添花。
- **回收锁(引用计数)**:每槽位带进程间引用计数 + 共享互斥。消费方拿到帧
  即持有一个引用,处理完(前处理拷贝进连续 device 缓冲之后)才释放;引用
  未清零前,发布线程**不得回收/覆写该槽位**,只能写下一个槽位。
  环深 = 槽位数,发布方遇全槽未释放则阻塞或丢最旧(策略可配)。
- **崩溃安全**:消费方持引用时崩溃会让槽位永远锁死 → 每个订阅方带租约
  心跳,超时未释放由发布方强制回收并记事件(首版可用固定超时)。
- **同步组**:一个帧消息 = 6 路相机同一触发时刻的一组指针(实测 nuScenes
  6 路是软同步,时间戳差毫秒级;回放按一组发布,消息里保留每路时间戳,
  供将来接真实相机的同步组语义)。

### 源适配器接口(量产形态的关键抽象)

```cpp
class IImageSource {           // 图像源 = 槽位的生产者,可替换
  virtual int  open(const SourceConfig&) = 0;
  virtual int  start() = 0;
  // 阻塞取一组同步帧:6 路只读平面指针 + 每路时间戳 + 场景/状态标志
  virtual int  acquire(FrameView& out) = 0;   // acquire 后持引用
  virtual void release(FrameView&) = 0;       // 处理完(含 fence 等待)后释放
  virtual int  stop() = 0;
};
```

- 首个实现 = **文件回放源**(manifest json 驱动:每帧 6 路 NV12 路径 +
  时间戳 + 场景 id,发布进 shm ring);
- 目标实现 = **dmabuf/NvBufSurface 采集源**(M5):池分配 + fd 发布 +
  消费侧 cudaExternalMemory 接入,model_node 一行不改;
- FrameView 带源状态位(丢帧/链路异常),文件源恒 OK,真相机直传;
- 回收升级为 **fence 语义**(M1.5):消费者前处理 kernel 后
  cudaEventRecord,引用释放在事件完成后,租约心跳只作崩溃兜底。

## 4. 算子清单

**前处理**(6 个订阅点指针 → 注册 → CUDA → 一块连续输出):
- P0 重采样 LUT 预计算(字典形式,CPU/离线一次性,按配置缓存):键 = 相机名,
  值 = undistort(K、畸变系数 D=(k1,k2,p1,p2,k3))+ resize + crop + flip
  折叠后的重采样表,常驻 device。运行时 kernel 只查表,零重复计算。
  **nuScenes 数据已去畸变**:D=0 时 LUT 退化为 resize/crop 部分;用测试数据
  把完整 undistort 流程跑一遍并逐位对齐验证(数值无任何变化),流程真实
  存在但对该数据集是恒等——为换真实相机预留,不引入数值差异。
- P1 NV12→RGB 重采样 kernel(6 路并行消费注册的只读指针,查 P0 LUT)+
  归一化融合,6 路输出写进**一块连续 device 缓冲**,布局即引擎输入
  `[1,6,3,256,704]`(FP16 或 FP32 可配)→ 引擎 input binding 直绑该缓冲,
  前处理到推理之间零额外拷贝、零中间全分辨率缓冲。
- P2 projection_mat 更新:ida 矩阵(参数化 scale/crop)× lidar2img。

**后处理**:
- Q1 det decode:sigmoid + topk(900→300)+ quality 融合 + exp(whl) + atan2(yaw)+vel。
- Q2 det 分数阈值(参数化)+ 时序状态回写(next_det_* → 下帧 prev_*)。
- Q3 map decode:sigmoid(map_cls)+ 阈值(参数化)+ 20 点折线组装。
- Q4 时序状态管理器:设备内存驻留,首帧零填(prev_* 全零、id=-1、identity t_matrix、
  time_interval=0.5),与链路二口径一致;**场景边界只重置 t_matrix=identity +
  dt=0.5,实例状态跨场景保留**(链路二 in_40 输入实测:id/feat/anchor/conf/
  id_count 全保留,M3 已按此对齐)。

**结果消息**(初版,版本化 struct + JSON 旁路):
header(seq/ts/frame_id/scene_flags/config_hash/plugin_path)+ 各级时间戳
(t_capture/t_preproc/t_infer/t_postproc)+ det[]{score,label,x,y,z,w,l,h,
yaw,vx,vy,id} + map[]{score,label,pts[20][2]}。发布走**单槽 latest-wins
信箱**(seqlock),下游永远取最新帧;发送前 init 预热 N 帧再对外服务。

## 5. 参数化配置(单一 JSON)

```
cameras[6]: { name, K[3][3], D[5], resize, crop[x,y,w,h], flip }
model:      { engine, plugin, det_topk, det_thr, map_thr }
            (2026-10-04 起 → engines{bb2,hd,mp} 三路径,见 6.1 节;
             engine 单路径字段保留作回退)
runtime:    { ring_depth, input_dir, output_dir, target_fps }
```

换车/换相机 = 换配置文件,不改代码。

## 5.5 长回放数据(形式已定:NV12 图像帧)

本地已具备完整 nuScenes mini(train+val):404 个关键帧 + sweeps +
v1.0-mini 元数据。回放源吃 **manifest json**(每帧:6 路 NV12 路径 +
时间戳 + 场景 id),三档预置:

| 档位 | 内容 | 节奏 | 用途 |
|---|---|---|---|
| R0 | val 81 关键帧(现状) | 2Hz,~40s | 与 det/map 基线同口径 |
| R1 | train+val 全部 404 关键帧 | 2Hz,~3.5min | 长时间稳定性/延迟统计 |
| R2 | 关键帧+sweeps 按 12Hz 真节奏 | 12Hz,~33min | 流水线压力/吞吐(明确标注:帧间隔 OOD,精度数字不可比) |

场景边界一律触发时序状态重置(与链路二口径一致);R2 只做链路压测,
不做精度口径。manifest 生成脚本随库交付(python,本地跑)。

## 6. 里程碑(每步独立可验证,遵守模块级编译纪律)

| 里程碑 | 内容 | 验证 |
|---|---|---|
| M1 ✅ **完成**(2026-10-03) | 总线库(shm ring 深度 4 + 槽位引用计数 + 进程间互斥 + 租约)
      + 源适配器接口 + 文件回放源(manifest) + model_node 空转订阅
      (只读映射 + cudaHostRegister **一次性注册整个 ring**) | 板上两进程收发 +
      延迟;崩溃注入(杀消费进程→租约超时强制回收)。
      **板端实测**:2Hz 回放 8 帧 crc_err=0/gaps=0;`cudaHostRegister 51.8MB`
      成功;崩溃注入 5 次 claim 超时(5s 租约)后 forced=1 回收,重加入
      consumer id=1 零错误,传输延迟 p50=3.4ms。本地(WSL)先验证:100fps
      精确节奏、背压、崩溃恢复、重加入、全平面图案校验。文件:
      deploy/prepost/{sp_bus,image_source,file_source,sp_filesrc}+
      {sp_pub,sp_sub,sp_inspect};驱动 deploy/board_m1.py(build/data/run/
      crash 四阶段);R0 数据 tools/make_nv12_manifest.py→work_dirs/nv12_r0
      (81 帧×6 路 BT.601 full-range,场景边界 k=40 复核,往返 MAE 0.1) |
| M1.5 ✅ **完成**(2026-10-03) | fence 回收(cudaEvent 记账,引用释放挂事件后)+ CUDA stream 纪律 | 回收正确性
      压测。**板端实测**:发布 300fps、消费端 kernel 延迟 30ms 后才读槽位,
      fence 模式 20 帧 fence_err=0(引用挂事件,发布端永不覆写);
      对照 race 模式(发射即释放)= **GPU 读-发布端写并发 → nvgpu 通道
      fault,进程直接崩** —— 覆写危害的最强演示,fence 在本板是硬要求。
      顺带完成零拷贝验证(设计风险项):`cudaHostRegisterMapped` +
      `cudaHostGetDevicePointer` 后 GPU 直读 registered shm 带宽实测
      **12.96 GB/s**(12.9MB/1ms);**Default 注册 + host 指针解引用 =
      illegal memory access**(GPU 页表无该映射),必须走 Mapped+设备指针。
      文件:deploy/prepost/sp_kernels.{h,cu} + sp_sub --fence/--race +
      board_m1.py fence 阶段 |
| M2 ✅ **完成**(2026-10-04) | P0 LUT(轴分离定点重采样表;**Pillow 12.3
      Resample.c 定点公式逐位复刻**: kint=round-half-away(w·2^22),
      ss=(2^21+Σkint·u8)>>22, 猜公式必错、以源码为准) + P1 三 kernel
      (NV12→RGB BT.601 full-range 浮点 → 水平定点查表 → 垂直+归一化融合,
      直出 [1,6,3,256,704] 连续缓冲) + P2(ida×lidar2img, host double) |
      **板端实测**: 帧 0/1/40(含跨场景帧) vs numpy/PIL 参考
      **100.000% 位级一致**, 1.32ms/帧; NV12 源地板 vs 链路二 JPG 管线
      MAE 0.0013(归一化域)。文件: preproc.{h,cu} + sp_preproc_test +
      board_m1.py m2 阶段 + preproc_ref.py + compare_m2.py |
| M3 ✅ **完成**(2026-10-04) | 引擎接入 + 时序闭环(sp_modelnode: 总线消费
      → P2/t_matrix/dt host 计算 → M2 前处理直绑 → enqueueV3 →
      next_*→prev_* D2D 反馈 → 19 路 dump; 零状态 f32=0/id=-1/count=0;
      **场景边界口径=链路二实测**: 只重置 tmat=identity+dt=0.5, 实例状态
      跨场景保留, id 计数器由引擎自驱动 +300/帧) | 81 帧闭环零丢帧
      (filesrc blocked=0 forced=0, seq 严格对齐), pre=1.45ms/帧,
      infer=44.8ms。位级隔离闭环(node 逐帧吃链路二 img.bin): f0 全输出
      rel_l2 ~1e-3; map 支路反馈锁定; **next_id_count 全 81 帧 100% 一致**;
      det 状态同实例集+逐 id 内容 ~1e-2 —— 元素级发散(f1 起)为引擎状态
      输出**行序非确定性**(同输入两进程自身 ~1e-1 抖动)+ NV12 源地板经
      top-k 选择重排放大, 集合级无害, 状态对比须按 id join。
      NV12 源整链 f0: det_cls 2.8e-2 / map_cls 9.7e-3。
      文件: sp_modelnode.cpp + board_m1.py m3 阶段 + compare_m3.py |
| M4 ✅ **完成**(2026-10-04; 性能/稳定性口径全达成, 链路一输入 mAP 复测
      单独挂起——等场景边界口径排查结论) | 双缓冲流水线(3 stream, 2 帧在飞:
      img/参数/输出按奇偶双份 + **双 IExecutionContext 固定绑定**免逐帧重绑;
      fence 挂事件异步归还; pinned D2H; 落盘出关键路径) + CUDA graph 实测
      (推理+反馈 D2D 按 parity 捕获两张, 1371 节点, DFA 插件可捕获) +
      分级时间戳(每帧 frame_log.tsv: pre/infer/post/gpu/svc/acq) | 板端实测
      (e_T6+libdfaplug_v8): 交付基线 38.99ms/25.6fps = 裸引擎无前后处理;
      闭环 serial e2e p50 41.1ms; 流水线不限速 23.9fps; **+graph: 节拍下
      e2e p50 36.5ms / p99 37.6ms, 不限速 27.5fps(超裸引擎基线), infer
      p50 36.3ms**。243 帧回绕压测(3 次场景边界): fps 零漂移(27.53 vs
      27.56), infer 抖动 p99-p50=0.64ms, RAM 无增长(4429≤4568MB),
      forced=0 零回收, GR3D 稳态 p50=99%。口径注意: 事件链 gpu_ms 含
      队列等待(饱和时 ≈2×单帧服务), 延迟(节拍跑)与吞吐(不限速跑)分开测。
      文件: sp_modelnode.cpp(重写) + board_m1.py m3(--graph/--serial/
      --fps/--fetch log/--loop) + deploy/_probe_m4_report.py |
| 算子库收尾 ✅ **完成**(2026-10-04) | Q1 det decode + Q2 阈值参数化 +
      Q3 map decode(postproc.h; **口径逐位对齐离线评测脚本**
      eval_t6_mini_v8 / eval_t6_mini_map: sigmoid → 摊平 9000 稳定
      top300 → ×sigmoid(quality[a][0]) 稳定重排 → exp(whl)/atan2(yaw);
      map 类别优先序 s>floor 全保留) + 结果消息(版本化 struct: header
      seq/ts_capture/scene_flags/config_hash/plugin_path + 分级时间戳 +
      det[]{score,label,x,y,z,w,l,h,yaw,vx,vy,id} +
      map[]{score,label,pts[20][2]}) + 单槽 latest-wins 信箱(sp_result.h
      **seqlock**: 单写多读, 写方永不阻塞读方) + sp_resultmon 读端工具
      (--wait-ms 等创建 / --json / crc 校验) + 源状态位贯穿
      (FrameMeta.flags bit0=source_ok, bits8..13=cam[0..5]_ok;
      结果消息低 8b 重映射打包) |
      板端实测: 12 帧 C++ decode vs numpy 参考全 PASS — det
      数量/label/id 全等, score max|d|=5.1e-7, box 9.5e-7,
      map 9.5e-7(f32 舍入量级); 信箱 live 并发(读端先起, 27fps 发布
      ×60 帧): **new=60 torn=0** crc 全对; decode+发布 0.49ms/帧,
      JSON 旁路仅验证跑(+5.9ms); flags=127(source_ok+6 路 cam_ok)
      端到端可见。文件: postproc.h + sp_result.h + sp_resultmon.cpp +
      sp_modelnode.cpp(--det-thr/--map-thr/--det-topk/--mailbox) +
      deploy/_probe_decode_check.py |
| DLA 实验 ✅ **完成**(2026-10-04; 用户批准的一次全量构建尝试) |
      结论: **本模型/本栈 DLA 离载不可行, 该优化杠杆关闭**。三级证据:
      (1) 全图 e_T6 trtexec --useDLACore=0 --allowGPUFallback:
      QDQ 网络必须加 --int8, 加后 DFA 插件 format 协商全灭
      (v8 插件 supportsFormatCombination 只收 kHALF/kLINEAR,
      DLA 混合编译器给出的组合全部被拒 → DeformableAggregation
      'could not find any supported formats'; 且 REDUCE/UNARY/
      ELEMENTWISE/SHUFFLE/非C-CONCAT 全报 Unsupported on DLA,
      可上 DLA 的只有 backbone 卷积);
      (2) backbone 单独探针(sp_backbone2.onnx, --int8 --fp16):
      **DLA 60.7ms vs GPU 15.3ms — DLA 慢 4 倍**(DCN 类算子回退 GPU
      + DLA↔GPU 每边界 reformat + DLA 低频), 'backbone 上 DLA' 的
      未来路径同样关闭;
      (3) 板侧 /dev/nvhost-ctrl-nvdla0/1 存在(硬件通道在), 阻碍在
      算子覆盖与插件契约, 不在设备节点。板上探针产物已清理,
      e_T6.engine 未受影响; 交付维持 GPU e_T6 + v8 + CUDA graph
      (27.5fps 已达标)。文件: _probe_dla_*.py(探针脚本留档) |
| M5 ✔ **完成**(2026-10-04) | 拆分链接入:sp_modelnode
      装载 e_bb2+e_hd;col_feats [1,89760,256] f16 边界张量按 parity 固定
      分配、D2D 直供(run_engine2 同款零拷贝);graph 按 bb2/hd × 2 parity
      捕 4 张;next_* 感知状态反馈不变(全来自 hd,bb2 无状态)。
      **闭环 det 0.4177/NDS 0.4735/map 0.7478,门禁全过**
      (基线 0.4160/0.4729/0.7481);legacy 0.4210/0.4740。
      过程抓出并修复 mat4_inv 双坑(M2 潜伏的 MESA 转录错 → 闭环
      -10pt,见 OPTIMIZATION_SUMMARY 第十二轮);复测驱动
      deploy/_m5_rerun.py,dump 留档 mini_m3fix_81/mini_m5fix_81 |
| M6a ✔ **完成**(2026-10-04) | MP 接入:e_mp 同进程第三级;9 感知张量自 hd 输出直供(名字映射
      **全等匹配**)+ 9 张历史状态外旋 D2D + **场景首帧全清零**
      (cudaMemsetAsync 同流);逐帧 dump。补齐状态输入缓冲分配
      (单缓冲双 parity 同址)。**mp 段 8.85ms;闭环 EPA 0.6072/0.5112、
      L2 0.7456、col 0.161%(离线参考量级过门);同源对拍 plan 侧
      全 81 帧 ≤0.5% 含边界,复位口径与离线一致** |
      板上 repro/mini_pipeline_mp.sh 以 M5 闭环 dump 为 SRC 生成**同源参考**
      (mpSRC 符号农场)→ 逐张量对拍(f0 干净/motion 侧混沌地板)+
      eval_mp_mini 端指标过门;dump 留档 mini_m6fix_mp/preproc_ref/mpref |
| M6b ✔ **完成**(2026-10-04) | 信箱 v2:版本号 bump,v1 字段全保留,+ motion_cls/reg +
      plan_cls/reg/status + final_plan[6,2](argmax 解码,口径=
      eval_mp_mini final_planning,--cmd 默认 2 直行,真 cmd 车辆接口)+
      时间戳加 t_mp;sp_resultmon 兼容并打印 v2 plan 摘要 |
      12 帧 C++ 解码 vs numpy 参考(同 cmd 复算 plan_cls/reg dump)
      **12/12 PASS**(mode 全一致,maxdiff ≤5e-4);坑:mon 名自带
      sp_res_ 前缀/文件重定向块缓冲被 pkill -9 丢(stdbuf -oL+SIGTERM) |
| M7(后置,可选) | bb2(k+1)∥hd(k)+mp(k) 重叠 + e_mp graph:3 引擎 ×
      2 parity = 6 context/graph,冲 ≥25fps 吞吐(稳态周期
      max(bb2 15.8, hd+mp 32.7)=32.7ms ≈ 30fps 上限) | 吞吐/延迟双口径
      (M4 表);精度门 = M5/M6 对拍口径复跑不劣化 |
| M8(原 M5 顺延) | dmabuf/NvBufSurface 采集源(池 + fd 发布 + cudaExternalMemory) | model_node
      零改动跑通同链路 |
| M9(原 M6 顺延,可选) | VIC 几何离载实测 | 吞吐/带宽报告 |

## 6.1 拆分链与 MP 接入设计(M5/M6/M7,2026-10-04 用户批准,精度优先)

背景:感知模块最终交付 = 拆分链 e_bb2+e_hd+run_engine2(39.86ms,det
0.4160 / NDS 0.4729 / map 0.7481);MP = 单时序引擎 e_mp(8.74ms,mini 81
帧指标≈FP32)。本节把两者接入链路一。**已批取舍:精度优先,串行节拍
~20fps 可接受,吞吐优化后置 M7。**

### 数据流(同进程三级引擎,单主流,沿用 M4 纪律)

```
filesrc → shm ring → sp_modelnode(单进程单主流)
  acquire → P1/P2 前处理(不动,1.45ms)
  → e_bb2(graph×2 parity)→ col_feats 边界张量 D2D 直供(~46MB f16,
    按 parity 固定分配,bb2 输出与 hd 输入绑同一缓冲,零 memcpy)
  → e_hd(graph×2 parity)→ det/map 输出 + ego_feature_map
       ├─ next_* 感知状态 D2D 反馈(现有逻辑不动)
       └─ 9 个感知张量直供 e_mp(det_cls/bbox/instance_feature/
          anchor_embed/instance_id、map_cls/instance_feature/
          anchor_embed、ego_feature_map)
  → e_mp ← 9 张 mp 历史状态 D2D 反馈(场景首帧全清零)
  → Q1-Q3 后处理(det/map 不动)+ motion/plan 段 → 信箱 v2 + 逐帧 dump
```

串行节拍 ≈ pre 1.45 + bb2 15.8 + hd 24 + mp 8.7 + post 0.5 ≈ 50.5ms
≈ 20fps,端到端延迟同量级;M7 重叠后稳态吞吐上限 ≈30fps。

### 决策点(已批)

- **MP = sp_modelnode 同进程第三级引擎**(非独立总线节点):状态反馈同流
  定序、感知张量零拷贝直供,与离线已验证形态一致。拆独立节点要为原始
  张量开 shm topic(~1.2MB/帧)+ 跨进程同步,后置。
- **信箱 v2**:v1 字段全保留,追加 motion_cls/reg、plan_cls/reg/status、
  final_plan[6,2](argmax 解码,口径 = eval_mp_mini 的 final_planning)、
  时间戳加 t_mp;原始张量不进信箱(体积),验证走逐帧 dump。
- **e_mp CUDA graph**:先实测裸 enqueueV3 发射开销,>1ms 则按 parity 捕
  两张(flag 可切换);M4 经验:发射 ~6ms/千节点。

### 两套场景边界口径并存(Q4 扩展)

- 感知(不变):场景边界只重置 t_matrix=identity + dt=0.5,实例状态跨
  场景保留(M3 实测口径)。
- MP:**场景首帧 9 张历史状态全清零**(f32=0、prev id=-1、period=0)——
  与离线 mini_mpstate0 口径一致。node 内两套状态各自管理,不得互相波及。

### 验证门

- **M5 门**:81 帧闭环 det/map dump vs evaldata/mini_sp2_81 对拍(M3
  口径:状态按 id join、行序非确定)+ 补 M4 挂起的链路一 det/map mAP
  复测(应复现拆分链离线口径 0.4160 / 0.7481 量级)。
- **M6 门**:**不得拿 mini_mp_eng 直接当参考**——其感知输入来自单引擎
  reset 链,与拆分链 hd 输出行序/数值不同,mp 匹配分支按 id 走,逐张量
  必发散。先在板上用 repro/mini_pipeline_mp.sh 以 M5 闭环 dump 为 SRC
  重跑生成同源参考,sp_modelnode 的 mp 输出对拍之;再 eval_mp_mini 出
  端指标(EPA≈0.594/0.509、L2≈0.743 量级为过门)。
- **M6b 门**:12 帧 C++ final_plan/motion 解码 vs numpy 参考全 PASS
  (照算子库收尾验收口径:集合全等 + 分数 ≤1e-6)。

### 风险与对策

- **col_feats 46MB × 2 parity 显存驻留**:graph 捕获要求地址固定 → 启动
  时按 parity 各分配一次,不逐帧 cudaMalloc。
- **hd→mp 张量名映射**(det_instance_feature→det_feat 等)在 node 内按
  **全等**匹配——M2/M3 教训:子串 find 会踩 prev_det_id 陷阱。
- **mp 状态清零用 cudaMemsetAsync 同流定序**,禁 host memset+H2D
  (与 M4 流水线纪律一致:异步源不能是栈变量/即时值)。
- **回退**:e_T6 单引擎路径代码保留不删,回退组合仍在交付清单;node 不做
  双模式运行时开关,要回退用 git 留档。

## 7. 风险与对策

- **前处理数值差异(定位:微乎其微,按确认项对待)**:CUDA 双线性 vs PIL
  双线性、JPEG↔NV12 色彩往返会有像素级差异;判断是影响可忽略,不作为设计
  风险。M2 逐像素 max-abs-diff + M4 mini mAP 复测只做量化留档,确认量级,
  不构成推进门槛。NV12 采集端固定 BT.601 full-range,前后端同矩阵。
- **统一内存零拷贝**:~~需 M1 先验证~~ **已验证(2026-10-03)**:
  `cudaHostRegisterMapped` + `cudaHostGetDevicePointer` 路径可用,GPU 直读
  registered shm 实测 12.96 GB/s;Default+host 指针解引用不可用(炸通道),
  见 IPC 内存契约红线。
- **回收锁死锁**:消费方崩溃导致槽位永不释放 → 租约心跳 + 超时强制回收
  (见 IPC 内存契约),M1 里带上崩溃注入测试。
- **不碰板子环境**:只交付 .cpp/.cu/.so/JSON;编译用板上 nvcc,命令均模块级。

## 8. 工业对齐待批清单(评审提出、尚未批准,先记录不实现)

- 发布清单 manifest(engine/plugin/onnx/config SHA256 + 编译命令行 + TRT/
  CUDA/JetPack + nvpmodel 电源模式 + 评测结果哈希);
- 全量 6019 帧 val 精度闭环(见 OPTIMIZATION_SUMMARY 下一步 #2);
- 进程监护(崩溃自动重启)与结果心跳/陈旧报警、NaN 合理性检查、
  reset 控制话题;
- 色彩矩阵配置化(BT.601/709 × full/limited,当前固定 601 full-range);
- INT8 标定集/缓存版本化记录 + "换前处理是否重标定"决策记录;
- 为什么不用 ROS 2/DDS 的定位说明(自研轻量总线 = 引擎/算子层交付的
  显式零拷贝契约,非通用中间件替代)。
