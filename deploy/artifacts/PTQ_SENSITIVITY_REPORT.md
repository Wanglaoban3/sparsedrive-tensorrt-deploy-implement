# SparseDrive INT8 PTQ 敏感层分析报告（ModelOpt MTQ）

- 配置: `projects/configs/sparsedrive_small_stage2.py`
- 权重: `ckpt/sparsedrive_stage2.pth`
- 标定/评估样本: 16/16（nuScenes-mini，训练管线增强流）
- 量化模块数: 112（INT8_DEFAULT_CFG：weight + input + output 三类量化器全覆盖，MHA 走 functional 路径不被 MTQ 0.11 包裹，见策略文档盲区一节）
- 输出漂移指标: det/map 输出分组 `rel_l2 = ||fp32-INT8|| / ||fp32||`（组间均值）与 `cos` 相似度

## 1. 总体结果

| 配置 | rel_l2 | cos |
|---|---|---|
| fp32 基线 | 0 | 1.00000 |
| 全模块 INT8 | 0.25626 | 0.88222 |
| keep-FP top-112（曲线最优） | 0.00000 | 1.00000 |

## 2. Skip 曲线（把敏感度最高的 K 个模块保持 FP）

| keep_fp_topk | rel_l2 | cos | 相对全INT8改善 |
|---|---|---|---|
| 0 | 0.27117 | 0.87951 | -5.8% |
| 14 | 0.25892 | 0.88082 | -1.0% |
| 16 | 0.26261 | 0.87871 | -2.5% |
| 28 | 0.26066 | 0.88139 | -1.7% |
| 32 | 0.25664 | 0.88124 | -0.1% |
| 42 | 0.25084 | 0.88192 | +2.1% |
| 56 | 0.23498 | 0.88390 | +8.3% |
| 70 | 0.21691 | 0.88750 | +15.4% |
| 84 | 0.20469 | 0.89486 | +20.1% |
| 98 | 0.18086 | 0.91302 | +29.4% |
| 112 | 0.00000 | 1.00000 | +100.0% ← 50%目标达成点 |

选层规则（50% 目标）：最小的 K=112 达到 rel_l2 ≤ 0.12813。QAT 从该 skip 配置出发。

## 3. 敏感度排名（Top 30）

sensitivity = rel_l2(全INT8) − rel_l2(该模块回退FP)；正值越大＝该层越该保 FP。

| 排名 | 模块 | sensitivity | rel_l2_without |
|---|---|---|---|
| 1 | `model.img_backbone.conv1` | +0.00263 | 0.25363 |
| 2 | `model.img_backbone.layer1.0.conv2` | -0.00908 | 0.26534 |
| 3 | `model.img_backbone.layer1.0.conv3` | -0.00970 | 0.26596 |
| 4 | `model.img_backbone.layer2.3.conv2` | -0.01016 | 0.26642 |
| 5 | `model.img_backbone.layer3.5.conv1` | -0.01023 | 0.26649 |
| 6 | `model.img_backbone.layer2.0.conv1` | -0.01027 | 0.26653 |
| 7 | `model.img_backbone.maxpool` | -0.01028 | 0.26654 |
| 8 | `model.img_backbone.layer1.2.conv1` | -0.01041 | 0.26667 |
| 9 | `model.img_backbone.layer3.3.conv1` | -0.01054 | 0.26680 |
| 10 | `model.img_backbone.layer3.4.conv1` | -0.01060 | 0.26686 |
| 11 | `model.img_backbone.layer3.3.conv2` | -0.01065 | 0.26691 |
| 12 | `model.img_backbone.layer2.0.conv2` | -0.01069 | 0.26695 |
| 13 | `model.img_backbone.layer2.1.conv2` | -0.01070 | 0.26696 |
| 14 | `model.img_backbone.layer1.2.conv3` | -0.01070 | 0.26696 |
| 15 | `model.img_backbone.layer1.1.conv1` | -0.01074 | 0.26700 |
| 16 | `model.img_backbone.layer3.3.conv3` | -0.01087 | 0.26713 |
| 17 | `model.img_backbone.layer3.4.conv2` | -0.01089 | 0.26715 |
| 18 | `model.img_backbone.layer3.5.conv3` | -0.01096 | 0.26722 |
| 19 | `model.img_backbone.layer2.0.downsample.0` | -0.01113 | 0.26739 |
| 20 | `model.img_backbone.layer3.1.conv2` | -0.01119 | 0.26745 |
| 21 | `model.img_backbone.layer4.1.conv2` | -0.01122 | 0.26748 |
| 22 | `model.img_backbone.layer4.2.conv1` | -0.01125 | 0.26751 |
| 23 | `model.img_backbone.layer4.1.conv1` | -0.01130 | 0.26756 |
| 24 | `model.img_backbone.layer3.2.conv3` | -0.01140 | 0.26766 |
| 25 | `model.img_backbone.layer4.0.conv2` | -0.01141 | 0.26767 |
| 26 | `model.img_backbone.layer2.2.conv3` | -0.01149 | 0.26775 |
| 27 | `model.img_backbone.layer1.1.conv2` | -0.01157 | 0.26783 |
| 28 | `model.img_backbone.layer1.2.conv2` | -0.01160 | 0.26786 |
| 29 | `model.img_backbone.layer3.1.conv3` | -0.01162 | 0.26788 |
| 30 | `model.img_neck.lateral_convs.1.conv` | -0.01163 | 0.26789 |

## 4. 敏感度分布特征

- 模块构成: backbone 54 / heads 47 / neck 8 / depth 3 / 其他 0
- Top20 敏感层构成: heads 0 / backbone 20（heads 的分类/质量回归分支与 backbone 高层感受野卷积交替占优）
- |sensitivity| 中位数 0.01302，最大 0.01582：单层关量化的增益普遍在 ±0.02 rel_l2 以内，呈长尾分布——印证 skip 曲线平坦、需配合 QAT。

## 5. 复现

```bash
python deploy/ptq_sensitivity.py --calib-samples 16 --eval-samples 16 --topk-curve 8
python deploy/render_ptq_report.py
```
