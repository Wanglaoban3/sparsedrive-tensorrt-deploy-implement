import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import scatter
from projects.mmdet3d_plugin.ops import feature_maps_format

def main():
    print("====== 1. 初始化模型与加载真实数据 ======")
    cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    model = build_model(cfg.model).cuda().eval()
    load_checkpoint(model, "ckpt/sparsedrive_stage2.pth", map_location="cpu")

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    data_iter = iter(data_loader)

    data = next(data_iter)
    data = scatter(data, [0])[0]
    img = data['img'][0].cuda()
    
    if isinstance(data['img_metas'][0], list):
        img_metas = data['img_metas'][0]
    else:
        img_metas = data['img_metas']

    print("\n====== 2. 执行你提供的正常推理流程 ======")
    with torch.no_grad():
        is_first_frame = True
        mask = torch.tensor([not is_first_frame], dtype=torch.bool, device=img.device)

        # 1. 提取特征图
        feature_maps = model.extract_feat(img)
        
        # 2. 推理 Det & Map 头 (完全使用真实 data 字典，不猜 metas)
        det_output = model.head.det_head(feature_maps, data)
        map_output = model.head.map_head(feature_maps, data)

        # 3. 为 ONNX 准备独立的 Tensor 输入
        det_cls_sigmoid = det_output["classification"][-1].sigmoid()
        map_cls_sigmoid = map_output["classification"][-1].sigmoid()

        # 完全采用你代码里的提取方式
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        ego_feature_map = feature_maps_inv[0][-1][:, 0]

        bs = img.shape[0]
        T_temp2cur = torch.eye(4).unsqueeze(0).expand(bs, -1, -1).to(img.device)

        anchor_encoder = model.head.det_head.anchor_encoder
        anchor_handler = model.head.det_head.instance_bank.anchor_handler

        det_instance_feature = det_output["instance_feature"]
        det_anchor_embed = det_output["anchor_embed"]
        det_anchors = det_output["prediction"][-1]
        det_instance_id = det_output["instance_id"]
        
        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        
        # 外部维护历史状态参数（首帧给0）
        Q = model.head.motion_plan_head.instance_queue.queue_length
        num_det = model.head.motion_plan_head.num_det
        dim = 256
        
        onnx_external_states = {
            "history_instance_feature": torch.zeros(bs, num_det, Q, dim, device=img.device),
            "history_anchor": torch.zeros(bs, num_det, Q, 11, device=img.device),
            "history_period": torch.zeros(bs, num_det, dtype=torch.long, device=img.device),
            "prev_instance_id": torch.zeros(bs, num_det, dtype=torch.long, device=img.device),
            "prev_confidence": torch.zeros(bs, num_det, device=img.device),
            "history_ego_feature": torch.zeros(bs, 1, Q, dim, device=img.device),
            "history_ego_anchor": torch.zeros(bs, 1, Q, 11, device=img.device),
            "history_ego_period": torch.zeros(bs, 1, dtype=torch.long, device=img.device),
            "prev_ego_status": torch.zeros(bs, 1, 10, device=img.device)
        }

        print("\n====== 3. 打印给 forward_onnx 的真实张量形状 ======")
        print(f"img: {img.shape}")
        print(f"det_instance_feature: {det_instance_feature.shape}")
        print(f"det_anchor_embed: {det_anchor_embed.shape}")
        print(f"det_classification_sigmoid: {det_cls_sigmoid.shape}")
        print(f"det_anchors: {det_anchors.shape}")
        print(f"det_instance_id: {det_instance_id.shape}")
        print(f"map_instance_feature: {map_instance_feature.shape}")
        print(f"map_anchor_embed: {map_anchor_embed.shape}")
        print(f"map_classification_sigmoid: {map_cls_sigmoid.shape}")
        print(f"ego_feature_map: {ego_feature_map.shape}")
        print(f"mask: {mask.shape}, dtype: {mask.dtype}")
        print(f"T_temp2cur: {T_temp2cur.shape}")
        for k, v in onnx_external_states.items():
            print(f"state [{k}]: {v.shape}, dtype: {v.dtype}")

        print("\n====== 4. 测试以这些真实张量执行 forward_onnx ======")
        try:
            mo_outs = model.head.motion_plan_head.forward_onnx(
                det_instance_feature=det_instance_feature,
                det_anchor_embed=det_anchor_embed,
                det_classification_sigmoid=det_cls_sigmoid,
                det_anchors=det_anchors,
                det_instance_id=det_instance_id,
                map_instance_feature=map_instance_feature,
                map_anchor_embed=map_anchor_embed,
                map_classification_sigmoid=map_cls_sigmoid,
                ego_feature_map=ego_feature_map,
                anchor_encoder=anchor_encoder,
                anchor_handler=anchor_handler,
                mask=mask,
                is_first_frame=is_first_frame,
                T_temp2cur=T_temp2cur,
                **onnx_external_states
            )
            print("✅ forward_onnx 真实张量推理成功！没有报错！")
        except Exception as e:
            print("❌ forward_onnx 失败！")
            raise e

if __name__ == '__main__':
    main()