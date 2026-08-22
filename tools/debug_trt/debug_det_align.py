import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
import projects.mmdet3d_plugin 

# 全局字典保存每一层的输出
NATIVE_VARS = {}
ONNX_VARS = {}

def hook_fn(dict_to_save, layer_name):
    def hook(module, input, output):
        # 如果像 refine 这种返回 Tuple (anchor, cls, qt) 的，分别保存
        if isinstance(output, tuple):
            dict_to_save[layer_name + "_out_0"] = output[0].clone().detach() if output[0] is not None else None
            dict_to_save[layer_name + "_out_1"] = output[1].clone().detach() if output[1] is not None else None
        else:
            dict_to_save[layer_name + "_out"] = output.clone().detach() if output is not None else None
    return hook

def run_layer_debug(model, data_loader):
    model.eval()
    det_head = model.head.det_head
    nh_det = det_head.instance_bank.num_temp_instances 
    
    prev_instance_feature = torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda')
    prev_anchor = torch.zeros((1, nh_det, 11), dtype=torch.float32, device='cuda')
    prev_confidence = torch.zeros((1, nh_det), dtype=torch.float32, device='cuda')
    prev_instance_id = torch.zeros((1, nh_det), dtype=torch.float32, device='cuda')
    prev_id_count = torch.zeros((1, 1), dtype=torch.float32, device='cuda')
    
    prev_global_mat = None
    prev_time = None

    for i, data in enumerate(data_loader):
        if i > 0: break # 我们只查出问题的 Frame 0
        print(f"\n{'='*20} Frame {i} Transformer 层级逐层比对 {'='*20}")
            
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            kwargs = scattered_data.copy()
            kwargs.pop('img')
            img_metas = kwargs['img_metas'][0]

            curr_time = img_metas['timestamp']
            dt = 0.5 if prev_time is None else curr_time - prev_time
            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            
            curr_global = img_metas['T_global']
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = img_metas['T_global_inv'] @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)

            # 骨干网络提取原始特征
            features_raw = model.extract_feat(img, metas=kwargs)

            # ==========================================
            # 1. 挂载 Hook 并运行 Native (使用深拷贝的特征防止污染)
            # ==========================================
            hooks = []
            for idx, op in enumerate(det_head.operation_order):
                if det_head.layers[idx] is not None:
                    hooks.append(det_head.layers[idx].register_forward_hook(hook_fn(NATIVE_VARS, f"iter_{idx}_{op}")))
            
            # 【关键】硬复制特征，防止 Native 原地修改
            features_nat = [f.clone() for f in features_raw]
            _ = det_head(features_nat, kwargs)
            
            for h in hooks: h.remove()
            hooks.clear()

            # ==========================================
            # 2. 挂载 Hook 并运行 ONNX (再次使用深拷贝的特征)
            # ==========================================
            for idx, op in enumerate(det_head.operation_order):
                if det_head.layers[idx] is not None:
                    hooks.append(det_head.layers[idx].register_forward_hook(hook_fn(ONNX_VARS, f"iter_{idx}_{op}")))
            
            features_onnx = [f.clone() for f in features_raw]
            _ = det_head.forward_onnx(
                feature_maps=features_onnx,
                prev_instance_feature=prev_instance_feature,
                prev_anchor=prev_anchor,
                instance_t_matrix=instance_t_matrix,
                time_interval=dt_tensor,
                prev_confidence=prev_confidence,
                prev_instance_id=prev_instance_id,
                prev_id_count=prev_id_count,
                metas=kwargs
            )
            for h in hooks: h.remove()

            # ==========================================
            # 3. 逐层对比
            # ==========================================
            for k in sorted(NATIVE_VARS.keys(), key=lambda x: int(x.split('_')[1])):
                v_nat = NATIVE_VARS[k]
                v_onnx = ONNX_VARS.get(k)
                if v_nat is not None and v_onnx is not None:
                    err = (v_nat - v_onnx).abs().max().item()
                    if err > 1e-4:
                        print(f"❌ {k:<25} MAX Error = {err:.6e}  <-- 【发散源头在这!】")
                    else:
                        print(f"✅ {k:<25} MAX Error = {err:.6e}")
                elif v_nat is None and v_onnx is None:
                    continue
                else:
                    print(f"❌ {k:<25} Type Mismatch")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.task_config['with_map'] = False
    cfg.task_config['with_motion_plan'] = False
    if 'head' in cfg.model:
        cfg.model.head.task_config = cfg.task_config

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    
    run_layer_debug(model, data_loader)

if __name__ == "__main__":
    main()