# Copyright (c) OpenMMLab. All rights reserved.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorrt as trt
import argparse
import mmcv
import os
from os import path as osp
import sys
import ctypes
from collections import OrderedDict

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

# 💡 核心导入：激活 MMDetection3D 和 SparseDrive 注册表
import mmdet3d
import projects.mmdet3d_plugin

# ==============================================================================
# 🚀 1. TensorRT 推理引擎封装 (消除警告版)
# ==============================================================================
class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT context.")

        self.inputs, self.outputs, self.bindings = OrderedDict(), OrderedDict(), []
        
        for i in range(self.engine.num_bindings):
            # 兼容 TRT 8.5+，避免弃用警告
            if hasattr(self.engine, 'get_tensor_name'):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            else:
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            shape = [s if s > 0 else 1 for s in shape]
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            gpu_mem = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
            self.bindings.append(gpu_mem.data_ptr())
            
            if is_input:
                self.inputs[name] = gpu_mem
            else:
                self.outputs[name] = gpu_mem

    def infer(self, feed_dict):
        for name, data in feed_dict.items():
            if name in self.inputs:
                self.inputs[name].copy_(data.to(self.inputs[name].dtype).contiguous())
        self.context.execute_v2(self.bindings)
        return {name: mem.clone() for name, mem in self.outputs.items()}

# ==============================================================================
# 🔄 2. 级联测试 Loop (Det/Map Engine -> Motion Engine)
# ==============================================================================
def trt_cascade_engine_test(model, 
                            det_map_engine_init, det_map_engine_temporal, 
                            motion_engine_init, motion_engine_temporal, 
                            data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    # 获取配置常数
    det_head = model.head.det_head
    map_head = model.head.map_head
    motion_head = model.head.motion_plan_head

    nh_det = det_head.instance_bank.num_temp_instances
    dim_det = det_head.instance_bank.anchor.shape[-1]
    nh_map = map_head.instance_bank.num_temp_instances
    dim_map = map_head.instance_bank.anchor.shape[-1]
    Q = motion_head.instance_queue.queue_length
    
    def get_zero_history_det():
        return {
            'prev_instance_feature': torch.zeros((1, nh_det, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_det, dim_det), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
            'prev_instance_id': torch.full((1, nh_det), -1, dtype=torch.int32, device='cuda'),
            'prev_id_count': torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
        }

    def get_zero_history_map():
        return {
            'prev_instance_feature': torch.zeros((1, nh_map, 256), dtype=torch.float32, device='cuda'),
            'prev_anchor': torch.zeros((1, nh_map, dim_map), dtype=torch.float32, device='cuda'),
            'prev_confidence': torch.zeros((1, nh_map), dtype=torch.float32, device='cuda'),
        }

    def get_zero_history_motion():
        return {
            "mo_history_instance_feature": torch.zeros((1, nh_det, Q, 256), dtype=torch.float32, device='cuda'),
            "mo_history_anchor": torch.zeros((1, nh_det, Q, 11), dtype=torch.float32, device='cuda'),
            "mo_history_period": torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_instance_id": torch.zeros((1, nh_det), dtype=torch.int32, device='cuda'),
            "mo_prev_confidence": torch.zeros((1, nh_det), dtype=torch.float32, device='cuda'),
            "mo_history_ego_feature": torch.zeros((1, 1, Q, 256), dtype=torch.float32, device='cuda'),
            "mo_history_ego_anchor": torch.zeros((1, 1, Q, 11), dtype=torch.float32, device='cuda'),
            "mo_history_ego_period": torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
            "mo_prev_ego_status": torch.zeros((1, 1, 10), dtype=torch.float32, device='cuda')
        }

    history_det = get_zero_history_det()
    history_map = get_zero_history_map()
    history_motion = get_zero_history_motion()

    prev_global_mat = None
    prev_time = None

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            scattered_data = scatter(data, [torch.cuda.current_device()])[0]
            img = scattered_data['img']
            proj_mat = scattered_data['projection_mat']
            img_metas = scattered_data['img_metas'][0]

            curr_time = img_metas['timestamp']
            if prev_time is None:
                dt = 0.5
                is_scene_start = True
            else:
                dt = curr_time - prev_time
                is_scene_start = (dt > 2.0 or dt < 0)

            if is_scene_start:
                dt = 0.5
                history_det = get_zero_history_det()
                history_map = get_zero_history_map()
                history_motion = get_zero_history_motion()
                prev_global_mat = None 

            dt_tensor = torch.tensor([dt], device='cuda', dtype=torch.float32)
            mask_tensor = torch.tensor([not is_scene_start], device='cuda', dtype=torch.bool)
            prev_time = curr_time

            curr_global = img_metas['T_global']
            curr_global_inv = img_metas['T_global_inv']
            if prev_global_mat is None:
                instance_t_matrix = torch.eye(4, device='cuda').unsqueeze(0)
            else:
                t_mat = curr_global_inv @ prev_global_mat
                instance_t_matrix = torch.from_numpy(t_mat).float().cuda().unsqueeze(0)
            prev_global_mat = curr_global

            # ==============================================================
            # 🔴 Step 1: 推理 Det & Map 引擎
            # ==============================================================
            feed_dict_perception = {
                'img': img,
                'projection_mat': proj_mat,
                'instance_t_matrix': instance_t_matrix,
                'time_interval': dt_tensor,
                'prev_det_feat': history_det['prev_instance_feature'],
                'prev_det_anchor': history_det['prev_anchor'],
                'prev_det_conf': history_det['prev_confidence'],
                'prev_det_id': history_det['prev_instance_id'],
                'prev_id_count': history_det['prev_id_count'],
                'prev_map_feat': history_map['prev_instance_feature'],
                'prev_map_anchor': history_map['prev_anchor'],
                'prev_map_conf': history_map['prev_confidence'],
            }

            if is_scene_start:
                trt_outs_perc = det_map_engine_init.infer(feed_dict_perception)
            else:
                trt_outs_perc = det_map_engine_temporal.infer(feed_dict_perception)

            # 更新感知历史
            history_det['prev_instance_feature'] = trt_outs_perc['next_det_feat']
            history_det['prev_anchor'] = trt_outs_perc['next_det_anchor']
            history_det['prev_confidence'] = trt_outs_perc['next_det_conf']
            history_det['prev_instance_id'] = trt_outs_perc['next_det_instance_id']
            history_det['prev_id_count'] = trt_outs_perc['next_id_count']

            history_map['prev_instance_feature'] = trt_outs_perc['next_map_feat']
            history_map['prev_anchor'] = trt_outs_perc['next_map_anchor']
            history_map['prev_confidence'] = trt_outs_perc['next_map_conf']

            # ==============================================================
            # 🟢 Step 2: 推理 Motion 引擎
            # 把感知引擎提取出来的特征直接喂给它！
            # ==============================================================
            feed_dict_motion = {
                'det_instance_feature': trt_outs_perc['det_instance_feature'],
                'det_anchor_embed': trt_outs_perc['det_anchor_embed'],
                'det_classification_sigmoid': trt_outs_perc['det_cls'].sigmoid(),
                'det_anchors': trt_outs_perc['det_bbox'],
                'det_instance_id': trt_outs_perc['det_instance_id'].to(torch.int32),
                'map_instance_feature': trt_outs_perc['map_instance_feature'],
                'map_anchor_embed': trt_outs_perc['map_anchor_embed'],
                'map_classification_sigmoid': trt_outs_perc['map_cls'].sigmoid(),
                'ego_feature_map': trt_outs_perc['ego_feature_map'],
                
                'instance_t_matrix': instance_t_matrix,
                'mask': mask_tensor,
            }
            feed_dict_motion.update(history_motion)

            if is_scene_start:
                trt_outs_mo = motion_engine_init.infer(feed_dict_motion)
            else:
                trt_outs_mo = motion_engine_temporal.infer(feed_dict_motion)

            # 更新 Motion 历史
            history_motion['mo_history_instance_feature'] = trt_outs_mo['next_mo_history_instance_feature']
            history_motion['mo_history_anchor'] = trt_outs_mo['next_mo_history_anchor']
            history_motion['mo_history_period'] = trt_outs_mo['next_mo_history_period']
            history_motion['mo_prev_instance_id'] = trt_outs_mo['next_mo_prev_instance_id']
            history_motion['mo_prev_confidence'] = trt_outs_mo['next_mo_prev_confidence']
            history_motion['mo_history_ego_feature'] = trt_outs_mo['next_mo_history_ego_feature']
            history_motion['mo_history_ego_anchor'] = trt_outs_mo['next_mo_history_ego_anchor']
            history_motion['mo_history_ego_period'] = trt_outs_mo['next_mo_history_ego_period']
            history_motion['mo_prev_ego_status'] = trt_outs_mo['next_mo_prev_ego_status']

            # ==============================================================
            # 🛠️ 桥接官方后处理
            # ==============================================================
            # Det 加上防崩溃噪声
            det_cls_safe = torch.clamp(trt_outs_perc['det_cls'].float() + torch.rand_like(trt_outs_perc['det_cls'].float()) * 1e-6, 0.0, 1.0)
            model_outs_det = {
                "classification": [det_cls_safe],
                "prediction": [trt_outs_perc['det_bbox'].float()],
                "quality": [None],    
                "instance_id": trt_outs_perc['det_instance_id']
            }
            decoded_det_res = det_head.post_process(model_outs_det)

            model_outs_map = {
                "classification": [trt_outs_perc['map_cls'].float()],
                "prediction": [trt_outs_perc['map_pts'].float()],
                "quality": [None], 
                "instance_id": None   
            }
            decoded_map_res = map_head.post_process(model_outs_map)

            model_outs_motion = {
                "motion_classification": [trt_outs_mo['motion_cls'].float()],
                "motion_prediction": [trt_outs_mo['motion_reg'].float()],
                "planning_classification": [trt_outs_mo['plan_cls'].float()],
                "planning_prediction": [trt_outs_mo['plan_reg'].float()],
                "planning_status": [trt_outs_mo['plan_status'].float()]
            }
            
            try:
                decoded_mo_res = motion_head.post_process(model_outs_motion, decoded_det_res)
            except TypeError:
                decoded_mo_res = motion_head.post_process(model_outs_motion)

            merged_res = decoded_det_res[0].copy()
            merged_res.update(decoded_map_res[0])
            merged_res.update(decoded_mo_res[0])
            
            result = {
                'img_bbox': merged_res, 
                'pts_bbox': merged_res,
            }
            results.append(result)

        prog_bar.update()

    return results

# ==============================================================================
# 🎮 3. 命令行参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Cascade Engine Test")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    
    # 接收 4 个 Engine
    parser.add_argument("--engine_perc_init", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead_first.engine")
    parser.add_argument("--engine_perc_temp", default="work_dirs/sparsedrive_small_stage2/sparsedrive_multihead.engine")
    parser.add_argument("--engine_mo_init", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine_first.engine")
    parser.add_argument("--engine_mo_temp", default="work_dirs/sparsedrive_small_stage2/motion_plan_engine.engine")
    
    parser.add_argument("--plugin", default="projects/trt_plugin/build/libSparseDrivePlugin.so", help="Plugin path")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--eval", type=str, nargs="+", default=['bbox', 'motion', 'planning'], help='evaluation metrics')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if os.path.exists(args.plugin):
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded Custom Plugin: {args.plugin}")

    cfg = Config.fromfile(args.config)
    
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir

    samples_per_gpu = 1
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    dataset.work_dir = cfg.work_dir
    
    data_loader = build_dataloader_origin(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.CLASSES = dataset.CLASSES

    print(f"\n🚀 Loading Perception Engine (Init): {args.engine_perc_init}")
    trt_perc_init = TRTInfer(args.engine_perc_init)
    print(f"🚀 Loading Perception Engine (Temp): {args.engine_perc_temp}")
    trt_perc_temp = TRTInfer(args.engine_perc_temp)

    print(f"\n🚀 Loading Motion Engine (Init): {args.engine_mo_init}")
    trt_mo_init = TRTInfer(args.engine_mo_init)
    print(f"🚀 Loading Motion Engine (Temp): {args.engine_mo_temp}")
    trt_mo_temp = TRTInfer(args.engine_mo_temp)

    print("\n🔥 Starting TensorRT CASCADE Evaluation (Det+Map -> Motion)...")
    outputs = trt_cascade_engine_test(model, trt_perc_init, trt_perc_temp, trt_mo_init, trt_mo_temp, data_loader)

    if args.out:
        print(f"\n💾 Writing results to {args.out}")
        mmcv.dump(outputs, args.out)
    
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        if 'eval_mode' in eval_kwargs:
            eval_kwargs['eval_mode']['with_tracking'] = True
            eval_kwargs['eval_mode']['with_motion'] = True
            eval_kwargs['eval_mode']['with_planning'] = True
            eval_kwargs['eval_mode']['with_det'] = True
            eval_kwargs['eval_mode']['with_map'] = False # 默认关掉最耗内存的 Map 评估
        
        eval_kwargs.update(dict(metric=args.eval))
        print(f"\n📊 Evaluating metrics: {eval_kwargs}")
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    main()