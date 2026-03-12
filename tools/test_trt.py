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
# 🚀 1. TensorRT 推理引擎封装
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
# 🔄 2. 级联测试 Loop
# ==============================================================================
def trt_cascade_engine_test(model, 
                            det_map_engine_init, det_map_engine_temporal, 
                            motion_engine_init, motion_engine_temporal, 
                            data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

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

            history_det['prev_instance_feature'] = trt_outs_perc['next_det_feat']
            history_det['prev_anchor'] = trt_outs_perc['next_det_anchor']
            history_det['prev_confidence'] = trt_outs_perc['next_det_conf']
            history_det['prev_instance_id'] = trt_outs_perc['next_det_instance_id']
            history_det['prev_id_count'] = trt_outs_perc['next_id_count']

            history_map['prev_instance_feature'] = trt_outs_perc['next_map_feat']
            history_map['prev_anchor'] = trt_outs_perc['next_map_anchor']
            history_map['prev_confidence'] = trt_outs_perc['next_map_conf']

            feed_dict_motion = {
                'det_instance_feature': trt_outs_perc['det_instance_feature'],
                'det_anchor_embed': trt_outs_perc['det_anchor_embed'],
                'det_classification': trt_outs_perc['det_cls'],
                'det_anchors': trt_outs_perc['det_bbox'],
                'det_instance_id': trt_outs_perc['det_instance_id'].to(torch.int32),
                'map_instance_feature': trt_outs_perc['map_instance_feature'],
                'map_anchor_embed': trt_outs_perc['map_anchor_embed'],
                'map_classification': trt_outs_perc['map_cls'],
                'ego_feature_map': trt_outs_perc['ego_feature_map'],
                'instance_t_matrix': instance_t_matrix,
                'mask': mask_tensor,
            }
            feed_dict_motion.update(history_motion)

            if is_scene_start:
                trt_outs_mo = motion_engine_init.infer(feed_dict_motion)
            else:
                trt_outs_mo = motion_engine_temporal.infer(feed_dict_motion)

            history_motion['mo_history_instance_feature'] = trt_outs_mo['next_mo_history_instance_feature']
            history_motion['mo_history_anchor'] = trt_outs_mo['next_mo_history_anchor']
            history_motion['mo_history_period'] = trt_outs_mo['next_mo_history_period']
            history_motion['mo_prev_instance_id'] = trt_outs_mo['next_mo_prev_instance_id']
            history_motion['mo_prev_confidence'] = trt_outs_mo['next_mo_prev_confidence']
            history_motion['mo_history_ego_feature'] = trt_outs_mo['next_mo_history_ego_feature']
            history_motion['mo_history_ego_anchor'] = trt_outs_mo['next_mo_history_ego_anchor']
            history_motion['mo_history_ego_period'] = trt_outs_mo['next_mo_history_ego_period']
            history_motion['mo_prev_ego_status'] = trt_outs_mo['next_mo_prev_ego_status']

            model_outs_det = {
                "classification": [trt_outs_perc['det_cls'].float()],
                "prediction": [trt_outs_perc['det_bbox'].float()],
                "quality": [trt_outs_perc['det_quality'].float()], # 👈 把它加回来！
                "instance_id": trt_outs_perc['det_instance_id']
            }
            decoded_det_res = det_head.post_process(model_outs_det)

            model_outs_map = {
                "classification": [trt_outs_perc['map_cls'].float()],
                "prediction": [trt_outs_perc['map_pts'].float()],
                "instance_id": None 
            }
            decoded_map_res = map_head.post_process(model_outs_map)

            motion_output = {
                "classification": [trt_outs_mo['motion_cls'].float()],
                "prediction": [trt_outs_mo['motion_reg'].float()],
                "period": trt_outs_mo['next_mo_history_period'],
                "anchor_queue": list(trt_outs_mo['next_mo_history_anchor'].float().unbind(dim=2))
            }
            
            planning_output = {
                "classification": [trt_outs_mo['plan_cls'].float()],
                "prediction": [trt_outs_mo['plan_reg'].float()],
                "status": [trt_outs_mo['plan_status'].float()],
                "period": trt_outs_mo['next_mo_history_ego_period'],
                "anchor_queue": list(trt_outs_mo['next_mo_history_ego_anchor'].float().unbind(dim=2))
            }
            
            decoded_mo_res = motion_head.post_process(model_outs_det, motion_output, planning_output, scattered_data)

            motion_res_list, planning_res_list = decoded_mo_res
            merged_res = decoded_det_res[0].copy()
            merged_res.update(decoded_map_res[0])
            merged_res.update(motion_res_list[0])
            merged_res.update(planning_res_list[0])
            
            results.append({'img_bbox': merged_res, 'pts_bbox': merged_res})
        prog_bar.update()

    return results

# ==============================================================================
# 🎮 3. 命令行参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Cascade Engine Test")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="pytorch checkpoint file")
    
    parser.add_argument("--load_results", help="path to the cached .pkl results file to skip inference")
    
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
    cfg = Config.fromfile(args.config)

    # 💡 无论推理还是加载，都统一初始化 work_dir 逻辑，防止 evaluate 报错
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # ---------------------------------------------------------
    # 💡 核心逻辑：判断是加载已有结果还是重新推理
    # ---------------------------------------------------------
    if args.load_results:
        if not osp.exists(args.load_results):
            print(f"❌ Error: Specified results file not found at {args.load_results}")
            sys.exit(1)
        print(f"📂 Loading cached results from: {args.load_results}")
        outputs = mmcv.load(args.load_results)
    else:
        # 正常推理路径
        if os.path.exists(args.plugin):
            ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
            print(f"✅ Loaded Custom Plugin: {args.plugin}")

        if hasattr(cfg, 'task_config'):
            cfg.task_config['with_det'] = True
            cfg.task_config['with_map'] = True
            cfg.task_config['with_motion_plan'] = True
            if 'head' in cfg.model:
                cfg.model.head.task_config = cfg.task_config

        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg["custom_imports"])

        cfg.data.test.work_dir = cfg.work_dir # 同步工作目录给数据集配置

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader_origin(
            dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False, shuffle=False,
        )

        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = model.cuda()
        model.CLASSES = dataset.CLASSES

        trt_perc_init = TRTInfer(args.engine_perc_init)
        trt_perc_temp = TRTInfer(args.engine_perc_temp)
        trt_mo_init = TRTInfer(args.engine_mo_init)
        trt_mo_temp = TRTInfer(args.engine_mo_temp)

        print("\n🔥 Starting TensorRT Inference...")
        outputs = trt_cascade_engine_test(model, trt_perc_init, trt_perc_temp, trt_mo_init, trt_mo_temp, data_loader)

        if args.out:
            print(f"\n💾 Saving results to {args.out}")
            mmcv.dump(outputs, args.out)

    # 👇 修改点 2：在 Evaluate 之前释放模型和引擎，并设置多进程启动方式
    if not args.load_results:
        print("\n🧹 Cleaning up GPU & CPU memory cache before evaluation...")
        try:
            del model
            del trt_perc_init
            del trt_perc_temp
            del trt_mo_init
            del trt_mo_temp
        except NameError:
            pass
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Memory cleared successfully.")

    import multiprocessing
    try:
        # 设置多进程启动模式为 fork，规避评估时的内存翻倍问题
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    # 👆

    # ---------------------------------------------------------
    # 📊 评估部分 (🔥 终极分组隔离版，0内存开销，防污染)
    # ---------------------------------------------------------
    if args.eval:
        cfg.data.test.work_dir = cfg.work_dir
        dataset = build_dataset(cfg.data.test)
        dataset.work_dir = cfg.work_dir 

        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
            
        if 'eval_mode' not in eval_kwargs:
            eval_kwargs['eval_mode'] = {}
            
        # 兜底：防止 KeyError
        if 'tracking_threshold' not in eval_kwargs['eval_mode']:
            eval_kwargs['eval_mode']['tracking_threshold'] = 0.2
        if 'motion_threshhold' not in eval_kwargs['eval_mode']:
            eval_kwargs['eval_mode']['motion_threshhold'] = 0.2
            
        # 🎯 核心修复 1：强制补全 metric，否则底层评估会无视 tracking
        eval_metrics = args.eval.copy() if args.eval else ['bbox', 'map', 'motion', 'planning']
        if 'track' not in eval_metrics and 'tracking' not in eval_metrics:
            eval_metrics.append('track') # 把 track 加进评估列表
        eval_kwargs.update(dict(metric=eval_metrics))
        
        # =================================================================
        # 🌟 核心修复 2：绝杀分组！
        # - 先跑 Map, Motion, Plan，它们绝不改数据。
        # - 最后把 DET 和 TRACKING 绑在一起跑！让 Track 能拿到 Det 的框！
        # =================================================================
        eval_groups = [
            {'name': 'MAP',              'flags': {'with_map': True}},
            {'name': 'MOTION',           'flags': {'with_motion': True}},
            {'name': 'PLANNING',         'flags': {'with_planning': True}},
            {'name': 'DET AND TRACKING', 'flags': {'with_det': True, 'with_tracking': True}},
        ]
        
        final_results_dict = {}
        print(f"\n📊 Evaluating metrics (work_dir: {cfg.work_dir})...")
        print(f"🎯 Metrics to evaluate: {eval_metrics}")
        
        for group in eval_groups:
            print(f"\n>>> 🏃 Running Evaluation for: {group['name']} ...")
            
            single_eval_kwargs = eval_kwargs.copy()
            single_eval_kwargs['eval_mode'] = eval_kwargs['eval_mode'].copy() 
            
            # 先把所有的任务开关置为 False
            for k in ['with_det', 'with_tracking', 'with_map', 'with_motion', 'with_planning']:
                single_eval_kwargs['eval_mode'][k] = False
            
            # 再精准开启当前组需要的开关
            for k, v in group['flags'].items():
                single_eval_kwargs['eval_mode'][k] = v
            
            # 🚀 直接传入原始 outputs，无需 deepcopy，毫无内存压力！
            task_res = dataset.evaluate(outputs, **single_eval_kwargs)
            final_results_dict.update(task_res)
            
        print("\n=======================================================")
        print("🏆 Final Cascade Evaluation Results (Isolated):")
        print("=======================================================")
        for k, v in final_results_dict.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()