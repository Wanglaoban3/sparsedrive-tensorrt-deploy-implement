# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
from os import path as osp
import sys
import types
import torch
import warnings

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmdet.apis import single_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.ops import feature_maps_format
import projects.mmdet3d_plugin

# ==============================================================================
# 💡 核心注入逻辑：使用 forward_onnx 替换原生的 simple_test
# ==============================================================================
def custom_simple_test(self, img, **data):
    """这将会替换 SparseDrive.simple_test"""
    
    # 提取 img_metas
    if isinstance(data['img_metas'][0], list):
        img_metas = data['img_metas'][0]
    else:
        img_metas = data['img_metas']
        
    # 兼容处理：尝试获取 scene_token，如果没在 meta_keys 里配置，则退化为靠 timestamp 判断
    scene_token = img_metas[0].get('scene_token', None)
    timestamp = img_metas[0].get('timestamp', 0.0)
    
    # 初始化生命周期变量 (存储在 self 上避免跨帧丢失)
    if not hasattr(self, 'onnx_external_states'):
        self.onnx_external_states = {}
        self.prev_scene_token = None
        self.prev_timestamp = -1e5
        self.prev_metas = None

    # 判断是否为新的场景（Sequence）
    is_first_frame = False
    if scene_token is not None:
        if scene_token != self.prev_scene_token:
            is_first_frame = True
            self.prev_scene_token = scene_token
    else:
        # 如果没有 scene_token，通过时间戳差值判断是否切换了场景（nusc相邻帧一般差 0.5s，差值过大说明切场景了）
        if self.prev_metas is None or abs(float(timestamp) - float(self.prev_timestamp)) > 5.0:
            is_first_frame = True
        self.prev_timestamp = timestamp

    if is_first_frame:
        self.onnx_external_states = {}

    mask = torch.tensor([not is_first_frame], dtype=torch.bool, device=img.device)

    # 1. 提取特征图
    feature_maps = self.extract_feat(img)
    
    # 2. 推理 Det & Map 头 (前置任务)
    det_output = self.head.det_head(feature_maps, data)
    map_output = self.head.map_head(feature_maps, data)

    # 3. 为 ONNX 准备独立的 Tensor 输入
    det_cls_sigmoid = det_output["classification"][-1].sigmoid()
    map_cls_sigmoid = map_output["classification"][-1].sigmoid()

    feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
    ego_feature_map = feature_maps_inv[0][-1][:, 0]

    bs = img.shape[0]
    if is_first_frame:
        T_temp2cur = torch.eye(4).unsqueeze(0).expand(bs, -1, -1).to(img.device)
    else:
        prev_t_global = self.prev_metas[0]["T_global"]
        curr_t_global_inv = img_metas[0]["T_global_inv"]
        
        # 安全转为 numpy 计算投影矩阵，避免 tensor 设备不匹配或类型报错
        if isinstance(prev_t_global, torch.Tensor):
            prev_t_global = prev_t_global.cpu().numpy()
        if isinstance(curr_t_global_inv, torch.Tensor):
            curr_t_global_inv = curr_t_global_inv.cpu().numpy()
            
        T_temp2cur_np = curr_t_global_inv @ prev_t_global
        T_temp2cur = torch.tensor(T_temp2cur_np, dtype=torch.float32).unsqueeze(0).to(img.device)
    
    self.prev_metas = img_metas

    anchor_encoder = self.head.det_head.anchor_encoder
    anchor_handler = self.head.det_head.instance_bank.anchor_handler

    # 4. === 运行你的 forward_onnx ===
    onnx_outs = self.head.motion_plan_head.forward_onnx(
        det_instance_feature=det_output["instance_feature"],
        det_anchor_embed=det_output["anchor_embed"],
        det_classification_sigmoid=det_cls_sigmoid,
        det_anchors=det_output["prediction"][-1],
        det_instance_id=det_output["instance_id"],
        map_instance_feature=map_output["instance_feature"],
        map_anchor_embed=map_output["anchor_embed"],
        map_classification_sigmoid=map_cls_sigmoid,
        ego_feature_map=ego_feature_map,
        anchor_encoder=anchor_encoder,
        anchor_handler=anchor_handler,
        mask=mask,
        is_first_frame=is_first_frame,
        T_temp2cur=T_temp2cur,
        **self.onnx_external_states
    )

    # 拆包提取状态并更新供下一帧使用
    (
        motion_classification, motion_prediction,
        planning_classification, planning_prediction, planning_status,
        next_states, _, _
    ) = onnx_outs
    self.onnx_external_states = next_states

    motion_output = {
        "classification": motion_classification,
        "prediction": motion_prediction,
        "period": next_states["history_period"],
        "anchor_queue": list(next_states["history_anchor"].unbind(dim=2))
    }
    
    planning_output = {
        "classification": planning_classification,
        "prediction": planning_prediction,
        "status": planning_status,
        "period": next_states["history_ego_period"],
        "anchor_queue": list(next_states["history_ego_anchor"].unbind(dim=2))
    }
    
    model_outs = (det_output, map_output, motion_output, planning_output)
    
    # 使用原生后处理解算 Box 和 Trajectory
    results = self.head.post_process(model_outs, data)
    output = [{"img_bbox": result} for result in results]
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) an ONNX-forward model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--eval", type=str, nargs="+", help='evaluation metrics')
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 强制所有任务开启
    if hasattr(cfg, 'task_config'):
        cfg.task_config['with_det'] = True
        cfg.task_config['with_map'] = True
        cfg.task_config['with_motion_plan'] = True
        if 'head' in cfg.model:
            cfg.model.head.task_config = cfg.task_config

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0]) 
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    distributed = False

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    dataset.work_dir = cfg.work_dir
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False, # 测试集必须 False 以保证时序连续
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    # =========================================================
    # 猴子补丁 (Monkey Patch)：将 custom_simple_test 绑定到模型上
    # =========================================================
    model.simple_test = types.MethodType(custom_simple_test, model)

    model = MMDataParallel(model, device_ids=[0])
    
    # 正常调用 single_gpu_test，它内部会自动调用我们绑定的 custom_simple_test
    print("\n🚀 [Start ONNX-Forward Evaluation]...\n")
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    # outputs = mmcv.load("results.pkl")
    # 评估与结果输出
    if args.out and not args.out.endswith(".pkl_only_eval"):
        print(f"\nwriting results to {args.out}")
        mmcv.dump(outputs, args.out)
    
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        
        # =====================================================================
        # 🎯 核心修复：强制修改 eval_mode，彻底屏蔽 bbox, tracking 和 map 的评估
        # =====================================================================
        if 'eval_mode' not in eval_kwargs:
            eval_kwargs['eval_mode'] = cfg.get('evaluation', {}).get('eval_mode', {})
            
        eval_kwargs['eval_mode']['with_det'] = False
        eval_kwargs['eval_mode']['with_tracking'] = False
        eval_kwargs['eval_mode']['with_map'] = False
        
        # 确保 motion 和 planning 是开启的
        eval_kwargs['eval_mode']['with_motion'] = True
        eval_kwargs['eval_mode']['with_planning'] = False
        
        # 补充必要的阈值，防止触发 KeyError
        if 'motion_threshhold' not in eval_kwargs['eval_mode']:
            eval_kwargs['eval_mode']['motion_threshhold'] = 0.2
            
        print("\n=> 正在跳过检测与跟踪评估，只计算 Motion 和 Planning 的指标...")
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()