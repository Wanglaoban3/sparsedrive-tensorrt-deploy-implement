from __future__ import division

import argparse
import copy
import importlib
import os
import time
import warnings
from datetime import timedelta
from os import path as osp

import cv2
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel.scatter_gather import scatter
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from torch import distributed as dist

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

from projects.mmdet3d_plugin.apis.train import custom_train_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

cv2.setNumThreads(8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantization-aware training with ModelOpt"
    )
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--modelopt-state-from",
        help="restore quantized model structure from a saved modelopt_state",
    )
    parser.add_argument(
        "--save-modelopt-state",
        help="path to save modelopt_state, default: <work_dir>/modelopt_state.pth",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--quant-cfg",
        default="int8",
        choices=["int8", "fp8"],
        help="built-in ModelOpt quantization config",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=32,
        help="number of training batches used for quantization calibration",
    )
    parser.add_argument(
        "--print-quant-summary",
        action="store_true",
        help="print ModelOpt quantizer summary after quantization",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use (only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use (only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument("--gpus-per-machine", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options
    if args.calib_batches < 1:
        raise ValueError("--calib-batches must be >= 1")
    return args


def import_custom_modules(cfg, config_path):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if not getattr(cfg, "plugin", False):
        return

    if hasattr(cfg, "plugin_dir"):
        module_root = os.path.dirname(cfg.plugin_dir)
    else:
        module_root = os.path.dirname(config_path)
    module_root = os.path.normpath(module_root)
    module_path = ".".join(
        [part for part in module_root.split(os.sep) if part and part != "."]
    )
    if module_path:
        importlib.import_module(module_path)


def get_quant_config(name):
    quant_cfg_map = {"int8": mtq.INT8_DEFAULT_CFG}
    if hasattr(mtq, "FP8_DEFAULT_CFG"):
        quant_cfg_map["fp8"] = mtq.FP8_DEFAULT_CFG
    if name not in quant_cfg_map:
        raise ValueError(f"Unsupported quant config: {name}")
    return copy.deepcopy(quant_cfg_map[name])


def build_calib_forward_loop(data_loader, max_batches):
    def forward_loop(model):
        was_training = model.training
        model.train()
        device_id = torch.cuda.current_device()
        for step, data in enumerate(data_loader):
            if step >= max_batches:
                break
            data = scatter(data, [device_id])[0]
            with torch.no_grad():
                model(**data)
        model.train(was_training)

    return forward_loop


def save_modelopt_state(model, path):
    save_dir = osp.dirname(path)
    if save_dir:
        mmcv.mkdir_or_exist(save_dir)
    torch.save(mto.modelopt_state(model), path)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_custom_modules(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    if args.launcher == "none":
        distributed = False
    elif args.launcher == "mpi_nccl":
        distributed = True
        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()

        device_ids_on_machines = list(range(args.gpus_per_machine))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            map(str, device_ids_on_machines)
        )
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=mpi_world_size,
            rank=mpi_local_rank,
            timeout=timedelta(seconds=3600),
        )
        cfg.gpu_ids = range(mpi_world_size)
    else:
        distributed = True
        init_dist(
            args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params
        )
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(
        "Environment info:\n" + dash_line + env_info + "\n" + dash_line
    )
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    if cfg.get("load_from") and not args.modelopt_state_from:
        logger.info(
            f"Loading floating-point checkpoint before QAT: {cfg.load_from}"
        )
        load_checkpoint(model, cfg.load_from, map_location="cpu")
        cfg.load_from = None

    if torch.cuda.is_available():
        if distributed:
            model = model.cuda()
        else:
            model = model.cuda(cfg.gpu_ids[0])

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]

    if args.modelopt_state_from:
        logger.info(
            "Restoring quantized model structure from modelopt_state: "
            f"{args.modelopt_state_from}"
        )
        model = mto.restore_from_modelopt_state(
            model, modelopt_state_path=args.modelopt_state_from
        )
    else:
        logger.info(
            f"Applying ModelOpt QAT config `{args.quant_cfg}` "
            f"with {args.calib_batches} calibration batches"
        )
        samples_per_gpu = cfg.data.get(
            "samples_per_gpu", cfg.data.train.get("samples_per_gpu", 1)
        )
        calib_loader = build_dataloader(
            datasets[0],
            samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
            runner_type=cfg.runner["type"] if "runner" in cfg else "EpochBasedRunner",
        )
        forward_loop = build_calib_forward_loop(
            calib_loader, args.calib_batches
        )
        model = mtq.quantize(model, get_quant_config(args.quant_cfg), forward_loop)

        if args.print_quant_summary and (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            mtq.print_quant_summary(model)

    modelopt_state_path = args.save_modelopt_state or osp.join(
        cfg.work_dir, "modelopt_state.pth"
    )
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        save_modelopt_state(model, modelopt_state_path)
        logger.info(f"Saved modelopt_state to: {modelopt_state_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    logger.info(f"Model:\n{model}")

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            modelopt_state_path=modelopt_state_path,
        )

    model.CLASSES = datasets[0].CLASSES
    if getattr(cfg, "plugin", False):
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )
    else:
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
