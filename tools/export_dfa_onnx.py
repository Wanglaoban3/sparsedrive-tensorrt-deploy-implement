import argparse
import os
import sys
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation


def feature_maps_format(feature_maps, inverse=False):
    """A lightweight copy of the project helper.

    The plugin branch needs the formatted multi-camera multi-level layout to
    enter the custom ONNX node. The inverse path is used only inside the PyTorch
    reference implementation of the custom op.
    """

    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        if torch.jit.is_tracing() or isinstance(split_size, torch.Tensor):
            split_size = split_size.detach().cpu().numpy().tolist()

        cam_split = [1]
        cam_split_size = [sum(split_size[0])]

        for i in range(num_cams - 1):
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])

        mc_feat = []
        for i, x in enumerate(col_feats.split(cam_split_size, dim=1)):
            bs, _, c = x.shape
            mc_feat.append(x.view(bs, cam_split[i], -1, c))

        if isinstance(spatial_shape, torch.Tensor):
            spatial_shape = spatial_shape.cpu().numpy().tolist()

        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat_splits = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat_splits):
                h, w = spatial_shape[shape_index][j]
                bs, n_cams, _, c = f.shape
                f_view = f.view(bs, n_cams, int(h), int(w), c)
                feat_splits[j] = f_view.permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat_splits)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for feat in feature_maps:
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape, dtype=torch.int64, device=col_feats.device
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [
            torch.tensor(
                [0], device=scale_start_index.device, dtype=scale_start_index.dtype
            ),
            scale_start_index[:-1],
        ]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)
    return [col_feats, spatial_shape, scale_start_index]


def find_first_deformable_layer(head):
    for layer in head.layers:
        if isinstance(layer, DeformableFeatureAggregation):
            return layer
    raise RuntimeError("No DeformableFeatureAggregation layer found in the selected head.")


def make_synthetic_projection(batch_size, num_cams, device, dtype):
    proj = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
        batch_size, num_cams, 1, 1
    )
    cam_offsets = torch.linspace(-0.2, 0.2, num_cams, device=device, dtype=dtype)
    proj[:, :, 0, 0] = 1.0
    proj[:, :, 1, 1] = 1.0
    proj[:, :, 2, 2] = 1.0
    proj[:, :, 2, 3] = 10.0
    proj[:, :, 0, 3] = cam_offsets
    proj[:, :, 1, 3] = -cam_offsets
    return proj


def build_dummy_feature_maps(batch_size, num_cams, embed_dims, img_h, img_w, strides, device, dtype):
    feature_maps = []
    for stride in strides:
        h = img_h // stride
        w = img_w // stride
        feature_maps.append(
            torch.randn(batch_size, num_cams, embed_dims, h, w, device=device, dtype=dtype)
        )
    return feature_maps


def reference_custom_daf(col_feats, spatial_shape, scale_start_index, sampling_location, weights):
    """Pure PyTorch reference for the plugin branch.

    This mirrors the logic of the custom CUDA operator closely enough for
    export-time tracing and for a sanity check against the native path.
    """
    mc_ms_feat = feature_maps_format([col_feats, spatial_shape, scale_start_index], inverse=True)
    feature_maps = [feat for group in mc_ms_feat for feat in group]

    bs, num_anchor, num_pts, num_cams, _ = sampling_location.shape
    num_levels = len(feature_maps)

    points_2d = (sampling_location * 2.0 - 1.0).permute(0, 3, 1, 2, 4).contiguous()
    points_2d = points_2d.reshape(bs * num_cams, num_anchor, num_pts, 2)

    sampled = []
    for fm in feature_maps:
        sampled.append(
            F.grid_sample(
                fm.flatten(end_dim=1),
                points_2d,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    sampled = torch.stack(sampled, dim=1)
    sampled = sampled.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts)
    sampled = sampled.permute(0, 4, 5, 1, 2, 3)

    num_groups = weights.shape[-1]
    group_dims = sampled.shape[-1] // num_groups
    sampled = sampled.reshape(*sampled.shape[:-1], num_groups, group_dims)
    sampled = (weights[..., None] * sampled).sum(dim=(2, 3, 4))
    sampled = sampled.reshape(bs, num_anchor, -1)
    return sampled


class PluginDeformableAggregationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        col_feats,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        return reference_custom_daf(
            col_feats,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

    @staticmethod
    def symbolic(
        g,
        col_feats,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        return g.op(
            "SparseDrive::DeformableAggregation",
            col_feats,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )


class DFASubgraphExporter(nn.Module):
    def __init__(self, dfa_module: DeformableFeatureAggregation, use_plugin: bool):
        super().__init__()
        self.dfa = dfa_module
        self.use_plugin = use_plugin

    def _reshape_weights(self, weights, bs, num_anchor, num_cams, num_levels):
        return (
            weights.permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .reshape(bs, num_anchor, self.dfa.num_pts, num_cams, num_levels, self.dfa.num_groups)
        )

    def _native_forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor,
    ):
        key_points = self.dfa.kps_generator(anchor, instance_feature)
        weights = self.dfa._get_weights(
            instance_feature,
            anchor_embed,
            metas={"projection_mat": projection_mat},
        )
        features = self.dfa.feature_sampling(
            list(feature_maps),
            key_points,
            projection_mat,
            image_wh,
        )
        features = self.dfa.multi_view_level_fusion(features, weights)
        features = features.sum(dim=2)
        output = self.dfa.proj_drop(self.dfa.output_proj(features))
        if self.dfa.residual_mode == "add":
            output = output + instance_feature
        elif self.dfa.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _plugin_forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        num_cams = feature_maps[0].shape[1]
        num_levels = len(feature_maps)

        key_points = self.dfa.kps_generator(anchor, instance_feature)
        weights = self.dfa._get_weights(
            instance_feature,
            anchor_embed,
            metas={"projection_mat": projection_mat},
        )

        points_2d = self.dfa.project_points(
            key_points, projection_mat, image_wh
        ).permute(0, 2, 3, 1, 4).contiguous().reshape(
            bs, num_anchor, self.dfa.num_pts, num_cams, 2
        )
        weights = self._reshape_weights(weights, bs, num_anchor, num_cams, num_levels)

        col_feats, spatial_shape, scale_start_index = feature_maps_format(feature_maps)
        features = PluginDeformableAggregationFunction.apply(
            col_feats,
            spatial_shape,
            scale_start_index,
            points_2d,
            weights,
        )
        output = self.dfa.proj_drop(self.dfa.output_proj(features))
        if self.dfa.residual_mode == "add":
            output = output + instance_feature
        elif self.dfa.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def forward(
        self,
        feature_map_0,
        feature_map_1,
        feature_map_2,
        feature_map_3,
        instance_feature,
        anchor,
        anchor_embed,
        projection_mat,
        image_wh,
    ):
        feature_maps = [feature_map_0, feature_map_1, feature_map_2, feature_map_3]
        if self.use_plugin:
            return self._plugin_forward(
                feature_maps,
                instance_feature,
                anchor,
                anchor_embed,
                projection_mat,
                image_wh,
            )
        return self._native_forward(
            feature_maps,
            instance_feature,
            anchor,
            anchor_embed,
            projection_mat,
            image_wh,
        )


def maybe_import_custom_modules(cfg):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])


def disable_deformable_plugin_in_cfg(cfg):
    if "model" in cfg and "use_deformable_func" in cfg.model:
        cfg.model.use_deformable_func = False
    if "img_backbone" in cfg.model and "pretrained" in cfg.model.img_backbone:
        cfg.model.img_backbone.pretrained = None
    head_cfg = cfg.model.get("head", None)
    if head_cfg is None:
        return
    for sub_key in ("det_head", "map_head"):
        sub_cfg = head_cfg.get(sub_key, None)
        if sub_cfg is None:
            continue
        deformable_cfg = sub_cfg.get("deformable_model", None)
        if deformable_cfg is not None and "use_deformable_func" in deformable_cfg:
            deformable_cfg["use_deformable_func"] = False


def build_model(cfg, checkpoint):
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.eval()
    return model


def export_onnx(
    wrapper,
    inputs,
    out_path,
    opset_version,
    custom_opsets=None,
):
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            out_path,
            input_names=[
                "feature_map_0",
                "feature_map_1",
                "feature_map_2",
                "feature_map_3",
                "instance_feature",
                "anchor",
                "anchor_embed",
                "projection_mat",
                "image_wh",
            ],
            output_names=["dfa_output"],
            opset_version=opset_version,
            do_constant_folding=False,
            custom_opsets=custom_opsets or {},
        )


def verify_onnx_model(path):
    if onnx is None:
        print(f"[skip] onnx package is not available, cannot verify {path}")
        return
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"[ok] ONNX verified: {path}")


def main():
    parser = argparse.ArgumentParser(description="Export SparseDrive DFA subgraph to ONNX")
    parser.add_argument(
        "--config",
        default="projects/configs/sparsedrive_small_stage2.py",
        help="Config file used to instantiate the model.",
    )
    parser.add_argument(
        "--checkpoint",
        default="ckpt/sparsedrive_stage2.pth",
        help="Checkpoint to load before export.",
    )
    parser.add_argument(
        "--out",
        default="work_dirs/sparsedrive_small_stage2/dfa.onnx",
        help="Output ONNX path prefix. The script will write *_plugin.onnx and *_native.onnx.",
    )
    parser.add_argument(
        "--task",
        choices=["det", "map"],
        default="det",
        help="Which DFA block to export.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=704)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    maybe_import_custom_modules(cfg)
    disable_deformable_plugin_in_cfg(cfg)

    model = build_model(cfg, args.checkpoint).cuda().eval()
    head = model.head.det_head if args.task == "det" else model.head.map_head
    dfa = find_first_deformable_layer(head).cuda().eval()

    device = next(dfa.parameters()).device
    dtype = torch.float32

    strides = cfg.get("strides", [4, 8, 16, 32])
    embed_dims = dfa.embed_dims
    feature_maps = build_dummy_feature_maps(
        args.batch_size,
        args.num_cams,
        embed_dims,
        args.img_h,
        args.img_w,
        strides,
        device,
        dtype,
    )

    instance_feature = head.instance_bank.instance_feature.unsqueeze(0).repeat(
        args.batch_size, 1, 1
    ).to(device=device, dtype=dtype)
    anchor = head.instance_bank.anchor.unsqueeze(0).repeat(
        args.batch_size, 1, 1
    ).to(device=device, dtype=dtype)
    anchor_embed = head.anchor_encoder(anchor)

    projection_mat = make_synthetic_projection(
        args.batch_size, args.num_cams, device, dtype
    )
    image_wh = torch.tensor(
        [args.img_w, args.img_h], device=device, dtype=dtype
    ).view(1, 1, 2).repeat(args.batch_size, args.num_cams, 1)

    plugin_wrapper = DFASubgraphExporter(dfa, use_plugin=True).to(device).eval()
    native_wrapper = DFASubgraphExporter(dfa, use_plugin=False).to(device).eval()

    with torch.no_grad():
        ref_plugin = plugin_wrapper(
            *feature_maps,
            instance_feature,
            anchor,
            anchor_embed,
            projection_mat,
            image_wh,
        )
        ref_native = native_wrapper(
            *feature_maps,
            instance_feature,
            anchor,
            anchor_embed,
            projection_mat,
            image_wh,
        )
        max_diff = (ref_plugin - ref_native).abs().max().item()
        print(f"[check] PyTorch plugin/native max abs diff: {max_diff:.6f}")

    base, ext = os.path.splitext(args.out)
    if ext.lower() != ".onnx":
        base = args.out
    plugin_path = f"{base}_plugin.onnx"
    native_path = f"{base}_native.onnx"

    plugin_inputs = (
        feature_maps[0],
        feature_maps[1],
        feature_maps[2],
        feature_maps[3],
        instance_feature,
        anchor,
        anchor_embed,
        projection_mat,
        image_wh,
    )
    native_inputs = plugin_inputs

    export_onnx(
        plugin_wrapper,
        plugin_inputs,
        plugin_path,
        args.opset,
        custom_opsets={"SparseDrive": 1},
    )
    export_onnx(
        native_wrapper,
        native_inputs,
        native_path,
        args.opset,
    )

    if args.verify:
        verify_onnx_model(native_path)
        if onnx is not None:
            try:
                onnx.load(plugin_path)
                print(f"[ok] ONNX loaded: {plugin_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[warn] plugin ONNX load check failed: {exc}")

    print(f"[done] plugin ONNX: {plugin_path}")
    print(f"[done] native  ONNX: {native_path}")


if __name__ == "__main__":
    main()
