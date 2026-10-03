"""Generate nuScenes trainval VAL infos on a box WITHOUT lidar files, CAN
bus expansion, or the vector map expansion.

Camera-only perception eval needs only: metadata (v1.0-trainval/*.json),
camera keyframes (samples/CAM_*), and LIDAR_TOP placeholder files so the
converter's file-existence checks pass (the eval pipeline never reads
lidar).  Ego status is reconstructed from keyframe ego poses; the map
extractor is stubbed (map_annos empty - det mAP/NDS unaffected).

Before running, point NUSCENES_ROOT at a camera-only layout:
  - $NUSCENES_ROOT/v1.0-trainval/  (metadata)
  - $NUSCENES_ROOT/samples/CAM_*/  (extracted)
  - placeholders: python tools/data_converter/make_lidar_placeholders.py

Run from project root:
    python tools/data_converter/create_trainval_val_infos_local.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "data_converter"))

import numpy as np
from pyquaternion import Quaternion

DATA_ROOT = os.environ.get("NUSCENES_ROOT",
                           os.path.join(ROOT, "data", "nuscenes"))
sys.argv = [
    "create_trainval_val_infos_local.py", "nuscenes",
    "--root-path", DATA_ROOT,
    "--canbus", DATA_ROOT,
    "--out-dir", os.path.join(ROOT, "data", "infos"),
    "--extra-tag", "nuscenes",
    "--version", "v1.0-trainval",
]

import nuscenes_converter as nc  # noqa: E402


def _pose_of(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return (np.array(ep["translation"], dtype=np.float64),
            Quaternion(ep["rotation"]))


def get_ego_status_local(nusc, nusc_can_bus, sample):
    out = [0.0] * 10
    try:
        cur = sample
        prev = nusc.get("sample", cur["prev"]) if cur["prev"] else None
        prev2 = (nusc.get("sample", prev["prev"])
                 if prev and prev["prev"] else None)
        tg, Rg = _pose_of(nusc, cur)
        yaw = Rg.yaw_pitch_roll[0]
        Rinv = Rg.rotation_matrix.T
        if prev is not None:
            tp, Rp = _pose_of(nusc, prev)
            vx, vy, _ = Rinv @ ((tg - tp) / 0.5)
            out[6], out[7] = float(vx), float(vy)
            yawp = Rp.yaw_pitch_roll[0]
            out[5] = float((yaw - yawp) / 0.5)
            if prev2 is not None:
                tp2, _ = _pose_of(nusc, prev2)
                vpx, vpy, _ = Rinv @ ((tp - tp2) / 0.5)
                out[0] = float((vx - vpx) / 0.5)
                out[1] = float((vy - vpy) / 0.5)
    except Exception:
        pass
    return np.array(out, dtype=np.float32)


nc.get_ego_status = get_ego_status_local


class _StubMapExtractor(object):
    def __init__(self, data_root, roi_size):
        self.roi_size = roi_size

    def get_map_geom(self, *args, **kwargs):
        return {}

    def get_vector(self, *args, **kwargs):
        return {}


def _all_scenes_available(nusc):
    """Camera-only box: expose ONLY the val scenes as available so the
    train split filters to empty (its lidar placeholders don't exist and
    we don't need train infos).  Returns the get_available_scenes shape:
    list of dicts with 'token' and 'name'."""
    from nuscenes.utils import splits as _splits

    val_names = set(_splits.val)
    return [dict(token=s["token"], name=s["name"])
            for s in nusc.scene if s["name"] in val_names]


class _ValOnlyNuScenes(nc.NuScenes):
    """Restrict nusc.sample to val scenes: _fill_trainval_infos iterates
    nusc.sample and runs per-sample work (get_sample_data reads the lidar
    placeholder) on EVERY sample, so train samples must be dropped here,
    not just filtered out of the scene lists."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from nuscenes.utils import splits as _splits

        val_names = set(_splits.val)
        keep = {s["token"] for s in self.scene if s["name"] in val_names}
        self.sample = [s for s in self.sample
                       if s["scene_token"] in keep]
        # NuScenes.get() = getattr(self, table)[self.getind(token)] - the
        # token->index mapping must match the FILTERED list, otherwise
        # every 'sample' lookup returns a WRONG record
        self._val_index = {s["token"]: i for i, s in enumerate(self.sample)}
        print(f"[create_trainval_val_infos_local] samples kept: "
              f"{len(self.sample)}")

    def getind(self, table_name, token):
        # during super().__init__ the attribute doesn't exist yet and the
        # full table is current - only remap after filtering is done
        if table_name == "sample" and hasattr(self, "_val_index"):
            return self._val_index[token]
        return super().getind(table_name, token)


if __name__ == "__main__":
    from projects.mmdet3d_plugin.datasets.map_utils import \
        nuscmap_extractor as _ne

    nc.NuscMapExtractor = _StubMapExtractor
    _ne.NuscMapExtractor = _StubMapExtractor

    # bypass lidar file-existence filtering in get_available_scenes
    import mmcv  # noqa: E402
    nc.get_available_scenes = _all_scenes_available
    nc.NuScenes = _ValOnlyNuScenes

    # debug: log any box-count mismatch with its exact sample_data record
    _orig_gsd = nc.NuScenes.get_sample_data

    def _gsd_dbg(self, token, **kw):
        path, boxes, intr = _orig_gsd(self, token, **kw)
        sd = self.get('sample_data', token)
        if sd['is_key_frame']:
            n_anns = len(self.get('sample', sd['sample_token'])['anns'])
            if n_anns != len(boxes):
                print(f"[MISMATCH] {sd['filename']} boxes={len(boxes)} "
                      f"anns={n_anns} prev={sd['prev'][:8]} "
                      f"sample={sd['sample_token'][:8]}", flush=True)
        return path, boxes, intr

    nc.NuScenes.get_sample_data = _gsd_dbg

    nc.nuscenes_data_prep(
        root_path=DATA_ROOT,
        can_bus_root_path=DATA_ROOT,
        info_prefix="nuscenes",
        version="v1.0-trainval",
        dataset_name="NuScenesDataset",
        out_dir=os.path.join(ROOT, "data", "infos"),
        max_sweeps=10,
    )
    print("TRAINVAL_VAL_INFOS_DONE")
