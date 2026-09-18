"""Generate nuScenes-mini infos on a box WITHOUT the CAN bus expansion.

- NuScenesCanBus only needs a 'can_bus' directory to exist (data/nuscenes/can_bus,
  may be empty); per-scene message lookups then raise inside get_ego_status.
- We patch get_ego_status: reconstruct the 10-dim status from consecutive
  keyframe ego poses (dt = 0.5 s):
      [accel(3), rotation_rate(3), vel(3), steer(1)]
  matching the layout produced by the CAN path (index 6 = vx, 7 = vy,
  steer last). Zero-filled fields are the ones CAN-only sensors provide.
- If the nuScenes map expansion is missing, NuscMapExtractor is stubbed to
  return empty vectors so infos generation still completes (map-head GT will
  be empty; det/plan unaffected).

Run from project root:
    python tools/data_converter/create_mini_infos_local.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "data_converter"))

import numpy as np
from pyquaternion import Quaternion

# nuscenes_converter parses argv at module import; feed it a valid arg set
_D = os.path.join(ROOT, "data")
sys.argv = [
    "create_mini_infos_local.py", "nuscenes",
    "--root-path", os.path.join(_D, "nuscenes"),
    "--canbus", os.path.join(_D, "nuscenes"),
    "--out-dir", os.path.join(_D, "infos"),
    "--extra-tag", "nuscenes",
    "--version", "v1.0-mini",
]

import nuscenes_converter as nc  # noqa: E402


def _pose_of(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ep["translation"], dtype=np.float64), Quaternion(ep["rotation"])


def get_ego_status_local(nusc, nusc_can_bus, sample):
    out = [0.0] * 10
    try:
        cur = sample
        prev = nusc.get("sample", cur["prev"]) if cur["prev"] else None
        prev2 = nusc.get("sample", prev["prev"]) if prev and prev["prev"] else None
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
    """Drop-in replacement when the map expansion package is absent."""

    def __init__(self, data_root, roi_size):
        self.roi_size = roi_size

    def get_map_geom(self, *args, **kwargs):
        return {}

    def get_vector(self, *args, **kwargs):
        return {}


if __name__ == "__main__":
    expansion = os.path.join(ROOT, "data", "nuscenes", "maps", "expansion")
    has_expansion = os.path.isdir(expansion) and any(
        f.endswith(".json") for f in os.listdir(expansion))
    if not has_expansion:
        print("[create_mini_infos_local] vector map expansion not found -> stubbing NuscMapExtractor")
        from projects.mmdet3d_plugin.datasets.map_utils import nuscmap_extractor as _ne

        nc.NuscMapExtractor = _StubMapExtractor
        _ne.NuscMapExtractor = _StubMapExtractor

    nc.nuscenes_data_prep(
        root_path=os.path.join(ROOT, "data", "nuscenes"),
        can_bus_root_path=os.path.join(ROOT, "data", "nuscenes"),
        info_prefix="nuscenes",
        version="v1.0-mini",
        dataset_name="NuScenesDataset",
        out_dir=os.path.join(ROOT, "data", "infos"),
        max_sweeps=10,
    )
    print("MINI_INFOS_DONE")
