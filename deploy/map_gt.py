# -*- coding: utf-8 -*-
"""Vectorized-map ground truth for nuScenes map evaluation, WITHOUT the
nuscenes-devkit (numpy + shapely only).

Faithfully reproduces the semantics of
  - nuscenes.map_expansion.map_api (NuScenesMapExplorer._get_layer_line /
    _get_layer_polygon / get_patch_coord): each layer geometry is cropped
    by the rotated patch box and returned in PATCH-LOCAL coordinates,
    which for patch_box centered at the lidar pose equals the LIDAR frame;
  - projects/mmdet3d_plugin/datasets/map_utils/nuscmap_extractor.py
    (divider / ped_crossing / boundary extraction, ped merging);
  - VectorizeMap(simplify=True) from the eval pipeline (GT line sampling).

Coordinate convention (triangulated against kmeans_map_100.npy and the
engine's map_pts outputs): for roi_size=(30, 60), point columns are
(x, y) = (+-15 m axis, +-30 m axis), matching normalize_line's
origin=(roi[0]/2, roi[1]/2), norm=(roi[0], roi[1]) in models/map/target.py.

Class order (geom2anno / MAP_CLASSES): {0: ped_crossing, 1: divider,
2: boundary}.
"""
import json
import os

import numpy as np
from shapely import affinity, ops
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

# projects/mmdet3d_plugin/datasets/map_utils/utils.py is shapely+scipy
# only, so it is importable here without the mmcv/mmdet stack.
import importlib.util as _ilu

_UTILS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "projects", "mmdet3d_plugin", "datasets",
    "map_utils", "utils.py")
_spec = _ilu.spec_from_file_location("sd_map_utils", _UTILS)
_sd_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sd_utils)
split_collections = _sd_utils.split_collections
get_drivable_area_contour = _sd_utils.get_drivable_area_contour
get_ped_crossing_contour = _sd_utils.get_ped_crossing_contour

MAP_CLASSES = ("ped_crossing", "divider", "boundary")


def rotation_yaw_deg(R):
    """Yaw of a 3x3 rotation matrix in degrees (nuScenes quaternion_yaw
    on the lidar2global rotation reduces to atan2(R[1,0], R[0,0]))."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


class NuScenesMapLite(object):
    """Minimal NuScenesMap: loads one map-expansion json and extracts
    line/polygon geometries from its node table."""

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.tables = json.load(f)
        self._idx = {}

    def _table(self, name):
        if name not in self._idx:
            self._idx[name] = {r["token"]: r for r in self.tables[name]}
        return self._idx[name]

    def get(self, table, token):
        return self._table(table)[token]

    def extract_line(self, line_token):
        rec = self.get("line", line_token)
        pts = [(self.get("node", t)["x"], self.get("node", t)["y"])
               for t in rec["node_tokens"]]
        return LineString(pts)

    def extract_polygon(self, polygon_token):
        rec = self.get("polygon", polygon_token)
        ext = [(self.get("node", t)["x"], self.get("node", t)["y"])
               for t in rec["exterior_node_tokens"]]
        # map expansion v1.3 stores holes as [{'node_tokens': [...]}];
        # v1.2 used a flat 'interior_node_tokens' list.
        holes = rec.get("holes") or [
            {"node_tokens": t} for t in rec.get("interior_node_tokens", [])]
        ints = [[(self.get("node", t)["x"], self.get("node", t)["y"])
                 for t in hole["node_tokens"]] for hole in holes]
        return Polygon(ext, ints)


class MapExplorerLite(object):
    """Reproduces NuScenesMapExplorer patch semantics: crop by the rotated
    patch, then rotate by -angle around the patch center and translate the
    center to the origin."""

    def __init__(self, nusc_map):
        self.map_api = nusc_map
        self._layer_index = {}

    def _layer_geoms(self, layer_name):
        """Build once per layer: all record geometries + STRtree. The devkit
        _get_layer_* iterate EVERY record and intersect with the rotated
        patch; querying the tree with the patch returns every record whose
        bounding box hits the patch envelope, a strict superset, so the
        filtered intersection results are identical to the full iteration."""
        if layer_name in self._layer_index:
            return self._layer_index[layer_name]
        geoms, recs = [], []
        for rec in self.map_api.tables[layer_name]:
            try:
                if "line_token" in rec:
                    g = self.map_api.extract_line(rec["line_token"])
                else:
                    g = self.map_api.extract_polygon(rec["polygon_token"])
            except Exception:
                continue
            if g.is_empty:
                continue
            geoms.append(g)
            recs.append(rec)
        entry = (STRtree(geoms), recs) if geoms else None
        self._layer_index[layer_name] = entry
        return entry

    def _layer_candidates(self, layer_name, patch):
        entry = self._layer_geoms(layer_name)
        if entry is None:
            return []
        tree, recs = entry
        return [(recs[i], tree.geometries[i]) for i in tree.query(patch)]

    def get_patch_coord(self, patch_box, rotation=0.0):
        # patch_box = (center_x, center_y, len_y, len_x) - nuScenes order;
        # shapely box(minx, miny, maxx, maxy) gets len_x on the x extent.
        cx, cy = patch_box[0], patch_box[1]
        patch = box(cx - patch_box[3] / 2, cy - patch_box[2] / 2,
                    cx + patch_box[3] / 2, cy + patch_box[2] / 2)
        return affinity.rotate(patch, rotation, origin=(cx, cy))

    def _to_local(self, geom, patch_box, patch_angle):
        geom = affinity.rotate(geom, -patch_angle,
                               origin=(patch_box[0], patch_box[1]))
        return affinity.affine_transform(
            geom, [1, 0, 0, 1, -patch_box[0], -patch_box[1]])

    def _get_layer_line(self, patch_box, patch_angle, layer_name):
        patch = self.get_patch_coord(patch_box, patch_angle)
        out = []
        for _, line in self._layer_candidates(layer_name, patch):
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                out.append(self._to_local(new_line, patch_box, patch_angle))
        return out

    def _get_layer_polygon(self, patch_box, patch_angle, layer_name):
        patch = self.get_patch_coord(patch_box, patch_angle)
        out = []
        for _, poly in self._layer_candidates(layer_name, patch):
            new_poly = poly.intersection(patch)
            if not new_poly.is_empty:
                out.append(self._to_local(new_poly, patch_box, patch_angle))
        return out


class MapExtractor(object):
    """Per-sample map GT in the lidar frame; mirrors NuscMapExtractor
    (divider / merged ped crossings / drivable-area boundary) and
    VectorizeMap(simplify=True) sampling.

    Args:
        map_root (str): directory holding <location>.json expansion files.
        roi_size (tuple): (len_x, len_y) of the BEV roi, default (30, 60).
        simplify (float): simplify tolerance passed to LineString.simplify;
            None disables simplification (keep raw geometry).
    """

    def __init__(self, map_root, roi_size=(30, 60), simplify=0.2):
        self.map_root = map_root
        self.roi_size = roi_size
        self.simplify = simplify
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                               roi_size[0] / 2, roi_size[1] / 2)
        self._maps = {}
        self._explorers = {}

    def _explorer_for(self, location):
        if location not in self._explorers:
            path = os.path.join(self.map_root, location + ".json")
            nmap = NuScenesMapLite(path)
            self._maps[location] = nmap
            self._explorers[location] = MapExplorerLite(nmap)
        return self._explorers[location]

    def _union_ped(self, ped_geoms):
        """Merge same-direction overlapping ped crossings (ported to
        shapely-2 STRtree index semantics from nuscmap_extractor)."""
        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:] - rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()
            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = STRtree(ped_geoms)
        final_pgeom = []
        remain_idx = list(range(len(ped_geoms)))
        for i, pgeom in enumerate(ped_geoms):
            if i not in remain_idx:
                continue
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)
            for o_idx in tree.query(pgeom):
                if o_idx not in remain_idx:
                    continue
                o = ped_geoms[o_idx]
                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees
                    final_pgeom[-1] = final_pgeom[-1].union(o)
                    remain_idx.pop(remain_idx.index(o_idx))

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results

    def _sample_line(self, geom):
        """VectorizeMap(simplify=True) equivalent for one LineString."""
        if self.simplify is not None:
            geom = geom.simplify(self.simplify, preserve_topology=True)
        coords = np.array(geom.coords)[:, :2]
        if len(coords) < 2:
            return None
        return coords.astype(np.float64)

    def get_map_annos(self, location, translation, rotation):
        """GT for one sample.

        Args:
            location (str): nuScenes log location (map name).
            translation (array): lidar2global translation, shape (3,).
            rotation (array): lidar2global 3x3 rotation matrix.

        Returns:
            annos (dict): {class_id: [line coords (N, 2) ...]} in the lidar
                frame, class order MAP_CLASSES = (ped, divider, boundary).
        """
        ex = self._explorer_for(location)
        patch_box = (translation[0], translation[1],
                     self.roi_size[1], self.roi_size[0])
        yaw = rotation_yaw_deg(rotation)

        all_dividers = []
        for layer in ("lane_divider", "road_divider"):
            for line in ex._get_layer_line(patch_box, yaw, layer):
                all_dividers += split_collections(line)

        ped_crossings = []
        for p in ex._get_layer_polygon(patch_box, yaw, "ped_crossing"):
            ped_crossings += split_collections(p)
        ped_crossings = self._union_ped(ped_crossings)
        ped_lines = []
        for p in ped_crossings:
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_lines.append(line)

        road_segments = ex._get_layer_polygon(patch_box, yaw, "road_segment")
        lanes = ex._get_layer_polygon(patch_box, yaw, "lane")
        if road_segments or lanes:
            union_roads = ops.unary_union(road_segments)
            union_lanes = ops.unary_union(lanes)
            drivable = split_collections(
                ops.unary_union([union_roads, union_lanes]))
            boundaries = get_drivable_area_contour(drivable, self.roi_size)
        else:
            drivable, boundaries = [], []

        geoms = dict(divider=all_dividers, ped_crossing=ped_lines,
                     boundary=boundaries, drivable_area=drivable)
        annos = {}
        for cls, label in zip(("ped_crossing", "divider", "boundary"),
                              (0, 1, 2)):
            annos[label] = []
            for geom in geoms[cls]:
                if geom.geom_type == "LineString":
                    sampled = self._sample_line(geom)
                    if sampled is not None:
                        annos[label].append(sampled)
        return annos
