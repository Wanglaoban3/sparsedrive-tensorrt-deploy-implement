# -*- coding: utf-8 -*-
"""Vectorized-map evaluation of e_T6 (DFA-v8) engine outputs on nuScenes
mini, standalone: no mmcv / mmdet / nuscenes-devkit needed.

Pipeline:
  1. GT   - MapExtractor (deploy/map_gt.py) rebuilds per-token vectorized
            map ground truth from the nuScenes map-expansion jsons
            (classes 0/1/2 = ped_crossing/divider/boundary, lidar frame,
            VectorizeMap(simplify=True)-equivalent sampling).
  2. Pred - decode map_cls/map_pts tensors dumped from the engine
            (sigmoid on [100, 3] class logits; [100, 20, 2] points are
            already absolute lidar-frame coordinates).
  3. Match- chamfer-distance instance matching + AP over thresholds
            {0.5, 1.0, 1.5} m, reusing the reference implementation from
            projects/mmdet3d_plugin/datasets/evaluation/map/AP.py
            (loaded by path; its package-relative import is rewritten).

Usage (from the repo root):
    python deploy/eval_t6_mini_map.py [--score-floor 0.0]
Requires: data/infos/mini/nuscenes_infos_val.pkl, the map-expansion jsons
under evaldata/map_expansion/, and engine dumps under
work_dirs/sparsedrive_small_stage2/evaldata/mini_eng_v8/.
"""
import argparse
import json
import os
import pickle
import sys
import types
import importlib.util as ilu

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENG = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_eng_v8")
PKL = os.path.join(ROOT, "data", "infos", "mini", "nuscenes_infos_val.pkl")
MAP_ROOT = os.path.join(ROOT, "evaldata", "map_expansion")
ART = os.path.join(ROOT, "deploy", "artifacts", "eval_t6_mini_map.json")

AP_PATH = os.path.join(
    ROOT, "projects", "mmdet3d_plugin", "datasets", "evaluation", "map",
    "AP.py")
DIST_PATH = os.path.join(
    ROOT, "projects", "mmdet3d_plugin", "datasets", "evaluation", "map",
    "distance.py")

sys.path.insert(0, os.path.join(ROOT, "deploy"))
from map_gt import MapExtractor, MAP_CLASSES  # noqa: E402

INTERP_NUM = 200
THRESHOLDS = [0.5, 1.0, 1.5]


def load_reference_matchers():
    """Load the reference AP.py/distance.py by path. AP.py uses a
    package-relative import ('from .distance import ...'), which is
    rewritten to a flat-module import of the same file."""
    spec = ilu.spec_from_file_location("map_distance", DIST_PATH)
    dist = ilu.module_from_spec(spec)
    spec.loader.exec_module(dist)

    with open(AP_PATH, "r", encoding="utf-8") as f:
        src = f.read().replace("from .distance import",
                               "from map_distance import")
    ap = types.ModuleType("map_ap")
    ap.__dict__["sys_modules_hint"] = None
    sys.modules["map_distance"] = dist
    exec(compile(src, AP_PATH, "exec"), ap.__dict__)
    return ap


def interp_fixed_num(vector, num_pts):
    """Same as VectorEvaluate.interp_fixed_num (shapely linspace interp)."""
    from shapely.geometry import LineString
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    sampled = np.array([list(line.interpolate(d).coords)
                        for d in distances]).squeeze()
    return sampled


def load_out(k, name, shape, eng, dt=np.float32):
    for pref in ("outv8_%02d", "out_%02d", "outv8_%03d", "out_%03d"):
        d = os.path.join(eng, pref % k)
        if os.path.isdir(d):
            return np.fromfile(os.path.join(d, name + ".bin"), dt).reshape(
                shape)
    raise FileNotFoundError("no frame dir for k=%d under %s" % (k, eng))


def decode_preds(k, score_floor, eng):
    """Engine map outputs -> per-token pred dict for one frame."""
    cls = 1.0 / (1.0 + np.exp(-load_out(k, "map_cls", (100, 3), eng)))
    pts = load_out(k, "map_pts", (100, 20, 2), eng)
    vectors, scores, labels = [], [], []
    for c in range(3):
        for a in range(100):
            s = float(cls[a, c])
            if s <= score_floor:
                continue
            vectors.append(pts[a])
            scores.append(s)
            labels.append(c)
    return {"vectors": vectors, "scores": scores, "labels": labels}


def build_gts(eng):
    sys.path.insert(0, ROOT)
    with open(PKL, "rb") as f:
        infos = pickle.load(f, encoding="latin1")["infos"]
    by_token = {info["token"]: info for info in infos}

    meta = np.load(os.path.join(eng, "mini_meta.npz"), allow_pickle=True)
    tokens = [str(t) for t in meta["tokens"]]
    missing = [t for t in tokens if t not in by_token]
    assert not missing, "infos pkl misses %d dump tokens" % len(missing)

    extractor = MapExtractor(MAP_ROOT)
    gts = {}
    from pyquaternion import Quaternion
    for k, token in enumerate(tokens):
        info = by_token[token]
        l2e = np.eye(4)
        l2e[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        l2e[:3, 3] = info["lidar2ego_translation"]
        e2g = np.eye(4)
        e2g[:3, :3] = Quaternion(
            info["ego2global_rotation"]).rotation_matrix
        e2g[:3, 3] = info["ego2global_translation"]
        l2g = e2g @ l2e
        annos = extractor.get_map_annos(
            info["map_location"], l2g[:3, 3], l2g[:3, :3])
        gts[token] = annos
        if (k + 1) % 20 == 0:
            counts = {c: len(annos[c]) for c in (0, 1, 2)}
            print("gt %2d/%d  %s  %s" % (k + 1, len(tokens), token[:8],
                                         counts), flush=True)
    return tokens, gts


def evaluate(tokens, gts, results, ap):
    """Aggregation ported from VectorEvaluate.evaluate (single process)."""
    id2cat = dict(enumerate(MAP_CLASSES))
    samples_by_cls = {label: [] for label in id2cat}
    num_gts = {label: 0 for label in id2cat}
    num_preds = {label: 0 for label in id2cat}

    for token in tokens:
        gt = gts[token]
        pred = results.get(token) or {"vectors": [], "scores": [],
                                      "labels": []}
        vectors_by_cls = {label: [] for label in id2cat}
        scores_by_cls = {label: [] for label in id2cat}
        for i in range(len(pred["labels"])):
            label = pred["labels"][i]
            vectors_by_cls[label].append(np.asarray(pred["vectors"][i]))
            scores_by_cls[label].append(float(pred["scores"][i]))
        for label in id2cat:
            new_sample = (vectors_by_cls[label], scores_by_cls[label],
                          gt.get(label, []))
            num_gts[label] += len(gt.get(label, []))
            num_preds[label] += len(scores_by_cls[label])
            samples_by_cls[label].append(new_sample)

    result_dict = {}
    sum_mAP = 0.0
    for label in id2cat:
        samples = samples_by_cls[label]
        result_dict[id2cat[label]] = {"num_gts": num_gts[label],
                                      "num_preds": num_preds[label]}
        sum_AP = 0.0
        tpfp_score_list = []
        for pred_vectors, scores, groundtruth in samples:
            # VectorEvaluate._evaluate_single: interpolate, instance_match,
            # then re-key by threshold with the score column hstacked.
            pred_lines = np.stack(
                [interp_fixed_num(v, INTERP_NUM) for v in pred_vectors]) \
                if pred_vectors else np.zeros((0, INTERP_NUM, 2))
            gt_lines = np.stack(
                [interp_fixed_num(v, INTERP_NUM) for v in groundtruth]) \
                if groundtruth else np.zeros((0, INTERP_NUM, 2))
            scores_arr = np.array(scores)
            tp_fp_list = ap.instance_match(
                pred_lines, scores_arr, gt_lines, THRESHOLDS, "chamfer")
            by_thr = {}
            for i, thr in enumerate(THRESHOLDS):
                tp, fp = tp_fp_list[i]
                by_thr[thr] = np.hstack(
                    [tp[:, None], fp[:, None], scores_arr[:, None]])
            tpfp_score_list.append(by_thr)
        for thr in THRESHOLDS:
            tp_fp_score = np.vstack([i[thr] for i in tpfp_score_list])
            sort_inds = np.argsort(-tp_fp_score[:, -1])
            tp = np.cumsum(tp_fp_score[sort_inds, 0])
            fp = np.cumsum(tp_fp_score[sort_inds, 1])
            eps = np.finfo(np.float32).eps
            recalls = tp / max(num_gts[label], eps)
            precisions = tp / np.maximum(tp + fp, eps)
            AP = ap.average_precision(recalls, precisions, "area")
            sum_AP += AP
            result_dict[id2cat[label]]["AP@%.1f" % thr] = float(AP)
        AP = sum_AP / len(THRESHOLDS)
        sum_mAP += AP
        result_dict[id2cat[label]]["AP"] = float(AP)

    result_dict["mAP"] = float(sum_mAP / len(id2cat))
    mAP_normal = sum(result_dict[id2cat[label]]["AP@%.1f" % thr]
                     for label in id2cat for thr in THRESHOLDS) / 9
    result_dict["mAP_normal"] = float(mAP_normal)
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng", default=ENG,
                        help="dump dir with outv8_XX/out_XX + mini_meta.npz")
    parser.add_argument("--art", default=ART, help="output results json")
    parser.add_argument("--score-floor", type=float, default=0.0,
                        help="drop pred candidates with score <= floor")
    args = parser.parse_args()

    ap = load_reference_matchers()
    print("loading engine dumps ...", flush=True)
    results = {}
    tokens, gts = build_gts(args.eng)
    for k, token in enumerate(tokens):
        results[token] = decode_preds(k, args.score_floor, args.eng)

    print("matching (chamfer, %s) ..." % THRESHOLDS, flush=True)
    result_dict = evaluate(tokens, gts, results, ap)

    header = ["category", "num_preds", "num_gts"] + \
        ["AP@%.1f" % t for t in THRESHOLDS] + ["AP"]
    widths = [max(len(h), 8) for h in header]
    print("|".join(h.ljust(w) for h, w in zip(header, widths)))
    for label in range(3):
        name = MAP_CLASSES[label]
        row = [name, str(result_dict[name]["num_preds"]),
               str(result_dict[name]["num_gts"])] + \
            ["%.4f" % result_dict[name]["AP@%.1f" % t] for t in THRESHOLDS] + \
            ["%.4f" % result_dict[name]["AP"]]
        print("|".join(v.ljust(w) for v, w in zip(row, widths)))
    print("mAP %.4f   mAP_normal %.4f" % (result_dict["mAP"],
                                          result_dict["mAP_normal"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.art)), exist_ok=True)
    with open(args.art, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print("saved", args.art)


if __name__ == "__main__":
    main()
