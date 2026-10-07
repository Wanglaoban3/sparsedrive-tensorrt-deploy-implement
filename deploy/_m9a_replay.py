# -*- coding: utf-8 -*-
"""M9a §7.5 一致性门禁 — numpy 参考规则实现 (C 触发器的镜像).

输入: ctx_%05d.bin 夹具 (sp_trigger ctx_dump, 布局钉死见 SDD ledger) +
thr.conf; 输出: 与 C 触发器 events_raw.jsonl 同构的事件流 (免配额).
镜像纪律: 公式逐条对齐 deploy/prepost/rules/*.c 与 sp_rule_util.h;
数值路径全 float64 (ctx 内 ego/det 为 float32 存储, 两侧同源).

ctx bin 布局 (小端):
  u32 magic=0x53505431, u32 abi, u32 flags, u64 seq, i64 ts_ns, u32 scene,
  u32 n_det, u32 n_hist, ego(6f32), n_det×trk(12f32 一条: score,label,id,
  x,y,z,w,l,h,yaw,vx,vy), n_hist×{ego(6f32), i64 ts, u32 n, u32 pad, n×trk}
"""
import math
import os
import re
import struct

import numpy as np

MAGIC = 0x53505431
EGO = np.dtype([("x", "<f4"), ("y", "<f4"), ("heading", "<f4"),
                ("speed", "<f4"), ("acc", "<f4"), ("yaw_rate", "<f4")])
TRK = np.dtype([("score", "<f4"), ("label", "<i4"), ("id", "<i4"),
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("w", "<f4"),
                ("l", "<f4"), ("h", "<f4"), ("yaw", "<f4"), ("vx", "<f4"),
                ("vy", "<f4")])


def read_ctx(path):
    b = open(path, "rb").read()
    magic, abi, flags = struct.unpack_from("<III", b, 0)
    assert magic == MAGIC, "bad magic in %s" % path
    seq, ts_ns = struct.unpack_from("<Qq", b, 12)
    scene, n_det, n_hist = struct.unpack_from("<III", b, 28)
    off = 40
    ego = np.frombuffer(b, EGO, 1, off)[0]
    off += EGO.itemsize
    det = np.frombuffer(b, TRK, n_det, off) if n_det else []
    off += n_det * TRK.itemsize
    hist_ego, hist_ts, hist_det = [], [], []
    for _ in range(n_hist):
        hist_ego.append(np.frombuffer(b, EGO, 1, off)[0])
        off += EGO.itemsize
        t, n, _pad = struct.unpack_from("<qII", b, off)
        hist_ts.append(t)
        off += 16
        hist_det.append(np.frombuffer(b, TRK, n, off) if n else [])
        off += n * TRK.itemsize
    return dict(seq=seq, ts_ns=ts_ns, scene=scene, ego=ego, det=det,
                n_hist=n_hist, hist_ego=hist_ego, hist_ts=hist_ts,
                hist_det=hist_det)


def parse_thr(path):
    secs = {}
    cur = None
    for ln in open(path, encoding="utf-8").read().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r"\[([a-z_]+)\]$", ln)
        if m:
            cur = m.group(1)
            secs[cur] = {}
        elif "=" in ln and cur:
            k, v = ln.split("=", 1)
            secs[cur][k] = float(v)
    return secs


def _chron(ctx, field):
    # ctx hist[0]=当前(新), 规则体要旧→新
    return np.array([float(ctx["hist_ego"][i][field])
                     for i in range(ctx["n_hist"] - 1, -1, -1)])


def _ts_chron(ctx):
    return np.array([float(ctx["hist_ts"][i])
                     for i in range(ctx["n_hist"] - 1, -1, -1)])


def _sustained(vals, ts, thr, ge, dur_s):
    # 镜像 ru_sustained: run 时间 = ts 实差和 >= dur_s (dt<0 记 0)
    n = len(vals)
    if n < 2:
        return False
    run, t = 0, 0.0
    for i in range(1, n):
        m = vals[i] >= thr if ge else vals[i] <= thr
        dt = (ts[i] - ts[i - 1]) / 1e9
        if dt < 0:
            dt = 0.0
        if m:
            run += 1
            t += dt
        else:
            run, t = 0, 0.0
        return_val = None
        if run >= 1 and t >= dur_s:
            return_val = True
            break
    return bool(return_val)


def _id_series(ctx, tid, want_ts=False):
    # 镜像 ru_id_series: 最老→当前, 每帧首条 id 匹配, 缺帧跳过
    sx, sy, svx, svy, sts = [], [], [], [], []
    for i in range(ctx["n_hist"] - 1, -1, -1):
        for d in ctx["hist_det"][i]:
            if int(d["id"]) == tid:
                sx.append(float(d["x"]))
                sy.append(float(d["y"]))
                svx.append(float(d["vx"]))
                svy.append(float(d["vy"]))
                sts.append(float(ctx["hist_ts"][i]))
                break
    return (sx, sy, svx, svy, sts if want_ts else None)


def _pick_lead(det, lon_min=2.0, lon_max=40.0, lat_max=2.5):
    best, bd = -1, 1e18
    for i, d in enumerate(det):
        if int(d["label"]) not in (0, 1, 2, 3, 4):
            continue
        if lon_min < float(d["x"]) < lon_max and abs(float(d["y"])) < lat_max:
            dd = math.hypot(float(d["x"]), float(d["y"]))
            if dd < bd:
                bd, best = dd, i
    return best


def _wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------- 14 条规则 (与 rules/*.c 逐条镜像) ----------
def r_hard_brake(c, s, thr):
    a, ts = _chron(c, "acc"), _ts_chron(c)
    n = len(a)
    if n < 2 or not _sustained(a, ts, thr["acc_thr"], 0, thr["dur_s"]):
        return None
    return min(1.0, -min(a) / 5.0)


def r_hard_accel(c, s, thr):
    a, ts = _chron(c, "acc"), _ts_chron(c)
    n = len(a)
    if n < 2 or not _sustained(a, ts, thr["acc_thr"], 1, thr["dur_s"]):
        return None
    return min(1.0, max(a) / 5.0)


def r_low_speed_crawl(c, s, thr):
    a, ts = _chron(c, "speed"), _ts_chron(c)
    n = len(a)
    if n < 2:
        return None
    win = thr["win_s"] * 1e9
    t_end = ts[-1]
    start = n - 1
    while start > 0 and (t_end - ts[start - 1]) <= win:
        start -= 1
    if (t_end - ts[start]) / 1e9 < thr["win_s"]:
        return None
    if float(np.mean(a[start:])) < thr["speed_thr"]:
        return 1.0
    return None


def r_reverse(c, s, thr):
    a, ts = _chron(c, "speed"), _ts_chron(c)
    if len(a) < 2 or not _sustained(a, ts, thr["speed_thr"], 0,
                                    thr["dur_s"]):
        return None
    return 1.0


def r_sharp_turn(c, s, thr):
    a = np.abs(_chron(c, "yaw_rate"))
    ts = _ts_chron(c)
    if len(a) < 2 or not _sustained(a, ts, thr["yaw_thr"], 1, thr["dur_s"]):
        return None
    return min(1.0, max(a) / 0.6)


def r_sharp_lat(c, s, thr):
    a = np.abs(_chron(c, "speed") * _chron(c, "yaw_rate"))
    ts = _ts_chron(c)
    if len(a) < 2 or not _sustained(a, ts, thr["lat_thr"], 1, thr["dur_s"]):
        return None
    return min(1.0, max(a) / 3.5)


def r_u_turn(c, s, thr):
    a, ts = _chron(c, "heading"), _ts_chron(c)
    n = len(a)
    if n < 2:
        return None
    win = thr["win_s"] * 1e9
    start = n - 1
    while start > 0 and (ts[-1] - ts[start - 1]) <= win:
        start -= 1
    total = sum(_wrap_pi(a[i] - a[i - 1]) for i in range(start + 1, n))
    if abs(total) * 180.0 / math.pi > thr["deg_thr"]:
        return 1.0
    return None


def r_lead_hard_brake(c, s, thr):
    li = _pick_lead(c["det"])
    if li < 0:
        return None
    d = c["det"][li]
    sx, _, svx, _, _ = _id_series(c, int(d["id"]))
    if len(svx) < 3:
        return None
    d0 = math.hypot(float(d["x"]), float(d["y"]))
    if (d0 < thr["dist_thr"] and min(svx) <= thr["decel_thr"]
            and max(svx) > thr["v0_thr"]):
        return min(1.0, -min(svx) / 5.0)
    return None


def r_slow_lead(c, s, thr):
    li = _pick_lead(c["det"])
    if li < 0:
        return None
    d = c["det"][li]
    _, _, svx, _, _ = _id_series(c, int(d["id"]))
    if len(svx) < 3:
        return None
    d0 = math.hypot(float(d["x"]), float(d["y"]))
    if (d0 < thr["dist_thr"] and float(np.mean(svx)) < thr["speed_thr"]
            and float(s["ego.speed"]) > thr["ego_thr"]):
        return 1.0
    return None


def r_stationary_approach(c, s, thr):
    li = _pick_lead(c["det"])
    if li < 0:
        return None
    d = c["det"][li]
    _, _, svx, _, _ = _id_series(c, int(d["id"]))
    if len(svx) < 3:
        return None
    d0 = math.hypot(float(d["x"]), float(d["y"]))
    ego = float(s["ego.speed"])
    if ego > thr["ego_thr"] and float(np.mean(svx)) < thr["vlead_thr"] \
            and d0 / ego < thr["ttc_thr"]:
        return min(1.0, thr["ttc_thr"] - d0 / ego)
    return None


def r_cut_in(c, s, thr):
    for d in c["det"]:
        if int(d["label"]) not in (0, 1, 2, 3, 4) or int(d["id"]) < 0:
            continue
        if float(d["x"]) > thr["dist_thr"]:
            continue
        _, sy, _, _, _ = _id_series(c, int(d["id"]))
        if len(sy) < 3:
            continue
        if (abs(sy[0]) >= thr["lat_edge"] and abs(sy[-1]) < thr["lat_in"]
                and (sy[-1] - sy[0]) * sy[0] < 0):
            return 1.0
    return None


def r_vru_near(c, s, thr):
    for d in c["det"]:
        if int(d["label"]) not in (6, 7, 8):
            continue
        if (thr["lon_min"] < float(d["x"]) < thr["lon_max"]
                and abs(float(d["y"])) < thr["lat_thr"]):
            return 1.0
    return None


def r_vru_cross(c, s, thr):
    for d in c["det"]:
        if int(d["label"]) not in (6, 7, 8) or int(d["id"]) < 0:
            continue
        if not (thr["lon_min"] < float(d["x"]) < thr["lon_max"]):
            continue
        _, sy, _, _, sts = _id_series(c, int(d["id"]), want_ts=True)
        if len(sy) < 3:
            continue
        lat_sp_max = 0.0
        for j in range(1, len(sy)):
            dt = (sts[j] - sts[j - 1]) / 1e9
            if dt <= 0:
                continue
            lat_sp_max = max(lat_sp_max, abs((sy[j] - sy[j - 1]) / dt))
        lat_min = min(abs(v) for v in sy)
        if lat_sp_max > thr["cross_thr"] and lat_min < thr["lat_thr"]:
            return 1.0
    return None


def r_construction_zone(c, s, thr):
    dmin = 1e18
    for d in c["det"]:
        if int(d["label"]) == 9:
            dmin = min(dmin, math.hypot(float(d["x"]), float(d["y"])))
    if dmin < thr["dist_thr"]:
        return 1.0 - dmin / thr["dist_thr"]
    return None


RULES = {
    "hard_brake": r_hard_brake, "hard_accel": r_hard_accel,
    "low_speed_crawl": r_low_speed_crawl, "reverse": r_reverse,
    "sharp_turn": r_sharp_turn, "sharp_lat": r_sharp_lat,
    "u_turn": r_u_turn, "lead_hard_brake": r_lead_hard_brake,
    "slow_lead": r_slow_lead, "stationary_approach": r_stationary_approach,
    "cut_in": r_cut_in, "vru_near": r_vru_near, "vru_cross": r_vru_cross,
    "construction_zone": r_construction_zone,
}


def replay_dir(ctx_dir, thr_path):
    """回放 ctx 目录 → 事件列表 [(seq, event, strength)] (按文件序=tick 序)."""
    secs = parse_thr(thr_path)
    events = []
    for fn in sorted(os.listdir(ctx_dir)):
        if not fn.endswith(".bin"):
            continue
        c = read_ctx(os.path.join(ctx_dir, fn))
        for name, fn_rule in RULES.items():
            thr = dict(secs.get(name, {}))
            strg = fn_rule(c, {"ego.speed": float(c["ego"]["speed"])}, thr)
            if strg is not None:
                events.append((c["seq"], name, strg))
    return events
