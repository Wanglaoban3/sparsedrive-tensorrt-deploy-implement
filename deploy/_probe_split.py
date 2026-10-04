import io, os, sys, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np

ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
with open(os.path.join(ROOT, "data", "infos", "mini", "nuscenes_infos_val.pkl"), "rb") as f:
    d = pickle.load(f)
infos = d["infos"] if isinstance(d, dict) and "infos" in d else d
print("n =", len(infos))
info = infos[0]
for k in ("scene_name", "scene_token", "token"):
    print(k, "->", info.get(k, "<missing>"))

import json
with open(os.path.join(ROOT, "data", "nuscenes", "v1.0-mini", "scene.json")) as f:
    scene_tab = {s["token"]: s["name"] for s in json.load(f)}

from nuscenes.utils.splits import create_splits_scenes
sp = create_splits_scenes()
mv = sp["mini_val"]
mt = sp["mini_train"]
print("mini_val scenes:", mv)
names = [scene_tab.get(x.get("scene_token", ""), "?scene") for x in infos]
n_val = sum(1 for n in names if n in mv)
n_train = sum(1 for n in names if n in mt)
print("frames in mini_val:", n_val, "mini_train:", n_train,
      "other:", len(names) - n_val - n_train)
