# -*- coding: utf-8 -*-
import json

base = r"<REPO>\deploy\artifacts"
p = base + r"\eval_fp32_val\metrics_summary.json"
m = json.load(open(p, encoding="utf-8"))
print("eval_fp32_val: mAP=%.4f NDS=%.4f" % (
    m.get("mean_ap", -1), m.get("nd_score", -1)))
for f in ("eval_t3_mini.json", "eval_t6_mini.json", "eval_v8_mini.json",
          "eval_fp32_val.json"):
    try:
        d = json.load(open(base + "\\" + f, encoding="utf-8"))
        print(f, {k: round(v, 4) for k, v in d.items()
                  if k.endswith(("mAP", "NDS"))})
    except Exception as e:
        print(f, "ERR", e)
