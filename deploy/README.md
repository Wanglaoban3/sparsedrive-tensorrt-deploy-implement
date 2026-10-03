# `deploy/` — Deployment & Optimization Toolchain

Everything that turned the upstream SparseDrive checkpoint into the
delivered TensorRT artifact lives here:

```
export_v5.py  →  v5_P1h.onnx  →  onnx2engine2  →  e_T6.engine
                                      +
                     dfaplug_v8.cu  →  libdfaplug_v8.so
                                      +
              run_engine / run_engines  +  eval_t6_mini_v8.py (accuracy gate)
```

The full story — every accepted and rejected optimization, measurements,
methodology and pitfalls — is [docs/OPTIMIZATION_SUMMARY.md](../docs/OPTIMIZATION_SUMMARY.md).

## Key files

### Delivery chain (the files that produce/verify the artifact)

| File | Role |
|---|---|
| `export_v5.py` | INT8+FP16 delivery ONNX export (logits-DFA: softmax moved into the plugin) |
| `dfaplug_v8.cu` | **delivery DFA plugin** — plan+gather dual kernel, self-allocating workspace, v3 fallback |
| `dfaplug_v3.cu` | single-kernel DFA (fallback path; logits semantics identical, ulp-equal) |
| `onnx2engine.cpp` / `onnx2engine3.cpp` | board-side ONNX→TRT compiler (`--int8 --fp16 --ws-mb`, per-layer precision overrides) |
| `run_engine.cpp` / `run_engine2.cpp` | single-engine runner (manifest input dirs, `--dump`, timing) |
| `run_engines.cpp` | N-engine chained runner, tensor-name D2D wiring (zero-copy on unified memory) |
| `prep_mini_inputs.py` | 81-frame mini engine-input preparation (scene-boundary state resets) |
| `repro mini pipelines` | see `../repro/mini_pipeline*.sh` (board side) |
| `eval_t6_mini_v8.py` (+`eval_t6_mini.py`/`_t12`/`_sp`) | mini closed-loop decoding + nuScenes mAP/NDS gate |
| `dfaplug_dump.cu` | runtime IO-dump plugin (env-gated) — captures real engine inputs for kernel A/B |
| `test_dfa_ab.cu` / `test_dfa_real2.cu` / `test_dfa_kernels.cu` | DFA harness: real-IO A/B, phase timing, kernel compare |
| `sim_dfa_v8.py` | numpy simulator of DFA semantics (variant validation, weight analysis) |
| `extract_dfa_consts.py` / `trace_dfa_inputs.py` | pull true shape/ssi constants from the source graph; verify logits chain |

### Graph surgery (rounds 1-8)

| File | Round |
|---|---|
| `export_fp32_onnx.py` / `export_quant_onnx.py` / `make_v5t3.py` / `make_v5p1h.py` / `make_v5_p2h.py` | quantization-state pipeline (head de-quantized → head full-FP16), P1 |
| `split_engine.py` / `split_head.py` / `requant_fpn.py` / `strip_casts.py` | P4 engine splitting (negative result, kept for the record) |
| `ssa_surgery2.py` / `probe_fa_io.py` / `_fa_part.cu` / `dfaplug_v10.cu` | P5 FlashAttention plugin (negative result, kept) |
| `selftest_fa*.cu` | wmma semantics probes (col_major verdict, determinism) |
| `census_det.py` / `analyze_spans.py` / `profile_buckets.py` | trtexec profile attribution |

### Board drivers (`board_*.py`, `push_*.py`)

One-shot paramiko drivers used during the optimization sessions: push a
source file, compile module-level on the board, run/re-profile, poll a DONE
marker, pull logs. They are the working record of rounds 1-8 (naming:
`board_<topic><n>.py` and `board_<topic><n>_poll.py` pairs). All of them read
two environment variables — `BOARD_HOST` and `BOARD_PASS` — and never embed
credentials. New board automation should follow the same pattern (see
`board_ssa4.py` for the canonical shape).

> Board scripts are pushed to a Linux board: keep them LF-only (see
> `.gitattributes`); CRLF silently breaks bash on the board.

### Session analysis (`_*.py`)

Scratch numerics/profiling checks kept as the audit trail of each round
(`_replay_check.py` = probe/replay numerics gate, `_diff_det.py` = A/B
diff, `_prof_dist3.py` = profile bucketing). Read them as documentation of
how each conclusion was reached; they are not part of the delivery chain.

### `artifacts/`

Pulled evaluation/profile evidence (mini mAP/NDS summaries per engine
variant, profile logs). Large blobs (pkl, results json) are gitignored.
