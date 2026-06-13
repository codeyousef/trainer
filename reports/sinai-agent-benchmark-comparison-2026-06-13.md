# sinai-agent benchmark comparison

Date: 2026-06-13

Issue context: FEL-593, FEL-600

Trainer branch: `feat/FEL-593-sinai-agent-training-roadmap`

Trainer commit: `284b80d FEL-600 export adapter dense module for ST packages`

Catbelly artifact commit: `0be01c018 FEL-600 refresh sinai-agent ST package` local only, not pushed.

## Scope

This report compares the Seen-native trainer path against the closest Catbelly Python/SentenceTransformer path for the `sinai-agent` MiniLM-based action evidence retriever.

The Seen trainer is the source of truth for training. Catbelly Python code was used only as the historical recipe reference and for SentenceTransformer package compatibility checks.

All Seen build, package, calibration, eval, and training commands were run with the required memory cap:

```bash
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
```

## Artifacts

Seen trainer repo:

`/mnt/Storage/Projects/seen/trainer`

Trainer config:

`/mnt/Storage/Projects/seen/trainer/config/sinai-agent.v0.1.adapter-full.resume.json`

Model package:

`/mnt/Storage/Projects/catbelly_studio/sinai/models/sinai-agent`

Dataset:

`/mnt/Storage/Projects/catbelly_studio/sinai/agentic_ei/v0.1`

Package contents relevant to benchmarking:

- `config.json` with `num_hidden_layers: 1`
- `modules.json` with `0_Transformer`, `1_Pooling`, `2_Dense`, `3_Normalize`
- `2_Dense/model.safetensors` containing the composed Seen adapter projection
- `thresholds.json`
- `eval_results.json`
- `training_config.json`

## Benchmark Summary

| Path | Result | Wall Time | Peak RSS | Notes |
|---|---:|---:|---:|---|
| Seen native eval | 86/100, answer rate 0.86 | 37.207s | 1,801,968 KB | GPU path confirmed with `backend gpu requires_gpu 1` and `Vulkan device type 2`. |
| SentenceTransformer package compatibility eval | 100/100, answer rate 1.00 | load 0.424s, eval 0.443s | 1,241,056 KB | CPU path under cap with tokenizer/BLAS single-thread caps. |
| Catbelly Python base MiniLM eval-only | 87/100, answer rate 0.87 | load 0.730s, calibration 0.292s, eval 0.167s | 1,752,704 KB | Eval-only baseline from the historical Python stack. |
| Catbelly Python batch-64 training | Failed | n/a | n/a | CUDA OOM during first transformer forward under the required cap. |
| Catbelly Python microbatch-8 training | Failed | n/a | n/a | CUDA OOM at AdamW optimizer step under the required cap. |

## Seen Trainer Measurements

Full trained Seen artifact:

- Training used the Seen trainer CLI, not Python ML training code.
- GPU path initialized during training/package/eval.
- Three effective GPU epochs completed via bounded runs.
- Loss progression: `0.464458 -> 0.459494 -> 0.454278`.

Current native eval after fresh calibration:

- Total: `100`
- Answered: `86`
- Answer rate: `0.86`
- Wall time: `37.207165s`
- Peak RSS: `1,801,968 KB`
- GPU confirmation: `backend gpu requires_gpu 1`, `Vulkan device type 2`

Per-domain native Seen eval:

| Domain | Answered | Total | Answer Rate |
|---|---:|---:|---:|
| customer_support | 10 | 10 | 1.000 |
| engineering | 8 | 8 | 1.000 |
| finance | 10 | 11 | 0.909 |
| hr | 7 | 13 | 0.538 |
| legal | 5 | 5 | 1.000 |
| marketing | 7 | 8 | 0.875 |
| operations | 13 | 13 | 1.000 |
| procurement | 4 | 8 | 0.500 |
| project_management | 9 | 9 | 1.000 |
| sales | 8 | 10 | 0.800 |
| security_it | 5 | 5 | 1.000 |

Earlier 64-example Seen benchmark:

- Train: final loss `0.475822`, wall `40.115s`, peak RSS `2,017,628 KB`.
- Calibrate: 11 domains, wall `16.046s`, peak RSS `1,490,612 KB`.
- Eval: answer rate `0.875`, wall `18.050s`, peak RSS `1,664,372 KB`.

## SentenceTransformer Compatibility

The previous package loaded in SentenceTransformer only after metadata cleanup, but it scored `0/100` with Seen thresholds because the Python path was effectively using base MiniLM embeddings and did not apply Seen adapters.

The fixed package now exports the Seen projection adapters as a standard SentenceTransformer Dense module:

- `minilm_layer_adapter`
- `layer_adapter`
- `adapter`

Those are composed into `2_Dense/model.safetensors` as `linear.weight`, followed by `3_Normalize`.

Current compatibility benchmark with refreshed thresholds:

- Total: `100`
- Answered: `100`
- Answer rate: `1.00`
- Load wall time: `0.423627s`
- Eval wall time: `0.442577s`
- Peak RSS: `1,241,056 KB`
- Score min: `0.869157`
- Score p5: `0.873494`
- Score mean: `0.894674`
- Score median: `0.894158`
- Score max: `0.921517`

SentenceTransformer CUDA load still failed under the required virtual-memory cap while moving tensors to CUDA. The compatibility benchmark therefore uses CPU under the cap, while Seen native eval confirms the GPU/Vulkan path.

## Catbelly Python Training Comparison

The closest Catbelly Python setup uses the historical MiniLM triplet recipe. It was tested as a reference path only.

Results under the required memory cap:

- Batch size 64 training failed with CUDA OOM during the first transformer forward.
- Microbatch size 8 training also failed with CUDA OOM at the AdamW optimizer step.
- Eval-only base MiniLM succeeded and answered `87/100`.

This means the Python setup can provide a baseline eval number, but not a successful capped training comparison on this machine without changing the recipe.

## Interpretation

The benchmarking goal is now executable:

- Seen trainer can package, calibrate, and evaluate the model under the cap.
- Seen native path runs on GPU/Vulkan.
- SentenceTransformer can load and use the packaged model through standard modules.
- Catbelly Python training does not complete under the same cap, so it is not a viable capped training path for this comparison.

Important caveat:

SentenceTransformer package scores are compatible with the package thresholds but are not numerically identical to Seen runtime scores. The likely causes are tokenizer/runtime math differences between HuggingFace/SentenceTransformer and the Seen MiniLM/tokenizer implementation. Exact Seen/ST parity should be tracked as a separate follow-up if required.

## Reproduction Commands

Seen check:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen check src/main.seen
```

Seen compile:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen compile src/main.seen target/trainer --fast --no-fork --emit-glsl --no-cache --jobs=1 --opt-jobs=1
```

Seen calibrate/eval/package:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
target/trainer calibrate --config config/sinai-agent.v0.1.adapter-full.resume.json
target/trainer eval --config config/sinai-agent.v0.1.adapter-full.resume.json
target/trainer package --config config/sinai-agent.v0.1.adapter-full.resume.json
```

SentenceTransformer compatibility eval:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /mnt/Storage/Projects/catbelly_studio/.venv/bin/python <compatibility-eval-snippet>
```
