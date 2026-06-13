# sinai-agent same-activity performance benchmark

Date: 2026-06-13

Issue context: FEL-593, FEL-600, FEL-601

Trainer branch: `perf/FEL-601-trainer-eval-performance`

Baseline trainer code commit: `02ec556 FEL-593 add sinai-agent benchmark report`

## What Was Benchmarked

This is an apples-to-apples performance comparison for the same retrieval activity:

1. Start a cold process.
2. Load the `sinai-agent` model/package.
3. Read the same 100 Agentic-EI test rows.
4. Encode the same 200 row texts: 100 query texts and 100 positive chunk texts.
5. Compute 100 cosine scores.
6. Apply the same packaged per-domain thresholds.
7. Report wall time, peak resident memory, throughput, answer rate, and GPU availability.

The FEL-601 optimized Seen path preserves the same row-level activity and outputs, but caches duplicate text embeddings inside eval. On this 100-row test split, the 200 row texts contain 173 unique texts.

This report does not compare Seen training against Python eval-only. Training feasibility is noted only at the end because it is a different activity.

All commands were run under the required memory cap:

```bash
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
```

## Inputs

Model/package:

`/mnt/Storage/Projects/catbelly_studio/sinai/models/sinai-agent`

Seen config:

`/mnt/Storage/Projects/seen/trainer/config/sinai-agent.v0.1.adapter-full.resume.json`

Test set:

`/mnt/Storage/Projects/catbelly_studio/sinai/agentic_ei/v0.1/test.jsonl`

Thresholds:

`/mnt/Storage/Projects/catbelly_studio/sinai/models/sinai-agent/thresholds.json`

## Median Results

Three cold-process runs were executed for each runnable path.

| Implementation | Device Path | Status | Median Cold Process Wall | Median Peak RSS | Throughput, Cold Process | Answer Rate |
|---|---|---|---:|---:|---:|---:|
| Seen trainer CLI baseline | Vulkan GPU | Passed | 38.780s | 1,796,844 KB | 2.58 pairs/s, 5.16 row-texts/s | 0.86 |
| Seen trainer CLI after FEL-601 cache | Vulkan GPU | Passed | 31.739s | 1,672,192 KB | 3.15 pairs/s, 6.30 row-texts/s | 0.86 |
| Seen trainer CLI after adapter/load optimization | Vulkan GPU | Passed | 15.083s | 1,433,248 KB | 6.63 pairs/s, 13.26 row-texts/s | 0.86 |
| Python SentenceTransformer | CPU | Passed | 4.404s | 1,242,472 KB | 22.70 pairs/s, 45.41 texts/s | 1.00 |
| Python SentenceTransformer | CUDA | Failed | Failed at model-to-CUDA | 1,314,296 KB at failure | n/a | n/a |

Performance ratios:

- FEL-601 made Seen CLI eval `1.22x` faster than the baseline, reducing median wall time by `18.2%`.
- FEL-601 reduced Seen CLI peak RSS by `6.9%`.
- The adapter/load optimization pass made Seen CLI eval `2.57x` faster than the baseline, reducing median wall time by `61.1%`.
- The adapter/load optimization pass made Seen CLI eval `2.10x` faster than the first FEL-601 cache pass.
- Python SentenceTransformer CPU cold-process eval remains `3.43x` faster than optimized Seen CLI GPU/Vulkan eval for this activity.
- Optimized Seen CLI peak RSS remains `1.15x` the Python CPU path for this activity.
- Python CUDA was not comparable under the cap because all full-activity attempts failed before encoding.

## Python CPU Internal Timing

The Python CPU process also recorded internal timings after imports completed:

| Phase | Median Time |
|---|---:|
| Model load | 0.454s |
| Encode 200 texts | 0.464s |
| Score and threshold 100 pairs | 0.000224s |
| Internal activity total | 0.928s |

Using the internal activity timer, Python CPU processed:

- 107.78 pairs/s
- 215.56 texts/s

The cold-process number is the fairer cross-implementation headline because the Seen measurement is also process-level.

## Run Details

Seen trainer CLI after adapter/load optimization, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed |
|---:|---:|---:|---:|---|
| 1 | 15.182s | 1,434,008 KB | 0.86 | yes |
| 2 | 15.083s | 1,431,832 KB | 0.86 | yes |
| 3 | 14.879s | 1,433,248 KB | 0.86 | yes |

The optimized eval path now:

- Uses a lean inference bundle for `mine`, `calibrate`, and `eval`, skipping training-only weight-map and parameter-registry inspection.
- Adds an eval-local flat embedding cache so duplicate query/chunk texts are encoded once.
- Scores cached normalized embeddings with direct dot product, avoiding a second normalize-per-score pass.
- Reuses low-rank adapter bottlenecks instead of recomputing them for every output dimension.
- Applies layer projection adapters directly over flat token sequences, avoiding row-vector and projected-vector allocation per token.
- Uses a rank-4 fast path for the active Sinai adapter shape.
- Releases the temporary normalized arrays allocated by `math_utils.cosine` for call sites that still use that helper.

Measured but reverted:

- Process-local Vulkan pipeline caching regressed median wall time to `16.189s` and peak RSS to `1,628,772 KB`.
- In-place GPU matmul bias addition regressed median wall time to `16.241s`.

Seen trainer CLI after FEL-601 cache, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed |
|---:|---:|---:|---:|---|
| 1 | 31.812s | 1,672,192 KB | 0.86 | yes |
| 2 | 31.662s | 1,669,240 KB | 0.86 | yes |
| 3 | 31.739s | 1,672,544 KB | 0.86 | yes |

Seen trainer CLI baseline, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed |
|---:|---:|---:|---:|---|
| 1 | 38.780s | 1,796,968 KB | 0.86 | yes |
| 2 | 38.770s | 1,796,844 KB | 0.86 | yes |
| 3 | 38.855s | 1,796,020 KB | 0.86 | yes |

Seen GPU confirmation came from the eval logs:

- `backend gpu requires_gpu 1`
- `Vulkan device type 2`

Python SentenceTransformer CPU:

| Run | Cold Process Wall | Internal Activity Wall | Load | Encode | Score | Peak RSS | Answer Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.426s | 0.928s | 0.452s | 0.475s | 0.000240s | 1,242,272 KB | 1.00 |
| 2 | 4.404s | 0.930s | 0.470s | 0.460s | 0.000218s | 1,242,620 KB | 1.00 |
| 3 | 4.378s | 0.918s | 0.454s | 0.464s | 0.000224s | 1,242,472 KB | 1.00 |

Python SentenceTransformer CUDA:

| Run | Status | Wall Time To Failure | Peak RSS At Failure | Failure |
|---:|---|---:|---:|---|
| 1 | failed | 4.038s | 1,316,496 KB | CUDA driver out of memory while moving model to CUDA |
| 2 | failed | 3.941s | 1,314,188 KB | CUDA driver out of memory while moving model to CUDA |
| 3 | failed | 4.093s | 1,314,296 KB | CUDA driver out of memory while moving model to CUDA |

## Interpretation

For this exact eval-style retrieval activity, Python SentenceTransformer CPU is still much faster than the Seen trainer CLI even though Seen initializes the Vulkan GPU path.

The Seen optimization work moved the median from `38.780s` to `15.083s`. This is a substantial improvement, but it does not close the full gap to Python SentenceTransformer CPU.

The most likely reasons are implementation-level rather than model-level:

- The Seen CLI performs this eval through the trainer runtime path, with per-unique-text encode behavior and many small GPU dispatches.
- The Python path uses optimized HuggingFace tokenizer/runtime and batched SentenceTransformer encode.
- The Seen GPU backend is active, but this eval path is not yet optimized to batch the full 200-text workload into a single high-throughput GPU execution plan.

The answer rates are included to prove each path completed the same activity, but they should not be read as pure quality parity:

- Seen uses the Seen tokenizer/MiniLM/runtime and adapter JSON path.
- Python uses the SentenceTransformer package path with the composed `2_Dense` adapter module and `3_Normalize`.
- These paths are compatible, but not numerically identical.

## Training Note, Not Part Of This Benchmark

The historical Catbelly Python training recipe was also tested earlier under the same memory cap:

- Batch size 64 training failed with CUDA OOM during the first transformer forward.
- Microbatch size 8 training failed with CUDA OOM at the AdamW optimizer step.

That is a training-feasibility result, not part of the same-activity performance comparison above.

## Reproduction

Seen CLI activity:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
target/trainer eval --config config/sinai-agent.v0.1.adapter-full.resume.json
```

Python CPU activity:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /mnt/Storage/Projects/catbelly_studio/.venv/bin/python <same-activity-sentencetransformer-script>
```

Python CUDA activity:

```bash
cd /mnt/Storage/Projects/seen/trainer
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /mnt/Storage/Projects/catbelly_studio/.venv/bin/python <same-activity-sentencetransformer-cuda-script>
```

## Follow-Up Needed

The benchmark shows the Seen trainer is functionally runnable on GPU and now much faster than the first measured implementation, but not performance competitive with Python SentenceTransformer CPU for this eval workload yet. FEL-601 removed duplicate eval encodes, redundant score normalization, training-only inference setup, and avoidable adapter recomputation/allocation. The next performance task should add a Seen benchmark mode that reports separate load, tokenize, encode, score, and threshold timings, then batch the eval text encoding path so Vulkan work amortizes dispatch overhead across the full text batch.
