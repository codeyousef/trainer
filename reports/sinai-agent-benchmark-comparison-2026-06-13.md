# sinai-agent same-activity performance benchmark

Date: 2026-06-13

Issue context: FEL-593, FEL-600

Trainer branch: `feat/FEL-593-sinai-agent-training-roadmap`

Benchmarked trainer code commit: `02ec556 FEL-593 add sinai-agent benchmark report`

## What Was Benchmarked

This is an apples-to-apples performance comparison for the same retrieval activity:

1. Start a cold process.
2. Load the `sinai-agent` model/package.
3. Read the same 100 Agentic-EI test rows.
4. Encode the same 200 texts: 100 query texts and 100 positive chunk texts.
5. Compute 100 cosine scores.
6. Apply the same packaged per-domain thresholds.
7. Report wall time, peak resident memory, throughput, answer rate, and GPU availability.

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
| Seen trainer CLI | Vulkan GPU | Passed | 38.780s | 1,796,844 KB | 2.58 pairs/s, 5.16 texts/s | 0.86 |
| Python SentenceTransformer | CPU | Passed | 4.404s | 1,242,472 KB | 22.70 pairs/s, 45.41 texts/s | 1.00 |
| Python SentenceTransformer | CUDA | Failed | Failed at model-to-CUDA | 1,314,296 KB at failure | n/a | n/a |

Performance ratios:

- Python SentenceTransformer CPU cold-process eval was `8.80x` faster than Seen CLI GPU/Vulkan eval for this activity.
- Seen CLI peak RSS was `1.45x` the Python CPU path for this activity.
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

Seen trainer CLI, Vulkan GPU:

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

For this exact eval-style retrieval activity, Python SentenceTransformer CPU is currently much faster than the Seen trainer CLI even though Seen initializes the Vulkan GPU path.

The most likely reasons are implementation-level rather than model-level:

- The Seen CLI performs this eval through the trainer runtime path, with per-example encode/scoring behavior and substantial model/runtime setup overhead.
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

The benchmark shows the Seen trainer is functionally runnable on GPU but not performance competitive for this eval workload yet. The next performance task should add a Seen benchmark mode that reports separate load, tokenize, encode, score, and threshold timings, then batch the eval text encoding path so Vulkan work amortizes startup and dispatch overhead.
