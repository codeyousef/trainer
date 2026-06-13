# sinai-agent same-activity performance benchmark

Date: 2026-06-13

Issue context: FEL-593, FEL-600, FEL-601, FEL-602

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

Three cold-process runs were executed for the earlier rows. The current Seen trainer row uses five cold-process runs after phase timing, sparse word-row caching, final-layer adapter pooling, transient trace-buffer cleanup, and the corrected GPU B-transposed matmul shader.

| Implementation | Device Path | Status | Median Cold Process Wall | Median Peak RSS | Throughput, Cold Process | Answer Rate |
|---|---|---|---:|---:|---:|---:|
| Seen trainer CLI baseline | Vulkan GPU | Passed | 38.780s | 1,796,844 KB | 2.58 pairs/s, 5.16 row-texts/s | 0.86 |
| Seen trainer CLI after FEL-601 cache | Vulkan GPU | Passed | 31.739s | 1,672,192 KB | 3.15 pairs/s, 6.30 row-texts/s | 0.86 |
| Seen trainer CLI after adapter/load optimization | Vulkan GPU | Passed | 15.083s | 1,433,248 KB | 6.63 pairs/s, 13.26 row-texts/s | 0.86 |
| Seen trainer CLI after encoder-layer cache optimization | Vulkan GPU | Superseded by shader fix | 10.171s | 724,300 KB | 9.83 pairs/s, 19.66 row-texts/s | 0.86 |
| Seen trainer CLI after corrected GPU matmul + sparse/cache cleanup | Vulkan GPU | Passed | 11.352s | 664,592 KB | 8.81 pairs/s, 17.62 row-texts/s | 1.00 |
| Python SentenceTransformer | CPU | Passed | 4.404s | 1,242,472 KB | 22.70 pairs/s, 45.41 texts/s | 1.00 |
| Python SentenceTransformer | CUDA | Failed | Failed at model-to-CUDA | 1,314,296 KB at failure | n/a | n/a |

Performance ratios:

- FEL-601 made Seen CLI eval `1.22x` faster than the baseline, reducing median wall time by `18.2%`.
- FEL-601 reduced Seen CLI peak RSS by `6.9%`.
- The adapter/load optimization pass made Seen CLI eval `2.57x` faster than the baseline, reducing median wall time by `61.1%`.
- The adapter/load optimization pass made Seen CLI eval `2.10x` faster than the first FEL-601 cache pass.
- The encoder-layer cache optimization row is retained as historical data, but it was superseded after discovering that the older installed compiler emitted an empty `tensorMatmulBTransposed` shader body. That row should not be treated as the corrected GPU projection path.
- The corrected GPU matmul + sparse/cache cleanup path makes Seen CLI eval `3.42x` faster than the baseline, reducing median wall time by `70.7%`.
- The corrected GPU matmul + sparse/cache cleanup path reduces Seen CLI peak RSS by `63.0%` versus baseline.
- Python SentenceTransformer CPU cold-process eval remains `2.58x` faster than the corrected Seen CLI GPU/Vulkan eval for this activity.
- Corrected Seen CLI peak RSS is now `0.53x` the Python CPU path for this activity.
- Python CUDA was not comparable under the cap because all full-activity attempts failed before encoding.

## Seen CLI Internal Timing

The corrected Seen CLI now records eval phase timings from inside the Seen executable:

| Phase | Median Time |
|---|---:|
| Read eval dataset | 0.308s |
| Tokenize 173 unique texts | 0.045s |
| Encode 173 unique texts | 6.519s |
| Score and threshold 100 pairs | 0.000313s |
| Write results | 0.000084s |
| Internal eval total | 6.879s |

The cold-process wall time includes roughly `4.47s` outside the measured eval loop, mostly model/tokenizer/safetensors load, adapter resume, backend initialization, and process startup.

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

Seen trainer CLI after corrected GPU matmul + sparse/cache cleanup, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed | Internal Encode |
|---:|---:|---:|---:|---|---:|
| 1 | 11.401s | 664,336 KB | 1.00 | yes | 6.578s |
| 2 | 11.076s | 664,592 KB | 1.00 | yes | 6.287s |
| 3 | 11.352s | 666,088 KB | 1.00 | yes | 6.519s |
| 4 | 11.280s | 664,504 KB | 1.00 | yes | 6.509s |
| 5 | 11.487s | 664,952 KB | 1.00 | yes | 6.686s |

This corrected path adds:

- Seen-native eval phase timing in `eval_results.json` and `trainer eval` logs.
- Tokenized encode entry points so eval can separate tokenization from model encode timing.
- Final-layer linear adapter pooling: when the MiniLM layer adapter is attached to the final ready layer, the trainer mean-pools first and applies the linear adapter once to the pooled vector.
- Sparse word-row caching for MiniLM embedding rows touched during the process, avoiding repeated safetensors row reads without loading the full vocabulary table.
- Raw embedding and training-trace transient buffer reclamation, reducing resident memory while preserving the corrected eval answer rate.
- A Seen compiler/runtime declaration fix so the newer compiler can build trainer and emit the real `tensorMatmulBTransposed` GPU shader body.

Seen trainer CLI after encoder-layer cache optimization, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed |
|---:|---:|---:|---:|---|
| 1 | 10.171s | 724,848 KB | 0.86 | yes |
| 2 | 10.274s | 723,164 KB | 0.86 | yes |
| 3 | 10.275s | 726,272 KB | 0.86 | yes |
| 4 | 9.832s | 722,176 KB | 0.86 | yes |
| 5 | 10.033s | 724,300 KB | 0.86 | yes |

The optimized eval path now:

- Caches ready MiniLM encoder layers when `cache_minilm_tensors` is enabled, avoiding repeated safetensors layer reloads for each unique eval text.
- Uses a lean inference bundle for `mine`, `calibrate`, and `eval`, skipping training-only weight-map and parameter-registry inspection.
- Adds an eval-local flat embedding cache so duplicate query/chunk texts are encoded once.
- Scores cached normalized embeddings with direct dot product, avoiding a second normalize-per-score pass.
- Reuses low-rank adapter bottlenecks instead of recomputing them for every output dimension.
- Applies layer projection adapters directly over flat token sequences, avoiding row-vector and projected-vector allocation per token.
- Uses a rank-4 fast path for the active Sinai adapter shape.
- Applies layer-norm weight/bias in-place after GPU row normalization instead of expanding per-row weight/bias tensors and launching extra elementwise GPU passes.
- Releases feed-forward pre-activation storage after GELU.
- Avoids computing attention layer norm twice when dense MiniLM adapter deltas are active.
- Releases the temporary normalized arrays allocated by `math_utils.cosine` for call sites that still use that helper.

Correctness note: this historical row was measured before the installed compiler was refreshed with the `tensorMatmulBTransposed` shader body. The corrected row above is the valid GPU projection result.

Seen trainer CLI after adapter/load optimization, Vulkan GPU:

| Run | Wall Time | Peak RSS | Answer Rate | GPU Confirmed |
|---:|---:|---:|---:|---|
| 1 | 15.182s | 1,434,008 KB | 0.86 | yes |
| 2 | 15.083s | 1,431,832 KB | 0.86 | yes |
| 3 | 14.879s | 1,433,248 KB | 0.86 | yes |

Measured but reverted:

- Process-local Vulkan pipeline caching regressed median wall time to `16.189s` and peak RSS to `1,628,772 KB`.
- In-place GPU matmul bias addition regressed median wall time to `16.241s`.
- GPU GELU dispatch regressed median wall time to `13.729s`.
- Folding layer-norm adapter deltas directly into the affine loop regressed median wall time to `13.226s`.
- Full word-embedding-table cache regressed median wall time to `11.174s` and peak RSS to `1,441,004 KB`; sparse word row loading remains better for this 100-row eval size.

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

The Seen optimization work moved the corrected median from `38.780s` to `11.352s` and reduced peak RSS from `1,796,844 KB` to `664,592 KB`. This is a substantial improvement and now restores answer-rate parity with the Python CPU package path, but it does not close the full performance gap.

The most likely reasons are implementation-level rather than model-level:

- The Seen CLI performs this eval through the trainer runtime path, with per-unique-text encode behavior and many small GPU dispatches/readbacks.
- The Python path uses optimized HuggingFace tokenizer/runtime and batched SentenceTransformer encode.
- The Seen GPU backend is active, but this eval path is not yet optimized to batch the full 200-text workload into a single high-throughput GPU execution plan.

The answer rates are included to prove each path completed the same activity, but they should not be read as pure quality parity:

- Seen uses the Seen tokenizer/MiniLM/runtime and adapter JSON path.
- Python uses the SentenceTransformer package path with the composed `2_Dense` adapter module and `3_Normalize`.
- The corrected GPU matmul path now reaches the same answer rate as Python on this 100-row benchmark, but the paths are still not guaranteed to be numerically identical.

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

The benchmark shows the Seen trainer is functionally runnable on GPU and now much faster than the first measured implementation, but not performance competitive with Python SentenceTransformer CPU for this eval workload yet. FEL-601 removed duplicate eval encodes, redundant score normalization, training-only inference setup, repeated encoder-layer safetensors loads, avoidable adapter/layer-norm allocation, repeated sparse word-row reads, raw embedding/trace-buffer leaks, and a compiler/runtime declaration blocker that prevented the corrected shader-emitting compiler from building trainer. The remaining high-leverage task is FEL-602: batch the eval text encoding path so Vulkan work amortizes dispatch/readback overhead across the full text batch.
