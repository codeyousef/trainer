# sinai-agent GPU-vs-GPU benchmark

Date: 2026-06-13

Activity: load the local sinai-agent package, read the 100-row Agentic-EI test set, deduplicate query/positive texts, encode 173 unique texts on GPU, score 100 pairs with the packaged thresholds, and report answer rate.

Config: `/mnt/Storage/Projects/seen/trainer/config/sinai-agent.v0.1.adapter-full.resume.json`

Benchmark command, intentionally without `ulimit -v`:

```bash
benchmarks/run_gpu_compare.py --cold-runs 7 --steady-runs 7 --warmup 2 --out-dir target/bench/gpu-compare-final --report reports/sinai-agent-gpu-vs-gpu-benchmark-2026-06-13.md
```

Seen validation/build commands stayed memory-capped:

```bash
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen check src/main.seen
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen compile src/main.seen target/trainer --fast --no-fork --emit-glsl --no-cache --jobs=1 --opt-jobs=1
```

| Implementation | GPU Path | Median Cold Process | Median Load | Median Steady Encode | Throughput | Median Score | RSS | GPU Memory | Answer Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen trainer | Vulkan | 5.492s | 4.160s | 1.148s | 150.7/s | 0.000s | n/a | n/a | 1.00 |
| Python SentenceTransformer | CUDA | 4.223s | 0.715s | 0.036s | 4823.5/s | 0.003s | 1638.2 MB | 450.4 MB | 1.00 |

Seen/Python ratios:

- Cold process speed ratio: `0.769x` where values above `1.0x` mean Seen is faster.
- Steady encode speed ratio: `0.031x` where values above `1.0x` mean Seen is faster.
- Acceptance with 5% margin: `failed`.

Correctness:

- Python CUDA answer rate: `1.00`.
- Seen Vulkan answer rate: `1.00`.
- Both paths used the same test JSONL and thresholds from the temporary benchmark config.

Current blocker:

- Seen is still using array-oriented GPU calls with host readback between kernels and naive Vulkan matmul/attention kernels. The fused hot-path kernels added for this run improved correctness-preserving execution, but the runtime still needs FEL-602 device-resident graph dispatch, batched descriptor/command reuse, and optimized tiled matmul/attention before it can plausibly beat warmed CUDA.
- Seen RSS and Vulkan VRAM are not yet exposed by the runtime harness. Python RSS and CUDA peak allocation are captured; Seen-side process/GPU memory counters should be added under FEL-602 instead of guessed in this report.

Raw JSON: `target/bench/gpu-compare-final/summary.json`
