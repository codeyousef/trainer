#!/usr/bin/env python3
"""Run the sinai-agent Seen Vulkan vs Python CUDA benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "sinai-agent.v0.1.adapter-full.resume.json"
DEFAULT_PYTHON = Path("/mnt/Storage/Projects/catbelly_studio/.venv/bin/python")
DEFAULT_MODEL = Path("/mnt/Storage/Projects/catbelly_studio/sinai/models/sinai-agent")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seconds(value: float) -> str:
    return f"{value:.3f}s"


def rate(unique_texts: float, encode_s: float) -> str:
    if encode_s <= 0:
        return "n/a"
    return f"{unique_texts / encode_s:.1f}/s"


def kb_to_mb(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / 1024.0:.1f} MB"


def mb_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} MB"


def run_process(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[float, int, str, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    return time.perf_counter() - start, proc.returncode, proc.stdout, proc.stderr


def make_seen_config(source: Path, target: Path) -> dict:
    cfg = read_json(source)
    bench_dir = target.parent
    cfg["eval_results_path"] = str(bench_dir / "seen-eval-results.json")
    cfg["backend"] = "gpu"
    write_json(target, cfg)
    return cfg


def run_python_cold(args, cfg: dict, out_dir: Path, env: dict[str, str]) -> list[dict]:
    results = []
    script = ROOT / "benchmarks" / "python_cuda_eval.py"
    for index in range(args.cold_runs):
        result_path = out_dir / f"python-cold-{index + 1}.json"
        cmd = [
            str(args.python),
            str(script),
            "--model-dir",
            str(args.python_model_dir),
            "--test-jsonl",
            cfg["test_jsonl_path"],
            "--thresholds",
            cfg["thresholds_path"],
            "--result",
            str(result_path),
            "--runs",
            "1",
            "--warmup",
            "0",
            "--batch-size",
            str(args.batch_size),
        ]
        wall_s, code, stdout, stderr = run_process(cmd, ROOT, env)
        if code != 0:
            raise RuntimeError(f"Python CUDA cold run {index + 1} failed:\n{stdout}\n{stderr}")
        data = read_json(result_path)
        data["process_wall_s"] = wall_s
        results.append(data)
    return results


def run_seen_cold(args, config_path: Path, out_dir: Path, env: dict[str, str]) -> list[dict]:
    results = []
    for index in range(args.cold_runs):
        result_path = out_dir / f"seen-cold-{index + 1}.json"
        cmd = [
            str(args.trainer),
            "bench-eval",
            "--config",
            str(config_path),
            "--runs",
            "1",
            "--warmup",
            "0",
            "--result",
            str(result_path),
        ]
        wall_s, code, stdout, stderr = run_process(cmd, ROOT, env)
        if code != 0:
            raise RuntimeError(f"Seen cold run {index + 1} failed:\n{stdout}\n{stderr}")
        data = read_json(result_path)
        data["process_wall_s"] = wall_s
        results.append(data)
    return results


def run_python_steady(args, cfg: dict, out_dir: Path, env: dict[str, str]) -> dict:
    result_path = out_dir / "python-steady.json"
    cmd = [
        str(args.python),
        str(ROOT / "benchmarks" / "python_cuda_eval.py"),
        "--model-dir",
        str(args.python_model_dir),
        "--test-jsonl",
        cfg["test_jsonl_path"],
        "--thresholds",
        cfg["thresholds_path"],
        "--result",
        str(result_path),
        "--runs",
        str(args.steady_runs),
        "--warmup",
        str(args.warmup),
        "--batch-size",
        str(args.batch_size),
    ]
    wall_s, code, stdout, stderr = run_process(cmd, ROOT, env)
    if code != 0:
        raise RuntimeError(f"Python CUDA steady run failed:\n{stdout}\n{stderr}")
    data = read_json(result_path)
    data["process_wall_s"] = wall_s
    return data


def run_seen_steady(args, config_path: Path, out_dir: Path, env: dict[str, str]) -> dict:
    result_path = out_dir / "seen-steady.json"
    cmd = [
        str(args.trainer),
        "bench-eval",
        "--config",
        str(config_path),
        "--runs",
        str(args.steady_runs),
        "--warmup",
        str(args.warmup),
        "--result",
        str(result_path),
    ]
    wall_s, code, stdout, stderr = run_process(cmd, ROOT, env)
    if code != 0:
        raise RuntimeError(f"Seen steady run failed:\n{stdout}\n{stderr}")
    data = read_json(result_path)
    data["process_wall_s"] = wall_s
    return data


def summarize_python_cold(rows: list[dict]) -> dict:
    return {
        "cold_process_wall_s": median([row["process_wall_s"] for row in rows]),
        "cold_internal_wall_s": median([row["cold_wall_s"] for row in rows]),
        "load_s": median([row["load_s"] for row in rows]),
        "encode_s": median([row["runs"][0]["encode_s"] for row in rows]),
        "answer_rate": median([row["runs"][0]["answer_rate"] for row in rows]),
        "rss_kb": median([row["rss_kb"] for row in rows]),
        "cuda_peak_allocated_mb": median([row["cuda_peak_allocated_mb"] for row in rows]),
    }


def summarize_seen_cold(rows: list[dict]) -> dict:
    return {
        "cold_process_wall_s": median([row["process_wall_s"] for row in rows]),
        "load_s": median([row["load_ms"] / 1000.0 for row in rows]),
        "encode_s": median([row["runs"][0]["encode_ms"] / 1000.0 for row in rows]),
        "answer_rate": median([row["runs"][0]["answer_rate"] for row in rows]),
        "unique_texts": median([row["runs"][0]["unique_texts"] for row in rows]),
    }


def summarize_python_steady(row: dict) -> dict:
    return {
        "process_wall_s": row["process_wall_s"],
        "load_s": row["load_s"],
        "encode_s": median([item["encode_s"] for item in row["runs"]]),
        "score_s": median([item["score_s"] for item in row["runs"]]),
        "answer_rate": median([item["answer_rate"] for item in row["runs"]]),
        "rss_kb": row["rss_kb"],
        "cuda_peak_allocated_mb": row["cuda_peak_allocated_mb"],
    }


def summarize_seen_steady(row: dict) -> dict:
    return {
        "process_wall_s": row["process_wall_s"],
        "load_s": row["load_ms"] / 1000.0,
        "encode_s": median([item["encode_ms"] / 1000.0 for item in row["runs"]]),
        "score_s": median([item["score_threshold_ms"] / 1000.0 for item in row["runs"]]),
        "answer_rate": median([item["answer_rate"] for item in row["runs"]]),
        "unique_texts": median([item["unique_texts"] for item in row["runs"]]),
    }


def write_markdown(path: Path, summary: dict) -> None:
    py_cold = summary["python_cuda_cold_summary"]
    seen_cold = summary["seen_vulkan_cold_summary"]
    py_steady = summary["python_cuda_steady_summary"]
    seen_steady = summary["seen_vulkan_steady_summary"]
    cold_ratio = py_cold["cold_process_wall_s"] / seen_cold["cold_process_wall_s"] if seen_cold["cold_process_wall_s"] else 0.0
    encode_ratio = py_steady["encode_s"] / seen_steady["encode_s"] if seen_steady["encode_s"] else 0.0
    passed = summary["seen_beats_python"]
    unique_texts = seen_steady.get("unique_texts", seen_cold.get("unique_texts", 173))
    python_unique_texts = summary["raw"]["python_steady"].get("unique_texts", unique_texts)
    config_path = summary.get("config_path", "config/sinai-agent.v0.1.adapter-full.resume.json")
    report_cmd = summary.get(
        "benchmark_command",
        "benchmarks/run_gpu_compare.py --cold-runs 7 --steady-runs 7 --warmup 2 --out-dir target/bench/gpu-compare-final --report reports/sinai-agent-gpu-vs-gpu-benchmark-2026-06-13.md",
    )
    text = f"""# sinai-agent GPU-vs-GPU benchmark

Date: 2026-06-13

Activity: load the local sinai-agent package, read the 100-row Agentic-EI test set, deduplicate query/positive texts, encode {int(unique_texts)} unique texts on GPU, score 100 pairs with the packaged thresholds, and report answer rate.

Config: `{config_path}`

Benchmark command, intentionally without `ulimit -v`:

```bash
{report_cmd}
```

Seen validation/build commands stayed memory-capped:

```bash
CAP_KB=$(awk '/MemAvailable/ {{ v=int($2/2); if (v>16777216) v=16777216; if (v<1048576) v=1048576; print v }}' /proc/meminfo)
ulimit -v "$CAP_KB"
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen check src/main.seen
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen compile src/main.seen target/trainer --fast --no-fork --emit-glsl --no-cache --jobs=1 --opt-jobs=1
```

| Implementation | GPU Path | Median Cold Process | Median Load | Median Steady Encode | Throughput | Median Score | RSS | GPU Memory | Answer Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen trainer | Vulkan | {seconds(seen_cold['cold_process_wall_s'])} | {seconds(seen_cold['load_s'])} | {seconds(seen_steady['encode_s'])} | {rate(float(unique_texts), seen_steady['encode_s'])} | {seconds(seen_steady['score_s'])} | n/a | n/a | {seen_steady['answer_rate']:.2f} |
| Python SentenceTransformer | CUDA | {seconds(py_cold['cold_process_wall_s'])} | {seconds(py_cold['load_s'])} | {seconds(py_steady['encode_s'])} | {rate(float(python_unique_texts), py_steady['encode_s'])} | {seconds(py_steady['score_s'])} | {kb_to_mb(py_steady.get('rss_kb'))} | {mb_value(py_steady.get('cuda_peak_allocated_mb'))} | {py_steady['answer_rate']:.2f} |

Seen/Python ratios:

- Cold process speed ratio: `{cold_ratio:.3f}x` where values above `1.0x` mean Seen is faster.
- Steady encode speed ratio: `{encode_ratio:.3f}x` where values above `1.0x` mean Seen is faster.
- Acceptance with 5% margin: `{'passed' if passed else 'failed'}`.

Correctness:

- Python CUDA answer rate: `{py_steady['answer_rate']:.2f}`.
- Seen Vulkan answer rate: `{seen_steady['answer_rate']:.2f}`.
- Both paths used the same test JSONL and thresholds from the temporary benchmark config.

Current blocker:

- Seen is still using array-oriented GPU calls with host readback between kernels and naive Vulkan matmul/attention kernels. The fused hot-path kernels added for this run improved correctness-preserving execution, but the runtime still needs FEL-602 device-resident graph dispatch, batched descriptor/command reuse, and optimized tiled matmul/attention before it can plausibly beat warmed CUDA.
- Seen RSS and Vulkan VRAM are not yet exposed by the runtime harness. Python RSS and CUDA peak allocation are captured; Seen-side process/GPU memory counters should be added under FEL-602 instead of guessed in this report.

Raw JSON: `{summary['result_json']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trainer", type=Path, default=ROOT / "target" / "trainer")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--python-model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "target" / "bench" / "gpu-compare")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "sinai-agent-gpu-vs-gpu-benchmark-2026-06-13.md")
    parser.add_argument("--cold-runs", type=int, default=7)
    parser.add_argument("--steady-runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()

    if args.report_only:
        summary = read_json(args.report_only)
        summary.setdefault("result_json", str(args.report_only))
        summary.setdefault("config_path", str(args.config))
        summary.setdefault(
            "benchmark_command",
            "benchmarks/run_gpu_compare.py --cold-runs 7 --steady-runs 7 --warmup 2 --out-dir target/bench/gpu-compare-final --report reports/sinai-agent-gpu-vs-gpu-benchmark-2026-06-13.md",
        )
        write_markdown(args.report, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary.get("seen_beats_python") else 3

    env = os.environ.copy()
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bench_config = args.out_dir / "seen-bench-config.json"
    cfg = make_seen_config(args.config, bench_config)

    py_cold = run_python_cold(args, cfg, args.out_dir, env)
    seen_cold = run_seen_cold(args, bench_config, args.out_dir, env)
    py_steady = run_python_steady(args, cfg, args.out_dir, env)
    seen_steady = run_seen_steady(args, bench_config, args.out_dir, env)

    py_cold_summary = summarize_python_cold(py_cold)
    seen_cold_summary = summarize_seen_cold(seen_cold)
    py_steady_summary = summarize_python_steady(py_steady)
    seen_steady_summary = summarize_seen_steady(seen_steady)
    seen_beats = (
        seen_cold_summary["cold_process_wall_s"] <= py_cold_summary["cold_process_wall_s"] * 0.95
        and seen_steady_summary["encode_s"] <= py_steady_summary["encode_s"] * 0.95
        and seen_steady_summary["answer_rate"] >= 1.0
    )

    result_json = args.out_dir / "summary.json"
    summary = {
        "result_json": str(result_json),
        "config_path": str(args.config),
        "benchmark_command": " ".join(os.sys.argv),
        "python_cuda_cold_summary": py_cold_summary,
        "seen_vulkan_cold_summary": seen_cold_summary,
        "python_cuda_steady_summary": py_steady_summary,
        "seen_vulkan_steady_summary": seen_steady_summary,
        "seen_beats_python": seen_beats,
        "raw": {
            "python_cold": py_cold,
            "seen_cold": seen_cold,
            "python_steady": py_steady,
            "seen_steady": seen_steady,
        },
    }
    write_json(result_json, summary)
    write_markdown(args.report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if seen_beats else 3


if __name__ == "__main__":
    raise SystemExit(main())
