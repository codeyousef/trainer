# Seen Trainer

Seen-native Sinai trainer for MiniLM/SentenceTransformer-style embedding models.

This project is implemented in Seen. It consumes local JSONL/source exports and local model artifacts, trains with mean pooling for training/evaluation/inference, can dispatch tensor kernels through Seen's Vulkan GPU runtime, and emits a SentenceTransformer-compatible package with Seen manifests.

## CLI

```sh
trainer mine --config config.json
trainer train --config config.json
trainer calibrate --config config.json
trainer eval --config config.json
trainer package --config config.json
trainer run-all --config config.json
```

If `--config` is omitted, the CLI reads `config/example.config.json`.

## Data

Training JSONL rows use this triplet schema:

```json
{
  "query_text": "question text",
  "positive_chunk_text": "matching answer chunk",
  "hard_negative_chunk_text": "hard negative chunk",
  "domain": "domain name",
  "source": "source id"
}
```

Source adapters can also normalize CSV/TSV/JSONL rows into `(query, positive)` pairs before mining.
Hard-negative mining precomputes normalized positive embeddings once, then uses Seen's
`tensorTopKInnerProduct` GPU kernel when `backend` is `gpu`; if dispatch is unavailable it
falls back to the same scalar top-k and domain/source exclusion semantics.
Loaded MiniLM forward passes also thread the configured backend into Q/K/V, attention-output,
intermediate, and output dense projections through Seen's `Tensor.matmul` dispatch while keeping
the scalar path as the correctness reference.
MiniLM backward tail gradients now thread the configured backend into dense projection input
gradients for FFN/output/attention paths, reusing `Tensor.matmul` through the dense-gradient
helpers while retaining scalar fallbacks for shape diagnostics and tests.

## Model Outputs

The package step writes SentenceTransformer-compatible files plus Seen manifests. When local MiniLM safetensors are loaded, training updates sparse embedding rows, embedding LayerNorm, and all ready encoder layer surfaces. These trainable base surfaces apply AdamW to the loaded safetensors base value plus the Seen delta, then persist the resulting delta so it can be materialized into `seen_trained_base_model.safetensors`.
Packages also include `minilm_parameter_registry.json`, a Seen-native object graph of loaded MiniLM safetensors tensors as `Parameter` buffers with value, gradient, and Adam moment arrays. When MiniLM slots are loaded, sparse embedding, embedding LayerNorm, and encoder layer-surface AdamW updates run through those registry buffers and sync the resulting deltas back into the package-compatible delta artifacts.

Safetensors metadata is inspected through header reads, tensor loads use byte-range reads, and materialization patches tensor-sized slices or sparse rows instead of loading the whole model file into a Seen byte array.

The direct update manifest is written to:

```text
<output_model_dir>/seen_base_weight_update_manifest.json
```

It reports whether the full MiniLM encoder surface was materialized for the loaded model.

## Capped Verification

Always run Seen builds/checks/tests under a memory cap:

```sh
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>8388608) v=8388608; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen check src/main.seen
SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen compile src/main.seen target/trainer --fast --no-fork --emit-glsl --no-cache --jobs=1 --opt-jobs=1
```

Test sources can be checked and run from the test project:

```sh
cd tests
CAP_KB=$(awk '/MemAvailable/ { v=int($2/2); if (v>8388608) v=8388608; print v }' /proc/meminfo)
ulimit -v "$CAP_KB"
for test in test_*.seen; do SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen check "$test" || exit 1; done
for test in test_*.seen; do
  name=${test%.seen}
  SEEN_JOBS=1 SEEN_OPT_JOBS=1 seen compile "$test" "../target/$name" --fast --no-fork --emit-glsl --no-cache --jobs=1 --opt-jobs=1 || exit 1
  "../target/$name" || exit 1
done
```
