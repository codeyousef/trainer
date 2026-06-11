# Sinai End-to-End Training Rewrite Guide

This document identifies the libraries, library features, and concrete algorithms needed to train the current Sinai models end to end. It is written so the stack can be reimplemented in another language without treating Python libraries as magic boxes.

## Scope

The active Sinai workspace is `sinai/`. The current preserved models are SentenceTransformer-style evidence/retrieval encoders, not standalone chat/generative models:

- `sinai/models/high_stakes/`
- `sinai/models/governance/`
- `sinai/models/commercial/`
- `sinai/models/industrial/`
- `sinai/models/institutional/`
- `sinai/models/sinai-assistant/`

The older JEPA/Phase4 and small ranking-head experiments remain useful for historical context, but the active training path is a bi-encoder evidence model:

1. Build or load `(query_text, positive_chunk_text, hard_negative_chunk_text, domain, source)` triplets.
2. Mine hard negatives with embedding search.
3. Fine-tune a MiniLM/SentenceTransformer encoder with a triplet margin loss.
4. Save the full SentenceTransformer-compatible package.
5. Calibrate per-domain cosine thresholds.
6. Evaluate answer/retrieval coverage and persist `thresholds.json` and `eval_results.json`.

## Core Data Contract

Every trainable Sinai domain dataset reduces to JSONL rows with this schema:

```json
{
  "query_text": "question or user need",
  "positive_chunk_text": "evidence chunk that should support answering",
  "hard_negative_chunk_text": "similar-looking chunk that should not answer",
  "domain": "domain_name",
  "source": "source identifier"
}
```

The current training files include:

- `sinai/synthetic_data/final/train_high_stakes_v1.jsonl` - 42,000 rows.
- `sinai/synthetic_data/final/train_governance_v2.jsonl` - 222,760 rows.
- `sinai/synthetic_data/final/train_commercial_v2.jsonl` - 42,000 rows.
- `sinai/synthetic_data/final/train_industrial_v1.jsonl` - 24,401 rows.
- `sinai/synthetic_data/final/train_institutional_v1.jsonl` - 15,780 rows.
- Older general splits: `train.jsonl`, `dev.jsonl`, `test.jsonl`, totaling about 934k rows.

## End-To-End Pipeline

### 1. Ingest Source Data

Current scripts use two source styles:

- Local HuggingFace Arrow datasets loaded from `sinai/datasets/...` via `load_from_disk`.
- Remote HuggingFace datasets loaded by name via `load_dataset`.
- Hand-written deterministic template expansion for governance/HR/compliance data.
- Existing local JSONL files for institutional datasets.

For a rewrite, the required functionality is just:

- Iterate rows from local structured datasets.
- Read JSONL.
- Read tabular/Arrow-like columns.
- Map source-specific columns into `(query, positive)` pairs.
- Filter out examples with too-short query or positive text.

### 2. Normalize To Triplet Candidates

Most scripts first build per-domain lists of `(query, positive)` pairs.

Examples:

- Commercial: `instruction -> response`, `question -> description`, chat `user -> assistant`, or `sentence -> sentence`.
- High-stakes: legal/medical/finance/safety datasets mapped to question/answer, prompt/code, problem/patch, or code/label pairs.
- Industrial: manufacturing/construction/logistics/energy Q&A, with simple template synthesis for small domains.
- Institutional: government, education, pharma, and software-doc Q&A JSONL.

The normalized positive chunk is often just the answer text. Some generators build an index chunk by concatenating query and positive text:

```text
index_chunk = query_text + " " + positive_chunk_text
```

### 3. Mine Hard Negatives

Most domain generators use the same semantic negative mining pattern:

1. Encode all candidate chunks with the base encoder.
2. Convert embeddings to `float32`.
3. L2-normalize every embedding.
4. Build an inner-product nearest-neighbor index.
5. For each query or positive chunk embedding, search the index.
6. Reject candidates from the same domain/topic boundary.
7. Pick either the first valid cross-domain hit or a random candidate from the top-k valid hits.
8. Fallback to a random chunk from another domain when search finds nothing.

With normalized vectors, inner product is cosine similarity:

```text
cosine(a, b) = dot(a / ||a||, b / ||b||)
```

The current FAISS-backed exact index is `IndexFlatIP`, so a correct rewrite can start with brute-force matrix multiplication:

```text
scores = query_matrix * chunk_matrix^T
top_indices = argsort(scores, descending=true)[:k]
```

Approximate nearest-neighbor search is an optimization, not a semantic requirement.

### 4. Optional Quality Filtering

The older Model 2 generator applies a quality filter:

```text
q = normalize(encode(query_text))
p = normalize(encode(positive_chunk_text))
n = normalize(encode(hard_negative_chunk_text))
sim_pos = dot(q, p)
sim_neg = dot(q, n)
keep if sim_pos - sim_neg >= 0.05
```

Some newer generators skip this because their negatives are already mined or cross-domain. If rewriting the full pipeline, keep the quality-filter primitive because it is cheap and useful for new sources.

### 5. Train The Encoder

The active per-domain training scripts fine-tune the encoder itself:

- Base model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Batch size: `64`.
- Epochs: `3`.
- Learning rate: `2e-5`.
- Weight decay: `0.01`.
- Margin: `0.2`.
- Gradient clipping: global norm `1.0`.
- Optimizer: AdamW.
- Shuffle examples each epoch.
- Drop the final partial batch.

Training loop semantics:

```text
for each batch of triplets:
    q_tokens = tokenize(query_texts)
    p_tokens = tokenize(positive_chunk_texts)
    n_tokens = tokenize(hard_negative_chunk_texts)

    q_emb = transformer(q_tokens).last_hidden_state[:, 0]
    p_emb = transformer(p_tokens).last_hidden_state[:, 0]
    n_emb = transformer(n_tokens).last_hidden_state[:, 0]

    sim_qp = cosine_similarity(q_emb, p_emb)
    sim_qn = cosine_similarity(q_emb, n_emb)
    loss = mean(max(0, sim_qn - sim_qp + 0.2))

    backpropagate(loss)
    clip_grad_norm(parameters, 1.0)
    adamw_step(parameters)
```

Important fidelity note: these scripts train on the first-token vector (`last_hidden_state[:, 0]`), while the saved SentenceTransformer package advertises mean pooling for inference. To reproduce the current code exactly, preserve that mismatch. To make a cleaner rewrite, train and infer with the same pooling function.

### 6. Save The Model Package

The full model directory must be preserved. `model.safetensors` alone is not enough.

Required artifact types:

- Transformer weights: `model.safetensors`.
- Transformer config: `config.json`.
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`, and special-token metadata when present.
- SentenceTransformer module graph: `modules.json`.
- Pooling config: `1_Pooling/config.json`.
- SentenceTransformer config: `sentence_bert_config.json`.
- Sinai metadata: `training_config.json`, `thresholds.json`, `eval_results.json`, and assistant-specific manifest/config files where present.

### 7. Calibrate Thresholds

The eval scripts turn embeddings into answer/abstention thresholds.

Common current pattern:

1. Load triplets grouped by `domain`.
2. Shuffle deterministically with seed `42`.
3. Split each domain into 80% train/calibration and 20% test.
4. Encode calibration queries and positives.
5. Compute `Q->P` cosine similarities.
6. Set per-domain threshold to a low percentile of positive scores.
7. Evaluate on held-out positives by counting `sim(query, positive) >= threshold`.

The main active domain scripts use the 5th percentile. Some governance scripts use the 10th percentile.

```text
threshold[domain] = percentile(pos_similarities, 5)
answered = count(test_qp_similarity >= threshold[domain])
answer_rate = answered / total
```

The typical pass gates are:

- Per-domain answer rate: `>= 90%`.
- Overall answer rate: `>= 88%`.

The Sinai Assistant webdev gate reports a calibrated threshold of about `0.40000000000000013`.

## Libraries And What To Reimplement

### PyTorch

Used by training scripts as the tensor/autograd/runtime layer.

Required features:

- Dense tensor allocation on CPU/GPU.
- Automatic differentiation through transformer weights.
- Matrix multiplication, broadcasting, slicing, stacking, padding.
- Neural network module/container abstraction.
- `Linear`, `ReLU`, `Dropout` for older ranking heads.
- Cosine similarity.
- Vector normalization.
- ReLU margin loss.
- Cross-entropy loss for older InfoNCE/gap-head experiments.
- AdamW optimizer.
- Gradient zeroing, backward pass, gradient clipping.
- Optional cosine annealing scheduler for older head experiments.
- Save/load weights.
- Dataset/DataLoader equivalent: batching, shuffling, multi-worker loading, optional drop-last.

AdamW update, if implementing directly:

```text
g = gradient(parameter)
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g * g)
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
parameter = parameter - lr * weight_decay * parameter
parameter = parameter - lr * m_hat / (sqrt(v_hat) + eps)
```

### sentence-transformers

Used as the high-level encoder wrapper.

Required features:

- Load a Transformer encoder plus tokenizer by model id or local directory.
- Batch tokenize texts.
- Run transformer forward pass.
- Produce sentence embeddings.
- Optionally normalize embeddings.
- Save a directory that records transformer, tokenizer, module graph, pooling config, and weights.
- Report embedding dimension.

The active model module graph is:

```json
[
  {"idx": 0, "type": "sentence_transformers.base.modules.transformer.Transformer"},
  {"idx": 1, "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"}
]
```

The pooling config is mean pooling with 384 output dimensions:

```text
embedding = sum(token_embeddings * attention_mask) / max(sum(attention_mask), 1e-9)
```

### transformers

Used underneath SentenceTransformers and in assistant/eval harnesses.

Required training-time features:

- BERT/MiniLM encoder forward pass returning `last_hidden_state`.
- Fast tokenizer behavior compatible with the saved tokenizer.
- Attention masks.
- Padding and truncation.
- Special tokens.
- Save/load model configuration and weights.

Current base encoder architecture details from `config.json`:

- `model_type`: `bert`.
- Hidden size: `384`.
- Layers: `12`.
- Attention heads: `12`.
- Intermediate size: `1536`.
- Activation: `gelu`.
- Max positions: `512`.
- Vocabulary size: `250037`.
- Dropout: `0.1`.
- LayerNorm epsilon: `1e-12`.
- Position embeddings: absolute.

The tokenizer config uses:

- `PreTrainedTokenizerFast`.
- Lowercasing enabled.
- Right padding.
- Max/model length in artifact: `128`.
- `<s>` as CLS/BOS, `</s>` as SEP/EOS, `<pad>`, `<unk>`, `<mask>`.

Assistant/eval harnesses also import causal-LM loaders, generation configs, and pipelines. These are not required for training the active SentenceTransformer domain models.

### HuggingFace Datasets

Used for ingestion, not model math.

Required features:

- Load local Arrow-style datasets from disk.
- Load remote named datasets and configs.
- Streaming iteration for very large datasets.
- Access rows by column name.
- Split selection by name, usually `train`.
- Optional trust-remote-code flag handling.

For a rewrite, this can be replaced with any row iterator that yields the source columns expected by the generator scripts.

### FAISS

Used for hard-negative mining and retrieval indexing.

Required features:

- L2-normalize a matrix of float32 embeddings.
- Exact inner-product search over normalized vectors.
- Add vectors to an index.
- Search top-k for one or many query vectors.
- Optional read/write of prebuilt indexes.

Minimum replacement:

```text
normalize rows of chunk_embeddings
normalize query_embedding
scores = chunk_embeddings dot query_embedding
return top-k scores and indices
```

FAISS is only needed for speed at larger corpus sizes.

### NumPy

Used as the array/statistics glue.

Required features:

- Float32 arrays.
- Stack/vstack.
- L2 norms.
- Dot products.
- Random seed, random sampling, random permutation.
- Percentiles.
- Mean/std/min/max.
- Boolean masks and counts.

### scikit-learn

Used lightly in eval scripts for `train_test_split`. It is not a core dependency.

Replacement:

```text
shuffle indices with seed 42
test_count = floor(len(items) * 0.2)
test = items[:test_count]
train = items[test_count:]
```

### tqdm

Progress bars only. No semantic dependency.

### PyYAML

Used by eval-suite and research helpers. Not required for the core domain-model training loop unless rewriting the assistant capability harnesses too.

### requests, duckduckgo_search, urllib

Used by web-evidence and freshness harnesses. Not required for training the active domain encoders.

## Older Ranking Heads

Some scripts train heads over frozen encoder embeddings. They are not the active preserved model artifact style, but they matter if rewriting the whole research path.

### MarginHead

Used in older Sinai v2/webdev experiments:

```text
q_proj = Linear(384, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 128)
d_proj = Linear(384, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 128)
score(q, docs) = normalize(q_proj(q)) dot normalize(d_proj(docs))^T
loss = mean(max_over_negatives(max(0, margin - pos_score + neg_score)))
```

### GapHead / InfoNCE Head

Used in governance/head experiments:

```text
proj = Linear(384, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 64)
q = normalize(proj(query))
p = normalize(proj(positive))
n = normalize(proj(negative))
logits = concat(dot(q, p), dot(q, n_1), ..., dot(q, n_k)) / temperature
labels = 0
loss = cross_entropy(logits, labels)
```

The project later moved back toward raw cosine and full SentenceTransformer directories because the heads were brittle or less useful for cross-topic confidence.

## Minimal Rewrite Checklist

To train a Sinai-equivalent model in another language, implement these pieces in order:

1. JSONL reader/writer for the triplet schema.
2. Source row adapters that produce `(query, positive)` pairs per domain.
3. A BERT/MiniLM-compatible tokenizer and encoder forward pass.
4. Sentence embedding generation with mean pooling and optional L2 normalization.
5. Exact or approximate normalized inner-product top-k search.
6. Hard-negative selection with domain/topic exclusion and random fallback.
7. Triplet margin fine-tuning with AdamW and gradient clipping.
8. Model package serialization for weights, tokenizer, config, pooling, thresholds, and eval metadata.
9. Per-domain calibration using percentile thresholds.
10. Eval reporting with per-domain and overall answer rates.

## Reproduction Notes

- Use seed `42` for deterministic shuffles/sampling when matching current scripts.
- Store embeddings as `float32`.
- Normalize embeddings before dot-product search and threshold calibration unless intentionally matching a path that relies on `SentenceTransformer.encode` defaults.
- Preserve full model directories. The tokenizer and pooling metadata are part of the model.
- The active training scripts contain absolute paths under `/mnt/Storage/Projects/catbelly_studio/sinai`; a rewrite should pass workspace paths through configuration instead.
- If the goal is exact compatibility, preserve the current CLS-vector training and mean-pool inference mismatch. If the goal is a cleaner second implementation, align the pooling function across training, evaluation, and inference.
