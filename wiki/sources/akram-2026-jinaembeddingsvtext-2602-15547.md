---
type: source
subtype: paper
title: "jina-embeddings-v5-text: Task-Targeted Embedding Distillation"
slug: akram-2026-jinaembeddingsvtext-2602-15547
date: 2026-04-20
language: en
tags: [embedding, distillation, retrieval, multilingual, lora]
processed: true
raw_file: raw/papers/akram-2026-jinaembeddingsvtext-2602-15547/paper.pdf
raw_md: raw/papers/akram-2026-jinaembeddingsvtext-2602-15547/paper.md
bibtex_file: raw/papers/akram-2026-jinaembeddingsvtext-2602-15547/paper.bib
possibly_outdated: false
authors:
  - Mohammad Kalim Akram
  - Saba Sturua
  - Nastia Havriushenko
  - Quentin Herreros
  - Michael Günther
  - Maximilian Werk
  - Han Xiao
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2602.15547
doi: 10.48550/arXiv.2602.15547
url: http://arxiv.org/abs/2602.15547
citation_key: akram2026jinaembeddingsvtext
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper presents two compact multilingual text embedding models, jina-embeddings-v5-text-small (`677M` parameters, `1024` dimensions) and jina-embeddings-v5-text-nano (`239M`, `768` dimensions), trained with a two-stage recipe that first distills Qwen3-Embedding-4B into smaller backbones and then learns task-specific LoRA adapters for retrieval, semantic textual similarity, clustering, and classification. The models use last-token pooling, query/document prefixes for asymmetric retrieval, Matryoshka representation learning for truncation, and long-context support up to `32k` tokens. Across MMTEB, English MTEB, BEIR, RTEB, and LongEmbed, the models are competitive with or better than similarly sized multilingual baselines. Ablations show that embedding-space distillation plus task-specific contrastive training is stronger than either ingredient alone, and GOR regularization improves robustness under binary quantization.

## Problem & Motivation

General-purpose embedding models are widely used in retrieval, clustering, classification, and semantic similarity, but compact models often lose too much quality when trained only with contrastive learning or only with distillation. The paper targets a practical gap: how to build small multilingual embedding models that keep strong retrieval quality, remain usable across multiple downstream task types, and still support long inputs, truncation, and quantization. The authors argue that distillation should provide the general semantic geometry, while task-specific adapters should resolve objective conflicts that arise when one embedding space is optimized for very different uses.

## Method

- **Backbones and outputs**: jina-embeddings-v5-text-small is built from Qwen3-0.6B-Base (`600M` base params, `1024`-dim embeddings, `theta = 1M` at inference), and jina-embeddings-v5-text-nano from EuroBERT-210M (`210M`, `768` dims, `theta = 250K` at inference). Both use last-token pooling and support up to `32k` tokens.
- **Task-specific adapters**: the model keeps four separate LoRA adapters for retrieval, semantic similarity, clustering, and classification; all adapters use rank `32` and alpha `32`, and the user selects the adapter at inference time.
- **Stage 1 distillation**: the student is trained on query-document pairs from `300+` datasets spanning `30+` languages for `50,000` steps. Small uses `8 x 512` batches, nano uses `8 x 1024`, both at sequence length `512` and learning rate `1e-4`.
- **Distillation objective**: student embeddings `z^S` are projected into teacher space with `psi(z) = Wz + b`, and the loss minimizes cosine distance to Qwen3-Embedding-4B teacher embeddings: `L_distill = sum_i sum_{z in {x,y}} [1 - phi(psi(z_i^S), z_i^T)]`.
- **Long-context extension**: the small model gets an extra `6,500`-step stage on `1,000-4096` token multilingual pairs, with sequence length `4096`, `2 x 64` batches, learning rate `1e-4`, and a reduced RoPE scale `theta = 500K` during training to improve long-context extrapolation.
- **Retrieval adapter**: asymmetric retrieval prepends `Query:` or `Document:` and optimizes `L_retrieval = lambda_NCE L_NCE + lambda_D L_distill + lambda_S L_GOR`, where InfoNCE uses hard negatives and a learnable temperature `tau`, and GOR penalizes squared inner products among non-matching query and positive embeddings.
- **Retrieval hyperparameters**: both retrieval adapters train for `8,000` steps at learning rate `2e-5`; small uses `2 x (256 / 64)` dynamic batches at `384 / 4096` tokens, while nano uses `2 x (384 / 96)` at `384 / 4096`.
- **STS adapter**: symmetric matching uses only the `Document:` prefix and a CoSENT ranking loss when graded scores exist, otherwise `L_sts = lambda_NCE L_NCE + lambda_D L_distill` with ratio `lambda_NCE : lambda_D = 1 : 2`; both models train for `20,000` steps at learning rate `5e-5`, sequence length `384`, and temperatures `tau = 0.02`, `tau' = 0.05`.
- **Clustering and classification adapters**: clustering re-runs distillation with a clustering instruction ("Identify the topic or theme of the given document:") for `20,000` steps at learning rate `1e-5`; classification converts labeled data into anchor-positive-seven-negative triplets and optimizes bidirectional InfoNCE plus relational distillation `L_r`, training for `30,000` steps at learning rate `4e-4`.
- **Efficiency features**: Matryoshka representation learning enables post-hoc embedding truncation, and the spread-out regularizer is explicitly designed to make the embedding space more uniform and more robust to ANN retrieval and binary quantization.

## Key Results

- On MMTEB v2, jina-embeddings-v5-text-small reaches `67.0` average task score and `58.9` average task-type score, while jina-embeddings-v5-text-nano reaches `65.5` and `57.7`; both outperform similarly sized jina-v3, snowflake-l-v2, and multilingual-e5-large-instruct on the global average.
- On English MTEB v2, the small model scores `71.7` average tasks, `60.1` retrieval, and `88.1` STS; the nano model scores `71.0`, `58.8`, and `88.3`, making it especially strong among models under `0.5B` parameters.
- On the aggregated retrieval view (MTEB-M, MTEB-E, RTEB, BEIR, LongEmbed), the small model reaches `63.28` average and beats Qwen3-0.6B on `3/5` benchmarks; the nano model reaches `61.43` average, `56.06` on BEIR, and `58.80` on MTEB-E while being the smallest evaluated model.
- Retrieval-loss ablations show that the full objective `L_NCE + L_distill + L_GOR` is best: `64.50` on MTEB retrieval and `66.45` on public RTEB, compared with `63.16` / `64.37` for distillation alone.
- GOR contributes modestly at BF16 (`64.50` vs. `64.21` on MTEB; `66.45` vs. `66.16` on RTEB) but materially improves binary quantization robustness, reducing the MTEB drop from `-3.08` to `-1.90` and the RTEB drop from `-3.92` to `-2.51`.
- Truncation experiments show the embeddings remain usable under dimensionality reduction, but retrieval quality drops sharply once the embedding size falls below `256` dimensions.

## Limitations

- The teacher model Qwen3-4B still has a clear margin over both students, reaching `69.5` MMTEB average tasks and `67.95` aggregated retrieval average versus `67.0` and `63.28` for the small student.
- The small model does not dominate every benchmark: Qwen3-0.6B remains stronger on English MTEB retrieval (`61.83` vs. `60.07`) and LongEmbed (`72.20` vs. `66.39`).
- Clustering is not best-in-class among compact multilingual models; KaLM-mini-v2.5 reports slightly higher clustering scores on both MMTEB (`53.8` vs. `53.4`) and English MTEB (`58.1` vs. `54.7`).
- Long-context enhancement is described only for the small model, so the long-document story is less complete for the nano variant.
- The paper reports aggregate benchmark numbers and some partially self-evaluated leaderboard entries, but gives limited detail on the exact composition and filtering of the `300+` distillation datasets.

## Concepts Extracted

- [[sentence-embedding]]
- [[knowledge-distillation]]
- [[contrastive-loss]]
- [[low-rank-adaptation]]
- [[matryoshka-representation-learning]]
- [[dense-retrieval]]
- [[semantic-textual-similarity]]
- [[long-context-training]]
- [[multilingual-pretraining]]
- [[binary-quantization]]

## Entities Extracted

- [[mohammad-kalim-akram]]
- [[saba-sturua]]
- [[nastia-havriushenko]]
- [[quentin-herreros]]
- [[michael-gunther]]
- [[maximilian-werk]]
- [[han-xiao]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
