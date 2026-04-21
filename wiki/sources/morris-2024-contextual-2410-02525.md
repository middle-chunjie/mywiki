---
type: source
subtype: paper
title: Contextual Document Embeddings
slug: morris-2024-contextual-2410-02525
date: 2026-04-20
language: en
tags: [dense-retrieval, document-embedding, contrastive-learning, domain-adaptation, benchmarks]
processed: true

raw_file: raw/papers/morris-2024-contextual-2410-02525/paper.pdf
raw_md: raw/papers/morris-2024-contextual-2410-02525/paper.md
bibtex_file: raw/papers/morris-2024-contextual-2410-02525/paper.bib
possibly_outdated: false

authors:
  - John X. Morris
  - Alexander M. Rush
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.02525
doi:
url: http://arxiv.org/abs/2410.02525
citation_key: morris2024contextual
paper_type: method
read_status: unread
domain: ir
---

## Summary

This paper argues that standard dense retrievers learn document vectors in isolation and therefore miss corpus-specific contextual signals analogous to inverse document frequency. It introduces Contextual Document Embeddings (CDE), combining a harder contrastive training scheme that clusters neighboring query-document pairs into pseudo-domains with a two-stage architecture that conditions each document or query embedding on sampled documents from the target corpus. The training objective explicitly filters likely false negatives and uses shared contextual inputs plus two-stage gradient caching to keep large-batch training tractable. Across BEIR-style experiments and full MTEB evaluation, both contextual batching and the contextual encoder improve over vanilla biencoders, and the final `cde-small-v1` model reaches a `65.00` MTEB mean without relying on large hard-negative setups or extreme batch engineering.

## Problem & Motivation

Dense retrieval models typically encode each document as `phi(d)` independently of the target corpus, so they cannot adapt their representations to corpus-specific term frequency shifts the way sparse methods using IDF can. This becomes problematic under domain shift, where terms that are informative in the training corpus may become ubiquitous in a new retrieval corpus. The paper targets this mismatch by asking how to make dense document embeddings context-aware at both training time and inference time, while preserving the practical retrieval pipeline of precomputing document vectors and performing vector search.

## Method

- **Retrieval formulation**: score documents with `f(d, q) = phi(d) · psi(q)` and approximate `p(d | q)` with contrastive learning over sampled negatives rather than the full corpus normalizer.
- **Adversarial contextual batching**: partition the training set into pseudo-domains `({B^1}, ..., {B^B})` so that each batch contains semantically close query-document pairs, making in-batch negatives harder. The paper relaxes the combinatorial objective in Eq. `2` with an asymmetric `k`-means objective in Eq. `3`.
- **Clustering implementation**: build pre-training batches using GTR embeddings and FAISS clustering, running `100` clustering steps and keeping the best of `3` attempts. Cluster sizes are swept from `64` up to `4,194,304` in the small setting.
- **False-negative filtering**: define a surrogate equivalence class `S(q, d) = {d' in D | f(q, d') >= f(q, d) + epsilon}` and remove `S(q, d)` from the contrastive denominator, yielding the filtered objective in Eq. `4`.
- **Contextual architecture**: compute corpus-aware embeddings with a two-stage encoder. First embed contextual documents `d^1, ..., d^J` with `M_1`; then feed `M_1(d^1), ..., M_1(d^J)` plus the target text tokens into `M_2` so `phi(d'; D) = M_2(M_1(d^1), ..., M_1(d^J), E(d'_1), ..., E(d'_T))` and analogously for queries.
- **Position-agnostic context**: contextual documents are treated as an unordered set, so positional information is removed for dataset-context tokens, including a rotary-positional-embedding modification described in the supplement.
- **Efficiency techniques**: reuse the same contextual documents within each batch, cache first-stage context embeddings at indexing time, and apply two-stage gradient caching so each transformer forward pass is rerun with gradients instead of storing all activations.
- **Training setup**: initialize `M_1` and `M_2` from a BERT-base/NomicBERT-style backbone with shared query/document weights but separate stage weights; train with Adam, `1000` warmup steps, learning rate `2e-5`, linear decay to `0`, contrastive temperature `tau = 0.02`, sequence dropout `p = 0.005`, sequence length `512`, and up to `512` contextual inputs in the large setting.
- **Evaluation regimes**: small experiments use a `6`-layer transformer with max sequence length `64` and `64` contextual tokens on a truncated BEIR setup; the large model evaluates on full MTEB after pre-training on `200M` weakly supervised pairs and supervised fine-tuning on `1.8M` retrieval pairs plus BGE meta-datasets.

## Key Results

- On the small BEIR-style setup, the vanilla biencoder scores `59.9` NDCG@10, contextual batching alone reaches `61.7`, the contextual architecture alone reaches `62.4`, and combining both yields `63.1`.
- The paper reports a strong positive correlation between harder in-batch negatives and downstream retrieval quality; with filtering enabled, smaller clusters consistently outperform larger ones.
- On full MTEB for models under `250M` parameters, `cde-small-v1` with true contextual documents reaches a mean score of `65.00`, compared with `63.81` when replacing context with random documents.
- Relative to the strongest listed same-scale baseline, `gte-base-en-v1.5` at `64.11`, the contextual model posts the best overall MTEB mean and improves particularly on clustering (`48.3`) and classification (`81.7`).
- On MTEB retrieval tasks specifically, the supervised contextual model improves the mean from `52.8` to `54.0`, with notable gains on ArguAna (`49.3 -> 53.8`) and TREC-COVID (`79.9 -> 82.6`).
- The contextual architecture is more computationally expensive: in the supplement, one unsupervised epoch takes about `2` days on `8` H100 GPUs versus about `1` day for the biencoder, but shorter-sequence experiments are `10-20x` faster.

## Limitations

- The approach assumes access to a representative corpus or sampled contextual documents at indexing time; if context is unavailable, performance falls back toward a standard biencoder.
- Training is materially more complex than standard contrastive retrieval, requiring offline clustering, false-negative filtering, context packing, and a two-stage model with gradient caching.
- The method is sensitive to false negatives and distributed training artifacts; the supplement notes that improper DDP and gradient-sync handling can silently diverge.
- Gains are uneven across task types: semantic textual similarity changes little under random versus true contextual documents, suggesting some embedding behavior remains mostly context-agnostic.
- The paper focuses on small-to-mid-scale text embedding models and retrieval-centric benchmarks, so it does not establish the same benefit profile for much larger encoders or non-text modalities.

## Concepts Extracted

- [[document-embedding]]
- [[contextual-document-embedding]]
- [[dense-retrieval]]
- [[contrastive-learning]]
- [[domain-adaptation]]
- [[pseudo-relevance-feedback]]
- [[hard-negative-mining]]
- [[false-negative-filtering]]
- [[gradient-cache]]
- [[sequence-dropout]]
- [[rotary-positional-embedding]]

## Entities Extracted

- [[john-x-morris]]
- [[alexander-m-rush]]
- [[cornell-university]]
- [[beir]]
- [[mteb]]
- [[gtr]]
- [[faiss]]
- [[nomic-bert]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
