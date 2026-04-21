---
type: source
subtype: paper
title: "BTR: Binary Token Representations for Efficient Retrieval-Augmented Language Models"
slug: unknown-nd-btrbinary-2310-01329
date: 2026-04-20
language: en
tags: [retrieval-augmentation, model-efficiency, compression, question-answering, llm]
processed: true

raw_file: raw/papers/unknown-nd-btrbinary-2310-01329/paper.pdf
raw_md: raw/papers/unknown-nd-btrbinary-2310-01329/paper.md
bibtex_file: raw/papers/unknown-nd-btrbinary-2310-01329/paper.bib
possibly_outdated: true

authors:
  - Qingqing Cao
  - Sewon Min
  - Yizhong Wang
  - Hannaneh Hajishirzi
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.01329
doi:
url: https://arxiv.org/abs/2310.01329
citation_key: unknownndbtrbinary
paper_type: method

read_status: unread
read_date:
rating:

domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

BTR targets the reader bottleneck in retrieval-augmented language models by precomputing token-level passage states as 1-bit vectors and caching them offline. Built on Atlas and Fusion-in-Decoder style readers, it decomposes query and passage encoding in lower encoder layers, inserts calibrated binarization inside the encoder, and restores accuracy with query-aware passage-token distillation plus passage-representation recovery losses. The system further applies offline Hamming-distance token merging and runtime intra-/cross-passage compression to cut both storage and inference cost. Across five knowledge-intensive tasks, BTR-Atlas preserves roughly `92%` to `97%` of Atlas performance while improving reader throughput by `2.5x` to `4.1x`, and compresses a Wikipedia-scale `3.2B`-token cache to `127 GB`.

## Problem & Motivation

Retrieval-augmented language models mitigate hallucination, staleness, and privacy issues, but their reader modules are slow because they must repeatedly encode many retrieved passages and compute query-passage interactions online. Prior decomposition approaches such as DeFormer reduce inference compute by caching passage representations, yet their continuous token states require terabytes of storage at Wikipedia scale. BTR is motivated by the need to keep the strong accuracy of retrieve-and-read systems while making reader inference practical on commodity GPUs through much smaller cached passage representations.

## Method

- **Reader decomposition**: BTR starts from Atlas readers built on [[fusion-in-decoder]] and [[t5]]. It encodes queries online through lower encoder layers and caches passage states offline, then rejoins them from layer `k + 1`; the decomposition layer is `k = 9` for BTR-Atlas base and `k = 20` for BTR-Atlas large.
- **Binary token representation**: for a passage token vector `h_k = [h_1, ..., h_d]`, BTR computes `b_k = sign(h_k)`, where each bit is `1` if `h_i > 0` and `-1` otherwise. This yields token-level cacheability instead of full-precision passage tensors.
- **Calibrated binarization**: instead of binarizing after the whole encoder block, BTR inserts binarization after layer normalization but before multi-head attention. It stores passage-token variance and uses layernorm weights to recover scale, while training uses a straight-through estimator so the forward pass stays discrete and the backward pass uses `tanh`.
- **Offline compression**: BTR further compresses cached binary passage tokens with Hamming-distance merging. Stopwords are collapsed to a mean binary vector, while non-stopword tokens merge `r_o%` of states using bipartite matching; the paper selects an offline compression ratio of `0.2`.
- **Three-stage training**: step 1 trains the original reader with `L_task`; step 2 trains the decomposed reader with `L_task + L_distill`; step 3 initializes from step 2 and trains the binarized model with `L_task + L_recovery`.
- **Query-aware distillation**: BTR distills only the top-`r` salient passage tokens selected by query-to-passage attention, with `r = 50%` of passage tokens by default and `L_distill = (1/r) Σ_i (h_i - h_i^decomp)^2`.
- **Passage recovery and runtime compression**: a linear projection on binary states minimizes `L_recovery = (1/d) Σ_i (h_i - b_i^proj)^2`. During inference BTR also applies intra-passage and cross-passage [[token-compression]], merging tokens at runtime with cosine similarity; the default decoder compression period is `g = 3`.
- **Implementation hyperparameters**: training uses `4` to `8` A40/A100 GPUs with BF16, `40` retrieved passages for QA/FEVER and `30` for MMLU, max passage length `320`, max answer length `32`, and task-specific learning rates ranging from `5e-6` to `8e-5`.

## Key Results

- On Wikipedia-scale caching (`32.1M` passages, about `3.2B` tokens), BTR stores the reader cache in `127 GB`, and the paper reports this is over `100x` smaller than prior continuous caching approaches.
- BTR-Atlas base reaches `49.5 / 66.7 / 43.8` EM on NaturalQuestions, TriviaQA, and WebQuestions, plus `70.2` FEVER accuracy and `35.4` MMLU accuracy, while speeding inference up by `3.1x / 2.5x / 2.6x / 3.1x / 2.6x` over Atlas base.
- BTR-Atlas large reaches `56.1 / 70.8 / 49.1` EM on the three QA sets, `75.9` FEVER accuracy, and `39.2` MMLU accuracy, with `4.0x / 3.9x / 3.6x / 4.1x / 2.5x` speedups over Atlas large.
- The ablation on NaturalQuestions shows binary passage representations cut storage from `12,804 GB` to `127 GB`; removing passage-recovery loss drops accuracy from `49.5` to `47.4`, and removing query-aware distillation drops it to `48.2`.
- For encoder-only readers, BTR-BERT still improves throughput by over `3x` while maintaining more than `92%` of baseline accuracy, showing the token binarization idea is not limited to encoder-decoder readers.

## Limitations

- BTR still sacrifices accuracy relative to the full Atlas readers, typically preserving `92%` to `97%` rather than matching the original models exactly.
- Even after aggressive compression, a Wikipedia-scale cache still requires `127 GB`, which is far smaller than prior work but not lightweight.
- The strongest speedups rely on encoder or encoder-decoder readers with decomposable passage computation; the paper explicitly says extending BTR to decoder-only readers is non-trivial.
- Runtime token compression cannot be applied to exact-span encoder readers such as BERT QA because compressed tokens would destroy recoverable span boundaries.
- The work is an arXiv preprint focused on five knowledge-intensive benchmarks and one 2018 Wikipedia snapshot, so broader robustness under newer RAG pipelines is not established.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[fusion-in-decoder]]
- [[binary-token-representation]]
- [[calibrated-binarization]]
- [[token-compression]]
- [[query-aware-distillation]]
- [[passage-representation-caching]]

## Entities Extracted

- [[qingqing-cao]]
- [[sewon-min]]
- [[yizhong-wang]]
- [[hannaneh-hajishirzi]]
- [[university-of-washington]]
- [[atlas]]
- [[t5]]
- [[triviaqa]]
- [[webquestions]]
- [[fever]]
- [[mmlu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
