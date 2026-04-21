---
type: source
subtype: paper
title: "PLAID: An Efficient Engine for Late Interaction Retrieval"
slug: santhanam-2022-plaid-2205-09707
date: 2026-04-20
language: en
tags: [retrieval, late-interaction, indexing, efficiency, neural-ir]
processed: true

raw_file: raw/papers/santhanam-2022-plaid-2205-09707/paper.pdf
raw_md: raw/papers/santhanam-2022-plaid-2205-09707/paper.md
bibtex_file: raw/papers/santhanam-2022-plaid-2205-09707/paper.bib
possibly_outdated: true

authors:
  - Keshav Santhanam
  - Omar Khattab
  - Christopher Potts
  - Matei Zaharia
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2205.09707
doi:
url: http://arxiv.org/abs/2205.09707
citation_key: santhanam2022plaid
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

PLAID is a serving engine for late-interaction retrieval built on top of ColBERTv2. The paper's core claim is that the centroid assignments already stored for residual-compressed token embeddings are sufficient to filter most low-value passages before expensive residual lookup and MaxSim scoring. PLAID therefore introduces a multi-stage pipeline with centroid-based candidate generation, centroid pruning, centroid interaction, and only then full residual decompression for the final ranking stage. Across MS MARCO v1, Wikipedia OpenQA, LoTTE, and MS MARCO v2, the engine preserves nearly all of ColBERTv2's retrieval quality while materially reducing latency, reaching up to `7x` GPU and `45x` CPU speedups in its conservative setting and scaling to collections with `140M` passages.

## Problem & Motivation

Late-interaction retrievers such as ColBERT and ColBERTv2 achieve strong passage-ranking quality by comparing query and passage token embeddings with MaxSim, but their serving path is expensive because each passage is a matrix rather than a single vector. Vanilla ColBERTv2 spends much of its latency budget on index lookup, padding ragged passage tensors, and residual decompression for large candidate sets. The paper asks whether the centroid IDs already used for residual compression can do more than save space: specifically, whether they can identify strong candidates early enough to avoid reconstructing most passage embeddings while keeping recall and end-to-end ranking quality essentially intact.

## Method

- **Late-interaction base score**: PLAID inherits ColBERT-style scoring, `S_{q,d} = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} Q_i \cdot D_j^T`, where each query token matches its most similar passage token and the scores are summed across query tokens.
- **Residual-compressed storage**: passage token vectors remain compressed as nearest-centroid IDs plus quantized residuals. For default `128`-dimensional vectors, each token stores a centroid reference plus a `16`-byte or `32`-byte residual for `1`-bit or `2`-bit encoding respectively.
- **Candidate generation from centroids**: given centroid matrix `C` and query embedding matrix `Q`, PLAID computes query-centroid scores once as `S_{c,q} = C \cdot Q^T`, then gathers passages associated with the top-`t` centroids per query token, where `t` is `nprobe`.
- **Centroid interaction**: instead of immediately reconstructing embeddings, PLAID substitutes each token vector by its assigned centroid ID, forms centroid-score tensors `\tilde{D}` from the centroid lookup rows, and ranks candidates with centroid-only MaxSim `S_{\tilde{D}} = \sum_i \max_j \tilde{D}_{i,j}`.
- **Centroid pruning**: before centroid interaction, PLAID drops centroid IDs whose maximum query relevance is too small, retaining centroid `i` only if `\max_j S_{c,q_{i,j}} \ge t_{cs}`. This sparsifies bag-of-centroids passage representations.
- **Stage scheduling**: the engine uses a four-stage pipeline: centroid-based candidate generation, centroid pruning, centroid interaction, and final exact reranking after residual decompression. The paper uses the heuristic that Stage 3 returns `ndocs / 4` passages for final scoring.
- **Serving hyperparameters**: for final depths `k = 10 / 100 / 1000`, PLAID uses `nprobe = 1 / 2 / 4`, `t_cs = 0.5 / 0.45 / 0.4`, and `ndocs = 256 / 1024 / 4096`. Most experiments use `2`-bit compression, except MS MARCO v2 which uses `1`-bit compression.
- **Engine optimizations**: PLAID changes the inverted list to store passage IDs instead of embedding IDs, reducing one MS MARCO v2 inverted list from `71 GB` to `27 GB`. It also adds padding-free CPU MaxSim kernels and lookup-table residual decompression on CPU and GPU.

## Key Results

- **Centroid-only recall**: retrieving `10k` passages with centroid-only scoring recovers `99%+` of the top-`k` passages returned by full vanilla ColBERTv2 on both MS MARCO v1 and LoTTE, motivating aggressive pre-filtering.
- **MS MARCO v1**: at `k = 1000`, PLAID reaches `MRR@10 = 39.8`, `R@100 = 91.3`, and `R@1k = 97.5` versus vanilla ColBERTv2's `39.7 / 91.4 / 98.3`, while cutting latency from `259.6 ms` to `38.4 ms` on GPU and from `4568.5 ms` to `101.3 ms` on `8` CPU threads.
- **Wikipedia OpenQA**: at `k = 1000`, PLAID obtains `Success@5 = 74.4` and `Success@100 = 88.9` versus vanilla `74.3 / 89.0`, with latency reduced from `204.1 ms` to `55.3 ms` on GPU and from `5077.9 ms` to `228.4 ms` on CPU.
- **LoTTE pooled**: at `k = 1000`, PLAID slightly improves quality over vanilla ColBERTv2 with `Success@5 = 69.6` and `Success@100 = 90.5` versus `69.3 / 90.3`, while reducing latency from `66.9 ms` to `27.3 ms` on GPU and from `1508.4 ms` to `163.1 ms` on CPU.
- **MS MARCO v2 scale**: on the `138.4M`-passage benchmark, PLAID reaches `MRR@100 = 18.0` and `R@100 = 68.4` at `251.3 ms` on `8` CPU threads for `k = 1000`, and the paper reports `20.8x` CPU speedup with no quality loss up to top-`100` passages.
- **Ablation**: centroid interaction alone yields `5.2x` GPU and `8.6x` CPU speedups, while optimized kernels add another `1.3x` GPU and `4.9x` CPU; optimized kernels alone would give only about `3x` CPU speedup versus `42.4x` for the full system.

## Limitations

- PLAID is not a standalone retriever; it is tightly coupled to ColBERT-style late interaction and ColBERTv2's residual-compressed indexing scheme.
- The quality/latency tradeoff depends on tuning `nprobe`, `t_cs`, `ndocs`, and final depth `k`; more aggressive pruning can lose recall.
- GPU support is incomplete relative to CPU optimizations: the paper explicitly leaves custom padding-free GPU MaxSim kernels as future work, and both vanilla and PLAID run out of memory on MS MARCO v2 at `k = 1000`.
- Evaluation focuses on retrieval latency and effectiveness, not on training cost, index-build cost, or portability to other multi-vector retrievers with different compression schemes.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[passage-retrieval]]
- [[multi-stage-retrieval]]
- [[late-interaction-retrieval]]
- [[centroid-interaction]]
- [[centroid-pruning]]
- [[residual-compression]]
- [[maxsim-scoring]]
- [[k-means-clustering]]

## Entities Extracted

- [[keshav-santhanam]]
- [[omar-khattab]]
- [[christopher-potts]]
- [[matei-zaharia]]
- [[stanford-university]]
- [[colbert]]
- [[colbertv2]]
- [[plaid]]
- [[wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
