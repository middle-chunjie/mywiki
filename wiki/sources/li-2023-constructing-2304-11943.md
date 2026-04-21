---
type: source
subtype: paper
title: Constructing Tree-based Index for Efficient and Effective Dense Retrieval
slug: li-2023-constructing-2304-11943
date: 2026-04-20
language: en
tags: [dense-retrieval, ann, indexing, information-retrieval, beam-search]
processed: true

raw_file: raw/papers/li-2023-constructing-2304-11943/paper.pdf
raw_md: raw/papers/li-2023-constructing-2304-11943/paper.md
bibtex_file: raw/papers/li-2023-constructing-2304-11943/paper.bib
possibly_outdated: true

authors:
  - Haitao Li
  - Qingyao Ai
  - Jingtao Zhan
  - Jiaxin Mao
  - Yiqun Liu
  - Zheng Liu
  - Zhao Cao
year: 2023
venue: SIGIR '23
venue_type: conference
arxiv_id: 2304.11943
doi:
url: http://arxiv.org/abs/2304.11943
citation_key: li2023constructing
paper_type: method

read_status: unread
read_date:
rating:

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper proposes JTR, a tree-based dense-retrieval index that jointly optimizes query encoding and index nodes instead of training the encoder and ANN structure separately. The core idea is to replace task-agnostic index training with a unified contrastive objective over tree nodes, then improve beam-search pruning through sibling-based negative sampling and improve leaf assignment through overlapped clustering. The method preserves sub-linear retrieval complexity while reducing the effectiveness drop that usually comes from ANN indexing. On MS MARCO passage and document retrieval, JTR achieves the best or near-best ranking quality among tested ANN baselines and materially outperforms prior tree-based indexes, while remaining substantially faster than brute-force-style strong baselines such as JPQ.

## Problem & Motivation

Dense retrieval is effective for first-stage retrieval, but brute-force maximum-inner-product search is too slow for web-scale corpora. Existing ANN indexes improve latency, yet tree-based indexes often lose relevance quality because they are trained with reconstruction-oriented objectives and optimized independently from the query encoder. The paper argues that this objective mismatch leads to sub-optimal pruning and poor cluster assignment. JTR is motivated by the need to keep the sub-linear efficiency of tree indexing while directly optimizing the index structure for supervised retrieval quality.

## Method

- **Dense retrieval setup**: a query-document pair is scored by inner product, `s(q, d) = <f(q), f(d)>`. JTR uses a BERT-based query encoder `Phi(q)` and document embeddings from STAR as the initialization for index construction.
- **Tree-based index**: documents are recursively partitioned by `k`-means. Each non-leaf node has branch balance factor `beta`, each leaf holds at most `gamma` documents, and node embeddings are initialized by cluster centroids. In experiments, the default setting is `beta = 10`, `gamma = 1000`, and embedding dimension `D = 768`.
- **Node scoring and retrieval**: each cluster node has a trainable embedding `e_c`. At retrieval time, node relevance is computed as `s = e_c^T Phi(q)`, beam search keeps the top `b` frontier nodes, and documents inside the returned leaf set are re-ranked by `e_d^T Phi(q)`.
- **Maximum-heap motivation**: the tree should satisfy the paper's heap-style constraint in Eq. (2), where a parent node's relevance is tied to the best child, so that beam search does not prune away the path to the true target leaf too early.
- **Unified contrastive learning loss**: instead of training the index with reconstruction loss, JTR treats the target leaf and its ancestors as positives and optimizes each level with `L(q, n^+, n_1^-, ..., n_m^-) = -log( exp(s(q, n^+)) / (exp(s(q, n^+)) + sum_j exp(s(q, n_j^-))) )`. This jointly updates the query encoder and cluster-node embeddings.
- **Tree-based negative sampling**: negatives are chosen from sibling nodes of the positive node at each level. Because sibling clusters are close in embedding space, they act as hard structured negatives and sharpen beam-search discrimination.
- **Overlapped clustering**: leaf assignment is optimized separately because cluster assignment is non-differentiable. With query-leaf matrix `M`, relevance matrix `Y`, and assignment matrix `C`, the paper approximates recall maximization through `C* = Proj(Y_bar^T M)`, where `Proj(.)` keeps the top `lambda` leaves per document and allows a document to appear in up to `lambda` clusters.
- **Complexity and implementation**: retrieval complexity is `O(beta * b * log K) + O(b * N / K)`. Training uses AdamW, batch size `32`, learning rate `5e-6`, and top `100` ADORE-STAR retrieved documents to build `Y_bar` for overlapped clustering.

## Key Results

- **MS MARCO Passage**: JTR reaches `MRR@100 = 0.318`, `R@100 = 0.778`, `NDCG@10 = 0.610` on DL19, and `NDCG@10 = 0.632` on DL20 with `AQT = 30 ms`.
- **MS MARCO Document**: JTR reaches `MRR@100 = 0.364`, `R@100 = 0.848`, `NDCG@10 = 0.590` on DL19, and `NDCG@10 = 0.565` on DL20 with `AQT = 18 ms`.
- **Against the strongest recall baseline**: JPQ attains higher recall (`0.832` passage, `0.889` doc) but is much slower (`152 ms` passage, `55 ms` doc). JTR trades a small recall drop for roughly `3x-5x` lower latency.
- **Against existing tree indexes**: on passage retrieval, JTR strongly improves over Annoy (`0.144` MRR@100, `132 ms`) and FLANN (`0.271` MRR@100, `18 ms`); similar gains hold on document retrieval.
- **Ablation on MS MARCO Doc Dev**: a plain tree index gets `0.256` MRR@100 / `0.556` R@100 / `5 ms`; adding joint optimization raises this to `0.296` / `0.640`; reorganizing clusters gives `0.303` / `0.678`; overlapped clustering reaches `0.327` / `0.743` with `8 ms`.

## Limitations

- The evaluation is limited to MS MARCO passage/document retrieval and TREC DL 2019/2020, so cross-domain generalization is not established.
- JTR is not the fastest ANN option in the comparison; IVFPQ, FLANN, and HNSW achieve lower latency on several settings but with weaker ranking quality.
- The paper notes that when recall requirements are extremely high (for example `R@100 > 0.8`), brute-force-style methods such as JPQ can become more efficient overall than indexed search.
- Cluster assignment is still optimized through a separate projection step rather than fully differentiable end-to-end learning.
- Memory usage is not reported, so the paper does not resolve the effectiveness-efficiency-memory tradeoff.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[dual-encoder]]
- [[tree-based-index]]
- [[contrastive-learning]]
- [[tree-based-negative-sampling]]
- [[overlapped-clustering]]
- [[k-means-clustering]]
- [[beam-search]]

## Entities Extracted

- [[haitao-li]]
- [[qingyao-ai]]
- [[jingtao-zhan]]
- [[jiaxin-mao]]
- [[yiqun-liu]]
- [[zheng-liu-huawei]]
- [[zhao-cao]]
- [[tsinghua-university]]
- [[renmin-university-of-china]]
- [[huawei-poisson-lab]]
- [[ms-marco]]
- [[bert]]
- [[faiss]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
