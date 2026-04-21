---
type: source
subtype: paper
title: Adaptive Retrieval and Scalable Indexing for k-NN Search with Cross-Encoders
slug: unknown-nd-adaptive-2405-03651
date: 2026-04-20
language: en
tags: [retrieval, knn-search, cross-encoder, matrix-factorization, indexing]
processed: true
raw_file: raw/papers/unknown-nd-adaptive-2405-03651/paper.pdf
raw_md: raw/papers/unknown-nd-adaptive-2405-03651/paper.md
bibtex_file: raw/papers/unknown-nd-adaptive-2405-03651/paper.bib
possibly_outdated: false
authors:
  - Nishant Yadav
  - Nicholas Monath
  - Manzil Zaheer
  - Rob Fergus
  - Andrew McCallum
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.03651
doi:
url: https://arxiv.org/abs/2405.03651
citation_key: unknownndadaptive
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper studies how to make `k`-NN retrieval feasible when relevance is defined by a cross-encoder rather than a cheap dot product. Its core idea is to align item embeddings with cross-encoder scores through sparse matrix factorization instead of building a dense query-item score matrix or fully distilling a dual encoder on the target domain. At test time, the proposed Axn procedure adaptively estimates a query embedding from exact cross-encoder scores on a small retrieved set, then alternates between approximate retrieval and exact rescoring over multiple rounds. Across ZeShEL and BeIR domains, this combination improves recall-cost tradeoffs over retrieve-and-rerank baselines and cuts offline indexing cost dramatically relative to adaCUR and dual-encoder distillation.

## Problem & Motivation

Cross-encoders produce stronger relevance estimates than dual-encoders because they jointly encode the query-item pair, but exact search requires `O(|I|)` cross-encoder calls per query and is impractical at scale. Existing approximations have complementary weaknesses: retrieve-and-rerank with dual-encoders can miss good items because retrieval is decoupled from the cross-encoder, while CUR-based methods require scoring every item against many train queries, making offline indexing prohibitively expensive for large item sets. The paper targets the common deployment setting where one already has a trained cross-encoder and must index a new target corpus efficiently without expensive target-domain distillation.

## Method

- The paper approximates cross-encoder similarity `f(q, i)` with an inner-product space `\hat{f}(q, i) = u_q v_i^\top`, so approximate search becomes `\arg topk_i u_q v_i^\top`.
- Offline indexing constructs a partially observed score matrix `G \in \mathbb{R}^{|Q_train| \times |I|}` using only `k_d` items per train query or `k_d` queries per item, reducing cross-encoder calls from dense `|Q_train||I|` to sparse sampling.
- Item and train-query embeddings are fit by minimizing the sparse reconstruction objective ``min_{U,V} ||(G - UV^\top)_{P_train}||_2`` over observed entries only.
- `MFTrns` treats `U` and `V` as trainable parameters, optionally initialized from `DE_src`; this is accurate on smaller domains but scales poorly because every item embedding becomes a free parameter.
- `MFInd` freezes `DE_src` and learns separate 2-layer MLPs with skip connections on top of query/item embeddings: ``x'_out = b_2 + W_2^\top gelu(b_1 + W_1^\top x_in)``, ``x_out = \sigma(w_skip)x'_out + (1-\sigma(w_skip))x_in``. The skip scalar is initialized to `-5`.
- Matrix-factorization training uses AdamW. Reported learning rates are `0.001` for ZeShEL and Hotpot-QA, `0.005` for SciDocs, with epochs varying from `4` to `20` depending on dataset, `|Q_train|`, and `k_d`.
- At test time, Axn keeps item embeddings fixed and runs adaptive retrieval for `R = 5` rounds on ZeShEL or `R = 10` on BeIR. In each round it solves the least-squares system ``V_{A_r} u_q = a_r`` and uses the pseudo-inverse solution ``u_q = (V_{A_r}^\top V_{A_r})^\dagger V_{A_r}^\top a_r`` to update the test-query embedding.
- New candidates are retrieved by approximate scores ``u_q V^\top`` under a fixed cross-encoder budget `B_ce`, then exact CE scores are computed for newly added items and the query embedding is updated again.
- To regularize underdetermined early-round solutions, the final query embedding is interpolated as ``u_q = (1-\lambda)u_q^(LinReg) + \lambda u_q^(param)`` with a parametric embedding from `DE_src` or `MFInd`; ZeShEL uses `\lambda = 0`, while BeIR tunes `\lambda` on a dev set.
- For BeIR, cross-encoder scores are linearly normalized as ``s_final(q, i) = \beta (s_init(q, i) - \alpha)`` so the CE score range better matches the dual-encoder initialization used by matrix factorization.

## Key Results

- Over the same `DE_src` retrieve-and-rerank baseline on YuGiOh, `Axn_DEsrc,DEsrc` improves `Top-1-Recall` by `5.2%` and `Top-100-Recall` by `54%`, with no additional offline indexing cost and only small inference overhead.
- The paper reports that sparse matrix factorization can match or improve baseline recall while giving up to `100x` speedup over adaCUR and up to `5x` speedup over target-domain dual-encoder distillation for indexing.
- On Hotpot-QA, `adaCUR_DEsrc` needs `1000+` GPU hours to embed `5` million items and reaches `Top-1-Recall@100 = 75.9`, `Top-100-Recall@500 = 44.8`; by contrast, `MFInd` with `|Q_train| = 10K`, `k_d = 100` fits in under `3` hours and `Axn_MFInd,DEsrc` reaches `80.5` and `42.6` respectively.
- For downstream tasks, better `k`-NN recall usually helps, but not uniformly: on YuGiOh entity linking, `RnR_DEsrc` attains `50.6` accuracy while exact brute-force CE search gives `49.8`.
- On large-scale domains, item-embedding computation remains manageable: the appendix reports about `2` hours just to compute initial Hotpot-QA item embeddings on a `12 GB` 2080Ti, versus `90` seconds for SciDocs.

## Limitations

- The approach still assumes access to target-domain train queries and repeated black-box cross-encoder scoring to build the sparse matrix `G`; it is cheaper than dense methods, but not annotation- or compute-free.
- `MFTrns` scales poorly on very large corpora because item embeddings are free parameters; the paper notes that Hotpot-QA with `5` million items and `d = 768` implies roughly `4` billion trainable parameters and required `80 GB` A100 memory.
- Axn can overfit early retrieved items when `|A_r| < d`, so it needs interpolation with a parametric query embedding and careful tuning of `\lambda` on some domains.
- On Hotpot-QA, inference is still not fully global: the implementation first shortlists top `10K` items with a baseline retriever before running adaptive updates, so quality depends partly on the initial shortlist.
- Improvements in nearest-neighbor recall do not always transfer monotonically to downstream metrics, indicating residual mismatch between the learned approximation objective and task-level behavior of the cross-encoder.

## Concepts Extracted

- [[cross-encoder]]
- [[dual-encoder]]
- [[approximate-nearest-neighbor-search]]
- [[matrix-factorization]]
- [[sparse-matrix-factorization]]
- [[inductive-matrix-factorization]]
- [[transductive-matrix-factorization]]
- [[retrieve-and-rerank]]
- [[nearest-neighbor-search]]
- [[information-retrieval]]
- [[zero-shot-retrieval]]
- [[entity-linking]]

## Entities Extracted

- [[nishant-yadav]]
- [[nicholas-monath]]
- [[manzil-zaheer]]
- [[rob-fergus]]
- [[andrew-mccallum]]
- [[google-deepmind]]
- [[university-of-massachusetts-amherst]]
- [[beir]]
- [[zeshel]]
- [[scidocs]]
- [[hotpot-qa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
