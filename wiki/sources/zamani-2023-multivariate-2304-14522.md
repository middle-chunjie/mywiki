---
type: source
subtype: paper
title: Multivariate Representation Learning for Information Retrieval
slug: zamani-2023-multivariate-2304-14522
date: 2026-04-20
language: en
tags: [dense-retrieval, representation-learning, information-retrieval, probabilistic-modeling, uncertainty]
processed: true
raw_file: raw/papers/zamani-2023-multivariate-2304-14522/paper.pdf
raw_md: raw/papers/zamani-2023-multivariate-2304-14522/paper.md
bibtex_file: raw/papers/zamani-2023-multivariate-2304-14522/paper.bib
possibly_outdated: true
authors:
  - Hamed Zamani
  - Michael Bendersky
year: 2023
venue: SIGIR 2023
venue_type: conference
arxiv_id: 2304.14522
doi: 10.1145/3539618.3591740
url: http://arxiv.org/abs/2304.14522
citation_key: zamani2023multivariate
paper_type: method
read_status: unread
domain: ir
---

## Summary

> ⚠️ **Possibly outdated** — published 2023 in a volatile IR/retrieval domain; check for follow-up work.

MRL (Multivariate Representation Learning) replaces the single-vector bi-encoder paradigm in dense retrieval with multivariate normal distributions: each query and document is represented as a mean vector and a diagonal covariance (variance) vector. Relevance scoring uses negative multivariate KL divergence, which the paper reformulates into a dot-product-compatible form so that existing ANN indexes (e.g., FAISS/HNSW) can be used without modification. Using DistilBERT initialized from TAS-B checkpoints, MRL achieves statistically significant gains over all single-vector baselines on MS MARCO and TREC Deep Learning tracks, and often outperforms ColBERTv2 while requiring 7× less storage and 5× lower query latency. An emergent by-product is that the learned query variance norm serves as an effective pre-retrieval query performance predictor.

## Problem & Motivation

Existing dense retrieval models represent queries and documents as fixed-length vectors (instantiations of Salton's vector space model). Such representations are "point estimates" that carry no notion of confidence or uncertainty. A model cannot distinguish whether its representation of an ambiguous query is reliable or not. This limitation was known in classical language model IR but had not been systematically addressed in the neural/dense retrieval era. Inspired by portfolio theory and Bayesian neural relevance work, the authors hypothesize that modeling uncertainty in the latent space would benefit retrieval quality, robustness, and interpretability.

## Method

**Representation.**
- Each query `q` and document `d` is encoded as a `k`-variate normal distribution: `Q ~ N_k(M_Q, Σ_Q)`, `D ~ N_k(M_D, Σ_D)`.
- Covariance matrices are diagonal, so each distribution is fully described by a `k`-dim mean vector and a `k`-dim variance vector.
- Implementation uses a BERT-style encoder with two special input tokens: `[CLS]` → mean projection via `W_M ∈ R^{768×k}`; `[VAR]` → variance projection via softplus activation: `Σ = (1/β) log(1 + exp(β · q_[VAR] · W_Σ))`.
- Softplus guarantees variance is always positive and differentiable; `β` is a tunable sharpness hyperparameter.
- `k = 381` (so that the concatenated ANN vector `2k+2 = 764 ≈ 768` dims, matching standard BERT output size for fair comparison).

**Scoring.**
- Relevance: `score(q,d) = −KLD_k(Q ∥ D)`.
- After algebraic simplification (diagonal Σ, dropping query-independent constants): `score(q,d) = rank −½[Σᵢ log σ²_di + Πᵢ σ²_qi / Πᵢ σ²_di + Σᵢ (μ_qi − μ_di)² / σ²_di]`.

**ANN compatibility.**
- Score is reformulated as a dot product between `q̃ = [1, Πq, μ²_q1, …, μ²_qk, μ_q1, …, μ_qk] ∈ R^{2k+2}` and a precomputed document vector `d̃` of the same dimension. Document vectors are indexed offline; query-time cost equals standard ANN search.

**Training.**
- Knowledge distillation from a BERT cross-encoder teacher using a listwise LambdaRank-inspired loss (Eq. 12).
- Hard negatives sampled from BM25 top-100 (`m_BM25`) + model-self top-100 (`m_hard`, refreshed every 5000 steps), plus in-batch negatives.
- Optimizer: Adam with linear warmup (4000 steps); batch size 512; learning rate from `[1e-6, 1e-5]`; initialized from TAS-B DistilBERT checkpoint.

## Key Results

- **MS MARCO Dev (MRR@10):** MRL 0.393 vs. best baseline CLDRD 0.382 (statistically significant).
- **MS MARCO Dev (MAP):** MRL 0.402 vs. CLDRD 0.386.
- **TREC-DL'19 (NDCG@10):** MRL 0.738 vs. CLDRD 0.725; vs. ColBERTv2 0.733.
- **TREC-DL'20 (NDCG@10):** MRL 0.701 vs. ColBERTv2 0.712 (ColBERTv2 wins here), vs. CLDRD 0.687.
- **Zero-shot (SciFact NDCG@10):** MRL 0.683 vs. Contriever-FT 0.677; vs. ColBERTv2 0.682.
- **Zero-shot (FiQA NDCG@10):** MRL 0.371 vs. ColBERTv2 0.359 (MRL wins).
- **Storage:** Single-vector (including MRL) 26 GB vs. ColBERTv2 192 GB on MS MARCO (8.8M passages).
- **Query latency:** Single-vector 89 ms/query vs. ColBERTv2 438 ms/query.
- **Query performance prediction (QPP):** MRL `|Σ_Q|` Pearson ρ = 0.271 (TREC-DL'19), 0.272 (TREC-DL'20), outperforming classical pre-retrieval QPP baselines (Max DC best competitor at 0.341/0.234).
- `β` sensitivity is low for `β ≥ 1`; best range `β ∈ [1, 10]`.

## Limitations

- MRL was evaluated only on passage retrieval (MS MARCO domain); document-level retrieval and structured retrieval were not tested.
- The ANN reformulation increases the effective vector dimensionality to `2k+2 ≈ 764` plus a pre-computed scalar `γ_d`, slightly increasing per-document storage vs. naive single-vector 768-dim.
- Diagonal covariance assumption discards correlations between representation dimensions; a full covariance matrix would be more expressive but computationally intractable.
- Zero-shot on TREC COVID (NDCG@10 = 0.668) slightly underperforms docT5query (0.713) and ColBERTv2 (0.696); the variance representation does not fully close the domain-shift gap in bio-medical retrieval.
- The model is initialized from TAS-B and uses knowledge distillation from a BERT cross-encoder — it is not self-contained as a standalone training recipe and depends on teacher model quality.
- Evaluation is single-run; the variance `β` was tuned on a validation set, meaning results may not transfer out-of-the-box to new domains.

## Concepts Extracted

- [[multivariate-representation-learning]]
- [[probabilistic-dense-retrieval]]
- [[dense-retrieval]]
- [[bi-encoder]]
- [[approximate-nearest-neighbor-search]]
- [[knowledge-distillation]]
- [[kl-divergence]]
- [[query-performance-prediction]]
- [[multi-vector-retrieval]]
- [[hard-negative-mining]]
- [[zero-shot-retrieval]]
- [[learning-to-rank]]
- [[representation-learning]]
- [[cross-encoder]]
- [[in-batch-negatives]]
- [[passage-retrieval]]

## Entities Extracted

- [[hamed-zamani]]
- [[michael-bendersky]]
- [[ms-marco]]
- [[ms-marco-passage-ranking]]
- [[beir]]
- [[colbertv2]]
- [[faiss]]
- [[bert]]
- [[trec-dl-2019]]
- [[trec-dl-2020]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
