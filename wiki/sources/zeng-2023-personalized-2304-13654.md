---
type: source
subtype: paper
title: A Personalized Dense Retrieval Framework for Unified Information Access
slug: zeng-2023-personalized-2304-13654
date: 2026-04-20
language: en
tags: [dense-retrieval, personalization, information-retrieval, recommendation, e-commerce]
processed: true
raw_file: raw/papers/zeng-2023-personalized-2304-13654/paper.pdf
raw_md: raw/papers/zeng-2023-personalized-2304-13654/paper.md
bibtex_file: raw/papers/zeng-2023-personalized-2304-13654/paper.bib
possibly_outdated: true
authors:
  - Hansi Zeng
  - Surya Kallumadi
  - Zaid Alibadi
  - Rodrigo Nogueira
  - Hamed Zamani
year: 2023
venue: SIGIR 2023
venue_type: conference
arxiv_id: 2304.13654
doi: 10.1145/3539618.3591626
url: http://arxiv.org/abs/2304.13654
citation_key: zeng2023personalized
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature in dense retrieval and recommendation.

This paper proposes UIA (Unified Information Access), a dense retrieval framework that handles multiple information access tasks — keyword search, query by example, and complementary item recommendation — within a single bi-encoder model. UIA encodes the information access functionality as a textual description alongside the user request, and introduces an Attentive Personalization Network (APN) that performs both content-based and collaborative personalization over user interaction history. Training uses a non-personalized pre-training stage followed by personalized fine-tuning with hard negative sampling. Experiments on a large-scale Lowe's e-commerce dataset and the Amazon ESCI dataset show that UIA significantly outperforms task-specific baselines as well as prior joint search-recommendation frameworks, with up to 45% NDCG@10 gains from joint optimization alone.

## Problem & Motivation

Information access systems (search, recommendation, Q&A) have historically been developed as separate specialized models despite overlapping user needs and shared data. A unified model could transfer knowledge across tasks, reduce engineering overhead, and improve cold-start task performance. Prior work (JSR, SRJGraph) achieved some joint search-recommendation modeling but was not extensible to new information access functionalities and did not incorporate expressive personalization. The key challenge is designing a single architecture flexible enough to distinguish between tasks (e.g., similar-item vs. complementary-item retrieval given identical inputs) while adapting to individual user preferences.

## Method

- **Task formulation**: A unified scoring function `f(F_t^u, R_t^u, H_t^u, I_i; θ)` takes four inputs: Information Access Functionality `F`, Request `R`, User History `H`, and Candidate Item `I`. Functionality is encoded as a natural-language description (e.g., "find complementary items"), enabling zero-parameter extension to new tasks.
- **Bi-encoder architecture**:
  - *Request encoder* `E_R`: BERT-base encodes `[CLS] R [SEP] F [SEP]` → dense vector `R_t^u ∈ R^768`.
  - *Item encoder* `E_I`: BERT-base encodes `[CLS] I [SEP]` → dense vector `I_i`.
  - Similarity: inner product `R · I` (both encoders constrained to same dimensionality).
- **User History Encoding**: Last `N=5` interactions encoded as `2×N` vectors — one per past (request, item) pair — using the shared encoders `E_R` and `E_I`.
- **Attentive Personalization Network (APN)**:
  - Content-based: multi-head attention (`N_h=12` heads, `l=l_v=64`) over past request matrix `H_t^u ∈ R^{N×d}` (keys) and clicked item matrix `C_t^u ∈ R^{N×d}` (values), with current request as query `Q_j = R_t^u · θ_j^Q`. Attention: `softmax(QK^T / √l)V`.
  - Collaborative: user embedding `u ∈ R^{l_u=128}` and functionality embedding `f ∈ R^{l_f=64}` appended to APN output, then projected through ReLU feed-forward → `R_t^{u*}`.
  - Item vectors adjusted via FFN to align with personalized semantic space → `I_i^*`.
- **Two-stage training**:
  - *Phase 1 non-personalized pre-training*: cross-entropy loss with BM25 hard negatives + in-batch negatives, ratio 1:1. Batch size 384.
  - *Phase 2 ANCE-style negatives*: self-mined negatives via ANN index (Faiss), same loss. Learning rate `7e-6`.
  - *Personalized fine-tuning*: adds APN; retrains on user-partitioned data (≥10 past interactions for search/QBE, ≥5 for CIR). Learning rate `7e-5`.
- **Backbone**: `bert-base-uncased` for Lowe's; `msmarco-bert-base-dot-v5` for Amazon ESCI.

## Key Results

- **Lowe's dataset** (890K users, 2.26M items):
  - UIA NDCG@10: Keyword Search 0.399, QBE 0.495, CIR 0.432 (all statistically significant vs. all baselines p<0.01 except MRR@10 for keyword search vs. joint training).
  - vs. best task-specific baseline (Context-Aware RocketQA): +2.6% Search, +34.1% QBE, +42.1% CIR NDCG@10.
  - vs. best joint baseline (SRJGraph): +1.8% Search, +3.6% QBE, +2.9% CIR NDCG@10.
  - APN ablation: removing APN drops NDCG@10 by 93%/72%/145% on Search/QBE/CIR.
  - Encoding functionality `F` critical: dropping it cuts QBE by 30% and CIR by 34%.
  - Joint optimization contributes 34% QBE and 45% CIR relative improvement over task-specific training.
  - 60%+ of users benefit from joint optimization in QBE and CIR tasks.
- **Amazon ESCI** (no user IDs, no personalization):
  - UIA MRR@10: KS 0.532, QBE 0.251, CIR 0.490 — all significant improvements over baselines.
  - Gains smaller due to absence of personalization; CIR benefits most from joint training.
- **Ablation findings**: content-based personalization drives most APN gains; collaborative personalization helps KS and CIR but not QBE; cross-task history (using all tasks' interactions) uniformly outperforms task-specific history.

## Limitations

- Both datasets are from e-commerce (Lowe's and Amazon); generalizability to other domains (e.g., document retrieval, biomedical) is unvalidated.
- No public multi-functional dataset with user identifiers exists; Amazon ESCI lacks user IDs preventing personalization evaluation there.
- Personalization hurts ~25% of users (cold-start, intent drift); the paper suggests predicting personalization need as future work but leaves it unsolved.
- APN relies on item content text; sparse or missing item descriptions degrade performance.
- Faiss ANN index is built offline and must be periodically refreshed; stale index may harm self-negative sampling.
- Scalability of user embedding matrix `E_U ∈ R^{|U|×128}` is not discussed for extremely large user bases.

## Concepts Extracted

- [[dense-retrieval]]
- [[bi-encoder]]
- [[personalization]]
- [[unified-information-access]]
- [[attentive-personalization-network]]
- [[personalized-dense-retrieval]]
- [[query-by-example]]
- [[complementary-item-recommendation]]
- [[collaborative-filtering]]
- [[sequential-recommendation]]
- [[hard-negative-mining]]
- [[approximate-nearest-neighbor-search]]
- [[multi-task-learning]]
- [[in-batch-negative-sampling]]
- [[cold-start]]

## Entities Extracted

- [[hansi-zeng]]
- [[surya-kallumadi]]
- [[zaid-alibadi]]
- [[rodrigo-nogueira]]
- [[hamed-zamani]]
- [[lowes-companies]]
- [[amazon-esci]]
- [[university-of-massachusetts-amherst]]
- [[university-of-campinas]]
- [[dpr]]
- [[ance]]
- [[rocketqa]]
- [[sasrec]]
- [[bm25]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
