---
type: source
subtype: paper
title: History-Aware Conversational Dense Retrieval
slug: mo-2024-historyaware-2401-16659
date: 2026-04-20
language: en
tags: [conversational-search, dense-retrieval, query-reformulation, contrastive-learning, topic-switching]
processed: true

raw_file: raw/papers/mo-2024-historyaware-2401-16659/paper.pdf
raw_md: raw/papers/mo-2024-historyaware-2401-16659/paper.md
bibtex_file: raw/papers/mo-2024-historyaware-2401-16659/paper.bib
possibly_outdated: false

authors:
  - Fengran Mo
  - Chen Qu
  - Kelong Mao
  - Tianyu Zhu
  - Zhan Su
  - Kaiyu Huang
  - Jian-Yun Nie
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.16659
doi:
url: http://arxiv.org/abs/2401.16659
citation_key: mo2024historyaware
paper_type: method

read_status: unread

domain: ir
---

## Summary

This paper proposes HAConvDR, a conversational dense retrieval method that explicitly decides which historical turns are useful for the current query and then uses those decisions twice: to denoise query reformulation and to mine extra supervision for training. The core idea is pseudo relevance judgment (PRJ): a historical turn is marked relevant if concatenating its query and gold passage improves retrieval for the current turn. Relevant history forms the reformulated query, while relevant historical passages become pseudo positives and irrelevant ones become historical hard negatives in contrastive training. Built on ANCE, HAConvDR reaches `MRR = 30.1` and `NDCG@3 = 28.5` on TopiOCQA, and `MRR = 47.7` and `NDCG@3 = 44.8` on QReCC, with especially clear gains on long, topic-shifting conversations.

## Problem & Motivation

Conversational dense retrieval usually feeds the whole interaction history into a retriever, assuming the model can learn which earlier turns matter. This paper argues that assumption is fragile: long conversations contain irrelevant turns, and using all of them creates shortcut history dependency, where the model over-focuses on historical needs and ranks historical gold passages above the current target passage. The authors therefore target two coupled problems: identifying which prior turns are actually relevant to the current information need, and converting those relevance judgments into stronger positive and negative supervision for retriever training. The motivation is strongest in realistic conversations with topic shifts, where naive history concatenation is especially noisy.

## Method

- **Task setup**: for current query `q_n`, the history contains turns `(q_i, p_i^*)` for `i < n`, where `p_i^*` is the historical ground-truth passage. The goal is to retrieve `p_n^*` from corpus `D`.
- **Pseudo relevance judgment (PRJ)**: for each historical turn, run retriever `phi` on the raw query and on the augmented query `q_n ∘ q_i ∘ p_i^*`. If retrieval metric `M` improves, mark the turn relevant; otherwise mark it irrelevant.
- **History partitioning**: PRJ splits historical passages into relevant and irrelevant sets, written as `P_h^+` and `P_h^-`. This is the paper's central supervision interface between history modeling and retrieval training.
- **Context-denoised query reformulation**: construct reformulated query `q_n^r = q_n ∘ ... ∘ p_i^* ∘ q_i ...` using only PRJ-relevant historical turns, instead of concatenating the whole session.
- **History-aware contrastive learning**: use dual encoders with similarity `S(q, p) = F_Q(q)^T · F_P(p)`. Positives become `P_n^+ = {p_n^*} ∪ P_h^+`, while negatives become `P_n^- = P_b^- ∪ P_r^- ∪ P_h^-`, where `P_b^-` are in-batch negatives and `P_r^-` are retrieved hard negatives.
- **Training objective**: optimize the averaged contrastive loss `L = (1/N) Σ_i exp(S(q_n^r, p_i^+)) / (exp(S(q_n^r, p_i^+)) + Σ_j exp(S(q_n^r, p_j^-)))` so the reformulated query is closer to current and historically useful passages than to distractors.
- **Backbone and implementation**: initialize dense retrievers from ANCE, use Faiss for retrieval, update only the query encoder while freezing the passage encoder, and train with Adam at `3e-5` and batch size `32` on one Nvidia A100 40G GPU.
- **Input lengths and sampling**: maximum query/passage lengths are `512/384` on TopiOCQA and `256/256` on QReCC. For each instance, the model samples one historical pseudo positive and one historical hard negative in addition to a top retrieved hard negative.

## Key Results

- **Main benchmark wins on TopiOCQA**: HAConvDR reaches `MRR 30.1`, `NDCG@3 28.5`, `R@10 50.8`, `R@100 72.8`, beating ConvDR's `27.2`, `26.4`, `43.5`, and `61.1`.
- **Strong results on QReCC**: HAConvDR reaches `MRR 47.7`, `NDCG@3 44.8`, `R@10 71.6`, `R@100 88.7`, slightly surpassing SDRConv on `MRR` (`47.3`) and improving over ConvDR by `+9.2 MRR` and `+9.1 NDCG@3`.
- **Ablation confirms both components matter**: on TopiOCQA, removing historical hard negatives drops `MRR` from `30.1` to `28.2`, removing pseudo positives drops it to `26.8`, and removing PRJ-based query reformulation drops it further to `25.0`.
- **Relevant history is sparse**: PRJ analysis shows relevant historical turns are only a small fraction of history, peaking at roughly `20%`, which supports explicit denoising instead of whole-session concatenation.
- **Substitution setting still helps**: when historical gold passages are replaced by top retrieved passages, the full model still improves TopiOCQA to `MRR 25.94` and `NDCG@3 24.32`, above `QR w/ PRJ` at `24.98` / `23.09`.

## Limitations

- The method assumes access to historical gold passages during the main setup; the more realistic substitution with top retrieved passages is workable but measurably weaker, and degrades further as `k` increases.
- Reformulated queries can include selected historical passages and therefore become hundreds of tokens long, which raises truncation and residual noise risks even after PRJ filtering.
- PRJ is still a heuristic based on observed retrieval improvement under a chosen metric and retriever, so errors in the underlying retriever can propagate into both reformulation and supervision mining.
- Gains are much larger on TopiOCQA than on QReCC, suggesting the method is especially beneficial when conversations contain topic shifts and noisier history, not uniformly across all conversational retrieval regimes.

## Concepts Extracted

- [[conversational-dense-retrieval]]
- [[conversational-search]]
- [[context-denoising]]
- [[query-reformulation]]
- [[pseudo-relevance-judgment]]
- [[pseudo-relevance-feedback]]
- [[contrastive-learning]]
- [[hard-negative-mining]]
- [[dual-encoder-retrieval]]
- [[topic-switching]]

## Entities Extracted

- [[fengran-mo]]
- [[chen-qu]]
- [[kelong-mao]]
- [[tianyu-zhu]]
- [[zhan-su]]
- [[kaiyu-huang]]
- [[jian-yun-nie]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
