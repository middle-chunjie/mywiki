---
type: source
subtype: paper
title: Incorporating Dynamic Semantics into Pre-Trained Language Model for Aspect-based Sentiment Analysis
slug: zhang-2022-incorporating
date: 2026-04-20
language: en
tags: [absa, sentiment-analysis, bert, pre-trained-model, nlp]
processed: true
raw_file: raw/papers/zhang-2022-incorporating/paper.pdf
raw_md: raw/papers/zhang-2022-incorporating/paper.md
bibtex_file: raw/papers/zhang-2022-incorporating/paper.bib
possibly_outdated: true
authors:
  - Kai Zhang
  - Kun Zhang
  - Mengdi Zhang
  - Hongke Zhao
  - Qi Liu
  - Wei Wu
  - Enhong Chen
year: 2022
venue: Findings of ACL 2022
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2022.findings-acl.285
url: https://aclanthology.org/2022.findings-acl.285
citation_key: zhang2022incorporating
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature in the NLP/sentiment domain.

This paper proposes DR-BERT (Dynamic Re-weighting BERT), a method for aspect-based sentiment analysis (ABSA) that extends BERT with a lightweight Dynamic Re-weighting Adapter (DRA) inserted after each BERT encoder layer. Inspired by psycholinguistic findings that human semantic comprehension focuses on roughly 7 words at a time in a sequential, attention-shifting manner, the DRA iteratively selects the single most aspect-relevant word at each step using a re-weighting attention function, then updates a sentence representation via GRU. The combined BERT + DRA model outperforms prior BERT-based ABSA methods (RGAT-BERT, T-GCN) on three standard benchmarks while adding minimal computation overhead compared to the base BERT model.

## Problem & Motivation

Standard ABSA models—both attention-based and pre-trained BERT variants—select all important words from a sentence simultaneously. They ignore the dynamic, sequential nature of human semantic comprehension, where a reader focuses on a small contextual window and iteratively refines understanding given a specific aspect. Vanilla BERT is further limited because it is pre-trained on sentence-level tasks (NSP, MLM) and has difficulty conditioning its representations on a particular aspect, especially in multi-aspect sentences where different aspects carry opposite sentiment polarities.

## Method

- **Input format**: Sentence `S = {w_1^s, …, w_l_s^s}` concatenated with aspect `A = {w_1^a, …, w_l_a^a}` via `[CLS] S [SEP]`. Multi-word aspects are represented as the average of their word embeddings.
- **BERT Encoder**: `N = 12` stacked layers (bert-base: `n_layers=12`, `n_heads=12`, `n_hidden=768`). Each layer runs multi-head self-attention followed by a position-wise FFN. Max-pooling over the FFN outputs produces the initial sentence representation `h_s = Max_Pooling(f_i)`.
- **Dynamic Re-weighting Adapter (DRA)**: Inserted after each BERT encoder layer. At each step `t ∈ [1, T]`:
  1. Re-weighting attention: `M = W_s·S + (W_d·h_{t-1} + W_a·a) ⊗ w`; scores `m = ω^T tanh(M)`.
  2. Soft argmax selection: `a_t = Σ_i softmax(λ·m_i) · s_i` where `λ = 100` approximates hard argmax while remaining differentiable.
  3. GRU update: `h_t = GRU(a_t, h_{t-1})` with hidden size `256`; `T = 7` re-weighting steps (matching the psycholinguistic 7-word memory threshold).
- **Fusion and Prediction**: Final representation `e = W_e·f + U_e·h_T + b_e` is fed through a 3-layer MLP (`l = 3`, `β = 0.8`) → softmax over 3 sentiment polarities.
- **Training**: Cross-entropy loss + L2 regularization; Adam optimizer; learning rate ∈ {2e-5, 5e-5, 1e-3}; batch size ∈ {16, 32, 64, 128}; dropout `0.2`.
- **Datasets**: SemEval-2014 Restaurant (2164/728 train/test pos), SemEval-2014 Laptop (994/341), Twitter (1561/173).

## Key Results

- **Restaurant**: DR-BERT Accuracy 87.72%, F1 82.31% — vs T-GCN 86.16%/79.95% and RGAT-BERT 86.60%/81.35%.
- **Laptop**: DR-BERT Accuracy 81.45%, F1 78.16% — vs T-GCN 80.88%/77.03%.
- **Twitter**: DR-BERT Accuracy 77.24%, F1 76.10% — vs T-GCN 76.45%/75.25%.
- Ablation on Laptop: adding only MLP gives 77.94/74.42; adding only DRA gives 80.66/77.13; full DR-BERT 81.45/78.16 — DRA contributes substantially more than MLP alone.
- Optimal re-weighting length `T = 7` (consistent with psycholinguistic 7-word memory); performance degrades for `T < 4` or `T > 8`.
- Runtime (Laptop): DR-BERT 157s/epoch × 10 epochs = 26.1 min, vs T-GCN 168s × 10 = 28.0 min — DRA adds negligible overhead.

## Limitations

- Evaluated only on three English review datasets; cross-lingual and domain generalization not assessed.
- The hard-argmax approximation via large `λ` is a heuristic; gradient flow through the re-weighting step is approximate.
- Hyperparameter sensitivity: `T`, `λ`, `l`, `β` all require tuning; no principled method for setting them.
- Multi-aspect disambiguation improves over baselines but remains imperfect on overlapping aspects (e.g., "Supplied software" vs "software" in the same sentence).
- BERT layers are frozen during fine-tuning; joint end-to-end training of BERT + DRA is not explored.
- The psychological analogy (7-word memory, eye-tracking) is qualitative and does not constitute a formal cognitive model.

## Concepts Extracted

- [[aspect-based-sentiment-analysis]]
- [[bert]]
- [[pre-trained-model]]
- [[fine-tuning]]
- [[attention-mechanism]]
- [[gated-recurrent-unit]]
- [[adapter]]
- [[multi-head-attention]]
- [[self-attention]]
- [[dynamic-semantic-comprehension]]
- [[sentiment-analysis]]
- [[graph-attention-network]]
- [[dependency-graph]]

## Entities Extracted

- [[kai-zhang]]
- [[kun-zhang-hfut]]
- [[mengdi-zhang]]
- [[hongke-zhao-tju]]
- [[qi-liu]]
- [[wei-wu]]
- [[enhong-chen]]
- [[university-of-science-and-technology-of-china]]
- [[hefei-university-of-technology]]
- [[meituan]]
- [[tianjin-university]]
- [[jacob-devlin]]
- [[semeval-2014-absa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
