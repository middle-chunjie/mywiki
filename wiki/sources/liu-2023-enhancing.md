---
type: source
subtype: paper
title: Enhancing Hierarchical Text Classification through Knowledge Graph Integration
slug: liu-2023-enhancing
date: 2026-04-20
language: en
tags: [hierarchical-text-classification, knowledge-graph, contrastive-learning, bert, text-classification]
processed: true
raw_file: raw/papers/liu-2023-enhancing/paper.pdf
raw_md: raw/papers/liu-2023-enhancing/paper.md
bibtex_file: raw/papers/liu-2023-enhancing/paper.bib
possibly_outdated: true
authors:
  - Ye Liu
  - Kai Zhang
  - Zhenya Huang
  - Kehang Wang
  - Yanghai Zhang
  - Qi Liu
  - Enhong Chen
year: 2023
venue: "Findings of the Association for Computational Linguistics: ACL 2023"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.findings-acl.358
url: "https://aclanthology.org/2023.findings-acl.358"
citation_key: liu2023enhancing
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

K-HTC augments hierarchical text classification with external knowledge graphs at three points: token-level representation, hierarchy-aware label learning, and document-level contrastive supervision. It first aligns document tokens and label names to ConceptNet concepts, pretrains concept embeddings with TransE on a pruned subgraph, and fuses BERT token states with GraphSAGE-aggregated neighbor information. It then derives label embeddings from the same knowledge-aware encoder, propagates them over the label hierarchy with a GCN, and applies label-conditioned attention to form class-enhanced document vectors. Finally, two contrastive objectives weight document pairs by shared concepts and shared labels. On BGC and WOS, K-HTC reaches the best Macro-F1 among compared systems and shows larger gains on deeper, harder hierarchy levels.

## Problem & Motivation

Hierarchical text classification is harder than flat text classification because each document may receive multiple labels arranged in a taxonomy or DAG rather than a single independent class. The paper argues that strong HTC models built on BERT and hierarchy modeling still lack external world knowledge, so they can overfit to shallow lexical cues and miss inferential evidence. The motivating examples show that a document mentioning "USA" can still belong to an Africa travel category once related concepts such as Sahara or Algiers are considered. The core motivation is therefore to inject knowledge graphs into both document encoding and hierarchical label learning, then exploit shared concepts across documents as an additional supervision signal.

## Method

- **Concept preparation**: run soft n-gram matching with lemmatization and stop-word filtering to map token sequence `x = {x_1, ..., x_N}` to concept sequence `c = {c_1, ..., c_N}`; unmatched tokens use `` `c_i = [PAD]` ``.
- **Concept pretraining**: retain matched concepts plus first-order neighbors to build a pruned KG `G_1'`, then pretrain concept embeddings with TransE as `` `U in R^(N_c x v)` `` with `` `v = 768` ``.
- **Knowledge-aware Text Encoder (KTE)**: encode text with `bert-base-uncased` to get token states `` `w_i = BERT(x_i)` ``, map concepts to embeddings `` `u_i = U(c_i)` ``, aggregate `k = 3` sampled neighbors with 1-layer mean-aggregator GraphSAGE `` `u_i' = GraphSAGE(u_i, G_k)` ``, then fuse by point-wise addition `` `m_i = w_i + u_i'` ``.
- **Knowledge-aware Hierarchical Label Attention (KHLA)**: encode each label name with KTE and mean-pool it as `` `R_l^i = mean(KTE(L_i))` ``, then propagate label representations over the hierarchy graph with 1-layer GCN `` `H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l))` ``.
- **Label-conditioned document representation**: compute document states `R_d = KTE(D)`, attention scores `` `W_att = softmax(H · tanh(W_o R_d^T))` ``, and class-enhanced vector `` `M_1 = mean(W_att R_d)` ``; combine this with a second randomly initialized label-attention branch `M_2` and BERT `[CLS]` as `` `R_f = W_f concat(M_1, M_2, H_[CLS]) + b_f` ``.
- **Knowledge-aware Contrastive Learning (KCL)**: define knowledge-driven weights from shared concepts `` `c_ij = |C_i ∩ C_j|` `` and `` `beta_ij = c_ij / sum_k c_ik` ``, then optimize weighted contrastive loss with temperatures `` `tau = 10` `` for concept sharing and `` `tau = 1` `` for hierarchy-driven sharing based on `` `l_ij = |Y_i ∩ Y_j|` ``.
- **Classifier and training**: predict labels with a two-layer MLP and sigmoid output, optimize `` `L = L_bce + lambda_c L_c + lambda_h L_h` `` with batch size `16`, Adam learning rate `` `2e-5` ``, early stopping patience `10`, `` `lambda_h = 1e-4` `` on both datasets, and `` `lambda_c = 1e-3` `` on BGC / `` `1e-2` `` on WOS.

## Key Results

- **Main benchmark**: K-HTC achieves `71.26/63.31/65.99/80.52` on BGC and `84.15/80.01/81.69/87.29` on WOS for Macro-Precision / Macro-Recall / Macro-F1 / Micro-F1.
- **Against strongest baseline**: versus HPT, K-HTC improves Macro-F1 by `+0.66` on BGC (`65.99` vs `65.33`) and `+0.59` on WOS (`81.69` vs `81.10`), while trailing HPT by only `0.20` Micro-F1 on BGC and leading it by `0.47` on WOS.
- **Ablation evidence**: removing KTE, KHLA, or KCL drops BGC Macro-F1 from `65.99` to `64.38`, `63.63`, and `64.02`; on WOS the corresponding Macro-F1 drops are to `80.57`, `80.04`, and `80.18`.
- **Depth effect**: level-wise analysis on the 4-level BGC hierarchy shows the performance gap between full K-HTC and its ablations widens as classification gets deeper and harder.
- **Statistical test**: repeated runs against HPT yield Macro-F1 p-values of `0.046` on BGC and `0.016` on WOS, supporting that K-HTC is usually stronger.

## Limitations

- The pipeline depends on concept recognition and concept-embedding pretraining, which adds preprocessing cost relative to plain BERT-based HTC.
- Matching-based concept recognition can introduce noisy knowledge; the bad-case analysis shows spurious links such as `Clark -> USA` can trigger wrong labels.
- KHLA assumes usable label names are available; datasets with only opaque label IDs would need proxy descriptions such as category keywords.
- Experiments are limited to two HTC datasets and one external KG, so transfer to other domains, hierarchies, or knowledge sources is not established.
- The paper focuses on effectiveness rather than efficiency and does not quantify model size or end-to-end computational budget beyond hardware setup.

## Concepts Extracted

- [[hierarchical-text-classification]]
- [[multi-label-text-classification]]
- [[knowledge-graph]]
- [[concept-recognition]]
- [[transe]]
- [[graphsage]]
- [[graph-convolutional-network]]
- [[contrastive-learning]]
- [[pretrained-language-model]]

## Entities Extracted

- [[ye-liu]]
- [[kai-zhang]]
- [[zhenya-huang]]
- [[kehang-wang]]
- [[yanghai-zhang]]
- [[qi-liu]]
- [[enhong-chen]]
- [[conceptnet]]
- [[bert]]
- [[blurb-genre-collection-en]]
- [[web-of-science]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
