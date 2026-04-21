---
type: source
subtype: paper
title: Rethinking Positional Encoding in Tree Transformer for Code Representation
slug: peng-nd-rethinking
date: 2026-04-20
language: en
tags: [code, transformer, ast, positional-encoding, code-intelligence]
processed: true
raw_file: raw/papers/peng-nd-rethinking/paper.pdf
raw_md: raw/papers/peng-nd-rethinking/paper.md
bibtex_file: raw/papers/peng-nd-rethinking/paper.bib
possibly_outdated: true
authors:
  - Han Peng
  - Ge Li
  - Yunfei Zhao
  - Zhi Jin
year: 2022
venue: EMNLP 2022
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2022.emnlp-main.210
url: https://aclanthology.org/2022.emnlp-main.210/
citation_key: pengndrethinking
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper revisits how Transformers should encode tree structure for source-code representation. Instead of describing each AST node only by sibling order or shortest-path distance, it defines a recursive 2D coordinate sequence whose elements record both a node's sibling index and its parent's fan-out. The model then injects this structure into self-attention through two complementary terms: a global absolute bias derived from the full coordinate list and a local relative bias defined only on adjacent parent-child pairs. Evaluated on Python150k and JavaScript150k code summarization and code completion, the method consistently beats vanilla Transformer, Shaw-style relative encoding, and prior tree Transformers, while ablations show that both coordinate dimensions and both bias types contribute measurable gains.

## Problem & Motivation

Prior tree Transformers for code either encode node positions with incomplete structural descriptions or collapse structure into distances that cannot uniquely recover the tree. The paper argues that source code ASTs require a lossless or near-lossless positional description so that Transformer attention can exploit sibling order, branching factor, and ancestry together. It also observes that existing soft structural bias is usually introduced either locally or globally, without clarifying whether the two paradigms are complementary. The goal is therefore to design a tree positional encoding that preserves more tree information and to test whether combining global absolute bias with local relative bias improves code understanding.

## Method

- **Input representation**: ASTs are linearized in depth-first order; each node is represented by type and value embeddings, with a special `<empty>` value for non-leaf nodes.
- **Recursive 2D tree description**: each node position is defined as `` `F(x) = F(f(x)) + {(x_i, x_j)}` `` for non-root nodes and `` `F(root) = {(1, 1)}` ``. Here `` `x_i` `` is sibling order and `` `x_j` `` is the total number of children of the parent, so each node gets a coordinate list rather than a single index.
- **Coordinate embedding sequence**: the coordinate list is embedded as `` `H_i = [h(i^1), h(i^2), ..., h(i^n)]` ``, where `` `n` `` is node depth. The authors map each 2D coordinate to a scalar index in an embedding table rather than embedding the two dimensions separately.
- **Global absolute tree bias**: absolute position is encoded as `` `a_i = LN(Linear(Concat(H_i)))` ``. Short coordinate lists are zero-padded and long ones are truncated to the configured maximum tree depth before concatenation.
- **Local relative tree bias**: only adjacent parent-child pairs get non-zero relative vectors, using `` `r_ij = LN(Linear(sum_h H_i - sum_h H_j))` `` when `` `f(i) = j` `` or `` `f(j) = i` ``, and `` `r_ij = 0` `` otherwise. This keeps relative encoding tractable while preserving one-hop directionality.
- **Attention decomposition**: the standard token attention score is `` `alpha_ij = (x_i W^Q)(x_j W^K)^T` ``. The model adds a global absolute term `` `beta_ij = (a_i W_a^Q)(a_j W_a^K)^T` `` and a local relative term `` `gamma_ij = (x_i W^Q)(r_ij W_r^K)^T + (r_ji W_r^Q)(x_j W^K)^T` ``.
- **Unified tree-aware attention**: final attention is computed as `` `A_ij = (alpha_ij + beta_ij + gamma_ij) / sqrt(2)` `` before softmax. The paper explicitly adopts disentangled attention so positional signals are modeled inside attention rather than by adding position embeddings to token embeddings.
- **Backbone and optimization**: all models use `` `6` `` layers, `` `8` `` heads, hidden size `` `D = 512` ``, and FFN size `` `D_FF = 2048` ``. Training uses Adam with learning rate `` `1e-4` `` and batch size `` `32` ``; completion uses cosine decay with `` `2k` `` warmup steps, while summarization uses gradient clipping `` `5` ``.
- **Tree-specific hyperparameters**: maximum children per node is `` `16` ``, giving a coordinate table size of `` `16(16+1)/2 = 136` ``. Coordinate embedding size is `` `32/32` `` for summarization and `` `32/16` `` for completion in Python/JavaScript; max tree depth is `` `16/16` `` for summarization and `` `16/32` `` for completion. The added positional machinery introduces only about `` `1.3M` `` parameters.

## Key Results

- **Code completion, Python150k**: the full model reaches `MRR(type) 91.58`, `MRR(value) 55.62`, `ACC(type) 86.15`, `ACC(value) 51.19`, and `ACC(all) 63.19`, improving over the strongest baseline from Kim et al. (2020), which reports `62.15` on `ACC(all)`.
- **Code completion, JavaScript150k**: the model achieves `MRR(type) 91.63`, `MRR(value) 64.08`, `ACC(type) 85.91`, `ACC(value) 58.99`, and `ACC(all) 66.48`, versus `65.42` `ACC(all)` for Kim et al. (2020).
- **Code summarization, Python150k**: the model scores `BLEU 57.28`, `F1 36.00`, and `Acc 19.97`, outperforming Kim et al. (2020) at `57.02 / 35.55 / 19.29`.
- **Code summarization, JavaScript150k**: it reaches `BLEU 57.72`, `F1 25.03`, and `Acc 13.21`, improving on the best baseline's `57.33 / 23.47 / 12.07`.
- **Ablation on 2D coordinates**: removing the first coordinate dimension drops Python completion `ACC(all)` from `63.19` to `59.36`; removing the second drops it to `62.04`. On JavaScript completion, the same ablations reduce `ACC(all)` to `64.02` and `65.47`.
- **Ablation on bias types**: removing global absolute bias reduces completion `ACC(all)` to `62.70` on Python and `66.34` on JavaScript; removing local relative bias yields `62.75` and `65.67`, showing that both components help and the fused model is best.

## Limitations

- The experiments focus on function-level or similarly short code units; the paper does not study scalability to very long source files or whole-program contexts.
- The model only encodes AST structure, not data-flow, control-flow, or extra code-token context, so it may miss important semantic relations for some program-analysis tasks.
- Relative structure is modeled only for one-hop parent-child adjacency; the method deliberately avoids explicit global relative path modeling.
- Baselines exclude structure-and-context joint models, so the comparison isolates tree encoding quality rather than the strongest multimodal code representation systems available by 2022.
- The authors retrain all models three times for precise comparison, which they note increases computational cost and carbon emissions.

## Concepts Extracted

- [[transformer]]
- [[abstract-syntax-tree]]
- [[code-representation-learning]]
- [[code-summarization]]
- [[code-completion]]
- [[absolute-positional-encoding]]
- [[relative-positional-encoding]]
- [[disentangled-attention]]
- [[tree-positional-encoding]]

## Entities Extracted

- [[han-peng]]
- [[ge-li]]
- [[yunfei-zhao]]
- [[zhi-jin]]
- [[peking-university]]
- [[python150k]]
- [[javascript150k]]
- [[github]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
