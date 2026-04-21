---
type: concept
title: Partitioning-Based Graph Neural Network
slug: partitioning-based-graph-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [PGNN, partitioning GNN, 基于划分的图神经网络]
tags: [graph-learning, program-analysis, code-understanding]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Partitioning-Based Graph Neural Network** (基于划分的图神经网络) — a GNN architecture that divides an input graph (e.g., a program's S-AST) into a sequence of subgraphs aligned with code statements, encodes each subgraph independently with GGNN, then feeds the resulting subgraph embedding sequence into an LSTM to capture local-to-global hierarchical structure, mimicking the human divide-and-conquer reading strategy.

## Key Points

- Motivated by cognitive studies showing that programmers reason about code statement-by-statement in a bottom-up, local-to-global manner.
- Partitioning algorithm: recover the underlying tree structure from S-AST, iterate root-subtrees left-to-right accumulating node count, seal a subgraph when count ≥ threshold `λ` (adding cross-subgraph variable-context nodes), merge undersized last subgraphs; empirically `λ ≈ S-AST_size / 5` works well.
- Each subgraph is encoded by GGNN: `m_i^{l+1} = Σ_{j∈N_i} MLP(h_j^l, e_ij)`, `h_i^{l+1} = GRU(m_i^{l+1}, h_i^l)`, with mean READOUT to get a fixed-size subgraph embedding.
- A skip connection applies GGNN on the full S-AST graph, whose output is concatenated with the LSTM final hidden state before a fully-connected projection, preserving global structural context.
- Ablation against GNN-EK (no partitioning) shows PGNN partitioning contributes +0.9 BLEU-4 on CSN and +0.009 F1 on BCB-F.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2022-neural]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2022-neural]].
