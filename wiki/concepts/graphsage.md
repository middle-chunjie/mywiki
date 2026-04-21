---
type: concept
title: GraphSAGE
slug: graphsage
date: 2026-04-20
updated: 2026-04-20
aliases: [graph sample and aggregate]
tags: [graph-learning, ast]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**GraphSAGE** — an inductive graph neural network that updates node representations by aggregating neighbor features rather than learning only transductive node embeddings.

## Key Points

- [[guo-2022-modeling]] uses GraphSAGE as the AST encoder because it can aggregate neighborhood structure efficiently on syntax graphs.
- The node update is written as `h_i^k = W_1 e_i^{k-1} + W_2 Aggr({e_j^{k-1}})`, followed by `ReLU` and residual layer normalization across `6` AST layers.
- Triplet positional embeddings are injected before GraphSAGE so the encoder can distinguish nodes with the same label but different structural roles.
- The paper reports that GraphSAGE slightly outperforms GCN and GAT variants on Java, with the full model beating V-GCN and V-GAT in BLEU, METEOR, and ROUGE-L.
- Residual connections inside the GraphSAGE stack are critical: removing them causes severe underfitting on Java (`13.03` BLEU).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2022-modeling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2022-modeling]].
