---
type: concept
title: Attention Mechanism
slug: attention-mechanism
date: 2026-04-20
updated: 2026-04-20
aliases: [Attention, 注意力机制]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Attention Mechanism** (注意力机制) — a learned weighting function that emphasizes the most relevant parts of an input or memory when forming representations or predictions.

## Key Points

- MMAN assigns separate attention weights to tokens, AST nodes, and CFG nodes instead of uniformly pooling each modality.
- Token and AST attention use softmax-normalized scores over transformed hidden states and learned context vectors, while CFG attention uses a sigmoid weighting function.
- The paper reports that sigmoid-based CFG attention performs better than softmax for the CFG modality.
- Adding attention raises the tri-modal model from `MRR = 0.343` to `0.452` and from `R@1 = 0.243` to `0.347`.
- Attention visualizations highlight function names, AST leaf and operator nodes, and salient CFG statements, providing interpretability for retrieval decisions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wan-2019-multimodal-1909-13516]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wan-2019-multimodal-1909-13516]].
