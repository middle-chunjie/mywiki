---
type: concept
title: Tree-LSTM
slug: tree-lstm
date: 2026-04-20
updated: 2026-04-20
aliases: [Tree LSTM, Tree-Structured LSTM, 树结构长短期记忆网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree-LSTM** (树结构长短期记忆网络) — an LSTM variant that propagates hidden and cell states over tree-structured inputs instead of a single linear sequence, enabling hierarchical composition.

## Key Points

- In this paper, Tree-LSTM is used to encode binarized ASTs extracted from C functions.
- Each AST node combines left and right child states through `([h_L; h_R], [c_L; c_R])`, while missing children are replaced with zero states.
- The hidden state of the root node becomes the AST modality representation used in semantic code retrieval.
- The AST-only model with attention reaches `R@1 = 0.288` and `MRR = 0.379`, which is stronger than CFG-only but still weaker than tri-modal fusion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wan-2019-multimodal-1909-13516]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wan-2019-multimodal-1909-13516]].
