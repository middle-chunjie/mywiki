---
type: concept
title: Code-Pretrained Model
slug: code-pretrained-model
date: 2026-04-20
updated: 2026-04-20
aliases: [code pretrained encoder, 代码预训练模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code-Pretrained Model** (代码预训练模型) — a model pretrained on programming-language data so that its representations transfer better to downstream code tasks.

## Key Points

- The paper uses CodeBERT and GraphCodeBERT as representative code-pretrained encoders, with RoBERTa as a natural-language-pretrained control.
- It argues that stronger downstream data adaptation can still unlock large gains even when code-pretraining is already available.
- `CodeBERT + DA + CL` outperforms the GraphCodeBERT baseline on POJ-104 MAP and BigCloneBench F1, showing that pretraining alone does not determine final task quality.
- `GraphCodeBERT + DA + CL` reaches the best reported code-search score in the paper at `0.720` MRR, indicating that code-pretrained models still benefit from augmentation and curriculum design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2022-bridging-2112-02268]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2022-bridging-2112-02268]].
