---
type: concept
title: Zero-Shot Adaptation
slug: zero-shot-adaptation
date: 2026-04-20
updated: 2026-04-20
aliases: [unsupervised adaptation, 零样本自适应]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Adaptation** (零样本自适应) — adapting a source-trained model to a target domain using unlabeled target data or structural heuristics, without target-domain question-answer annotations.

## Key Points

- The paper varies context, answer, and question distributions one at a time to study which zero-shot interventions help ODQA transfer.
- For answer shift, uniform sampling across coarse entity types improves BioASQ retriever `Acc@100` from `45.35` to `50.02`, outperforming random and most-frequent sampling.
- For question shift, both standard question generation and [[cloze-question]] augmentation improve several datasets, though the gains remain limited on full-shift cases.
- Average zero-shot gains are much larger on label-shift and covariate-shift datasets than on full-shift datasets, showing that zero-shot adaptation is not uniformly effective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
