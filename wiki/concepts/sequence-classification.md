---
type: concept
title: Sequence Classification
slug: sequence-classification
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sequence Classification** (序列分类) — assigning a label to an entire input sequence based on its overall content.

## Key Points

- SoT-R formulates routing as a binary sequence-classification problem: trigger SoT (`1`) or fall back to normal decoding (`0`).
- The paper fine-tunes `roberta-base` (`120M` parameters) as the classifier.
- Training uses the annotated LIMA dataset, plus Vicuna-80 and WizardLM annotations for evaluation.
- The classifier is optimized with `AdamW`, Tversky loss (`alpha = 0.7`, `beta = 0.3`), and label smoothing (`epsilon = 0.2`).
- The paper emphasizes low false-positive rate because incorrectly triggering SoT harms answer quality more than missing a speed-up opportunity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ning-2024-skeletonofthought-2307-15337]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ning-2024-skeletonofthought-2307-15337]].
