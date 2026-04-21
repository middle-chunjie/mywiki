---
type: concept
title: Language Model Conditioning
slug: language-model-conditioning
date: 2026-04-20
updated: 2026-04-20
aliases: [language model conditioning, 语言模型条件控制, language model control]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Model Conditioning** (语言模型条件控制) — the process of modifying a language model's generation distribution so outputs reflect a desired condition such as style, sentiment, toxicity, or stance.

## Key Points

- LM-Switch frames conditioning as a transformation of the frozen model's word embedding space rather than full-model fine-tuning.
- The paper uses a scalar `ε` to express both polarity and strength of the condition, allowing the same learned switch to serve multiple intensities.
- Positive and negative labeled texts are handled symmetrically by training with `M(+εW)` and `M(-εW)`.
- The method is intended to preserve prompt continuation ability and avoid the overhead of auxiliary decoding-time classifiers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
