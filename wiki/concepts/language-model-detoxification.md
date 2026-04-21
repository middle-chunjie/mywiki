---
type: concept
title: Language Model Detoxification
slug: language-model-detoxification
date: 2026-04-20
updated: 2026-04-20
aliases: [language model detoxification, 语言模型去毒, detoxification]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Model Detoxification** (语言模型去毒) — the task of reducing toxic, harmful, or offensive generations from a language model while preserving fluency and diversity.

## Key Points

- LM-Switch uses toxicity labels to learn a single steering matrix instead of retraining the whole backbone model.
- The paper trains on the Jigsaw Unintended Bias in Toxicity Classification dataset and evaluates on `10K` non-toxic RealToxicityPrompts prompts.
- At generation time, detoxification uses a positive switch value of `5ε_0`, balancing lower toxicity with manageable perplexity increase.
- The reported gains are strong on toxicity metrics while keeping diversity near the baseline range, making detoxification the clearest empirical win in the paper.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
