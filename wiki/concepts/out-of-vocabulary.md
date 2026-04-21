---
type: concept
title: Out-of-Vocabulary
slug: out-of-vocabulary
date: 2026-04-20
updated: 2026-04-20
aliases: [OOV, unseen token, 未登录词]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Out-of-Vocabulary** (未登录词) — the condition where an input token is absent from the model's fixed vocabulary and must be mapped to an unknown or fallback representation.

## Key Points

- The paper uses OOV behavior to explain why Deepcom scales worse than other code summarization models.
- Deepcom's OOV ratio remains high even with larger training sets, e.g. `91.90%` to `88.32%` on FCM size variants.
- For the other models, OOV ratios fall much more sharply as dataset size grows, from `63.36%` to `48.60%` on the same variants.
- Identifier splitting is highlighted as one practical way to reduce OOV pressure in code summarization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
