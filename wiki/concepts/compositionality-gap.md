---
type: concept
title: Compositionality Gap
slug: compositionality-gap
date: 2026-04-20
updated: 2026-04-20
aliases: [compositional gap]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compositionality Gap** — the fraction of multi-step questions a model gets wrong among the subset for which it can answer all required sub-questions correctly in isolation.

## Key Points

- The paper introduces the term to separate factual recall from the ability to compose known facts into a correct 2-hop answer.
- On Compositional Celebrities, the gap stays around `40%` across GPT-3 and InstructGPT scale variants instead of shrinking with larger models.
- The metric is operationalized as `#(2-hop wrong and both 1-hop right) / #(both 1-hop right)`.
- Lower perplexity on the correct sub-question answers correlates with a smaller gap, suggesting confident fact recall improves composition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[press-2023-measuring-2210-03350]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[press-2023-measuring-2210-03350]].
