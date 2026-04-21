---
type: concept
title: Abstention
slug: abstention
date: 2026-04-20
updated: 2026-04-20
aliases: [giving up, refusal to answer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Abstention** — a failure mode where an agent declines to produce an answer instead of using its remaining search capacity to continue gathering evidence.

## Key Points

- [[yen-2025-lost-2510-18939]] treats abstention as undesirable for search agents because the agent could often continue searching.
- The analysis pipeline uses an LLM judge to determine whether the system actually attempted to answer.
- Slim lowers hallucination substantially but still records `27.7` abstention in the matched-budget comparison.
- HF-ODR is even more prone to abstention at `32.3`, showing that complex multi-agent workflows can still fail to convert search into answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
