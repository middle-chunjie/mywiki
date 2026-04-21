---
type: concept
title: Chain-of-Thought
slug: chain-of-thought
date: 2026-04-20
updated: 2026-04-20
aliases: [CoT, chain of thought]
tags: [llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Thought** (思维链) — an explicit sequence of intermediate reasoning steps generated before a model emits its final answer.

## Key Points

- The paper assumes that longer CoT traces usually increase reasoning accuracy by allocating more test-time compute.
- LCPO controls CoT length directly through prompt conditioning and RL rewards rather than through post-hoc truncation.
- L1 shows that short CoTs can preserve many qualitative reasoning patterns of long CoTs while using far fewer tokens.
- Keyword analysis suggests longer CoTs increase self-verification and conclusion-drawing behaviors.
- The ratio of thinking tokens to solution tokens stays relatively stable across different requested budgets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
