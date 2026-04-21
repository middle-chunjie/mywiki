---
type: concept
title: Span-Level Memory Processing
slug: span-level-memory-processing
date: 2026-04-20
updated: 2026-04-20
aliases: [span-level processing, chunk-level memory, 段落级记忆处理]
tags: [agents, memory, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Span-Level Memory Processing** (段落级记忆处理) — an approach to agent memory construction that groups interaction traces into contiguous text spans and processes each span as a unit, rather than updating memory turn by turn.

## Key Points

- MemSkill splits each interaction trace (e.g., a dialogue) into contiguous spans of a fixed token budget (default `512` tokens) and processes them sequentially with a controller-executor pair.
- Span-level granularity reduces the total number of LLM executor calls compared to per-turn processing, improving efficiency on long interaction histories.
- The controller's skill selection operates at the span level, conditioning on both the current text span and retrieved memories, enabling larger-context extraction behaviors that per-turn approaches cannot express.
- At evaluation time the same span-level formulation is applied unchanged from training, providing consistent behavior across contexts and improving scalability.
- The approach is not tied to a fixed unit and can accommodate different chunk sizes; the paper reports experiments with `50`, `100`, and `200` concatenated documents for HotpotQA transfer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2026-memskill-2602-02474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2026-memskill-2602-02474]].
