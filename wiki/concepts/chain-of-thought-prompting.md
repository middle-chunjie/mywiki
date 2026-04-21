---
type: concept
title: Chain-of-Thought Prompting
slug: chain-of-thought-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [chain of thought prompting, CoT prompting, 思维链提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Thought Prompting** (思维链提示) — a prompting technique that provides intermediate reasoning steps so a language model can better solve multi-step tasks.

## Key Points

- LOFT attaches chain-of-thought rationales to few-shot demonstrations rather than only asking for final answers.
- The paper reports that chain-of-thought is most helpful on tasks requiring multi-hop retrieval, RAG, and compositional reasoning.
- In the 128k ablations, removing chain-of-thought reduces the average score from `0.64` to `0.61`.
- The benchmark treats reasoning traces as part of prompt design rather than as a separate training procedure.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-can-2406-13121]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-can-2406-13121]].
