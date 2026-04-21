---
type: concept
title: Synthetic Programming Elicitation
slug: synthetic-programming-elicitation
date: 2026-04-20
updated: 2026-04-20
aliases: [SPE, synthetic programming elicitation]
tags: [llm, code-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Programming Elicitation** - a study-driven process that probes how LLMs naturally write code in order to choose an intermediate language interface that better aligns generation with a target programming task.

## Key Points

- The paper structures SPE as setup, execution, and analysis over prompts and model outputs rather than hand-designing the interface a priori.
- For UCLID5, the study uses `33` tasks and `2` LLM subjects to compare direct target-language generation against imaginary host-language API generation.
- Direct UCLID5 prompting fails to produce any parsing outputs across `660` attempts.
- Python prompts yield more stable output patterns, especially an object-oriented style that mirrors transition-system structure.
- The study output determines both the parent language `P` and the compilable child subset `C`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
