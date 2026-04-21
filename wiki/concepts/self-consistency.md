---
type: concept
title: Self-Consistency
slug: self-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [CoT-SC, self consistency]
tags: [llm, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Consistency** (自洽性采样) — an inference strategy that samples multiple reasoning traces and aggregates them to choose an answer that is most consistent across samples.

## Key Points

- The paper uses CoT-SC as a baseline that selects the majority answer from `10` CoT samples.
- Self-consistency improves robustness over single CoT in some settings, but it still depends entirely on LLM-generated reasoning rather than external search.
- In the paper's experiments, CoT-SC remains inefficient relative to XoT because it scales LLM calls without adding an external planner or world model.
- CoT-SC performs especially poorly on 8-Puzzle and Pocket Cube, where long-horizon spatial planning is the main bottleneck.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
