---
type: concept
title: Demonstration Selection
slug: demonstration-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [example selection, demonstration retrieval, 示例选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Demonstration Selection** (示例选择) — the process of choosing in-context examples for a prompt so that they best support a language model's performance on the target query.

## Key Points

- [[geng-2024-large]] compares random demonstrations against retrieval-based demonstrations for multi-intent code summarization.
- The paper studies both token-level and semantic retrieval strategies, each constrained to examples whose comments match the requested intent.
- Better demonstrations matter more than simply adding more examples: selected `1`-shot prompts can outperform randomly chosen `10`-shot prompts.
- Semantic selection with `10` shots yields the strongest average gains in the reported experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[geng-2024-large]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[geng-2024-large]].
