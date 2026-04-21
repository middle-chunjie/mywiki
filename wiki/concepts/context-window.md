---
type: concept
title: Context Window
slug: context-window
date: 2026-04-20
updated: 2026-04-20
aliases: [context length, 上下文窗口]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Window** (上下文窗口) — the maximum amount of preceding input a model can condition on during inference, usually measured in tokens.

## Key Points

- [[li-2023-compressing]] treats fixed context-window size as a core bottleneck for long-document summarization and multi-turn conversation.
- The paper argues that input-side compression can effectively increase usable context by discarding low-information spans before inference.
- In the reported experiments, examples longer than `2048` tokens are excluded because some evaluated LLMs cannot process them reliably.
- The method is motivated partly by avoiding context truncation when raw inputs would exceed the model's limit.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
