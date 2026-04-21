---
type: concept
title: Tool-Integrated Reasoning
slug: tool-integrated-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [TIR, 工具集成推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool-Integrated Reasoning** (工具集成推理) — a reasoning paradigm in which a language model interleaves internal chain-of-thought generation with calls to external tools such as search or code execution.

## Key Points

- The paper treats effective TIR as a joint problem of answer correctness, necessary tool use, and efficient reasoning after tool results arrive.
- It identifies three common TIR failure modes: insufficient tool use, excessive tool use, and overthinking after observing tool outputs.
- Tool-Light improves TIR through both data construction and post-training rather than through a handcrafted RL reward for each tool.
- The concrete tool setting in this paper includes a code interpreter and a search tool.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
