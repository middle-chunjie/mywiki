---
type: concept
title: Constrained Beam Search
slug: constrained-beam-search
date: 2026-04-20
updated: 2026-04-20
aliases: [lexically constrained decoding, 约束束搜索]
tags: [decoding, agents, tool-use]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Constrained Beam Search** (约束束搜索) — beam search decoding with explicit constraints that rule out invalid next tokens or sequences during generation.

## Key Points

- ToolWeaver uses constrained beam search at inference time so the model cannot emit invalid tool-code continuations.
- The constraints are active only during tool selection; free-form language generation for reasoning and answers remains unconstrained.
- This decoding strategy is necessary because hierarchical identifiers are multi-token sequences whose prefixes must stay valid.
- The paper treats constrained decoding as a practical mechanism for preserving the benefits of generative retrieval without allowing malformed tool IDs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
