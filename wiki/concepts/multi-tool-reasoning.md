---
type: concept
title: Multi-Tool Reasoning
slug: multi-tool-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-tir, 多工具推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Tool Reasoning** (多工具推理) — reasoning in which a model may invoke multiple tool types and multiple tool calls within one problem-solving trajectory.

## Key Points

- The paper formalizes multi-tool reasoning as a sequence of reasoning states conditioned on previous tool outputs and prior reasoning content.
- It argues that single-tool training methods do not generalize reliably to problems requiring heterogeneous tool behavior.
- Tool-Light is explicitly designed for multi-tool settings and benchmarks both knowledge-intensive and mathematical reasoning tasks.
- The method addresses both redundant tool calls and cases where the model should have invoked a tool but failed to do so.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2026-effective-2509-23285]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2026-effective-2509-23285]].
