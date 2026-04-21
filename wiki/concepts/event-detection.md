---
type: concept
title: Event Detection
slug: event-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [ED, event detection, 事件检测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Event Detection** (事件检测) — the task of identifying event trigger words or spans in text and classifying them into predefined event types.

## Key Points

- [[ma-2023-large-2303-08559]] benchmarks ED on ACE05, MAVEN, and ERE, covering label spaces from `33` to `168`.
- The paper shows that LLMs struggle more as event label spaces become fine-grained, with MAVEN highlighted as especially difficult.
- Direct LLM prompting underperforms supervised SLMs on ED, but hard-sample reranking still gives sizable local gains.
- The adaptive pipeline improves ACE05 `20-shot` ED performance from `65.8` to `67.3` with InstructGPT and to `67.8` with an ensembled SLM filter plus GPT-4.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
