---
type: concept
title: Long-Term Memory
slug: long-term-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [long-term memory, 长期记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Term Memory** (长期记忆) — a persistent memory mechanism that stores salient information beyond the current context window so an AI system can reuse it across sessions and long interaction horizons.

## Key Points

- The paper frames long-term memory as the missing capability that prevents LLM agents from maintaining coherent multi-session dialogue.
- Mem0 implements long-term memory by extracting salient facts from each new message pair instead of repeatedly sending full history to the model.
- Retrieved long-term memories act as compact context for answering new questions, reducing token cost and latency.
- The reported gains are strongest on single-hop and multi-hop memory questions on LOCOMO, showing that selective persistence helps factual recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
