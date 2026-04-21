---
type: concept
title: Topic Switching
slug: topic-switching
date: 2026-04-20
updated: 2026-04-20
aliases: [topic shift, 话题切换]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Topic Switching** (话题切换) — a conversational behavior in which successive turns remain related at a session level but shift to different subtopics or aspects, making naive use of earlier turns increasingly noisy.

## Key Points

- The paper treats topic switching as a main source of retrieval noise in long conversational search sessions.
- TopiOCQA is highlighted as a benchmark with stronger topic-switch phenomena than QReCC, which explains HAConvDR's larger gains there.
- PRJ turns topic switching from a problem into a supervision source by distinguishing which earlier turns still help the current query.
- The authors argue that realistic search conversations are often on related but different topics, so topic-switch robustness matters in practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-historyaware-2401-16659]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-historyaware-2401-16659]].
