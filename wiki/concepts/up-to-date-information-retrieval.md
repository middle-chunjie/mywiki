---
type: concept
title: Up-to-Date Information Retrieval
slug: up-to-date-information-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [fresh retrieval, current-information retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Up-to-Date Information Retrieval** (最新信息检索) — retrieval aimed at finding the most current evidence needed to answer questions about changing facts or newly occurring events.

## Key Points

- REALTIME QA uses Google Custom Search over recent news as its main source of fresh evidence for present-time QA.
- Open-book GPT-3 with current web retrieval substantially outperforms both closed-book baselines and retrieval from the 2018 Wikipedia dump.
- When Google Custom Search returns fewer than `5` documents, the benchmark backfills evidence with DPR results to maintain a fixed document budget.
- The paper's error analysis shows that retrieval failure, not reader failure, is the dominant obstacle to timely QA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kasai-2024-realtime-2207-13332]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kasai-2024-realtime-2207-13332]].
