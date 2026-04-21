---
type: concept
title: Temporal Misalignment
slug: temporal-misalignment
date: 2026-04-20
updated: 2026-04-20
aliases: [temporal drift, time mismatch]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Temporal Misalignment** (时间错位) — the mismatch between the time period reflected in a model's stored knowledge or evidence and the time period required by the evaluation or user query.

## Key Points

- REALTIME QA is explicitly designed to expose failures caused by models relying on outdated pretraining or stale retrieval corpora.
- Closed-book T5 and GPT-3 cannot directly incorporate post-training events, making them weaker baselines for present-time questions.
- DPR retrieval from the December 2018 Wikipedia dump is a controlled way to measure how much old evidence hurts current QA performance.
- The paper links temporal misalignment to both retrieval lag and answer-generation errors that remain outdated even after partial evidence updates.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kasai-2024-realtime-2207-13332]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kasai-2024-realtime-2207-13332]].
