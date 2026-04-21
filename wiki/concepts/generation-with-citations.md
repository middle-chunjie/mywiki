---
type: concept
title: Generation with Citations
slug: generation-with-citations
date: 2026-04-20
updated: 2026-04-20
aliases: [citation-grounded-generation, cited-generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generation with Citations** (带引用生成) — a long-context generation setting where a model must answer using provided evidence and attach correct citation markers or passage references to support each claim.

## Key Points

- HELMET includes generation with citations as one of its seven core benchmark categories because answer correctness and citation correctness are separable capabilities.
- The paper uses ALCE subsets ASQA and QAMPARI, with models required to generate long-form answers while citing passage IDs at sentence level.
- Evaluation averages answer correctness and citation quality, making the task harder than pure retrieval or short-form QA.
- The paper reports that citation-heavy tasks are among the clearest separators between strong closed-source and weaker open-source long-context models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2024-helmet-2410-02694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2024-helmet-2410-02694]].
