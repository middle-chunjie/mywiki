---
type: concept
title: Document Masking
slug: document-masking
date: 2026-04-20
updated: 2026-04-20
aliases: [document masks, cross-document attention masking, 文档掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Masking** (文档掩码) — an attention constraint that prevents tokens from attending across packed document boundaries within the same training sequence.

## Key Points

- The paper shows that enabling document masks during continued long-context training improves both long-context and short-context performance relative to full cross-document attention.
- In the reported ablation, document masking improves the average long-context score from `53.6` to `54.6`.
- The final ProLong recipe keeps cross-document attention disabled for packed short documents.
- Document masking also interacts with efficiency, because variable-length attention over separate documents can improve throughput compared with naively masking one full packed sequence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-how-2410-02660]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-how-2410-02660]].
