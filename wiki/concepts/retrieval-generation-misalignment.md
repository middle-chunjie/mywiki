---
type: concept
title: Retrieval-Generation Misalignment
slug: retrieval-generation-misalignment
date: 2026-04-20
updated: 2026-04-20
aliases: [检索-生成失配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Generation Misalignment** (检索-生成失配) — a failure mode in retrieve-then-generate systems where improvements in retrieval quality do not reliably translate into better generated outputs.

## Key Points

- The paper identifies this problem in end-to-end task-oriented dialogue, where generators often fail to exploit better retrieved entities.
- The authors show that Q-TOD, FiD, and ChatGPT can exhibit weak or even counterintuitive correlations between retriever quality and Entity F1.
- The proposed explanation is that retrieved entities are highly homogeneous, so the generator lacks strong inductive bias to distinguish among them.
- MK-TOD mitigates the misalignment by training the retriever with MML and exposing the generator to retrieval-related meta knowledge.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-retrievalgeneration]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-retrievalgeneration]].
