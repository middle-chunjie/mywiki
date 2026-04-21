---
type: concept
title: Non-Parametric Knowledge
slug: non-parametric-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [external knowledge, 非参数化知识]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Non-Parametric Knowledge** (非参数化知识) — knowledge stored outside model parameters, typically in an external document corpus or datastore that can be retrieved at inference time.

## Key Points

- In ITRG, non-parametric knowledge comes from retrieved Wikipedia paragraphs rather than from the model's internal memory.
- The paper argues that retrieved evidence can complement incomplete or disconnected parametric knowledge in the LLM.
- Refresh and refine both consume top-`k` retrieved passages as non-parametric context for document generation.
- The framework explicitly studies how non-parametric evidence and generated parametric knowledge can reinforce each other across iterations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2023-retrievalgeneration-2310-05149]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2023-retrievalgeneration-2310-05149]].
