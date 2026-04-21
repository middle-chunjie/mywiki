---
type: concept
title: Relation Extraction
slug: relation-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [RE, relation extraction, 关系抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relation Extraction** (关系抽取) — the task of identifying semantic relations between entities mentioned in text and assigning them predefined relation labels.

## Key Points

- [[ma-2023-large-2303-08559]] evaluates RE on TACRED and TACREV under settings from `1-shot` to `100-shot`.
- The paper finds that direct LLM prompting can be competitive only in very low-shot regimes, but supervised SLMs dominate once more labeled samples are available.
- RE is one of the tasks where the adaptive reranking pipeline works well, with InstructGPT reranking lifting TACREV `100-shot` performance from `72.6` to `75.4`.
- The reranking formulation casts each RE decision as a multiple-choice problem over top candidate relations, which is easier for LLMs than open-ended extraction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
