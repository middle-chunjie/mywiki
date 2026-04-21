---
type: concept
title: Knowledge Traceability
slug: knowledge-traceability
date: 2026-04-20
updated: 2026-04-20
aliases: [traceable reasoning, knowledge provenance, 知识可追溯性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Traceability** (知识可追溯性) — the property that a model's answer can be traced back to explicit evidence units and reasoning steps rather than only to opaque parametric memory.

## Key Points

- ToG represents reasoning as explicit KG paths, so a predicted answer can be inspected through the retrieved triples that supported it.
- The paper argues that this makes LLM reasoning more explainable than LLM-only prompting or looser retrieval-augmentation pipelines.
- In the case study, ToG can localize an incorrect answer to a suspicious triple along a specific reasoning path.
- Traceable paths are presented as a prerequisite for downstream expert review and KG maintenance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-thinkongraph-2307-07697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-thinkongraph-2307-07697]].
