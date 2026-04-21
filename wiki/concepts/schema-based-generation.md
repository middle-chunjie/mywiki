---
type: concept
title: Schema-Based Generation
slug: schema-based-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [schema-driven generation, schema-guided data generation]
tags: [data-generation, rag, evaluation, nlp]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Schema-Based Generation** (基于模式的生成) — a data synthesis paradigm in which an abstract structural template (schema) is first extracted from seed documents, then populated with concrete configurations to generate diverse, factually consistent domain-specific documents and evaluation data.

## Key Points

- A schema captures the structural skeleton of domain knowledge (e.g., financial report sections, legal case fields, medical record categories) without containing actual data, enabling reuse across many instances.
- Schema construction: LLM-driven extraction from seed documents followed by iterative human refinement that normalizes keys from instance-specific strings to generic field names (e.g., "Major Asset Acquisition" → `{event, time, description, impact}`), improving universal code handling and extensibility.
- Configuration generation fills schema fields via a hybrid rule-based + LLM approach: rule-based assignment ensures factual consistency for structured fields; LLMs handle complex narrative fields requiring creativity.
- Separating schema from instance data reduces hallucination risk during document generation and allows fine-grained control over dataset diversity and coverage.
- Applied in RAGEval to generate the DragonBall benchmark, yielding documents that outperform zero-shot and one-shot baselines in clarity, safety, conformity, and richness across all scenarios.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
