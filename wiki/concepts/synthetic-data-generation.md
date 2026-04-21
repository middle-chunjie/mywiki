---
type: concept
title: Synthetic Data Generation
slug: synthetic-data-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [合成数据生成, synthetic benchmark generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Data Generation** (合成数据生成) — the automated creation of benchmark or training examples using models or procedural pipelines instead of direct human authoring.

## Key Points

- mtRAG introduces a companion synthetic benchmark, `mtRAG-S`, to study scalable alternatives to fully human-authored multi-turn RAG evaluation.
- The synthetic pipeline reuses the same corpora, question-type schema, multi-turn patterns, and answerability categories as the human benchmark.
- Mixtral `8x22B Instruct v0.1` is used to generate synthetic conversations, capped at `8` turns to reduce repetition and hallucination.
- The paper finds the synthetic benchmark is easier and less diverse than mtRAG, with fewer turns and far lower passage diversity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[katsis-2025-mtrag-2501-03468]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[katsis-2025-mtrag-2501-03468]].
