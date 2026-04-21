---
type: concept
title: Data Synthesis
slug: data-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic data generation, 数据合成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Synthesis** (数据合成) — the automated construction of training or evaluation data by programmatically generating tasks, annotations, or trajectories from existing knowledge sources and rules.

## Key Points

- InfoSeek is presented as an open data-synthesis pipeline for deep-research QA rather than as a manually curated benchmark.
- The pipeline mines entities, hyperlinks, and factual claims from webpages and Wikipedia to build structured reasoning trees before generating natural-language questions.
- Structural complexity is controlled through explicit tree actions instead of only through prompt heuristics or surface difficulty measures.
- The synthesis process keeps intermediate metadata such as tree structure, retrieval labels, and reasoning trajectories, which later supports SFT and RL.
- Quality assurance removes easy, ambiguous, or unverifiable samples to improve the usefulness of the synthesized data.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2025-open-2509-00375]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2025-open-2509-00375]].
