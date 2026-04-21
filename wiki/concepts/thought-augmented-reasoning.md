---
type: concept
title: Thought-Augmented Reasoning
slug: thought-augmented-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [thought augmented reasoning, BoT reasoning]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Thought-Augmented Reasoning** — a reasoning paradigm in which an LLM retrieves reusable high-level thought templates from prior solved tasks and instantiates them for a new problem instead of reasoning from scratch.

## Key Points

- BoT frames thought-augmented reasoning as a pipeline of problem distillation, template retrieval, template instantiation, and template distillation back into memory.
- The method uses abstract reusable guidance rather than task-specific exemplars or exhaustive search over many candidate traces.
- The framework is designed to improve accuracy, inference efficiency, and robustness at the same time.
- Retrieved templates can be instantiated as prompt-based, procedure-based, or programming-based reasoning structures.
- The paper reports gains over both single-query prompting and multi-query search baselines on `10` reasoning tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-buffer-2406-04271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-buffer-2406-04271]].
