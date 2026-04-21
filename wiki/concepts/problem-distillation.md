---
type: concept
title: Problem Distillation
slug: problem-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [problem distiller, task distillation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Problem Distillation** — a preprocessing step that extracts key variables, objectives, and constraints from a raw task and rewrites it into a normalized structure that is easier for downstream retrieval and reasoning.

## Key Points

- BoT computes a distilled problem representation as `x_d = LLM(φ(x))` using a dedicated meta prompt.
- The distiller separates extraction and comprehension from final reasoning, reducing the burden on a single prompt stage.
- The distilled output explicitly includes key information, restrictions, and a higher-level reformulation of the task.
- The paper argues that problem distillation matters most on tasks with implicit constraints and complex variable interactions, such as Game of 24 or Checkmate-in-One.
- Ablation results show measurable drops when the problem distiller is removed, especially on harder reasoning benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-buffer-2406-04271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-buffer-2406-04271]].
