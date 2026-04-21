---
type: concept
title: Procedural Generation
slug: procedural-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [long procedural generation, 程序化生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Procedural Generation** (程序化生成) — a generation setting in which a model must follow an explicit multi-step procedure to emit a sequence of structured intermediate outputs.

## Key Points

- LONGPROC defines gold outputs as entry sequences `Y^* = {y_1^*, ..., y_n^*}` where each next entry is deterministically derived from the input and prior entries.
- The benchmark provides both detailed instructions and few-shot traces so that the model is asked to execute a procedure rather than infer one implicitly.
- The paper argues that procedural generation naturally stresses dispersed evidence integration, state maintenance, and long-form output consistency.
- Search-heavy tasks such as Countdown and Travel Planning instantiate procedural generation through explicit DFS traces with backtracking.
- The benchmark uses procedural generation to make long outputs objectively scorable instead of relying on subjective open-ended judgments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
