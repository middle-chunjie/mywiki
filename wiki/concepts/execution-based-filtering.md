---
type: concept
title: Execution-Based Filtering
slug: execution-based-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [execution filtering, 基于执行的过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution-Based Filtering** (基于执行的过滤) — a data-selection procedure that retains model-generated code outputs only when they pass executable tests in a controlled runtime environment.

## Key Points

- [[wei-2024-selfcodealign-2410-24198]] pairs each generated response with self-generated tests and executes them in a Docker sandbox before the example can enter the training set.
- For each instruction, the pipeline samples multiple candidate `(response, tests)` pairs and randomly keeps one passing response after filtering out failures.
- Ablations show strong benefit from execution filtering: `passes only` reaches `65.2` HumanEval+ `pass@1`, versus `61.6` for random selection and `57.9` for failures only.
- The paper treats execution feedback as the main mechanism that turns noisy self-generated data into usable instruction-tuning supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2024-selfcodealign-2410-24198]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2024-selfcodealign-2410-24198]].
