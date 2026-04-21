---
type: concept
title: Rule-Based Evaluation
slug: rule-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [deterministic evaluation, 基于规则的评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rule-Based Evaluation** (基于规则的评测) — an evaluation protocol that scores model outputs using deterministic rules or executable checks instead of subjective human or LLM judgments.

## Key Points

- LONGPROC is designed so every task has a structured output format and a deterministic scoring rule.
- HTML to TSV is scored with row-level F1, Pseudocode to Code with unit-test execution, and Path Traversal / ToM Tracking with exact match.
- Countdown and Travel Planning are scored by verifying final-solution correctness and constraint satisfaction rather than by judging the entire reasoning trace.
- The paper positions rule-based evaluation as a key advantage for long-output benchmarks, where free-form judgments are noisy and expensive.
- The authors use output tags and normalization rules to tolerate minor formatting variation while still preserving deterministic scoring.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
