---
type: concept
title: Self-Refinement
slug: self-refinement
date: 2026-04-20
updated: 2026-04-20
aliases: [self refine, self-refine, 自我修正]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Refinement** (自我修正) — an iterative process where a model analyzes its own incorrect solution and revises it to improve task performance.

## Key Points

- [[ding-2024-semcoder-2406-01006]] evaluates self-refinement as a static debugging task: generate code, explain the bug verbally, then patch it.
- The paper introduces `PYX-R`, a dataset of buggy programs, rubber-duck debugging rationales, and validated fixes for training this behavior.
- SEMCODER-S reaches `84.8` / `79.3` on HumanEval / HumanEval+ and `86.8` / `74.3` on MBPP / MBPP+ in zero-shot self-refinement.
- Fine-tuning on PYX-R further improves several baselines, though gains for stronger models are limited by occasional noise in debugging rationales.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-semcoder-2406-01006]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-semcoder-2406-01006]].
