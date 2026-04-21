---
type: concept
title: Coding Style Transformation
slug: coding-style-transformation
date: 2026-04-20
updated: 2026-04-20
aliases: [Style Transformation, 代码风格变换]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Coding Style Transformation** (代码风格变换) — a rewrite of source code that changes stylistic attributes, often while preserving program semantics, to imitate or perturb author-specific patterns.

## Key Points

- RoPGen's attack model transforms only discrepant attributes between the source author and the target style.
- Numeric attributes are transformed when the style gap exceeds threshold `` `τ` ``, while non-numeric attributes are transformed when source values are not subsets of target values.
- The paper uses style transformation both offensively, for imitation and hiding attacks, and defensively, for data augmentation during training.
- Transformation sequences can be replayed from adversarial examples or synthesized attribute-by-attribute for manipulated training programs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
