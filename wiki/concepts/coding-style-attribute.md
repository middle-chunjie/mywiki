---
type: concept
title: Coding Style Attribute
slug: coding-style-attribute
date: 2026-04-20
updated: 2026-04-20
aliases: [Coding Style Feature, 代码风格属性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Coding Style Attribute** (代码风格属性) — a measurable property of source code style, such as lexical, syntactic, or semantic usage patterns, that can help characterize an author's programming habits.

## Key Points

- The paper systematizes `23` attributes spanning token, statement, basic-block, and function granularity.
- Attributes are divided into exhaustive and non-exhaustive domains because the latter require more perturbed examples during robust training.
- Representative examples include identifier naming, variable-definition placement, namespace usage, loop structures, and function nesting depth.
- These attributes are the operational basis for both the attack procedures and RoPGen's data augmentation pipeline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
