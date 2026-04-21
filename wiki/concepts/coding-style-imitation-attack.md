---
type: concept
title: Coding Style Imitation Attack
slug: coding-style-imitation-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [Style Imitation Attack, 代码风格模仿攻击]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Coding Style Imitation Attack** (代码风格模仿攻击) — a targeted authorship attack that transforms a program to mimic the coding style of a chosen target author while preserving the original program's functionality.

## Key Points

- The attack is black-box: it does not require internal access to the victim attribution model.
- It extracts source-program style attributes, synthesizes target-author style attributes from reference programs, and transforms only the discrepant attributes.
- The paper reports that these attacks can be more effective and far cheaper to generate than adversarial examples.
- On baseline systems, targeted imitation attack success reaches as high as `74.6%` on GitHub-Java for PbNN.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
