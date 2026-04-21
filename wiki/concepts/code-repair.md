---
type: concept
title: Code Repair
slug: code-repair
date: 2026-04-20
updated: 2026-04-20
aliases: [automated program repair, bug fixing]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Repair** — a code generation task that transforms buggy source code into a corrected version.

## Key Points

- [[jha-2023-codeattack-2206-00052]] evaluates CodeAttack on Java code repair using the small Tufano dataset.
- The victim models include [[codet5]], [[codebert]], and [[graphcodebert]], all attacked in a sequence-to-sequence setting.
- CodeAttack reaches `99.36-99.52%` success across the three repair models while using far fewer queries than BERT-Attack.
- These results show that automated bug-fixing systems remain vulnerable even when perturbations are limited and similarity constraints are enforced.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
