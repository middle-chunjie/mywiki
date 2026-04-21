---
type: concept
title: Program Obfuscation
slug: program-obfuscation
date: 2026-04-20
updated: 2026-04-20
aliases: [code obfuscation, 程序混淆]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Obfuscation** (程序混淆) — semantics-preserving transformation of source code intended to reduce human readability or reverse-engineering ease without changing program behavior.

## Key Points

- The paper uses obfuscation transformations as the natural perturbation space for adversarial attacks on code models.
- It groups the transformations into replace operations and insert operations so the same optimization framework can handle both.
- The implemented set includes renaming local variables, function parameters, and object fields, plus boolean-literal rewrites, print insertion, and dead-code insertion.
- The attack assumes good obfuscations preserve functionality while still shifting the learned representation enough to fool a downstream model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
