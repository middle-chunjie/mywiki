---
type: concept
title: Executability
slug: executability
date: 2026-04-20
updated: 2026-04-20
aliases: [code compileability, 可执行性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Executability** (可执行性) — a binary property indicating whether generated code can run successfully on the provided inputs without compilation or runtime failure.

## Key Points

- [[dong-2023-codescore-2301-09043]] introduces Executability to separate compile/runtime failures from merely wrong outputs when `PassRatio = 0`.
- UniCE predicts an `Exec` probability jointly with CodeScore instead of relying only on continuous regression.
- The paper reports that the Exec component achieves above `90%` precision, F1, and accuracy even on zero-shot transfer to HE-Eval.
- Case studies show that the learned Exec signal is sensitive to syntax failures such as mismatched parentheses.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2023-codescore-2301-09043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2023-codescore-2301-09043]].
