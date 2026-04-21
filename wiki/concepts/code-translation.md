---
type: concept
title: Code Translation
slug: code-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [program translation]
tags: [software-engineering, transfer]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Translation** (代码翻译) — the task of converting source code from one programming language to another while preserving semantics.

## Key Points

- [[ahmad-2021-unified]] evaluates PLBART on Java↔C# translation as a test of cross-language semantic transfer.
- PLBART outperforms CodeBERT on BLEU, EM, and CodeBLEU in both directions despite not being pretrained on C#.
- The paper highlights semantically equivalent outputs that differ structurally from the reference, such as `else if` versus nested `else { if ... }`.
- Strong translation performance is presented as evidence that unified code-text pretraining captures reusable program semantics rather than memorizing syntax alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
