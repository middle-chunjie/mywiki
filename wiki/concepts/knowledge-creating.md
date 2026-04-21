---
type: concept
title: Knowledge Creating
slug: knowledge-creating
date: 2026-04-20
updated: 2026-04-20
aliases: [KC]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Creating** (知识创造) — the ability of a model to generate novel yet reasonable knowledge that remains faithful to real-world constraints.

## Key Points

- KoLA restricts this level to narrative generation settings, where plausible future events and relations can be annotated and evaluated more directly.
- The benchmark builds two creating tasks, one on encyclopedic narrative Wikipedia texts and one on evolving news or fiction, and annotates fine-grained event knowledge for the references.
- KoLA evaluates this level with a mixture of `partial(T, R)`, `partial(T, T_k)`, and `partial(T_k, R)` so that generation quality is not judged solely by surface overlap with a reference continuation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
