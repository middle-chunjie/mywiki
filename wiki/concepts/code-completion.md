---
type: concept
title: Code Completion
slug: code-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [program completion]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code completion** (代码补全) — the task of predicting the continuation of partial source code, ranging from next-token suggestions to method- or repository-level generation.

## Key Points

- The paper studies method-level completion where prompts end at dereference points inside real Java methods.
- DOTPROMPTS contains `10,538` dereference prompts extracted from `1,420` methods to evaluate this setting.
- The work argues that completion quality should be measured not just by lexical match, but also by compilability and semantic consistency with repository state.
- Fill-in-the-middle prompting and retrieval-based prompt augmentation improve completion, but the paper shows that decoding-time monitoring adds further gains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
