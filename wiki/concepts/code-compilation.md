---
type: concept
title: Code Compilation
slug: code-compilation
date: 2026-04-20
updated: 2026-04-20
aliases: [code compilation, compilation prediction, 代码编译判定]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Compilation** (代码编译判定) — the task of predicting whether a given program will compile or interpret successfully rather than fail with a syntax- or build-level error.

## Key Points

- [[unknown-nd-xcodeevalan-2303-03004]] defines Code Compilation as a binary classification task within xCODEEval.
- It is evaluated with `accuracy` rather than `pass@k`, separating compileability from full functional correctness.
- The paper reports `19,915,150` training examples, `6,394` validation examples, and `30,388` test examples across `11` languages.
- `gpt-3.5-turbo-0301` reaches `63.27` average accuracy, with comparatively strong results on PHP and Go.
- The benchmark connects this task to one of the six execution outcomes tracked by [[exec-eval]], but the task itself isolates compile/no-compile prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-xcodeevalan-2303-03004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-xcodeevalan-2303-03004]].
