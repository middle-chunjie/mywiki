---
type: concept
title: Program Translation
slug: program-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [code translation, programming language translation, 程序翻译]
tags: [software-engineering, nlp, code-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Program Translation** (程序翻译) — the task of automatically converting source code written in one programming language into semantically equivalent code in another programming language.

## Key Points

- Distinct from compilation: program translation operates between high-level languages (e.g., Java → Python, C++ → C#) rather than from high-level to machine code.
- Practical use cases include legacy code migration (e.g., FORTRAN/COBOL → Java/C++) and platform adaptation (e.g., Android app → iOS, JS).
- Supervised approaches require parallel code data; the scarcity of such data — especially for low-resource language pairs — is the primary bottleneck identified in [[zhu-2022-multilingual]].
- Unsupervised methods such as [[transcoder]] use cross-lingual masked language modeling and back-translation to avoid the need for parallel data, but performance on low-resource pairs remains limited.
- Fine-grained snippet-level alignment provides stronger supervision than method-level or program-level alignment, enabling multilingual training that benefits low-resource language pairs.
- Evaluation typically uses BLEU and CodeBLEU; functional correctness via test-case execution is a stricter but less commonly adopted metric.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2022-multilingual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2022-multilingual]].
