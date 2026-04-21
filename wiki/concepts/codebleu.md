---
type: concept
title: CodeBLEU
slug: codebleu
date: 2026-04-20
updated: 2026-04-20
aliases: [Code BLEU, AST-augmented BLEU]
tags: [evaluation, code, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**CodeBLEU** (CodeBLEU) — an extension of [[bleu]] for code generation that augments n-gram matching with Abstract Syntax Tree (AST) matching and data-flow matching to partially account for structural code equivalence.

## Key Points

- Proposed by Ren et al. (2020), CodeBLEU combines token-level n-gram overlap, keyword match, AST node match, and data-flow graph match into a weighted composite score.
- Despite the structural augmentations, partial code snippets may not parse, making CodeBLEU inapplicable to incomplete predictions.
- [[zhou-2023-codebertscore-2302-05527]] reports that CodeBLEU does not consistently outperform plain BLEU in functional correctness correlation on HumanEval.
- The paper cites Wang et al. (2022) who found CodeBLEU does not correlate well with execution accuracy, motivating embedding-based alternatives.
- CodeBLEU achieves Kendall-τ of `0.496` (Java) on functional correctness vs. CodeBERTScore's `0.553`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
