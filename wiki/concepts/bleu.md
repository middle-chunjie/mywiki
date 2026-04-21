---
type: concept
title: BLEU
slug: bleu
date: 2026-04-20
updated: 2026-04-20
aliases: [BLEU score, Bilingual Evaluation Understudy, n-gram precision metric]
tags: [evaluation, nlp, machine-translation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**BLEU** (BLEU分数) — an n-gram precision metric for text generation quality that counts overlapping n-gram sequences between a hypothesis and one or more references, originally designed for machine translation.

## Key Points

- BLEU computes the geometric mean of 1-gram through 4-gram modified precision scores, multiplied by a brevity penalty to discourage short outputs.
- When applied to code generation, BLEU is a common but weak baseline: it fails to recognize semantically equivalent code with different variable names or iteration styles.
- [[zhou-2023-codebertscore-2302-05527]] shows BLEU prefers non-equivalent candidates over functionally equivalent ones when surface lexical overlap is misleading.
- On CoNaLa human preference correlation, BLEU achieves Kendall-τ of `0.374` vs. CodeBERTScore's `0.517`.
- CodeBLEU and CrystalBLEU are code-specific variants of BLEU that partially address its limitations via AST/data-flow matching and trivial n-gram suppression respectively.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
