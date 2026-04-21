---
type: entity
title: CompLex LCP Dataset
slug: complex-lcp-dataset
date: 2026-04-20
entity_type: tool
aliases: [CompLex, LCP 2021, SemEval-2021 Task 1, CompLex Corpus]
tags: [dataset, nlp, lexical-complexity, benchmark]
---

## Description

CompLex is the benchmark corpus for the SemEval-2021 Task 1 (Lexical Complexity Prediction). It contains English single-word and multi-word target tokens annotated with continuous complexity scores (0–1) from Likert-scale human judgments, spanning three domains: biblical text, biomedical literature, and Europarl (parliamentary) proceedings.

## Key Contributions

- Defines lexical complexity prediction as a regression task; provides 7,662 / 421 / 917 (train/trial/test) entries for single-word and 1,517 / 99 / 184 for multi-word.
- Enables evaluation across distinct domains within English, exposing cross-domain generalization challenges.
- Best LCP 2021 competition Pearson scores: `.7886` (single-word), `.8612` (multi-word) on test.

## Related Concepts

- [[lexical-complexity-prediction]]
- [[complex-word-identification]]

## Sources

- [[zaharia-2022-domain-2205-07283]]
