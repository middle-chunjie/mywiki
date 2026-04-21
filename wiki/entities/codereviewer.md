---
type: entity
title: CodeReviewer
slug: codereviewer
date: 2026-04-20
entity_type: tool
aliases: [CodeReviewer]
tags: [dataset, code, evaluation]
---

## Description

CodeReviewer is a dataset of pull-request review comments paired with code edits, used in [[allamanis-2024-unsupervised-2402-08699]] to evaluate RTC for [[code-editing]].

## Key Contributions

- Supplies `1,000` sampled editing examples for the paper's EDITINGRTC experiments.
- Highlights the weakness of lexical-overlap supervision because PR comments and actual edits can be misaligned.

## Related Concepts

- [[code-editing]]
- [[round-trip-correctness]]
- [[semantic-equivalence]]

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]
