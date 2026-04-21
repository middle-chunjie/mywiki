---
type: entity
title: MultiArith
slug: multiarith
date: 2026-04-20
aliases: [Multi-Arith, MultiArith dataset]
tags: []
---

## Description

MultiArith is an arithmetic word problem benchmark with `600` multi-step math questions (average `31.8` words per question, numeric answers), drawn from the Math World Problem Repository (Roy and Roth, 2015). It is a standard evaluation dataset for chain-of-thought reasoning in language models.

## Key Contributions

- Primary evaluation dataset used to analyze the misleading-by-similarity failure mode of similarity-based demonstration retrieval.
- Auto-CoT achieves `92.0%` accuracy on MultiArith vs. `91.7%` for Manual-CoT and `78.7%` for Zero-Shot-CoT using GPT-3.

## Related Concepts

- [[arithmetic-reasoning]]
- [[chain-of-thought-prompting]]
- [[auto-cot]]

## Sources

- [[zhang-2022-automatic-2210-03493]]
