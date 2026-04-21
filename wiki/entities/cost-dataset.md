---
type: entity
title: CoST Dataset
slug: cost-dataset
date: 2026-04-20
entity_type: tool
aliases: [CoST, Code Snippet Translation Dataset]
tags: [dataset, program-translation, benchmark]
---

## Description

CoST (Code Snippet Translation) is a multilingual parallel code dataset introduced in [[zhu-2022-multilingual]], containing snippet-level and program-level parallel data from 7 programming languages (C, C++, C#, Java, Python, JavaScript, PHP) spanning 1,625 programming problems scraped from GeeksForGeeks.

## Key Contributions

- Provides 132,046 pairwise code snippet samples across up to 42 language pairs, the only dataset with snippet-level alignment for 7 languages as of 2022.
- Snippet alignment is achieved by matching code comments across languages following a contributor template, enabling one-to-one correspondence finer than method or program-level alignment.
- Also includes program-level pairwise data (~1,400–1,500 pairs for major language pairs) for end-to-end program translation evaluation.

## Related Concepts

- [[program-translation]]
- [[code-translation]]
- [[multilingual-snippet-translation]]
- [[low-resource-language]]

## Sources

- [[zhu-2022-multilingual]]
