---
type: entity
title: TL-CodeSum
slug: tl-codesum
date: 2026-04-20
entity_type: dataset
aliases: [TL-CodeSum, TLC]
tags: []
---

## Description

TL-CodeSum is a Java code summarization dataset used in [[shi-2022-evaluation-2107-07112]]. The paper describes it as a smaller benchmark with project mixing and substantial duplication, making it a useful stress case for evaluation methodology.

## Key Contributions

- Supplies `87,136` method-summary pairs from `9,732` Java projects.
- Exposes the effect of duplicated samples, including about `20%` exact code duplication across partitions.
- Produces the highest average scores among the paper's three base datasets, highlighting evaluation sensitivity to dataset construction.

## Related Concepts

- [[code-summarization]]
- [[data-splitting]]
- [[code-duplication]]

## Sources

- [[shi-2022-evaluation-2107-07112]]
