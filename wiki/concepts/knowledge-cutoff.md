---
type: concept
title: Knowledge Cutoff
slug: knowledge-cutoff
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge cutoff, cutoff date, 知识截止日期]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Cutoff** (知识截止日期) — the latest date of information assumed to be available to a model before it is evaluated or deployed.

## Key Points

- [[unknown-nd-xcodeevalan-2303-03004]] uses Codeforces release timestamps to analyze model behavior before and after a public cutoff date.
- The paper covers problems released between `2010-02-19` and `2022-11-21`, enabling contamination-aware slicing of the benchmark.
- The authors use this metadata to discuss possible post-cutoff degradation in ChatGPT on Codeforces problems.
- They argue that contamination analysis is only meaningful when there are enough post-cutoff examples in the benchmark.
- The paper explicitly asks model builders to disclose cutoff dates so that benchmark results can be interpreted more cleanly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-xcodeevalan-2303-03004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-xcodeevalan-2303-03004]].
