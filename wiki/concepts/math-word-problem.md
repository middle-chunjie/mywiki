---
type: concept
title: Math Word Problem
slug: math-word-problem
date: 2026-04-20
updated: 2026-04-20
aliases: [MWP, 数学文字题]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Math Word Problem** (数学文字题) — a natural-language problem that requires extracting the relevant quantities and relations from text before performing symbolic or arithmetic reasoning.

## Key Points

- The paper uses GSM-IC to test whether irrelevant sentences degrade math reasoning in large language models.
- Baseline LLaMA-2-70B-chat is sensitive to distractor sentences, especially when they are topically related to the problem.
- S2A improves performance by first extracting the relevant problem text and only then solving it.
- The setup shows that attention failures in LLMs are not limited to opinion bias, but also affect reasoning under irrelevant context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weston-2023-system-2311-11829]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weston-2023-system-2311-11829]].
