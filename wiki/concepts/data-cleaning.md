---
type: concept
title: Data Cleaning
slug: data-cleaning
date: 2026-04-20
updated: 2026-04-20
aliases: [dataset cleaning, 数据清洗]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data Cleaning** (数据清洗) — the process of detecting, removing, or correcting noisy instances so a dataset better matches the target task distribution.

## Key Points

- The paper treats data cleaning as a primary lever for improving neural code search, not merely a preprocessing detail.
- It decomposes cleaning into a rule-based syntactic stage and a model-based semantic stage.
- The rule filter removes or edits comments with artifacts such as HTML tags, URLs, Javadoc tags, and non-English text.
- The semantic filter keeps comments whose reconstruction loss under a VAE is closer to trusted natural-query data.
- Cleaning reduces the CodeSearchNet Java training set from `394,471` to `192,031` pairs while improving retrieval metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
