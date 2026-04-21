---
type: concept
title: Knowledge Denoising
slug: knowledge-denoising
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge denoising, 知识去噪]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Denoising** (知识去噪) — a process for improving noisy supervision by estimating which extracted or generated facts are reliable enough to keep, revise, or remove.

## Key Points

- [[sun-2024-consistency-2401-13598]] denoises synthetic relation labels by comparing original synthetic triplets with pseudo labels predicted by a pre-denoising model.
- The method constructs two cross-document knowledge graphs, `KG_s` and `KG_p`, and scores each triplet by summed frequency `` `s_ijk = F_ijk^s + F_ijk^p` ``.
- Reliability is filtered with a per-relation dynamic threshold based on mean minus standard deviation of consistency scores, rather than with one global cutoff.
- The denoising stage can both remove incorrect triplets and add missing ones when the same fact is consistently expressed across multiple synthetic documents.
- In ablations, removing denoising drops ZeroDocRTE test `F1` from `13.1 -> 11.4` on Re-DocRED and from `14.2 -> 12.1` on DocRED.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-consistency-2401-13598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-consistency-2401-13598]].
