---
type: concept
title: Normalized Information
slug: normalized-information
date: 2026-04-20
updated: 2026-04-20
aliases: [NI, 归一化信息量]
tags: [information-theory, sampling, filtering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Normalized Information** (归一化信息量) — a length-normalized information-theoretic score that estimates how atypical a document is under a probabilistic language model.

## Key Points

- The paper defines `NI(x) = I(x) / (|x| * log |V|)` to correct the raw information score for document length.
- Documents with very low or very high NI are treated as outliers because they tend to be repetitive, corrupted, overly generic, or hard to interpret.
- NI is estimated with both pretrained transformer language models and finite-context models, with pretrained transformers favored for direct use.
- Synthetic datasets built after excluding NI outliers yield higher question-quality scores than datasets built from NI extremes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
