---
type: concept
title: Label Shift
slug: label-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [标签偏移]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Label Shift** (标签偏移) — a shift setting where the input distribution remains broadly compatible across domains but the target output distribution differs from the source.

## Key Points

- In the paper's ODQA framing, label shift means `p_s(x) ~= p_t(x)` while `p_s(y | x) != p_t(y | x)`.
- BioASQ and Quasar-T are categorized as label-shift datasets by the proposed generalizability test.
- These datasets are more amenable to zero-shot interventions than full-shift datasets, because the source model still captures useful question-context structure.
- The paper treats answer-distribution mismatch as a major driver of label shift and studies answer sampling strategies to address it.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
