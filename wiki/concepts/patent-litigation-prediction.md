---
type: concept
title: Patent Litigation Prediction
slug: patent-litigation-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [专利诉讼预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Patent Litigation Prediction** (专利诉讼预测) — the task of estimating whether a patent is likely to trigger a lawsuit between specific companies, often framed as a ranking or risk-scoring problem.

## Key Points

- The paper defines the task at the triplet level `(plaintiff, defendant, patent)` rather than as patent-only classification.
- Prediction is cast as ranking risky patents for a company pair, which matches proactive legal-risk management better than binary labeling alone.
- Sparse historical lawsuits make pure collaborative methods weak, so the paper supplements them with patent text and metadata.
- The study treats litigation prediction as an implicit-feedback problem because only observed lawsuits are directly available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
