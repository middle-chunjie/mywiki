---
type: concept
title: Multiple Choice Grade
slug: multiple-choice-grade
date: 2026-04-20
updated: 2026-04-20
aliases: [MC grade, 多项选择评分]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multiple Choice Grade** (多项选择评分) — a discrete evaluation metric that assigns `1` when the correct option receives the highest predicted probability mass and `0` otherwise.

## Key Points

- [[schaeffer-2023-emergent-2304-15004]] identifies Multiple Choice Grade as one of the main metrics under which emergent abilities are reported.
- The paper treats it as a discontinuous metric that can convert smooth probability improvements into abrupt binary transitions.
- In BIG-Bench analyses, this metric accounts for a large fraction of annotated emergence cases.
- Replacing it with a continuous metric such as [[brier-score]] removes apparent emergence for the same LaMDA tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[schaeffer-2023-emergent-2304-15004]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[schaeffer-2023-emergent-2304-15004]].
