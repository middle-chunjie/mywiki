---
type: concept
title: Prompt Augmentation
slug: prompt-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt expansion]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt augmentation** (提示增强) — the addition of retrieved, structured, or otherwise derived context to the model input in order to improve generation.

## Key Points

- The paper compares MGD against and alongside two prompt augmentation variants: `classExprTypes` and RLPG.
- `classExprTypes` reserves `20%` of the prompt budget for files defining types referenced by expressions in the class.
- RLPG and FIM each improve baseline completion quality, but MGD still yields further gains on top of them.
- The paper argues that prompt augmentation and output-side monitoring are complementary because they intervene at different stages of generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
