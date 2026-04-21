---
type: concept
title: Label Space
slug: label-space
date: 2026-04-20
updated: 2026-04-20
aliases: [output space, 标签空间]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Label Space** (标签空间) — the set or distribution of allowable output labels for a task, which can itself provide strong supervision signals even when individual input-label pairs are incorrect.

## Key Points

- The paper identifies label space as one of four separable factors that demonstrations provide in in-context learning.
- Replacing correct task labels with random labels from the true label set causes only small losses, implying that knowing the available labels matters more than pairing them correctly.
- Replacing the label set with random English words hurts direct models by `5-16%` absolute, showing that direct generation depends strongly on the actual output vocabulary.
- Channel models show only `0-2%` degradation under label-space removal, suggesting that their dependence on label-space information is much weaker.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[min-2022-rethinking-2202-12837]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[min-2022-rethinking-2202-12837]].
