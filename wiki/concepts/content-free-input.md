---
type: concept
title: Content-Free Input
slug: content-free-input
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic-free input, null-content probe]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Content-Free Input** — an input string that intentionally carries no task-relevant semantics and is used to probe a prompt-induced output distribution.

## Key Points

- The paper uses a placeholder such as `[N/A]` as the content-free input `η`.
- Because the input should contain no class evidence, the ideal predictive distribution over labels is uniform.
- The entropy of the model's predictions on this probe becomes the paper's fairness score for the prompt.
- This design lets prompt quality be estimated without a development set and with only a single forward computation per candidate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-fairnessguided-2303-13217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-fairnessguided-2303-13217]].
