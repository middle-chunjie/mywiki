---
type: concept
title: Perplexity
slug: perplexity
date: 2026-04-20
updated: 2026-04-20
aliases: [PPL, 困惑度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Perplexity** (困惑度) — an intrinsic language-model evaluation metric that measures how well a model assigns probability mass to the observed token sequence.

## Key Points

- The paper uses perplexity as the primary quality metric while varying the Fraction of Saved Searches (FoSS).
- RETOMATON improves perplexity over both `kNN-LM` and ADAPTRET in the reported in-domain and domain-adaptation settings.
- One notable result is that RETOMATON still lowers perplexity even at `FoSS = 0`, because pointer continuation can reinforce good retrieved neighbors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
