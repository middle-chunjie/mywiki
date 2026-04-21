---
type: concept
title: Retrieval-Based Language Model
slug: retrieval-based-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [R-LM, retrieval-based LM, 基于检索的语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Based Language Model** (基于检索的语言模型) — a language model that combines a parametric LM with examples retrieved from an external datastore at inference time.

## Key Points

- The paper frames retrieval-based LMs as a way to improve accuracy, provenance, and domain adaptability without relying only on model weights.
- Its central systems problem is that retrieval may be invoked at every decoding step, making nearest-neighbor search the main inference bottleneck.
- RETOMATON preserves the retrieval signal even when skipping fresh search by approximating neighbor continuation through an automaton.
- The paper argues that retrieval trajectories themselves contain reusable symbolic structure, not just isolated nearest-neighbor hits.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
