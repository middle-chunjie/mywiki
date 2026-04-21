---
type: concept
title: Path Pruning
slug: path-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [document path pruning]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Path pruning** — pruning full prompt paths in a graph-structured prompt based on a learned or inferred relevance score before final generation.

## Key Points

- The paper scores each document path with a Bayesian saliency metric `P(d_i | q_i, p)` derived from language-model logits.
- Pruning removes entire document-query KV paths, so the retained context can be concatenated directly for response generation.
- The mechanism improves both efficiency and quality by filtering distracting documents rather than merely reordering them.
- On NaturalQuestions-Open with `mpt-7b-instruct`, Bayesian path pruning reaches `0.465` accuracy versus `0.224` without pruning.
- The paper's ablations show accuracy usually peaks around intermediate `top-k` values (`k = 2` to `4`) rather than keeping all retrieved paths.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[merth-2024-superposition-2404-06910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[merth-2024-superposition-2404-06910]].
