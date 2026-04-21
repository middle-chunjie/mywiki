---
type: concept
title: Weighted Finite Automaton
slug: weighted-finite-automaton
date: 2026-04-20
updated: 2026-04-20
aliases: [WFA, weighted automaton, 加权有限自动机]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Weighted Finite Automaton** (加权有限自动机) — a finite-state structure whose transitions carry weights, here used to approximate retrieval dynamics over a language-model datastore.

## Key Points

- RETOMATON constructs an automaton `A = <Q, \Sigma, q_0, \delta, \phi>` on top of datastore states created by clustering hidden representations.
- Unlike classical static WFAs, its transition weights depend on the current neural context vector as well as the previous state and token.
- State transitions are induced by successor pointers between consecutive datastore entries, allowing the model to continue retrieval without a new search.
- The automaton is unsupervised: it is obtained from datastore structure plus clustering rather than from labeled transition supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
