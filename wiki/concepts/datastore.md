---
type: concept
title: Datastore
slug: datastore
date: 2026-04-20
updated: 2026-04-20
aliases: [external memory datastore, 数据存储库]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Datastore** (数据存储库) — an external collection of key-value examples, here context representations and next-token targets, that a model can retrieve from during inference.

## Key Points

- In this paper each datastore entry initially contains a key `f(c_i)` and target token `w_i`, and is then extended with a successor pointer.
- RETOMATON turns the flat datastore into structured memory by clustering entries into automaton states and preserving their sequential links.
- The in-domain datastore is built from WIKIText-103 training data, while the domain-adaptation datastore is built from Law-MT.
- Datastore scale is large enough to matter operationally: about `103M` entries for WIKIText-103 and `19M` for Law-MT.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
