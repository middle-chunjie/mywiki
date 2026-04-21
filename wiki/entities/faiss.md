---
type: entity
title: FAISS
slug: faiss
date: 2026-04-20
entity_type: tool
aliases: [Facebook AI Similarity Search]
tags: []
---

## Description

FAISS is the nearest-neighbor search library used by the RETOMATON implementation and the compared `kNN-LM` baselines. It also supports the one-time clustering step used to build automaton states.

## Key Contributions

- Provides the approximate nearest-neighbor infrastructure for full datastore search.
- Supports the `k`-means preprocessing used to create RETOMATON clusters.

## Related Concepts

- [[nearest-neighbor-search]]
- [[clustering]]
- [[datastore]]

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]
