---
type: entity
title: RETOMATON
slug: retomaton
date: 2026-04-20
entity_type: tool
aliases: [retrieval automaton]
tags: []
---

## Description

RETOMATON is the automaton-augmented retrieval method introduced in this paper. It approximates repeated nearest-neighbor search by traversing clustered datastore states linked by pointers.

## Key Contributions

- Saves up to `83%` of kNN searches without discarding retrieval information entirely.
- Improves perplexity over both `kNN-LM` and ADAPTRET on WIKIText-103 and Law-MT.
- Shows how symbolic state transitions can complement neural language-model representations.

## Related Concepts

- [[weighted-finite-automaton]]
- [[retrieval-based-language-model]]
- [[nearest-neighbor-search]]

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]
