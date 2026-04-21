---
type: concept
title: Prefix Tree
slug: prefix-tree
date: 2026-04-20
updated: 2026-04-20
aliases: [trie, DocID prefix tree, 前缀树]
tags: [data-structure, generative-retrieval, decoding]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Prefix Tree** (前缀树) — a trie data structure that indexes all valid DocID token sequences in a generative retrieval corpus, used during constrained beam search to efficiently enumerate and filter valid next tokens at each decoding step.

## Key Points

- In generative retrieval, the prefix tree is built offline from the set of all DocIDs `{c^d : d ∈ C}`; it guarantees that every prefix expanded by beam search is a valid prefix of at least one real document's DocID.
- The masking function `g(prefix)` returns 0 if the prefix is valid (exists in the trie) and `-∞` otherwise, which is added to the token score before softmax so that invalid tokens are eliminated.
- Prevents the model from decoding sequences that have no corresponding document — without this constraint, many decoded token sequences would map to no document.
- Used as a component in PAG's planning-ahead constrained beam search: the trie also enables efficient pre-computation of which documents from the top-n simultaneous-score candidates share a given prefix.
- Scales with corpus size and DocID length; for MSMARCO (8.8M passages, L=8, V=2048) the trie fits in memory alongside inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2024-planning-2404-14600]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2024-planning-2404-14600]].
