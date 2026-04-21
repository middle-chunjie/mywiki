---
type: concept
title: Hierarchical Embedding
slug: hierarchical-embedding
date: 2026-04-20
aliases: [Hierarchy Embedding, hierarchical-code-embedding]
tags: [code-representation, program-analysis, embedding]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Hierarchical Embedding** (层次嵌入) — a token-level property encoding the structural position of each source code token within its program's syntax tree, decomposed into a statement-level global component and an intra-statement local component.

## Key Points

- Captures two orthogonal aspects of code structure: *global hierarchy* (the block-nesting depth and position of the statement containing a token, from the CST root to the statement node) and *local hierarchy* (the syntactic role of the token within its statement, from the statement node to the leaf).
- Represented as a fixed-size vector obtained by mean-pooling Transformer-encoded node-type sequences along the root-to-leaf CST path.
- Empirically, global hierarchy is more critical for structurally complex, long programs (C++ datasets), while local hierarchy dominates on simpler programs, but both together outperform either alone.
- Injected by concatenating the hierarchical embedding vector with the standard token embedding before passing to the sequence encoder, adding minimal parameters (~1–5%).
- Unlike pairwise relational encoding (e.g., AST node distances in GREAT or CodeTransformer), hierarchical embedding is a per-token property, not a pairwise relation, enabling broader structural coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
