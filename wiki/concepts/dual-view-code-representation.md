---
type: concept
title: Dual-View Code Representation
slug: dual-view-code-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [dual-view representation, 双视角代码表示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dual-View Code Representation** (双视角代码表示) — a document representation strategy that models a code artifact with both its natural-language documentation and its source code so generation can use complementary semantic cues.

## Key Points

- In [[li-2024-building]], each retrieved document is represented as `` `d_doc ⊕ d_code ⊕ q` ``, where documentation acts as a gist that helps the model focus on the relevant semantics of the code.
- The method is designed for retrieval-augmented code generation, summarization, and completion, where raw code alone may be long and noisy.
- Ablations show that removing the dual-view design hurts average results, especially for code generation and completion: for example, CgCSN-Java drops from `37.7/45.4` to `35.0/41.9` BLEU/CodeBLEU.
- Gains on code summarization are smaller because only a minority of retrieved code snippets have paired documentation in the underlying retrieval database.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-building]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-building]].
