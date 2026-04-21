---
type: concept
title: Retrieval-Augmented Code Generation
slug: retrieval-augmented-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval-based code generation, RACG]
tags: [code-generation, retrieval-augmented-generation, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Augmented Code Generation** (检索增强代码生成) — a class of code generation methods that retrieve external information (documentation, code examples, or repository context) at inference time and condition the code generator on that retrieved content alongside the NL intent.

## Key Points

- Two main retrieval targets: (1) NL–code example pairs (ExPrompting) and (2) code documentation (DocPrompting). Retrieving documentation substantially outperforms retrieving examples, especially for generalization to unseen functions (e.g., CMD Acc 25.32% vs 6.68% for GPT-Neo-125M on tldr).
- Documentation is more likely to be available for new libraries than NL–code examples, making it better suited for the long-tail of API coverage.
- Unlike open-domain QA retrieval augmentation, code generation requires the retrieved content to contain exact function names and argument signatures that appear verbatim in the output — precision of retrieved content is critical.
- Both sparse (BM25) and dense retrievers can be used; the best choice is dataset-dependent: BM25 suffices for Bash (vocabulary overlap with manuals), while dense retrieval is required for Python (semantic gap between StackOverflow questions and API docs).
- The number of retrieved documents `k` has diminishing returns: optimal is `5 ≤ k ≤ 10`; too many docs introduce noise the generator cannot filter.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-docprompting-2207-05987]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-docprompting-2207-05987]].
