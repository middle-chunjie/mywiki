---
type: concept
title: Code Search
slug: code-search
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic code retrieval, 代码搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Search** (代码搜索) — retrieving source-code snippets that satisfy a natural-language or code-like query by modeling semantic relevance rather than relying only on lexical overlap.

## Key Points

- The paper formulates domain-specific code search as semantic matching between natural-language descriptions and code snippets.
- CDCS handles code search with a joint `<NL, PL>` encoder and predicts relevance scores for each query-code pair.
- The work emphasizes that code search in domain-specific languages suffers from scarce labeled data compared with Java or Python.
- Evaluation is reported with `MRR`, `Acc@1`, `Acc@5`, and `Acc@10` over `1000` test queries.
- Results on SQL and Solidity show that stronger transfer initialization materially improves code-search quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gu-2018-deep]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gu-2018-deep]].
