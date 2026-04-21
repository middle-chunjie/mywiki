---
type: concept
title: Combinatory Categorial Grammar
slug: combinatory-categorial-grammar
date: 2026-04-20
updated: 2026-04-20
aliases: [CCG, combinatory categorial grammars, 组合范畴语法]
tags: [grammar, parsing, semantics]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Combinatory Categorial Grammar** (组合范畴语法) — a grammar formalism that derives compositional syntactic and semantic structures through typed lexical categories and combinatory rules.

## Key Points

- NS3 uses a CCG-based parser to derive the semantic layouts that drive module composition for code-search queries.
- The parser relies on hand-built lexical rules for common action verbs and entity nouns plus catch-all rules for less common words.
- Query preprocessing removes question words and punctuation to improve CCG parse success.
- In this paper, CCG acts as the symbolic scaffold that exposes query compositionality to the neural reranker.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[arakelyan-2022-ns-2205-10674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[arakelyan-2022-ns-2205-10674]].
