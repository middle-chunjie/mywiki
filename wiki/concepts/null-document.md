---
type: concept
title: Null Document
slug: null-document
date: 2026-04-20
updated: 2026-04-20
aliases: [空文档]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Null Document** (空文档) — an explicit empty retrieval candidate that lets a model represent cases where no external document is needed for prediction.

## Key Points

- REALM adds the null document `∅` to the top-`k` retrieved set during pre-training.
- The null document acts as a sink for examples where local context or parametric memory is already sufficient.
- It prevents the retriever from being forced to assign credit to irrelevant real passages.
- The paper later defines retrieval utility by comparing a document's contribution against conditioning on `∅`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
