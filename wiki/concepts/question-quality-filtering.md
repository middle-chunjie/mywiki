---
type: concept
title: Question Quality Filtering
slug: question-quality-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [问题质量过滤]
tags: [filtering, retrieval, synthetic-data]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Question Quality Filtering** (问题质量过滤) — a post-generation selection step that removes synthetic questions unlikely to be relevant to their source document or useful for retrieval training.

## Key Points

- The paper formalizes filtering with a binary acceptance function `f_k(x; m)` over question-document pairs.
- Two criteria are enforced: the question must be answerable from the paired document and must resemble a realistic retrieval query rather than a generic prompt.
- BM25 and monoT5 are evaluated as filter models, with BM25 preferred because it better captures retrieval suitability and is cheaper to run.
- Downstream reranker performance is much stronger when trained on accepted questions than on rejected or unfiltered questions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]
- [[guinet-2024-automated-2405-13622]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
