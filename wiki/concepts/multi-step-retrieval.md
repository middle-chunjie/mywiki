---
type: concept
title: Multi-Step Retrieval
slug: multi-step-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative retrieval, 多步检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Step Retrieval** (多步检索) — a retrieval setup in which evidence is selected iteratively, with each retrieval decision conditioned on the query and previously retrieved evidence.

## Key Points

- Q-RAG models retrieval as a finite-horizon sequential decision problem rather than a single similarity lookup.
- The state at step `t` is the ordered list of the query plus earlier selected chunks, so retrieval decisions can depend on partial evidence already collected.
- Retrieved chunks are removed from the action space after selection, preventing duplicate retrieval and enforcing explicit trajectory construction.
- The paper shows strong multi-step retrieval results on both long-context narrative benchmarks and open-domain multi-hop QA.
- Appendix B adds a planning variant that applies beam search over learned `Q` values to improve trajectory selection at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sorokin-2026-qrag-2511-07328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sorokin-2026-qrag-2511-07328]].
