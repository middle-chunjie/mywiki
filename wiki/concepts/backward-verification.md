---
type: concept
title: Backward Verification
slug: backward-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [reverse verification, 后向验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Backward Verification** (后向验证) — a post-generation checking procedure that treats a candidate answer as an added condition and then asks the model to verify whether the original problem facts remain recoverable or mutually consistent.

## Key Points

- [[weng-2023-large-2212-09561]] rewrites each candidate answer into a declarative conclusion `f_y` before verification.
- The verification stage does not train a separate verifier; it reuses the same LLM that generated the original chain of thought.
- Candidate answers are reranked by repeated backward checks, with `P = 10` verification samples per candidate in the reported setup.
- The paper frames backward verification as the source of interpretable verification scores for candidate selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weng-2023-large-2212-09561]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weng-2023-large-2212-09561]].
