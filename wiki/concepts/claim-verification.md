---
type: concept
title: Claim Verification
slug: claim-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [claim checking, 断言验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Claim Verification** (断言验证) — the process of deciding whether a candidate claim is supported by a specified evidence set, typically through semantic-equivalence checking.

## Key Points

- In PIC, a claim is supported if it is semantically equivalent to at least one claim in the source claim set.
- The verifier operates against the provided input claims rather than external search results, making the grounding space closed and explicit.
- Precision, recall, and `F_1@K` are all computed from supported versus unsupported verification outcomes.
- The appendix uses GPT-4o mini as the claim verifier and reports `81.0%` agreement with human annotators.
- Verification enables the paper to diagnose unsupported generations as distortion, fabrication, conflation, or verifier error.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2025-precise-2506-06589]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2025-precise-2506-06589]].
