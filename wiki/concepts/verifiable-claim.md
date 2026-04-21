---
type: concept
title: Verifiable Claim
slug: verifiable-claim
date: 2026-04-20
updated: 2026-04-20
aliases: [verifiable claims, 可验证断言]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Verifiable Claim** (可验证断言) — a short, self-contained information unit that is specific enough to be unambiguously checked against a reference knowledge source.

## Key Points

- The paper positions verifiable claims between sentence-level and fully atomic facts: they retain enough context for reliable checking while isolating distinct assertions.
- PIC inputs are defined as sets of verifiable claims, and generated outputs are re-decomposed into verifiable claims before scoring.
- Claim sets must be non-empty, deduplicated, non-contradictory, and sufficiently relevant to the instruction for PIC evaluation to be meaningful.
- Full PIC requires coverage of all source claims, while partial PIC requires only that generated claims stay within the source set.
- The paper uses automatic extraction of verifiable claims as a central interface for both benchmark construction and evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2025-precise-2506-06589]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2025-precise-2506-06589]].
