---
type: concept
title: Automated Theorem Proving
slug: automated-theorem-proving
date: 2026-04-20
updated: 2026-04-20
aliases: [ATP, automatic theorem proving, 自动定理证明]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Automated Theorem Proving** (自动定理证明) — the problem of automatically constructing formal proofs for statements expressed in a logical system, typically under exact symbolic verification.

## Key Points

- [[yang-nd-leandojo]] presents ATP as a demanding reasoning problem with prohibitively large search spaces, especially in realistic formal mathematics libraries.
- The paper treats formal proof generation as a special kind of code generation where outputs are checked by a proof assistant, leaving no room for unverifiable hallucinations.
- Its ReProver system combines learned tactic generation with retrieval and [[best-first-search]] rather than relying on a single forward generation pass.
- The proposed benchmark emphasizes harder generalization by testing on theorems that require premises unseen during training.
- The reported gains over GPT-4 and the no-retrieval baseline suggest that explicit search and library access remain critical for practical ATP with LLMs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-nd-leandojo]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-nd-leandojo]].
