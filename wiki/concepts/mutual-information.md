---
type: concept
title: Mutual Information
slug: mutual-information
date: 2026-04-20
updated: 2026-04-20
aliases: [MI, 互信息]
tags: [information-theory, statistics]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Mutual Information** (互信息) — a measure of how much knowing one random variable reduces uncertainty about another.

## Key Points

- The paper uses `I(X;T)` to quantify index compactness and `I(T;Q)` to quantify how informative identifiers are for retrieval.
- Under the bottleneck derivation, `I(X;Q|T) = -I(T;Q) + const`, linking conditional distortion to retrieval-relevant information.
- The authors estimate these quantities empirically by predicting ID prefixes of lengths `l = 2, 3, m`.
- The resulting curves diagnose overfitting and indexing quality in ways that plain Rec@1 cannot.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
