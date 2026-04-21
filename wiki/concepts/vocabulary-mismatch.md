---
type: concept
title: Vocabulary Mismatch
slug: vocabulary-mismatch
date: 2026-04-20
updated: 2026-04-20
aliases: [词汇失配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Vocabulary Mismatch** (词汇失配) — the retrieval failure case where a relevant query and document use different surface terms for the same concept, so exact lexical overlap alone misses the match.

## Key Points

- The paper contrasts vocabulary mismatch with semantic mismatch: the former concerns different words for the same concept, while the latter concerns the same word used with different meanings.
- COIL-tok does not solve vocabulary mismatch because its score sums only over exact shared tokens.
- COIL-full introduces a projected `[CLS]` similarity term as a coarse semantic backoff that partially mitigates vocabulary mismatch.
- The authors interpret the improvement from COIL-tok to COIL-full as evidence that dense global matching and exact token matching are complementary.
- Prior lexical improvements such as DeepCT and DocT5Query are presented as still suffering from vocabulary mismatch despite using language models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-coil-2104-07186]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-coil-2104-07186]].
