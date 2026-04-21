---
type: concept
title: Domain Adaptation
slug: domain-adaptation
date: 2026-04-20
updated: 2026-04-20
aliases: [domain transfer, 领域自适应]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Domain Adaptation** (领域自适应) — improving model behavior on a target domain by incorporating target-domain data or structure without fully retraining the original system from scratch.

## Key Points

- The paper demonstrates adaptation by building the datastore and automaton from Law-MT while the base LM is trained on WMT News Crawl.
- RETOMATON sharply lowers perplexity in this setting because it preserves non-parametric target-domain evidence even when searches are skipped.
- The paper further shows that automaton-augmented retrieval still helps after standard fine-tuning on the target domain.
- Gains are larger on Law-MT than on WIKIText-103, which the paper attributes to stronger `n`-gram repetitiveness.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
