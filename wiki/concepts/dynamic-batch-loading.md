---
type: concept
title: Dynamic Batch Loading
slug: dynamic-batch-loading
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic data loading, DBL, 动态批加载]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Batch Loading** (动态批加载) — a domain-reweighting training procedure that updates the sampling mixture of pretraining data according to each domain's residual loss relative to a target reference loss.

## Key Points

- The algorithm reevaluates per-domain validation losses every `m` steps and computes residual gaps against domain-specific reference losses.
- Sampling weights are updated multiplicatively with an exponential rule, giving more probability mass to domains that are recovering more slowly.
- The method is used in both pruning and continued pre-training rather than only in the recovery stage.
- In this paper, dynamic loading strongly increases C4 and Books exposure while decreasing GitHub and ArXiv exposure for the `1.3B` run.
- The authors attribute improved downstream accuracy to more balanced loss reduction across domains, not merely to lower average LM loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2024-sheared-2310-06694]].
