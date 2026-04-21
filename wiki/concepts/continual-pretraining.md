---
type: concept
title: Continual Pretraining
slug: continual-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [continued pretraining, domain-adaptive pretraining]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Continual pretraining** — additional large-scale pretraining performed after an initial foundation-model phase in order to shift capability balance or adapt the model to new data distributions without training from scratch.

## Key Points

- Lemur applies continual pretraining on top of `Llama-2-70B` rather than training a new base model from zero.
- The paper uses a `90B`-token corpus with a `10:1` code-to-text ratio to raise coding performance while retaining natural-language ability.
- Deduplication and source balancing are central to the recipe because the goal is not pure code specialization but text-code harmonization.
- The paper frames mixture-ratio selection as an open problem and notes that optimal ratios likely depend on model size and domain transfer behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lemurharmonizing-2310-06830]]
- [[research-2026-composer-2603-24477]]
- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lemurharmonizing-2310-06830]].
