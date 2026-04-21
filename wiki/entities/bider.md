---
type: entity
title: BIDER
slug: bider
date: 2026-04-20
entity_type: tool
aliases: [BrIDging knowledge inconsistency for efficient Retrieval-augmented LLMs, BIDER refiner]
tags: []
---

## Description

BIDER is the retrieval-document refiner proposed in [[jin-2024-bider-2402-12174]]. It transforms noisy retrieved text into compact key supporting evidence that is better aligned with a downstream LLM's answering preferences.

## Key Contributions

- Introduces a three-stage pipeline combining knowledge synthesis, supervised seq2seq learning, and PPO-based preference alignment.
- Improves average benchmark performance from `0.363` to `0.390` while reducing average input length from `759` to `90` tokens.
- Cuts end-to-end latency on `NQ` from `1.33s` to `1.18s` per query.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[key-supporting-evidence]]
- [[preference-alignment]]
- [[sequence-to-sequence]]

## Sources

- [[jin-2024-bider-2402-12174]]
