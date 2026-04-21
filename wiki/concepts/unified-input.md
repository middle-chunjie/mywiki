---
type: concept
title: Unified Input
slug: unified-input
date: 2026-04-20
updated: 2026-04-20
aliases: [unified representation, 统一输入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unified Input** (统一输入) — a shared representation format that lets one model operate over multiple evaluation conditions by concatenating the relevant evidence fields into a single input sequence.

## Key Points

- [[dong-2023-codescore-2301-09043]] uses unified input to cover `g + r`, `g + n`, and `g + r + n` within one evaluation framework.
- The implementation serializes inputs as `"[CLS] g [SEP] r [SEP] n [SEP]"`, omitting unavailable fields depending on the scenario.
- This design enables [[multi-task-learning]] over Ref-only, NL-only, and Ref&NL evaluation formats with a single model.
- The paper reports that the unified objective improves correlation across all three input settings relative to training each setting in isolation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2023-codescore-2301-09043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2023-codescore-2301-09043]].
