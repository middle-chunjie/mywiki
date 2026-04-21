---
type: concept
title: Paraphrase Robustness
slug: paraphrase-robustness
date: 2026-04-20
updated: 2026-04-20
aliases: [robustness to paraphrases, 改写鲁棒性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Paraphrase Robustness** (改写鲁棒性) — the ability of a model to preserve the same correct prediction across semantically equivalent reformulations of an input.

## Key Points

- The paper measures this on CounterFact using three paraphrased prompts per example.
- For questions GPT-J already answers correctly, LASER increases all-paraphrases-correct robustness by about `24.8` percentage points.
- This indicates that LASER changes answer stability across prompt variants, not just single-prompt accuracy.
- The robustness gain supports the view that the intervention reduces unstable or conflicting response modes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sharma-2023-truth-2312-13558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sharma-2023-truth-2312-13558]].
