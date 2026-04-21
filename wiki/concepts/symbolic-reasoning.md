---
type: concept
title: Symbolic Reasoning
slug: symbolic-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [symbolic reasoning, 符号推理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Symbolic Reasoning** (符号推理) — the ability to manipulate discrete symbols or states through explicit stepwise rules rather than relying only on surface pattern matching.

## Key Points

- The paper evaluates symbolic reasoning with two controlled tasks: last-letter concatenation and coin-flip state tracking.
- Chain-of-thought prompting substantially improves symbolic reasoning only at large model scale, especially for PaLM `62B` and `540B`.
- The method improves out-of-domain length generalization: PaLM `540B` rises from `0.2%` to `94.8%` on four-word last-letter concatenation.
- In-domain symbolic tasks approach saturation with chain-of-thought prompting, reaching `99.4%` on two-word concatenation and `100.0%` on in-domain coin flip for PaLM `540B`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2023-chainofthought-2201-11903]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2023-chainofthought-2201-11903]].
