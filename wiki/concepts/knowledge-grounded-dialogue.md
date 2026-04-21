---
type: concept
title: Knowledge-Grounded Dialogue
slug: knowledge-grounded-dialogue
date: 2026-04-20
updated: 2026-04-20
aliases: [KGD]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge-Grounded Dialogue** (知识驱动对话) — a dialogue generation setting where responses are conditioned on both conversational context and external knowledge such as retrieved documents or knowledge graph triples.

## Key Points

- This paper formulates KGD as generating a response `u_n` from dialogue context `C` and relevant knowledge triples `K`.
- The work emphasizes that robustness is a core challenge for KGD because noise can appear in both the dialogue context and the grounding knowledge.
- EnCo treats semantic-preserving paraphrases and incomplete knowledge as positive variants that should stay close to the original sample representation.
- It treats entity deletion or replacement in the context and KG as semantic-changing perturbations that should be separated through contrastive learning.
- The empirical study uses three public KGD benchmarks: KdConv, DuConv, and DuRecDial.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-improving-2401-04361]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-improving-2401-04361]].
