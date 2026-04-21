---
type: concept
title: Intra-Consistency
slug: intra-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [within-perspective consistency, intra consistency, 视角内一致性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Intra-Consistency** (视角内一致性) — the degree to which an output agrees with other outputs generated from the same perspective, serving as a confidence prior over candidates within that perspective.

## Key Points

- [[unknown-nd-enhancing-2309-17272]] defines lexical intra-consistency with an MBR-style similarity score based on aggregate BLEU against other candidates.
- The paper also defines semantic intra-consistency through structural equivalence classes and shared neighborhood patterns in the consistency graph.
- Intra-consistency is treated as supervision for the reranking score via `L_intra = 1/2 ||f - y||^2`, where `y` stores the consistency prior `phi(v)`.
- MPSC-Semantic and MPSC-Lexical both outperform the uniform variant, showing that better intra-consistency priors help select stronger programs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-enhancing-2309-17272]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-enhancing-2309-17272]].
