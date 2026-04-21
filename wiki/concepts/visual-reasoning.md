---
type: concept
title: Visual Reasoning
slug: visual-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [视觉推理]
tags: [reasoning, multimodal]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Visual Reasoning** (视觉推理) — solving tasks by inferring latent rules or transformations from visual input-output examples.

## Key Points

- [[tang-2024-code-2405-17503]] evaluates REx on ARC, where the model must synthesize image-transforming programs from training examples.
- The paper uses a `40`-problem ARC subset with human-written hypotheses as hints for the LLM.
- Candidate programs are scored by the fraction of training tasks they solve, which becomes the heuristic `h(rho)` for refinement search.
- REx beats the fixed-width Hypothesis Search baseline while using fewer LLM calls on ARC.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-code-2405-17503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-code-2405-17503]].
