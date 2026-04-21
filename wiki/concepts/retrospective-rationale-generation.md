---
type: concept
title: Retrospective Rationale Generation
slug: retrospective-rationale-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [retrospective thought annotation, synthetic rationale generation]
tags: [reasoning, data-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrospective Rationale Generation** — synthesizing a natural-language rationale for a proof step after the correct tactic is known, in order to create supervision for thought-augmented proving.

## Key Points

- [[unknown-nd-leanstar]] uses [[gpt-4]] as an oracle that receives the current proof state and the ground-truth tactic to produce a thought.
- The prompting is designed to avoid hindsight-heavy explanations that simply restate the tactic choice after the fact.
- The paper samples `17,256` successful proof trajectories and annotates `52,438` thoughts to create the first thought-augmented dataset for Lean theorem proving.
- This synthetic dataset is the training signal used to produce the initial Lean-CoT model before expert iteration begins.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
