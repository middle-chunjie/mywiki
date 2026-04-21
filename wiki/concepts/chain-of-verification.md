---
type: concept
title: Chain-of-Verification
slug: chain-of-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [CoVe, COVE, chain of verification, 链式验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Verification** (链式验证) — an inference-time prompting procedure in which a model drafts an answer, formulates verification questions, answers them, and revises the original response using the verification evidence.

## Key Points

- The CoVe pipeline has `4` stages: baseline generation, verification planning, verification execution, and final verified regeneration.
- The paper studies joint, two-step, factored, and factor+revise variants, with the factored families preventing direct conditioning on the original draft during question answering.
- Independent verification questions are motivated by the observation that a model may answer short factual checks more accurately than a long-form composite query.
- Across list generation, closed-book QA, and biography generation, CoVe improves factual precision over few-shot, instruction-tuned, and chain-of-thought baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dhuliawala-2023-chainofverification-2309-11495]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dhuliawala-2023-chainofverification-2309-11495]].
