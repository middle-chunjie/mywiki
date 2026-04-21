---
type: concept
title: Step-Level Verification
slug: step-level-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [step-level verification, step-wise verification, intermediate step verification, 步骤级验证]
tags: [llm, reasoning, evaluation, reward-model]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-Level Verification** (步骤级验证) — the assessment of correctness at the level of individual intermediate reasoning steps in a multi-step solution, as opposed to evaluating only the final output.

## Key Points

- Step-level verification enables identification of the earliest erroneous step in a chain-of-thought or multi-step solution, which is more informative than a binary correct/incorrect label on the final answer.
- [[processbench]] formalises this as: given solution `S = {s_0, ..., s_{n-1}}`, output index `i ∈ {-1, 0, ..., n-1}` where `-1` means all steps are correct.
- Two main implementations: (1) [[process-reward-model]] trained to output per-step correctness scores; (2) general LLMs prompted as [[critic-model]] to judge each paragraph.
- Step-level labels are harder to obtain than outcome labels—human experts with doctoral-level mathematics knowledge still disagree on ~30% of solutions in Olympiad-level problems before reaching consensus.
- A key finding from [[processbench]]: existing auto-labeled PRMs fail to generalize to harder problems, while human-annotated PRM training (PRM800K) produces substantially more robust step-level verifiers.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-processbench-2412-06559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-processbench-2412-06559]].
