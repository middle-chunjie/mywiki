---
type: concept
title: Feedback Generation
slug: feedback-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [feedback synthesis, 反馈生成]
tags: [llm, debugging, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Feedback Generation** (反馈生成) — the step that turns raw execution errors or failing behavior into natural-language explanations that a model can condition on for subsequent repair.

## Key Points

- [[unknown-nd-selfrepair-2306-09896]] explicitly separates feedback generation from repair so the diagnostic stage can be analyzed in isolation.
- In the formalism, feedback is sampled as `f_ij ~iid M_F(psi; p_i; e_i)`, where the feedback model conditions on the task, the buggy program, and its error message.
- Replacing a weaker model's self-generated feedback with explanations from a stronger model consistently improves repair success on both HumanEval and APPS.
- Human-written feedback improves GPT-4 repair much more than GPT-4's own feedback, suggesting that high-quality diagnosis remains a core bottleneck.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-selfrepair-2306-09896]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-selfrepair-2306-09896]].
