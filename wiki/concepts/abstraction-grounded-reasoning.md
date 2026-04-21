---
type: concept
title: Abstraction-Grounded Reasoning
slug: abstraction-grounded-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [abstraction grounded reasoning, abstraction-based reasoning, 抽象引导推理]
tags: [reasoning, prompting, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Abstraction-Grounded Reasoning** (抽象引导推理) — the second stage of [[step-back-prompting]], where an LLM derives the final answer to a specific question by reasoning over high-level concepts or first principles extracted in the prior abstraction step, rather than reasoning directly over the low-level details of the original question.

## Key Points

- Avoids error propagation caused by fixating on irrelevant surface details: by grounding reasoning on abstract principles, the model is less likely to deviate mid-chain.
- In STEM settings, the "first principles" (e.g., Ideal Gas Law, Gibbs free energy) are recalled from the LLM's parametric knowledge in the abstraction step, then used as explicit context for reasoning.
- In knowledge-intensive QA settings, facts about the abstract concept (e.g., a person's full employment or education history) are retrieved via [[retrieval-augmented-generation]] using the step-back question, providing broader context than direct retrieval on the original query.
- Empirically, the reasoning step remains the performance bottleneck: `>90%` of Step-Back errors occur here (Reasoning Error, Math Error, Context Loss, Factual Error), while Principle Error (failure at abstraction) is `<10%`.
- Distinct from task decomposition: reasoning is performed once over abstracted context, not recursively over sub-questions.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-take-2310-06117]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-take-2310-06117]].
