---
type: concept
title: Step-Back Prompting
slug: step-back-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [step back prompting, StepBack Prompting, 后退一步提示]
tags: [prompting, reasoning, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-Back Prompting** (后退一步提示) — a two-stage prompting technique for LLMs that first prompts the model to reformulate a specific question into a more abstract, higher-level step-back question (eliciting underlying concepts or first principles), then grounds the final answer on the abstracted knowledge via a second LLM call.

## Key Points

- **Two-stage pipeline**: (1) Abstraction — generate a step-back question at a higher conceptual level; (2) [[abstraction-grounded-reasoning]] — answer the original question conditioned on facts retrieved for the step-back question.
- The step-back question has a many-to-one mapping from original questions to abstract concepts, unlike [[chain-of-thought-prompting]] (which decomposes one question into many sub-questions) — abstraction is orthogonal to decomposition.
- Sample-efficient: as few as one few-shot demonstration suffices to teach the abstraction step, because LLMs already have implicit abstraction capability that needs only a light trigger.
- Integrates naturally with [[retrieval-augmented-generation]] for knowledge-intensive tasks: the step-back question replaces the original question as the retrieval query, retrieving broader, more reliably relevant context.
- Yields substantial gains over [[chain-of-thought-prompting]] on STEM (MMLU Physics +7%, Chemistry +11%), Knowledge QA (TimeQA +27%), and Multi-Hop Reasoning (MuSiQue +7%) using PaLM-2L.
- Error analysis shows `>90%` of remaining failures stem from the Reasoning step, not the Abstraction step, pointing to LLM reasoning capacity as the binding constraint.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-take-2310-06117]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-take-2310-06117]].
