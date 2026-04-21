---
type: concept
title: Knowledge Sufficiency Evaluation
slug: knowledge-sufficiency-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge adequacy assessment, 知识充分性评估]
tags: [rag, metacognition, knowledge-conflict, multi-hop-qa]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Sufficiency Evaluation** (知识充分性评估) — a diagnostic procedure that independently assesses whether an LLM's parametric (internal) knowledge and its retrieved (external) knowledge are each sufficient to answer a given question, used to classify failure conditions and select repair strategies.

## Key Points

- MetaRAG distinguishes four knowledge conditions: (1) no knowledge, (2) only internal, (3) only external, (4) both — each calling for a different planning strategy.
- Internal sufficiency is evaluated via `LLM_Eval-Critic(q, Prompt_Eval)` producing a binary "can I answer this from my own knowledge?" judgment; consistency with human annotation is 0.76.
- External sufficiency is evaluated by an NLI model (TRUE, T5-XXL): `f([d_i], q) ∈ {0, 1}` — whether retrieved passages entail an answer to the question; human annotation consistency is 0.84.
- The framework was validated with a preliminary human annotation study on 100 HotpotQA questions, which motivated the design of these automatic evaluators.
- External knowledge sufficiency is more impactful than internal: ablating the external check causes a larger EM drop (`37.4`) than ablating the internal check (`41.4`) on 2WikiMultihopQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-metacognitive-2402-11626]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-metacognitive-2402-11626]].
