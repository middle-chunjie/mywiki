---
type: concept
title: Declarative Knowledge
slug: declarative-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [factual knowledge, 陈述性知识]
tags: [metacognition, reasoning, knowledge]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Declarative Knowledge** (陈述性知识) — knowledge about facts, concepts, and patterns relevant to a task domain; in the context of LLM metacognition, it refers to an agent's awareness of common error types and their characteristics.

## Key Points

- MetaRAG uses declarative knowledge in its evaluating stage to detect three error patterns: Incomplete Reasoning (failure to follow a full chain-of-thought), Answer Redundance (overly verbose or repetitious answers), and Ambiguity Understanding (misinterpretation of query nuances).
- Each error type is described with a name, description, and examples formatted as `{Error name - Error description - Examples}` and fed to the evaluator-critic LLM.
- Declarative knowledge complements procedural knowledge: procedural knowledge checks whether sufficient knowledge exists, while declarative knowledge checks whether reasoning is logically sound given available knowledge.
- Ablation results show that incomplete-reasoning detection has the highest individual impact among the three declarative knowledge components.
- Contrasts with [[procedural-knowledge]], which concerns know-how about solving tasks rather than knowledge about error patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-metacognitive-2402-11626]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-metacognitive-2402-11626]].
