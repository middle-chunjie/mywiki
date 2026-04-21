---
type: concept
title: Metacognitive Regulation
slug: metacognitive-regulation
date: 2026-04-20
updated: 2026-04-20
aliases: [metacognitive control, 元认知调节]
tags: [metacognition, reasoning, rag]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Metacognitive Regulation** (元认知调节) — the active management and control of one's cognitive processes, comprising three sequential stages: monitoring (assessing whether the current output is satisfactory), evaluating (diagnosing why it may be flawed), and planning (selecting a corrective strategy).

## Key Points

- MetaRAG operationalizes metacognitive regulation as a three-step pipeline: Monitoring → Evaluating → Planning, executed iteratively up to a maximum of 5 rounds.
- Monitoring gates the evaluation stage via semantic similarity between the base LLM's answer and an expert model's answer; only when similarity falls below threshold `k` is full regulation triggered.
- Evaluating employs two types of metacognitive knowledge: procedural knowledge (checks internal and external knowledge sufficiency using LLM and NLI model respectively) and declarative knowledge (checks for incomplete reasoning, answer redundance, and ambiguity understanding).
- Planning produces tailored actions per diagnosis: query expansion for insufficient knowledge, prompt override for conflicting knowledge, NLI-based statement verification and suggestion generation for erroneous reasoning.
- The regulation loop mirrors Schraw & Moshman's (1995) cognitive psychology theory of metacognitive regulation applied to LLM inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-metacognitive-2402-11626]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-metacognitive-2402-11626]].
