---
type: concept
title: Black-Box Knowledge Injection
slug: black-box-knowledge-injection
date: 2026-04-20
updated: 2026-04-20
aliases: [black-box KG injection, 黑盒知识注入]
tags: [knowledge-graph, llm, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Knowledge Injection** (黑盒知识注入) — the paradigm of enriching a closed-source LLM with structured external knowledge (e.g., from a knowledge graph) via prompt engineering alone, without access to model weights or architecture.

## Key Points

- Formally defined in KnowGPT as learning a prompting function `f_prompt(Q, G)` that incorporates KG factual knowledge into a text prompt such that the black-box LLM `f(x)` produces correct answers — using only the model's API.
- Contrasts with white-box injection (which modifies attention, embeddings, or weights) and is the only feasible approach for proprietary models like ChatGPT and GPT-4.
- Two core challenges: (1) efficiently selecting the most informative knowledge triples from million-entity KGs within tight token budgets, and (2) encoding that knowledge in a prompt format the LLM can best exploit.
- KnowGPT addresses these with deep RL for path extraction and a Multi-Armed Bandit for adaptive prompt format selection, achieving 23.7% average gain over ChatGPT on three QA benchmarks.
- The approach avoids the high computation cost of white-box methods (weight updates) and can be applied to any black-box LLM that exposes a text API.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowgpt-2312-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowgpt-2312-06185]].
