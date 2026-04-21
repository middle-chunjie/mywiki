---
type: concept
title: Agentic LLM
slug: agentic-llm
date: 2026-04-20
updated: 2026-04-20
aliases: [agentic large language model, LLM agent]
tags: [agents, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic LLM** (智能体式大语言模型) — a large language model deployed as an interactive policy that reasons over observations, invokes tools and memory operations, and optimizes delayed task rewards across multi-step trajectories.

## Key Points

- The paper frames agentic LLMs as sequential decision-makers rather than single-turn text generators.
- Tool calls and memory updates are treated as first-class actions that should be logged and evaluated, not hidden implementation details.
- ASG-SI argues that self-improvement for such agents should be mediated by auditable skill promotion rather than unrestricted policy drift.
- The proposed training loop separates online interaction from offline compilation and verification to keep agent behavior inspectable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
