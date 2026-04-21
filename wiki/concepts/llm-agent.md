---
type: concept
title: LLM Agent
slug: llm-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [LLM agent, language-model agent, LLM 智能体]
tags: [agents, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LLM Agent** (LLM 智能体) — a large-language-model-based system that iteratively reasons, invokes tools, and updates state to complete multi-step tasks in an external environment.

## Key Points

- The paper models an agent as a component `pi : H x E -> H x E x R` that alternates between conversation updates and environment interactions.
- TOOLMAKER relies on tool-augmented LLM agents for repository installation, repository exploration, and error diagnosis rather than using a single monolithic prompt.
- The work argues that agent capability is fundamentally constrained by the breadth and quality of the tools available to it.
- In this setting, LLM agents are evaluated not only on reasoning but on their ability to interact with filesystems, dependency managers, and executable code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[w-lflein-2025-llm-2502-11705]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[w-lflein-2025-llm-2502-11705]].
