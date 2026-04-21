---
type: concept
title: Tool Overuse
slug: tool-overuse
date: 2026-04-20
updated: 2026-04-20
aliases: [excessive tool use, 工具过度使用]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Overuse** (工具过度使用) — the unnecessary invocation of external tools on problems that a model could already solve correctly using its own parametric knowledge.

## Key Points

- The paper formalizes tool overuse as `O = alpha * beta`, combining a model's no-tool competence with its tendency to still call tools on those solvable questions.
- Preliminary experiments show that both plain LLMs and full agent systems exhibit substantial overuse, with Llama-3.1-8B exceeding `50%` on GSM8K.
- The authors argue that overuse is not merely inefficient; it can directly lower answer accuracy by introducing extra interaction steps and tool-related failure modes.
- SMART is designed specifically to mitigate tool overuse by teaching explicit justifications for when internal reasoning is sufficient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qian-2025-smart-2502-11435]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qian-2025-smart-2502-11435]].
