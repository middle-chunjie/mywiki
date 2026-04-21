---
type: concept
title: Tool-Augmented Language Model
slug: tool-augmented-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [tool-using language model, 工具增强语言模型]
tags: [agents, llm, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool-Augmented Language Model** (工具增强语言模型) — a language model extended with external tools or APIs so it can retrieve information, invoke functions, and act beyond its parametric knowledge.

## Key Points

- ToolWeaver studies tool-augmented LLMs in the large-scale setting where the tool inventory contains `46,985` APIs.
- The paper models interaction as trajectories containing planning, tool selection, parameter generation, execution feedback, and final answer synthesis.
- It argues that external retrieval pipelines add latency and context-window pressure, motivating native generative tool selection inside the model.
- ToolWeaver focuses on making tool-augmented LLMs scale without injecting one isolated token per tool.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
