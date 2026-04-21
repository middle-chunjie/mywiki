---
type: concept
title: Generative Tool Use
slug: generative-tool-use
date: 2026-04-20
updated: 2026-04-20
aliases: [generative tool calling, 生成式工具使用]
tags: [agents, llm, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generative Tool Use** (生成式工具使用) — an approach where a language model directly generates tool identifiers, arguments, and related reasoning as part of autoregressive decoding.

## Key Points

- ToolWeaver follows the generative paradigm because it unifies selection and execution inside one model rather than relying on a separate retriever at inference time.
- The paper criticizes the common one-token-per-tool version of generative tool use for linear vocabulary growth and semantically isolated tool IDs.
- ToolWeaver keeps the generative formulation but replaces atomic identifiers with multi-token hierarchical codes.
- End-to-end training on `183,336` trajectories teaches the model to generate reasoning, tool codes, arguments, and final answers jointly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
