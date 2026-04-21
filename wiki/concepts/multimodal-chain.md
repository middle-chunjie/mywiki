---
type: concept
title: Multimodal Chain
slug: multimodal-chain
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal chaining, 多模态链式调用]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Chain** (多模态链式调用) — a multi-step system pattern where a model coordinates external tools or experts across sequential thought-action-observation steps while passing multimodal information between steps.

## Key Points

- The paper extends ReAct-style chaining from language-only settings to multimodal settings where tools can consume or return image-derived information.
- In the PPE example, the chain runs for `2` rounds: person detection first, then image cropping and helmet checking on the resulting person crops.
- The authors argue that multimodal chains overcome limits of single-shot prompting by decomposing tasks and exposing intermediate observations back to GPT-4V.
- This pattern is presented as a natural systems direction for stronger multimodal agents that combine reasoning, perception, and tool use.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-dawn-2309-17421]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-dawn-2309-17421]].
