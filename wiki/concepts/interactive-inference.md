---
type: concept
title: Interactive Inference
slug: interactive-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [tool-interactive inference, 交互式推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Interactive Inference** (交互式推理) — an inference procedure where a model alternates between generating reasoning, issuing tool calls, receiving tool outputs, and continuing from the updated context until it reaches a final answer.

## Key Points

- SMARTAgent is evaluated in an interactive loop that parses tool calls, executes them externally, and appends both the call and the result back into context.
- The training data is structured so that each input-output pair advances reasoning up to the next tool-reliant step, matching the interaction pattern used at test time.
- Because every extra tool call lengthens the trajectory, interactive inference makes tool-overuse directly visible as latency and error accumulation.
- The paper argues that strategic switching is especially important under this inference regime because poor decisions propagate across multiple rounds.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qian-2025-smart-2502-11435]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qian-2025-smart-2502-11435]].
