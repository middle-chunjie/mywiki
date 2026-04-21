---
type: concept
title: Instruction Synthesis
slug: instruction-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic instruction generation, 指令合成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Synthesis** (指令合成) — the generation of synthetic user prompts or task instructions intended to represent realistic downstream requests for an LLM.

## Key Points

- The paper uses personas to simulate heterogeneous end users and generate instructions that reflect their likely requests.
- It studies both zero-shot instruction synthesis and persona-enhanced few-shot synthesis seeded from an existing instruction dataset such as WildChat.
- The synthesized instructions are positioned as the first turn of simulated user-assistant conversations.
- The authors argue that scaling this process can produce large conversational corpora for improving instruction following and dialogue behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chan-2024-scaling-2406-20094]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chan-2024-scaling-2406-20094]].
