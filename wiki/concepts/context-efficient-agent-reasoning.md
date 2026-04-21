---
type: concept
title: Context-Efficient Agent Reasoning
slug: context-efficient-agent-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [上下文高效智能体推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context-Efficient Agent Reasoning** (上下文高效智能体推理) — improving agent decision quality while using less prompt budget by supplying compact high-value guidance instead of long raw histories.

## Key Points

- SkillRL replaces retrieved raw trajectories with distilled skills that fit more easily inside limited prompt windows.
- The reported average prompt length drops from roughly `1450` tokens for raw-memory prompting to below `1300` tokens with SkillRL.
- Despite using less context, the skill-based system outperforms both prompt-only memory methods and memory-augmented RL baselines.
- The paper presents this efficiency gain as evidence that abstraction improves information density rather than merely compressing text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
