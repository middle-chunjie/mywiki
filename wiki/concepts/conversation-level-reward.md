---
type: concept
title: Conversation-level Reward
slug: conversation-level-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [dialogue-level reward, 对话级奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Conversation-level Reward** (对话级奖励) — a reward function defined over an entire dialogue trajectory that measures whether the interaction as a whole achieves the user's goal with acceptable process quality.

## Key Points

- The paper approximates conversation-level reward by combining extrinsic task success and intrinsic interaction quality.
- Extrinsic reward is computed from the extracted final solution and a task-specific scoring function against the reference target.
- Intrinsic reward penalizes excessive token count and adds an LLM-judged interactivity term.
- This full-trajectory reward is the target quantity that MR tries to estimate at the response level.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
