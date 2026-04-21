---
type: concept
title: Black-Box Prompt Optimization
slug: black-box-prompt-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [BPO, black-box prompt optimization]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Prompt Optimization** (黑盒提示优化) — an alignment method that rewrites user prompts to better match a target model's preferences and capabilities without updating the target model's parameters.

## Key Points

- BPO constructs optimization targets by asking an LLM to compare preferred and dispreferred responses and then rewrite the original instruction accordingly.
- The paper builds about `14k` `(X_user, X_opt)` pairs from OASST1, HH-RLHF, Chatbot Arena, and Alpaca-GPT4 preference data.
- A `llama-2-7b-chat-hf` optimizer is trained for `3` epochs to learn the mapping from raw prompts to optimized prompts.
- The method is model-agnostic at application time: the same optimizer is used to improve API-only systems such as GPT-4 and open models such as Vicuna.
- Unlike PPO or DPO, BPO exposes alignment changes directly in rewritten prompts, making its behavior easier to inspect and debug.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-blackbox-2311-04155]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-blackbox-2311-04155]].
