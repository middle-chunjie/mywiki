---
type: concept
title: Self-Summarization
slug: self-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-summarization** (自总结) — a long-horizon agent technique in which the model periodically compresses prior interaction state into its own summary and then continues generation conditioned on that compressed state.

## Key Points

- Composer 2 uses self-summarization so a rollout can span multiple generations linked by model-written summaries instead of a single prompt-response pair.
- The final rollout reward is applied to all tokens in the chain, so summaries that preserve useful state are reinforced alongside successful action sequences.
- The report says self-summarization lets the agent process more information under a limited context window and may be invoked multiple times on difficult tasks.
- Compared with separate prompt-based compaction, the paper reports lower error, fewer tokens, and better KV-cache reuse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[research-2026-composer-2603-24477]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[research-2026-composer-2603-24477]].
