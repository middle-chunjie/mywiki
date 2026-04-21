---
type: concept
title: Cross-Task Transfer
slug: cross-task-transfer
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-domain transfer, 跨任务迁移]
tags: [transfer-learning, agents]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Task Transfer** (跨任务迁移) — performance gains on a target task that arise from training on data drawn from other tasks or domains rather than only from target-specific supervision.

## Key Points

- The paper compares mixed ADP training against task-specific-only tuning while keeping the evaluation harness fixed.
- On SWE-Bench, mixed ADP data reaches `10.4%` versus `1.0%` for Qwen2.5-7B-Instruct trained only on SWE-smith, and `16.6%` versus `11.0%` or `0.2%` for Qwen3-8B.
- On WebArena, ADP training reaches `20.1%` versus `16.0%` for Go-Browse-only tuning; on AgentBench OS it reaches `25.7%` versus `21.5%` for AgentInstruct-only tuning.
- The GAIA comparison (`9.1%` vs `0.6%`) suggests mixed-domain agent data can avoid the negative transfer induced by narrow single-domain tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[song-2026-agent-2510-24702]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[song-2026-agent-2510-24702]].
