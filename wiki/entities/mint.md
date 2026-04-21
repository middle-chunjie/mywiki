---
type: entity
title: MINT
slug: mint
date: 2026-04-20
entity_type: benchmark
aliases: [MINT benchmark]
tags: []
---

## Description

MINT is the agent-task evaluation benchmark used in [[wang-2024-executable-2402-01030]] to assess CodeActAgent on both in-domain and out-of-domain CodeAct tasks. The paper reports results with interaction budget `k = 5`.

## Key Contributions

- Serves as the main CodeAct evaluation suite for the trained open-source agents.
- Separates in-domain and out-of-domain performance because some training domains overlap MINT.
- Helps show that CodeActAgent improves on executable-action tasks beyond raw backbone baselines.

## Related Concepts

- [[tool-augmented-agent]]
- [[multi-turn-interaction]]
- [[instruction-tuning]]

## Sources

- [[wang-2024-executable-2402-01030]]
