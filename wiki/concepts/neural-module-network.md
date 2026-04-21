---
type: concept
title: Neural Module Network
slug: neural-module-network
date: 2026-04-20
updated: 2026-04-20
aliases: [NMN, neural modular network, 神经模块网络]
tags: [modularity, compositionality, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural Module Network** (神经模块网络) — a dynamically assembled neural architecture whose computation graph is composed from reusable modules according to the structure of an input query or task.

## Key Points

- NS3 instantiates an NMN from the semantic parse of each query, with noun phrases mapped to entity modules and verbs mapped to action modules.
- The top-level node is always an action module, while nested actions are handled by multiplying action-level compatibility scores.
- Entity modules localize candidate token spans in code, and action modules predict a masked argument conditioned on the remaining query arguments.
- The modular layout provides a more interpretable staged retrieval signal than a monolithic text-code encoder.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[arakelyan-2022-ns-2205-10674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[arakelyan-2022-ns-2205-10674]].
