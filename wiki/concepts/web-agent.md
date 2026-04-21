---
type: concept
title: Web Agent
slug: web-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [web agent, browsing agent]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Web Agent** — an LLM-based agent that interacts with search engines, webpages, or browser environments to gather evidence and complete tasks on the web.

## Key Points

- WebSailor treats web agents as systems that must iteratively think, call tools, read observations, and decide when to answer.
- The paper argues that strong web-agent performance depends on reducing uncertainty in large unstructured information spaces, not just executing a fixed multi-hop chain.
- WebSailor trains web agents with synthetic high-difficulty QA, short reconstructed reasoning traces, and reinforcement learning.
- The reported gains are largest on BrowseComp, where browsing strategy and evidence synthesis matter more than parametric recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-websailor-2507-02592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-websailor-2507-02592]].
