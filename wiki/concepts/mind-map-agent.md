---
type: concept
title: Mind-Map Agent
slug: mind-map-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [Mind-Map, mind map agent, 思维导图代理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Mind-Map Agent** (思维导图代理) — a structured memory agent that transforms an evolving reasoning trace into a knowledge graph, clusters it, and serves summarized or query-targeted context back to the main reasoner.

## Key Points

- The agent extracts entities and relations from the current reasoning chain to build a knowledge graph online.
- It applies community clustering to partition the graph and summarize each cluster for context compression.
- The main reasoner can query the Mind-Map when long reasoning chains or many tool calls cause context drift.
- In the paper's ablations, Mind-Map yields the best GAIA performance among the compared memory strategies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-agentic-2502-04644]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-agentic-2502-04644]].
