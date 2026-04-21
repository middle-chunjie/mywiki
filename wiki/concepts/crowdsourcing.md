---
type: concept
title: Crowdsourcing
slug: crowdsourcing
date: 2026-04-20
updated: 2026-04-20
aliases: [众包]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Crowdsourcing** (众包) — collecting annotations or judgments from a distributed pool of human workers under a shared task protocol.

## Key Points

- DL-MIA uses crowdsourcing to validate and refine LLM-generated intent candidates with human supervision.
- Annotators see a query, its passages, and proposed intents, and they may add, modify, or reject intents during labeling.
- Queries with more than `30` relevant passages are split into chunks of `30` passages to keep annotation manageable.
- The paper reports `22` annotation sets by `16` annotators, with each query annotated at least twice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[anand-2024-understanding-2408-17103]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[anand-2024-understanding-2408-17103]].
