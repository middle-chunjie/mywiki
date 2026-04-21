---
type: concept
title: Event Argument Extraction
slug: event-argument-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [EAE, event argument extraction, 事件论元抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Event Argument Extraction** (事件论元抽取) — the task of identifying the participants or roles associated with a detected event and assigning them argument-role labels.

## Key Points

- [[ma-2023-large-2303-08559]] includes EAE in the benchmark using ACE05, ERE, and RAMS.
- The paper evaluates EAE with head-F1 rather than full-span F1, following prior work.
- Direct LLM prompting on EAE shows mixed behavior but still does not overturn the overall conclusion that supervised SLMs are stronger few-shot IE systems in realistic settings.
- EAE is part of the evidence base for the paper's broader claim that LLMs do not scale well with complex label schemas and flexible IE output formats.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
