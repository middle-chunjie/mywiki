---
type: concept
title: Evidence Pattern
slug: evidence-pattern
date: 2026-04-20
updated: 2026-04-20
aliases: [EP, 证据模式]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Evidence Pattern** (证据模式) — a variable-substituted graph pattern that captures how topic entities, intermediate variables, and relations must connect to support an answer for a given question.

## Key Points

- The paper defines `pat(SG, q)` by replacing entities not mentioned in the question with variables while preserving the structural dependencies among remaining resources.
- Evidence patterns are used as the latent target of subgraph extraction, replacing the weaker objective of collecting individually relevant facts.
- Candidate patterns are considered valid only if they cover all topic entities and satisfy constrained endpoint structures that avoid arbitrary path growth.
- A BERT cross-encoder ranks candidate evidence patterns after retrieval and enumeration, selecting the best structure for downstream reasoning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-enhancing-2402-02175]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-enhancing-2402-02175]].
