---
type: entity
title: DL-MIA
slug: dl-mia
date: 2026-04-20
entity_type: dataset
aliases: [MS MARCO Intent Annotations]
tags: []
---

## Description

DL-MIA is the intent-annotated ranking dataset introduced in [[anand-2024-understanding-2408-17103]], derived from TREC-DL queries and passage judgments.

## Key Contributions

- Contributes `2655` `(query, intent, passage, label)` tuples over `24` queries and `69` finalized intents.
- Supports evaluation for ranking, diversification, intent coverage, and query suggestion.
- Makes user-intent mismatch measurable at passage level through intent-aware QRel scores.

## Related Concepts

- [[ranking-dataset]]
- [[user-intent]]
- [[relevance-judgment]]

## Sources

- [[anand-2024-understanding-2408-17103]]
