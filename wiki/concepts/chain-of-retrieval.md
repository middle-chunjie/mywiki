---
type: concept
title: Chain-of-Retrieval
slug: chain-of-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [CoR, chain of retrieval, 链式检索提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Retrieval** (链式检索提示) — a multi-step prompting procedure that incrementally elicits related relations, documents, entities, triplets, reasoning, and supporting evidence instead of asking an LLM to generate a final labeled sample in one shot.

## Key Points

- [[sun-2024-consistency-2401-13598]] uses chain-of-retrieval to first choose relations related to an unseen target relation, then generate a document, then derive entities and triplets step by step.
- The prompting process keeps conversational memory between steps so later outputs can condition on earlier generated artifacts.
- During document generation, the paper sets ChatGPT `temperature = 1` to encourage diversity in synthetic documents.
- Compared with vanilla prompting, chain-of-retrieval improves ZeroDocRTE test `F1` from `9.04 -> 13.23` on Re-DocRED and `9.77 -> 13.38` on DocRED.
- The same prompt design also improves ZeroDocRE test `F1` from `42.45 -> 49.21` on Re-DocRED and `34.98 -> 48.30` on DocRED.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-consistency-2401-13598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-consistency-2401-13598]].
