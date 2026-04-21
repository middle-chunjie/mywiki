---
type: concept
title: Semantic Parsing
slug: semantic-parsing
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic parse, 语义解析]
tags: [parsing, semantics, compositionality]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Parsing** (语义解析) — the process of converting natural-language input into a structured representation of meaning that can drive downstream reasoning or execution.

## Key Points

- NS3 uses semantic parsing to convert code-search queries into a shallow action-argument structure before retrieval.
- The parse separates noun phrases from verbs so that entity discovery and action reasoning can be handled by different modules.
- Parse quality materially constrains the system: only about 40% of CodeSearchNet and 70% of CoSQA examples survive the parser in the main setup.
- The paper links performance degradation on deeper queries to probable error propagation from automatically produced parses.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[arakelyan-2022-ns-2205-10674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[arakelyan-2022-ns-2205-10674]].
