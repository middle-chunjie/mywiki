---
type: concept
title: LLM-as-a-Judge
slug: llm-as-a-judge
date: 2026-04-20
updated: 2026-04-20
aliases: [大语言模型充当评审, model-based evaluation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LLM-as-a-Judge** (大语言模型充当评审) — an evaluation setup in which a language model grades generated outputs against criteria such as support, relevance, coverage, or entailment.

## Key Points

- AstaBench uses judge models for long-form QA, literature-table generation, hypothesis matching, and end-to-end scientific discovery.
- ScholarQA-CS2 uses judge prompts for citation recall, citation precision, answer relevance, and answer coverage.
- DiscoveryBench evaluates predicted hypotheses against gold hypotheses through the structured facets of context, variables, and relationship.
- The paper reports a `0.995` Pearson correlation between `gemini-2.5-flash` and `gemini-2.5-pro` on its ScholarQA-CS2 judging setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]
- [[wu-2025-webdancer-2505-22648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
