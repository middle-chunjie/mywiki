---
type: entity
title: WikiEval
slug: wikieval
date: 2026-04-20
entity_type: tool
aliases: [WikiEval dataset]
tags: []
---

## Description

WikiEval is the human-annotated benchmark introduced in [[es-2023-ragas-2309-15217]] to compare automated judgments of RAG outputs against human preferences.

## Key Contributions

- Contains examples built from `50` recent Wikipedia pages and evaluates faithfulness, answer relevance, and context relevance.
- Uses pairwise human judgments from `2` annotators, with disagreements resolved by discussion.
- Serves as the empirical testbed showing that RAGAS outperforms GPT scoring and ranking baselines.

## Related Concepts

- [[reference-free-evaluation]]
- [[faithfulness]]
- [[answer-relevance]]
- [[context-relevance]]

## Sources

- [[es-2023-ragas-2309-15217]]
