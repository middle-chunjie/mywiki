---
type: entity
title: TimeQA
slug: timeqa
date: 2026-04-20
entity_type: dataset
aliases: [Time QA, Time-sensitive QA]
tags: [benchmark, qa, temporal-reasoning]
---

## Description

TimeQA is an open-domain question answering dataset focused on time-sensitive queries (Chen et al., 2021). Questions require resolving temporal constraints (e.g., "which school did X attend between Aug 1954 and Nov 1954?"), making them challenging for LLMs that must retrieve and reason over temporally scoped facts. The dataset provides Easy and Hard splits of 2,613 questions each, totaling 5,226 test examples.

## Key Contributions

- Introduced as a benchmark requiring time-constrained factual reasoning, where both PaLM-2L and GPT-4 achieve only ~41–46% baseline accuracy.
- Used in [[zheng-2024-take-2310-06117]] to demonstrate that [[step-back-prompting]] with [[retrieval-augmented-generation]] achieves `68.7%` accuracy, a `+27%` gain over PaLM-2L baseline.
- The Hard split (time constraints more precise) particularly benefits from Step-Back Prompting: `40.4%` → `62.3%`.

## Related Concepts

- [[temporal-reasoning]]
- [[multi-hop-reasoning]]
- [[retrieval-augmented-generation]]

## Sources

- [[zheng-2024-take-2310-06117]]
