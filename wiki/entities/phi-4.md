---
type: entity
title: Phi-4
slug: phi-4
date: 2026-04-20
entity_type: tool
aliases: [phi4, Phi 4]
tags: []
---

## Description

Phi-4 is Microsoft's 14B-parameter decoder-only language model described in the Phi-4 technical report. Its defining contribution is a data-centric training recipe emphasizing synthetic data, curation, and post-training rather than a new backbone architecture.

## Key Contributions

- Demonstrates that a `14B` model can surpass GPT-4o on GPQA and MATH through data quality and post-training.
- Combines `10T`-token pretraining, `4K`-to-`16K` midtraining, and two-stage DPO with pivotal-token preference data.
- Serves as the central artifact evaluated throughout the report's benchmark and safety analysis.

## Related Concepts

- [[large-language-model]]
- [[synthetic-data]]
- [[pretraining-data-mixture]]
- [[direct-preference-optimization]]
- [[long-context-training]]

## Sources

- [[abdin-2024-phi-2412-08905]]
