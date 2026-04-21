---
type: entity
title: FLAN
slug: flan
date: 2026-04-20
entity_type: model
aliases: [FLAN 137B]
tags: []
---

## Description

FLAN is the instruction-tuned large language model used as the query generator in [[dai-2022-promptagator-2209-11755]]. The paper specifically uses a `137B` checkpoint to synthesize task-specific queries from target-corpus documents.

## Key Contributions

- Generates up to `8` synthetic queries per sampled document for PROMPTAGATOR training.
- Enables task adaptation from only `2-8` exemplars without directly training on MS MARCO or Natural Questions retrieval labels.
- Supports an ablation showing that removing NQ and Quora from FLAN training slightly lowers but does not erase retrieval gains.

## Related Concepts

- [[large-language-model]]
- [[question-generation]]
- [[prompt-engineering]]

## Sources

- [[dai-2022-promptagator-2209-11755]]
