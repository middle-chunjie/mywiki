---
type: concept
title: Prompt Proposal
slug: prompt-proposal
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt proposal, prompt proposals, 提示候选]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Proposal** (提示候选) - a discrete, human-interpretable rule that maps a target completion instance and repository to a candidate context string for prompting a language model.

## Key Points

- In RLPG, each prompt proposal is defined by a prompt source and a prompt context type.
- The paper instantiates `10` prompt sources and `7` context types, producing `63` proposal types in total.
- Prompt proposals encode domain expertise explicitly, for example by extracting method bodies from imported files or post lines from the current file.
- Applicability depends on the example, so not every proposal is available for every hole.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shrivastava-2023-repositorylevel-2206-12839]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shrivastava-2023-repositorylevel-2206-12839]].
