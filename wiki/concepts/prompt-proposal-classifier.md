---
type: concept
title: Prompt Proposal Classifier
slug: prompt-proposal-classifier
date: 2026-04-20
updated: 2026-04-20
aliases: [PPC, prompt proposal classifier, 提示候选分类器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Proposal Classifier** (提示候选分类器) - a model that predicts which prompt proposal is most likely to yield a successful completion for a specific example.

## Key Points

- The paper trains PPC with a multi-label target where a proposal is positive if it is applicable and makes Codex exactly match the target hole.
- RLPG-H encodes only the hole window and maps a frozen CodeBERT representation through an MLP to `63` proposal logits.
- RLPG-R conditions on both the hole window and proposal context through multi-head attention before scoring proposals.
- The classifier is what turns a hand-designed proposal inventory into automatic, example-specific prompt generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shrivastava-2023-repositorylevel-2206-12839]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shrivastava-2023-repositorylevel-2206-12839]].
