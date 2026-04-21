---
type: concept
title: Misleading by Similarity
slug: misleading-by-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [misleading by similarity, similarity-based misleading]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Misleading by Similarity** — a failure mode in similarity-based demonstration retrieval for chain-of-thought prompting, where semantically similar wrong demonstrations reinforce the same reasoning error in the language model's prediction for a test question.

## Key Points

- When Zero-Shot-CoT generates an incorrect reasoning chain for a question, retrieving questions that are semantically similar to that question creates demonstrations that share the same mistake; the LLM then replicates the error.
- Empirically, questions with similar surface form tend to cluster in "frequent-error clusters" where Zero-Shot-CoT has a disproportionately high failure rate (e.g., `52.3%` error in one MultiArith cluster).
- Unresolved-rate analysis: Retrieval-Q-CoT leaves `46.9%` of Zero-Shot-CoT failures unresolved vs. `25.8%` for Random-Q-CoT, confirming that similarity amplifies error propagation.
- The phenomenon motivates diversity-based demonstration selection (Auto-CoT): sampling one question per cluster ensures no more than `1/k` demonstrations come from any single frequent-error cluster.
- This analysis implies that quality of demonstration diversity matters more than semantic proximity to the test question when ground-truth annotations are unavailable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-automatic-2210-03493]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-automatic-2210-03493]].
