---
type: concept
title: Retrieval-Augmented Verification
slug: retrieval-augmented-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval augmented verification, 检索增强验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Augmented Verification** (检索增强验证) — a reasoning control mechanism that uses retrieved evidence to score and optionally refine intermediate model outputs rather than injecting retrieval directly into every generation step.

## Key Points

- RAG-Star applies retrieval after a sub-query and answer have already been generated from the policy model's internal knowledge.
- The verification module evaluates both whether the answer is supported by retrieved documents and whether the planned sub-query is logically consistent with the prior reasoning path.
- Retrieved evidence can trigger answer refinement when the model's proposed answer conflicts with external documents.
- The paper frames this design as a way to reduce knowledge conflict between parametric memory and external retrieval while still benefiting from factual grounding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragstar-2412-12881]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragstar-2412-12881]].
