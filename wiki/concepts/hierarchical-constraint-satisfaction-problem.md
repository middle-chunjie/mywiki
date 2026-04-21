---
type: concept
title: Hierarchical Constraint Satisfaction Problem
slug: hierarchical-constraint-satisfaction-problem
date: 2026-04-20
updated: 2026-04-20
aliases: [HCSP, hierarchical CSP, 层次化约束满足问题]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Constraint Satisfaction Problem** (层次化约束满足问题) — a reasoning formulation in which the final answer is obtained by satisfying both local constraints and recursively defined sub-questions over multiple levels.

## Key Points

- The paper uses HCSP as the formal definition of verifiable deep-research QA rather than treating the task as ordinary multi-hop retrieval.
- Its decomposition is written as `H(x) = \bigcap_{i=1}^{k} S(c_i) \cap \bigcap_{j=1}^{m} H(y_j)`, combining direct constraints and recursively solved sub-questions.
- Standard CSP and multi-hop reasoning are framed as special cases within the HCSP family.
- The hierarchy is intended to force progressive pruning of candidates instead of allowing a single shortcut constraint to identify the answer.
- The authors explicitly note two failure modes for HCSP construction: underdetermined trees and overdetermined trees.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2025-open-2509-00375]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2025-open-2509-00375]].
