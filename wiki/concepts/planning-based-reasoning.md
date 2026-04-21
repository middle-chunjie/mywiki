---
type: concept
title: Planning-Based Reasoning
slug: planning-based-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning with planning, plan-guided reasoning]
tags: [reasoning, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Planning-Based Reasoning** (基于规划的推理) — an approach to reasoning that explicitly selects and sequences intermediate actions according to their estimated future utility instead of generating one fixed reasoning trace greedily.

## Key Points

- CR-Planner treats reasoning and retrieval as actions inside one decision process rather than as a fixed pipeline.
- The next action is chosen by maximizing critic-estimated reward over available sub-goals or execution candidates.
- This design lets the system decide not only what rationale to produce, but also when retrieval is worth doing.
- The paper argues this is especially useful on tasks where one early mistake can corrupt the rest of the reasoning chain.
- The method improves hard competitive programming and theorem-driven math more than standard CoT or vanilla RAG.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
