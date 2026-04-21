---
type: concept
title: Task-Specific Skill Retrieval
slug: task-specific-skill-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [任务特定技能检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task-Specific Skill Retrieval** (任务特定技能检索) — retrieving only the subset of stored skills whose semantic content matches the current task description, so the agent receives specialized guidance without excessive context expansion.

## Key Points

- SkillRL always includes general skills, then ranks task-specific skills by similarity between task and skill embeddings.
- Retrieval follows `\mathrm{TopK}(\{s \in \mathcal{S}_k : \mathrm{sim}(e_d, e_s) > \delta\}, K)` with `K = 6` in the reported implementation.
- This design lets the prompt mix broad strategic priors with category-level heuristics tailored to the current task instance.
- The retrieval stage is central to making distilled skills more useful than long raw memory traces under a fixed context window.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
