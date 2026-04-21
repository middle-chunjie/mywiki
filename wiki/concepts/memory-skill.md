---
type: concept
title: Memory Skill
slug: memory-skill
date: 2026-04-20
updated: 2026-04-20
aliases: [memory skills, 记忆技能]
tags: [agents, memory, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory Skill** (记忆技能) — a structured, reusable unit of procedural guidance that specifies *when* and *how* interaction traces should be transformed into memory entries, enabling learnable and evolvable memory operations in LLM agents.

## Key Points

- In MemSkill, each skill consists of a short *description* (used for embedding-based selection) and a detailed *content* specification that instructs the LLM executor on how to perform memory extraction or revision.
- The skill bank is initialized with four general-purpose primitives (`Insert`, `Update`, `Delete`, `Skip`) and expanded by an LLM designer that analyzes hard failure cases.
- Skills are *reusable*: unlike trace-specific memories, they are shared across all interaction traces and can be applied to different span lengths and domains without retraining.
- The skill-conditioned formulation enables *compositional* memory construction—a controller selects a Top-K subset of relevant skills, and the executor applies them in a single LLM call.
- Qualitative analysis shows that evolved skills develop clear domain specialization: LoCoMo skills emphasize temporal context and activity structure, while ALFWorld skills focus on action constraints and object locations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2026-memskill-2602-02474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2026-memskill-2602-02474]].
