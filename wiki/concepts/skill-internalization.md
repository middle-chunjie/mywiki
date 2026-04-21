---
type: concept
title: Skill Internalization
slug: skill-internalization
date: 2026-04-20
updated: 2026-04-20
aliases: [internalised skills, internalized skills]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Skill Internalization** — the process of transferring externally provided procedural skills into model parameters so the policy can execute them without runtime retrieval or prompting.

## Key Points

- Skill0 treats skill internalization as the central training objective rather than a byproduct of better prompting.
- The paper operationalizes internalization by training with skills present in context and evaluating without any skill retrieval at inference.
- Helpfulness scores `Delta_k` are used to decide when a skill file still contributes to learning and when it can be removed.
- Positive transfer after removing skill prompts (`+1.6` on ALFWorld) is presented as evidence that behavior moved from context into parameters.
- The method contrasts internalization with conventional skill augmentation, where the model follows instructions but does not acquire them intrinsically.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2026-skill-2604-02268]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2026-skill-2604-02268]].
