---
type: concept
title: Exam Informativeness
slug: exam-informativeness
date: 2026-04-20
updated: 2026-04-20
aliases: [item information function, exam information, 考试信息量]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exam Informativeness** (考试信息量) — the degree to which an exam or question helps discriminate between systems with different latent ability levels on a target task.

## Key Points

- The paper measures informativeness with the IRT item information function `I(theta | g_i, d_i, b_i)`, which is ability-dependent rather than a single global score.
- Questions are most informative around their difficulty level, so a good exam should cover the ability range of the candidate RAG systems.
- The authors aggregate question-level information curves to compare subsets of questions defined by Bloom categories or semantic types.
- Informativeness is also the optimization target in the iterative exam-improvement loop, where low-discrimination items are removed to sharpen evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guinet-2024-automated-2405-13622]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guinet-2024-automated-2405-13622]].
