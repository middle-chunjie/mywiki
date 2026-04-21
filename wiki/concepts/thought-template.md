---
type: concept
title: Thought Template
slug: thought-template
date: 2026-04-20
updated: 2026-04-20
aliases: [thought template, high-level thought template]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Thought Template** — a reusable high-level reasoning pattern that abstracts a class of problem-solving procedures and can be instantiated for a specific task instance.

## Key Points

- In BoT, thought templates are distilled from complete problem-solving processes rather than written manually for each benchmark.
- Each template is paired with a description used for retrieval and a category used to narrow the search space.
- Retrieved templates are instantiated into concrete reasoning traces, procedures, or code depending on the task.
- When no matching template is found, BoT falls back to a small set of coarse-grained default templates.
- The buffer manager distills new templates through core-task summarization, solution-step description, and a general answer template.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-buffer-2406-04271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-buffer-2406-04271]].
