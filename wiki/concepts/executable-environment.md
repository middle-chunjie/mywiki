---
type: concept
title: Executable Environment
slug: executable-environment
date: 2026-04-20
updated: 2026-04-20
aliases: [runnable environment]
tags: [software-engineering, evaluation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Executable Environment** (可执行环境) — a task environment whose dependencies and runtime are configured well enough to execute code and tests for validation or feedback.

## Key Points

- SWE-Gym keeps only instances whose environments can be configured and validated against human-written tests.
- The paper emphasizes that executable environments are necessary for incremental unit-test feedback during agent interaction.
- Building these environments required manual dependency recovery from repository files, CI scripts, and documentation.
- The released benchmark includes pre-built Docker images to preserve reproducibility at the instance level.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pan-2024-training-2412-21139]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pan-2024-training-2412-21139]].
