---
type: concept
title: File-Level Localization
slug: file-level-localization
date: 2026-04-20
updated: 2026-04-20
aliases: [file localization, 文件级定位]
tags: [software-engineering, localization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**File-Level Localization** (文件级定位) — the task of identifying which repository files are most likely relevant to fixing a reported software issue.

## Key Points

- Agentless Mini simplifies localization to predicting relevant file paths from the issue description and repository structure.
- During RL data construction, the training context includes both changed files and predicted relevant but unchanged files.
- The authors intentionally avoid fine-grained localization stages during training so the model spends more capacity on reasoning about repair edits over full files.
- At inference time, multiple localization samples are generated and combined before repair generation.
- The paper later identifies this simplified localization design as a limitation because it may omit richer repository context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2025-swerl-2502-18449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2025-swerl-2502-18449]].
