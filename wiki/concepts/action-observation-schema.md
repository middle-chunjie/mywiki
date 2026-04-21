---
type: concept
title: Action-Observation Schema
slug: action-observation-schema
date: 2026-04-20
updated: 2026-04-20
aliases: [typed trajectory schema, 动作-观测模式]
tags: [agents, schema]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Action-Observation Schema** (动作-观测模式) — a typed representation of agent interaction traces that explicitly separates what the agent does from what it perceives from the environment.

## Key Points

- ADP represents trajectories as alternating actions and observations instead of opaque conversation strings.
- API actions store tool names and structured arguments, code actions store executable code plus language, and message actions store natural-language communication.
- Text observations capture user or environment feedback, while web observations capture `html`, `axtree`, `url`, `viewport_size`, and optional screenshots.
- This typed separation makes it easier to validate data quality and translate the same trajectory into different agent harness formats.
- The schema is expressive enough to cover browsing, coding, software engineering, and general tool workflows in one protocol.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[song-2026-agent-2510-24702]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[song-2026-agent-2510-24702]].
