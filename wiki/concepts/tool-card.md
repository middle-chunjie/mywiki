---
type: concept
title: Tool Card
slug: tool-card
date: 2026-04-20
updated: 2026-04-20
aliases: [tool cards, 工具卡片]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Card** (工具卡片) — a standardized wrapper for an external tool that exposes its interface, capabilities, constraints, and execution entry points in a form an agent can reliably inspect and use.

## Key Points

- OctoTools represents each tool with metadata describing input-output formats, command demonstrations, limitations, and best practices.
- Every tool card implements `execute()` for the tool's primary functionality and `get_metadata()` so the planner and executor can inspect capabilities dynamically.
- The paper's main systems claim is that tool cards make heterogeneous tools easy to add, replace, or upgrade without retraining the framework.
- Tool cards also improve reliability by surfacing warnings such as counting failure modes or detection errors that the planner can use for verification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2025-octotools-2502-11271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2025-octotools-2502-11271]].
