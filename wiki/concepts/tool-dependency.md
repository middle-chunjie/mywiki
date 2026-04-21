---
type: concept
title: Tool Dependency
slug: tool-dependency
date: 2026-04-20
updated: 2026-04-20
aliases: [工具依赖, tool prerequisite]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Dependency** (工具依赖) — a prerequisite relation in which one tool must be invoked before another because its output, verification result, or authorization is required downstream.

## Key Points

- The paper defines dependency through two cases: required intermediate output and required prior verification or permission.
- Dependency is modeled as a directed three-way classification problem over ordered tool pairs rather than as a symmetric similarity signal.
- Missing prerequisite tools is identified as a core failure mode of semantic-only tool retrieval for LLM agents.
- TGR uses predicted dependencies to propagate information from downstream tools to their prerequisites during retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-tool-2508-05152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-tool-2508-05152]].
