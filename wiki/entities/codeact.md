---
type: entity
title: CodeAct
slug: codeact
date: 2026-04-20
entity_type: framework
aliases: [Code Act]
tags: []
---

## Description

CodeAct is the executable-code action framework introduced in [[wang-2024-executable-2402-01030]]. It lets an LLM agent act by emitting Python code, executing it, and observing the resulting outputs or errors.

## Key Contributions

- Unifies tool use into a code-based action interface instead of text or JSON schemas.
- Supports control flow, data flow, library reuse, and execution-feedback-driven revision.
- Improves success rate and reduces interaction turns on complex multi-tool tasks.

## Related Concepts

- [[code-as-actions]]
- [[action-space]]
- [[multi-turn-interaction]]

## Sources

- [[wang-2024-executable-2402-01030]]
