---
type: entity
title: SWE-Agent
slug: swe-agent
date: 2026-04-20
entity_type: tool
aliases: [SWE Agent]
tags: [agents, code-repair]
---

## Description

SWE-Agent is a code agent originally designed for repository-level software repair and adapted in this paper for test generation. It exposes search, view, and edit operations over a limited shell-like environment.

## Key Contributions

- Achieves `S = 15.9%` on SWT-Bench and `18.5%` as SWE-Agent+ when asked to run generated tests.
- Outperforms or matches specialized non-agent baselines for issue-reproducing test generation.

## Related Concepts

- [[code-agent]]
- [[code-repair]]
- [[test-case-generation]]

## Sources

- [[m-ndler-2024-code-2406-12952]]
