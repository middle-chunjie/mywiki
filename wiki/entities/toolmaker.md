---
type: entity
title: TOOLMAKER
slug: toolmaker
date: 2026-04-20
entity_type: tool
aliases: [ToolMaker]
tags: []
---

## Description

TOOLMAKER is the agentic framework introduced in this paper for converting scientific papers and associated repositories into executable tools for LLM agents. It combines repository setup, code exploration, planning, implementation, and iterative self-debugging inside a reproducible execution environment.

## Key Contributions

- Achieves `12/15` correct tools on TM-BENCH, substantially exceeding [[openhands]] on the same benchmark.
- Couples environment snapshotting with a closed-loop debugging cycle rather than relying on one-shot code generation.
- Demonstrates that repository-backed tool creation can solve multi-step scientific workflows requiring installation and orchestration.

## Related Concepts

- [[tool-creation]]
- [[closed-loop-self-correction]]
- [[execution-environment]]

## Sources

- [[w-lflein-2025-llm-2502-11705]]
