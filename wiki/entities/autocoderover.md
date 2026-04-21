---
type: entity
title: AutoCodeRover
slug: autocoderover
date: 2026-04-20
entity_type: tool
aliases: [Auto Code Rover]
tags: [agents, code-repair]
---

## Description

AutoCodeRover is a code agent baseline in this paper that separates context gathering from one-shot patch generation. The authors adapt it from repair prompts to test-generation prompts and evaluate it on SWT-Bench.

## Key Contributions

- Represents a staged code-agent architecture with explicit retrieval before editing.
- Achieves `S = 9.1%` and `Delta C_all = 17.9%` in the paper's benchmark.

## Related Concepts

- [[code-agent]]
- [[code-repair]]
- [[test-case-generation]]

## Sources

- [[m-ndler-2024-code-2406-12952]]
