---
type: concept
title: Code Agent
slug: code-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [coding agent, software engineering agent, 代码智能体]
tags: [agents, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Agent** (代码智能体) — an LLM-based agent that can iteratively inspect, edit, and execute code through tools while maintaining state across multiple interaction steps.

## Key Points

- The paper studies whether code agents designed for code repair can also synthesize issue-reproducing tests.
- SWE-Agent, Aider, and AutoCodeRover are adapted mainly through instruction changes rather than architectural redesign.
- Compared with one-shot prompting baselines, code agents achieve stronger practical success on SWT-Bench.
- Test generation exposes a structured evaluation setting for code agents because both fail-to-pass behavior and patch-specific coverage can be measured.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[m-ndler-2024-code-2406-12952]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[m-ndler-2024-code-2406-12952]].
