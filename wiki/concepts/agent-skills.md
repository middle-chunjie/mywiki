---
type: concept
title: Agent Skills
slug: agent-skills
date: 2026-04-20
updated: 2026-04-20
aliases: [智能体技能, agent skill]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agent Skills** (智能体技能) — modular packages of procedural guidance and optional resources that augment an agent at inference time without changing model weights.

## Key Points

- SkillsBench defines a skill as a reusable artifact with a required `SKILL.md` file and optional scripts, templates, references, or examples.
- The paper restricts skills to procedural knowledge for a class of tasks, explicitly excluding plain prompts, factual retrieval snippets, and generic tool documentation.
- Skills are injected through filesystem directories that different harnesses can discover, making them portable across Claude Code, Codex CLI, and Gemini CLI.
- Curated skills improve average pass rate by `+16.2` percentage points, while self-generated skills fail to reproduce the same benefit.
- The paper finds that focused skills with `2-3` modules outperform larger skill bundles and that overly comprehensive skills can hurt performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-skillsbench-2602-12670]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-skillsbench-2602-12670]].
