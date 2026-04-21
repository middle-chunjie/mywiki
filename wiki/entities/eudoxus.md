---
type: entity
title: Eudoxus
slug: eudoxus
date: 2026-04-20
entity_type: tool
aliases: [Eudoxus prototype]
tags: [tool, code-generation]
---

## Description

Eudoxus is the prototype implementation of SPEAC introduced in this paper for generating UCLID5 code from natural-language task descriptions. It combines LLM prompting, static repair, and compilation.

## Key Contributions

- Instantiates the SPEAC pipeline for UCLID5 using Python as the parent language.
- Uses tree-sitter parsing, MAX-SMT repair with Z3, and iterative hole filling.
- Achieves much higher parse rates than prompting, self-repair, and fine-tuning baselines on the reported benchmarks.

## Related Concepts

- [[synthetic-programming-elicitation]]
- [[program-repair]]
- [[text-to-code]]

## Sources

- [[mora-2024-synthetic-2406-03636]]
