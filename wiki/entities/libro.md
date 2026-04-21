---
type: entity
title: LIBRO
slug: libro
date: 2026-04-20
entity_type: tool
aliases: [Libro]
tags: [llm, software-testing]
---

## Description

LIBRO is a prior LLM-based test generation method used in this paper as the strongest specialized baseline. The authors adapt it from its original Java setting to Python and use its error-trace heuristics for top-1 test selection.

## Key Contributions

- Serves as the paper's main non-agent baseline for issue-reproducing test generation.
- Reaches `S = 14.1%` and `Delta C_all = 23.8%` on SWT-Bench-Lite in the reported setup.

## Related Concepts

- [[test-case-generation]]
- [[issue-reproduction]]
- [[large-language-model]]

## Sources

- [[m-ndler-2024-code-2406-12952]]
