---
type: entity
title: Jedi
slug: jedi
date: 2026-04-20
entity_type: tool
aliases: [Python Jedi, jedi]
tags: []
---

## Description

Jedi is the Python static analysis library used in [[unknown-nd-coeditorleveraging-2305-18584]] to recover symbol usages, function signatures, and assignment sites for edit-context construction. It acts as the lightweight analysis backend behind Coeditor's signature retrieval.

## Key Contributions

- Enables retrieval of function signatures for used functions in the target edit region.
- Enables retrieval of first assignment statements for variables and class members referenced by the target code.

## Related Concepts

- [[static-analysis]]
- [[repository-level-context]]
- [[code-completion]]

## Sources

- [[unknown-nd-coeditorleveraging-2305-18584]]
