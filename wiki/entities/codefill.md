---
type: entity
title: CodeFill
slug: codefill
date: 2026-04-20
entity_type: tool
aliases: [CodeFill]
tags: []
---

## Description

CodeFill is the Python autocompletion model introduced by [[izadi-2022-codefill-2202-06689]]. It combines token-name, token-type, and statement-level prediction with three GPT-2-style Transformers under soft-parameter-sharing multi-task learning.

## Key Contributions

- Unifies structural and naming signals by jointly modeling aligned value and type sequences.
- Extends code completion from next-token ranking to statement-level generation with explicit `EOS` stopping.
- Adds a scope-aware re-ranking layer that improves practical completion ranking inside files, classes, and functions.

## Related Concepts

- [[code-completion]]
- [[multi-task-learning]]
- [[transformer]]
- [[statement-completion]]

## Sources

- [[izadi-2022-codefill-2202-06689]]
