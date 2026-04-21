---
type: entity
title: RealiT
slug: realit
date: 2026-04-20
entity_type: tool
aliases: [RealiT repair model]
tags: []
---

## Description

RealiT is the external code-repair model used in [[dinh-2023-large-2306-03438]] for the completion-then-rewriting baseline. It is chosen because it targets realistic bug-fixing patterns and aligns with several bug types studied in the paper.

## Key Contributions

- Serves as the repair backend for the completion-then-rewriting mitigation pipeline.
- Is especially suitable for wrong-variable, wrong-literal, and wrong-operator style errors.

## Related Concepts

- [[program-repair]]
- [[buggy-code-completion]]
- [[semantic-bug]]

## Sources

- [[dinh-2023-large-2306-03438]]
