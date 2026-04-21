---
type: entity
title: DOTPROMPTS
slug: dotprompts
date: 2026-04-20
entity_type: dataset
aliases: []
tags: []
---

## Description

DOTPROMPTS is the method-level Java completion benchmark derived from PRAGMATICCODE, where prompts end at dereference sites inside real repository code.

## Key Contributions

- Contains `1,420` target methods and `10,538` dereference prompts.
- Evaluates whether models can complete methods consistently with cross-file repository state rather than only local syntax.

## Related Concepts

- [[code-completion]]
- [[repository-level-context]]
- [[type-consistency]]

## Sources

- [[agrawal-nd-monitorguided]]
