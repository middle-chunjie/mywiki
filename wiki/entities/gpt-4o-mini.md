---
type: entity
title: GPT-4o-mini
slug: gpt-4o-mini
date: 2026-04-20
entity_type: tool
aliases: [gpt-4o mini]
tags: []
---

## Description

GPT-4o-mini is the LLM used throughout [[chhikara-nd-mem]] for memory extraction, update decisions, and several baseline or judging procedures.

## Key Contributions

- Powers the extraction function that produces candidate memories from dialogue context.
- Handles tool-calling decisions for `ADD`, `UPDATE`, `DELETE`, and `NOOP` operations.
- Supports the graph-memory extraction and update modules in `Mem0^g`.

## Related Concepts

- [[large-language-model]]
- [[persistent-memory]]
- [[graph-memory]]

## Sources

- [[chhikara-nd-mem]]
