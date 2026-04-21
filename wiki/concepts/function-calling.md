---
type: concept
title: Function Calling
slug: function-calling
date: 2026-04-20
updated: 2026-04-20
aliases: [tool calling, tool use, 函数调用, 工具调用]
tags: [llm, agents, tools]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Function Calling** (函数调用) — the capability of a language model to decide when to invoke an external function or API and to emit executable structured arguments that satisfy the tool schema.

## Key Points

- ToolACE treats function calling as more than single-tool selection: the training data explicitly covers single, parallel, dependent, and non-tool-use cases.
- The paper argues that strong function calling requires both correct tool choice and correct argument formatting, so verification checks function names, required fields, and parameter patterns.
- ToolACE measures sample difficulty by the learner's average token loss and uses that signal to keep function-calling supervision within a productive complexity band.
- Reported BFCL gains show that better synthetic function-calling data improves AST accuracy, executable accuracy, relevance detection, and irrelevance detection together.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-toolace-2409-00920]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-toolace-2409-00920]].
