---
type: entity
title: dr-agent-lib
slug: dr-agent-lib
date: 2026-04-20
entity_type: tool
aliases: [dr-agent-1ib, DR agent library]
tags: []
---

## Description

`dr-agent-lib` is the paper's MCP-based agent infrastructure for training and running deep-research systems with multiple search and browsing tools. It manages concurrency, caching, and prompt-layer composition for tool use.

## Key Contributions

- Provides the unified backend behind `google_search`, `web_browse`, and `paper_search`.
- Enables asynchronous, high-throughput tool calling during RL training.
- Supports flexible tool swapping and prompt iteration for open deep-research development.

## Related Concepts

- [[tool-augmented-language-model]]
- [[asynchronous-tool-calling]]
- [[deep-research]]

## Sources

- [[shao-2025-dr-2511-19399]]
