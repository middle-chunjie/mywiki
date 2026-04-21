---
type: entity
title: Qwen-Agent
slug: qwen-agent
date: 2026-04-20
entity_type: tool
aliases: [Qwen Agent]
tags: []
---

## Description

Qwen-Agent is the agent framework used to implement the ReAct-style interaction loop in [[li-2025-websailor-2507-02592]].

## Key Contributions

- Provides the execution scaffold for search, visit, and final-answer actions.
- Supports the paper's multi-turn trajectory format with explicit thought, tool call, and tool response segments.
- Enables capped long-horizon trajectories with up to `30` tool calls in the reported setup.

## Related Concepts

- [[react]]
- [[web-agent]]
- [[long-horizon-reasoning]]

## Sources

- [[li-2025-websailor-2507-02592]]
