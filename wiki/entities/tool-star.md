---
type: entity
title: Tool-Star
slug: tool-star
date: 2026-04-20
entity_type: tool
aliases: [ToolStar, toolstar]
tags: []
---

## Description

Tool-Star is a prior multi-tool reasoning framework used as both a baseline and an initialization reference in this paper. Tool-Light inherits Tool-Star's SFT setup and then adds entropy-guided sampling plus self-evolved DPO.

## Key Contributions

- Provides the SFT recipe that Tool-Light adopts as its first training stage.
- Serves as the main multi-tool baseline, scoring `56.6` average versus Tool-Light's `58.0`.

## Related Concepts

- [[multi-tool-reasoning]]
- [[supervised-fine-tuning]]
- [[self-evolved-preference-learning]]

## Sources

- [[chen-2026-effective-2509-23285]]
