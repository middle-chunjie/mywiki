---
type: entity
title: Tool-Light
slug: tool-light
date: 2026-04-20
entity_type: tool
aliases: [Tool Light]
tags: []
---

## Description

Tool-Light is the framework proposed in this paper for improving multi-tool tool-integrated reasoning. It combines entropy-guided data construction with supervised fine-tuning and self-evolved DPO alignment.

## Key Contributions

- Introduces entropy-guided branching to construct higher-quality preference data for tool use.
- Uses a two-stage training recipe of SFT plus iterative DPO rather than bespoke RL rewards.
- Improves average performance, tool-call efficiency, and necessity on 10 reasoning benchmarks.

## Related Concepts

- [[tool-integrated-reasoning]]
- [[entropy-guided-sampling]]
- [[direct-preference-optimization]]
- [[self-evolved-preference-learning]]

## Sources

- [[chen-2026-effective-2509-23285]]
