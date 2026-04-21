---
type: entity
title: AgentFlow
slug: agentflow
date: 2026-04-20
entity_type: tool
aliases: [Agent Flow]
tags: []
---

## Description

AgentFlow is the trainable agentic system proposed in the paper. It coordinates a planner, executor, verifier, and generator through shared evolving memory to solve multi-turn reasoning tasks with tools.

## Key Contributions

- Decomposes tool-using reasoning into explicit planning, execution, verification, and final generation modules.
- Enables on-policy planner optimization inside the live multi-turn interaction loop.
- Reports strong gains over monolithic tool-integrated reasoning systems and larger proprietary models.

## Related Concepts

- [[agentic-system]]
- [[evolving-memory]]
- [[flow-based-group-refined-policy-optimization]]

## Sources

- [[li-2026-intheflow-2510-05592]]
