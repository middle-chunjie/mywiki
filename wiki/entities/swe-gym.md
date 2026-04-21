---
type: entity
title: SWE-Gym
slug: swe-gym
date: 2026-04-20
entity_type: tool
aliases: [Software Engineering Gym]
tags: [dataset, benchmark, environment]
---

## Description

SWE-Gym is the executable training environment introduced in [[pan-2024-training-2412-21139]]. It packages real GitHub issue tasks with repository snapshots, configured runtimes, and unit-test-based feedback for software engineering agents.

## Key Contributions

- Provides `2,438` validated Python task instances from `11` repositories.
- Supports both policy training from successful trajectories and verifier training from success/failure signals.
- Releases per-instance Docker images to preserve reproducible execution.

## Related Concepts

- [[training-environment]]
- [[executable-environment]]
- [[software-engineering-agent]]

## Sources

- [[pan-2024-training-2412-21139]]
