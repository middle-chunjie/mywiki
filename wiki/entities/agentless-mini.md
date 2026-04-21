---
type: entity
title: Agentless Mini
slug: agentless-mini
date: 2026-04-20
entity_type: tool
aliases: [Agentless Mini]
tags: []
---

## Description

Agentless Mini is the simplified pipeline scaffold used in [[wei-2025-swerl-2502-18449]] to evaluate and scale repository-level issue resolution. It performs localization, repair, test generation, and reranking around the base model.

## Key Contributions

- Simplifies Agentless by emphasizing file-level localization and full-file repair reasoning.
- Supports scaling over both repair samples and reproduction-test samples for better reranking on SWE-bench Verified.
- Uses a consensus-based reranking objective to select the final patch.

## Related Concepts

- [[file-level-localization]]
- [[reproduction-test-generation]]
- [[software-issue-resolution]]

## Sources

- [[wei-2025-swerl-2502-18449]]
