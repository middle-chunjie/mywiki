---
type: entity
title: ASG-SI
slug: asg-si
date: 2026-04-20
entity_type: tool
aliases: [Audited Skill-Graph Self-Improvement]
tags: []
---

## Description

ASG-SI is the framework proposed in [[huang-2025-audited-2512-23760]] for self-improving agentic LLMs. It promotes reusable skills into an audited graph only after replay-based verification yields evidence sufficient for promotion and reward reconstruction.

## Key Contributions

- Defines a verifier-gated skill-promotion loop instead of unconstrained parameter-only improvement.
- Couples reward shaping, evidence bundles, experience synthesis, and continual memory control within one architecture.
- Ships with a reference prototype centered on deterministic verification and append-only audit traces.

## Related Concepts

- [[skill-graph]]
- [[verifiable-reward]]
- [[continual-memory]]

## Sources

- [[huang-2025-audited-2512-23760]]
