---
type: entity
title: T-Eval
slug: t-eval
date: 2026-04-20
entity_type: tool
aliases: [T Eval]
tags: []
---

## Description

T-Eval is an external tool-planning benchmark used in [[unknown-nd-workflowllm]] to test out-of-distribution transfer. The paper converts WorkflowBench into T-Eval's JSON-style format and retrains WorkflowLlama for this evaluation.

## Key Contributions

- Provides an external OOD benchmark for measuring plan-level API sequencing ability.
- Shows that WorkflowLlama reaches `77.5` F1, outperforming vanilla Llama-3.1-8B and several larger open-source baselines.

## Related Concepts

- [[out-of-distribution-generalization]]
- [[tool-learning]]
- [[workflow-orchestration]]

## Sources

- [[unknown-nd-workflowllm]]
