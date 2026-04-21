---
type: entity
title: WorkflowBench
slug: workflowbench
date: 2026-04-20
entity_type: tool
aliases: [Workflow Bench]
tags: []
---

## Description

WorkflowBench is the workflow-orchestration dataset introduced in [[unknown-nd-workflowllm]]. It combines collected and synthesized Python-style workflows together with queries, plans, comments, and API documentation.

## Key Contributions

- Provides `106,763` workflow instances spanning `1,503` APIs, `83` applications, and `28` categories.
- Supplies the supervised fine-tuning data used to train WorkflowLlama.
- Transfers beyond its native setting, supporting `77.5` F1 on the external T-Eval PLAN benchmark after adaptation.

## Related Concepts

- [[workflow-orchestration]]
- [[synthetic-data]]
- [[query-expansion]]

## Sources

- [[unknown-nd-workflowllm]]
