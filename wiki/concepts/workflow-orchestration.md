---
type: concept
title: Workflow Orchestration
slug: workflow-orchestration
date: 2026-04-20
updated: 2026-04-20
aliases: [工作流编排]
tags: [agents, automation, workflows]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Workflow Orchestration** (工作流编排) — the task of composing ordered actions, control flow, and parameters into an executable workflow that satisfies a user goal.

## Key Points

- WorkflowLLM treats workflow orchestration as a harder setting than simple tool calling because real workflows include long action sequences, branches, loops, and nested control structures.
- The paper measures workflow-orchestration quality with both structural code similarity metrics and a model-based Pass Rate metric.
- WorkflowBench is designed so that workflow orchestration is learned from Python-style code, API documentation, task plans, and action-level comments together.
- Fine-tuning on WorkflowBench substantially improves orchestration performance on both unseen instructions and unseen APIs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-workflowllm]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-workflowllm]].
