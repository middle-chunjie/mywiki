---
type: entity
title: WorkflowLlama
slug: workflowllama
date: 2026-04-20
entity_type: tool
aliases: [Workflow Llama]
tags: []
---

## Description

WorkflowLlama is the Llama-3.1-8B model fine-tuned on WorkflowBench in [[unknown-nd-workflowllm]]. It is designed to generate long, structured API workflows from natural-language requests.

## Key Contributions

- Achieves `39.3` CodeBLEU and `76.9%` Pass Rate on unseen-instruction evaluation.
- Maintains strong unseen-API generalization with `35.1` CodeBLEU and `70.4%` Pass Rate.
- Reaches `77.5` F1 on the T-Eval PLAN benchmark after format adaptation.

## Related Concepts

- [[workflow-orchestration]]
- [[fine-tuning]]
- [[out-of-distribution-generalization]]

## Sources

- [[unknown-nd-workflowllm]]
