---
type: concept
title: Task Instruction
slug: task-instruction
date: 2026-04-20
updated: 2026-04-20
aliases: [instruction prompt, 任务指令]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task Instruction** (任务指令) — a short natural-language description of the task objective that is prepended to inputs so a model can condition its representation on task identity.

## Key Points

- UDR concatenates each task instruction `I_i` with both the query example and the candidate demonstration before encoding.
- This gives one shared retriever a lightweight way to separate sentiment analysis, summarization, semantic parsing, code summarization, and other tasks.
- The paper uses concrete instructions such as "Summarize the text" for CNN/DailyMail and task-specific templates for scoring and inference.
- Ablation shows removing task prompts slightly degrades average performance, indicating they help recover task-specific features inside the shared retriever.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-unified]].
