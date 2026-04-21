---
type: concept
title: Corpus-in-Context Prompting
slug: corpus-in-context-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [CiC prompting, CiC]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Corpus-in-Context Prompting** — a prompting strategy that places an entire task corpus, plus instructions and demonstrations, directly inside a long-context model's prompt so retrieval and reasoning happen in-context.

## Key Points

- The prompt is organized as instructions, corpus, few-shot examples, and then the final query.
- Each candidate item in the corpus receives an explicit identifier so the model can retrieve or reference passages, images, audio, or tables.
- Few-shot demonstrations are grounded in the same corpus rather than separate toy corpora, which helps teach the model how to navigate the real evaluation context.
- The paper shows CiC can rival specialized retrieval and RAG pipelines at `128k` context length without task-specific fine-tuning.
- Ablations show performance depends strongly on design choices such as where the query appears and whether IDs are echoed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-can-2406-13121]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-can-2406-13121]].
