---
type: entity
title: LOCOMO
slug: locomo
date: 2026-04-20
entity_type: tool
aliases: [LoCoMo]
tags: []
---

## Description

LOCOMO is a benchmark for very long-term conversational memory, containing 10 multi-session conversations with roughly 600 dialogues and about 26k tokens each on average, plus around 200 questions per conversation.

## Key Contributions

- Provides single-hop, multi-hop, temporal, and open-domain question categories for evaluating conversational memory.
- Serves as the main benchmark used in [[chhikara-nd-mem]] to compare Mem0 against RAG, full-context, and memory-system baselines.
- Makes long-horizon coherence and timestamp-sensitive recall measurable for dialogue agents.

## Related Concepts

- [[long-term-memory]]
- [[temporal-reasoning]]
- [[large-language-model]]

## Sources

- [[chhikara-nd-mem]]
