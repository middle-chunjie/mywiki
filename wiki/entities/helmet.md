---
type: entity
title: HELMET
slug: helmet
date: 2026-04-20
entity_type: tool
aliases: [How to Evaluate Long-context Language Models Effectively and Thoroughly]
tags: []
---

## Description

HELMET is the long-context evaluation suite used in [[gao-2024-how-2410-02660]] to guide model development and report final results. In this paper it functions as the primary benchmark for realistic long-context tasks such as recall, RAG, re-ranking, ICL, QA, and summarization.

## Key Contributions

- Provides the downstream evaluation signal the paper prefers over perplexity and needle-in-a-haystack tests.
- Covers diverse long-context abilities rather than a single synthetic retrieval behavior.
- Serves as the main benchmark on which ProLong is compared against Llama-3.1, GPT-4o, Gemini-1.5-Pro, and other baselines.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[in-context-learning]]
- [[long-context-training]]

## Sources

- [[gao-2024-how-2410-02660]]
