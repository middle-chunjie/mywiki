---
type: entity
title: Prompt Cache
slug: prompt-cache
date: 2026-04-20
entity_type: tool
aliases: [PromptCache, prompt cache]
tags: []
---

## Description

Prompt Cache is the modular attention reuse baseline compared against in [[merth-2024-superposition-2404-06910]] for low-latency retrieval-augmented generation.

## Key Contributions

- Serves as a strong cached-prompt baseline that always attends to all retrieved documents.
- On NaturalQuestions-Open with `mpt-7b-instruct`, it reaches `0.278` accuracy and `91.8x` theoretical speedup, both below superposition prompting.

## Related Concepts

- [[prompt-caching]]
- [[kv-cache]]
- [[retrieval-augmented-generation]]

## Sources

- [[merth-2024-superposition-2404-06910]]
