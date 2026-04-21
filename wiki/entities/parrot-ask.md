---
type: entity
title: Parrot-Ask
slug: parrot-ask
date: 2026-04-20
entity_type: tool
aliases: [PARROT Ask]
tags: []
---

## Description

Parrot-Ask is the question-asking model introduced in [[sun-2024-parrot-2310-07301]]. It is trained to imitate how real users ask follow-up questions in multi-turn conversations.

## Key Contributions

- Optimizes a question-only loss over dialogue sequences to model user-side behavior.
- Extends ShareGPT and UltraChat seeds into longer, more context-dependent sessions.
- Produces the user turns used to build the Parrot-40K dataset.

## Related Concepts

- [[question-generation]]
- [[user-simulation]]
- [[instruction-tuning]]

## Sources

- [[sun-2024-parrot-2310-07301]]
