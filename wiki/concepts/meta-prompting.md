---
type: concept
title: Meta-Prompting
slug: meta-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [meta prompting]
tags: [llm, prompting, scaffolding]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta-Prompting** — a prompting framework in which a language model orchestrates a multi-step solution process by acting as a conductor that can spawn and coordinate expert-style subcalls, tools, and verification steps.

## Key Points

- The paper implements both the conductor and the experts with the same underlying LM, changing only the prompt role and local context.
- The Meta Model maintains a global interaction history and decides whether to delegate, integrate an expert reply, or terminate with a final answer.
- Expert calls are isolated: an expert only sees the instructions and context chosen by the Meta Model rather than the full conversation history.
- The framework is task-agnostic and zero-shot, using one reusable meta instruction set across arithmetic, reasoning, coding, multilingual math, and constrained writing tasks.
- With GPT-4 plus a Python interpreter, meta-prompting reaches a macro average of `72.9`, substantially above the paper's standard and expert-prompting baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[suzgun-2024-metaprompting-2401-12954]].
