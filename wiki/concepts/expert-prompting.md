---
type: concept
title: Expert Prompting
slug: expert-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [expert prompting]
tags: [llm, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expert Prompting** — a prompting strategy that asks a language model to answer under an explicitly constructed expert identity tailored to the task or input.

## Key Points

- In this paper, expert prompting is a zero-shot baseline with two variants: a fixed static expert persona and a dynamic expert persona generated per input.
- The approach uses role conditioning to make the answer sound more informed, but it does not introduce the multi-step conductor-and-expert loop used by meta-prompting.
- Dynamic expert prompting reaches a macro average of `54.6`, well below meta-prompting with Python (`72.9`) and also below meta-prompting without Python (`61.4`).
- The comparison isolates a key claim of the paper: changing persona alone is weaker than decomposition, coordination, and verification across multiple subcalls.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[suzgun-2024-metaprompting-2401-12954]].
