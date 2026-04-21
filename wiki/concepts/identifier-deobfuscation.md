---
type: concept
title: Identifier Deobfuscation
slug: identifier-deobfuscation
date: 2026-04-20
updated: 2026-04-20
aliases: [DOBF, identifier deobfuscation, 标识符去混淆]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Identifier Deobfuscation** (标识符去混淆) — a pretraining objective that masks or replaces program identifiers and trains a model to recover the original names from code structure and surrounding context.

## Key Points

- The paper applies identifier deobfuscation to encoder-only models rather than sequence-to-sequence decoders.
- CodeSage obfuscates class names, function names, arguments, and variables into special tokens such as `` `c_i` ``, `` `f_i` ``, and `` `v_i` ``.
- Recovering identifier names pushes the encoder to model semantics, data dependencies, and syntactic structure rather than only surface token co-occurrence.
- Natural-language comments and docstrings are left unobfuscated, so the model can align identifier prediction with textual explanations.
- In the ablation study, DOBF is stronger than random masking on classification and NL2Code retrieval, but weaker on Code2Code search unless paired with MLM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2402-01935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2402-01935]].
