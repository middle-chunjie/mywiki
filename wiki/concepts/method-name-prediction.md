---
type: concept
title: Method Name Prediction
slug: method-name-prediction
date: 2026-04-20
aliases: [Function Name Prediction, method-name-generation]
tags: [code-intelligence, code-generation, software-engineering]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Method Name Prediction** (方法名预测) — a code generation task in which a model receives a method body with its name masked and must generate the original method name, typically evaluated at the subtoken level using precision, recall, and F1.

## Key Points

- Evaluation uses subtoken-level F1: method names are split on camelCase/underscore boundaries into subtokens, and precision/recall are computed over subtoken sets.
- Requires both semantic understanding (what the method does) and lexical awareness (naming conventions), making it a good proxy for code comprehension quality.
- HiT addresses this as a generation task using a Transformer decoder with a pointer copy mechanism, enabling out-of-vocabulary tokens from the method body to be copied into the name.
- CodeSearchNet (CSN) benchmarks for Ruby and Python are standard evaluation datasets; CSN-Ruby F1 baseline (vanilla Transformer) is ~21.71 and CSN-Python ~29.96.
- HiT achieves 29.06 F1 on CSN-Ruby and 35.41 on CSN-Python, outperforming both CodeTransformer and GTNM in the same setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
