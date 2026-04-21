---
type: concept
title: Statement Completion
slug: statement-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-token completion, statement-level prediction, 语句补全]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Statement Completion** (语句补全) — code completion that generates the remaining token sequence of a partial program statement, rather than only the immediate next token.

## Key Points

- CodeFill trains statement completion as a third task alongside token-value and token-type prediction.
- Pre-processing inserts an explicit `EOS` marker so generation can stop at statement boundaries.
- The statement task is designed to teach longer-range dependencies among identifier names and syntactic elements.
- For `n = 4` tokens, the paper reports `70.2` METEOR and `63.8` ROUGE-L, outperforming TravTrans+.
- The ablation study shows that adding the statement task improves any-token performance from `78.9/79.5` to `80.6/81.7` accuracy/MRR under soft parameter sharing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[izadi-2022-codefill-2202-06689]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[izadi-2022-codefill-2202-06689]].
