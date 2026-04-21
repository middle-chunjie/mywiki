---
type: concept
title: Identifier Splitting
slug: identifier-splitting
date: 2026-04-20
updated: 2026-04-20
aliases: [subtoken splitting, identifier tokenization, 标识符切分]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Identifier Splitting** (标识符切分) — the preprocessing step that breaks compound identifiers such as camelCase or snake_case names into smaller subtokens.

## Key Points

- In the paper this operation is denoted by `S` and is the only preprocessing choice with statistically significant average gains.
- Average BLEU-DC improves from `11.34` without splitting to `12.15` with splitting across the evaluated models.
- The authors connect the gain to lower OOV rates and better handling of unseen compound words in code.
- Identifier splitting is beneficial for CodeNN, Astattgru, Rencos, and NCS under the reported settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
