---
type: concept
title: Input-Label Mapping
slug: input-label-mapping
date: 2026-04-20
updated: 2026-04-20
aliases: [input-label correspondence, 输入-标签映射]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Input-Label Mapping** (输入-标签映射) — the correspondence between each input example and its correct task label, typically treated as the core supervised signal in labeled data.

## Key Points

- This paper directly tests whether in-context learning gains depend on preserving the ground-truth mapping inside demonstrations.
- Randomly replacing demonstration labels usually causes only `0-5%` absolute degradation, implying that the mapping contributes far less than commonly assumed.
- Even demonstrations with `0%` correct labels can remain much better than zero-shot prompting, especially for MetaICL and GPT-J multi-choice settings.
- The authors argue that large language models may recover task mappings from pretrained priors while using demonstrations mainly for distributional and formatting cues.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[min-2022-rethinking-2202-12837]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[min-2022-rethinking-2202-12837]].
