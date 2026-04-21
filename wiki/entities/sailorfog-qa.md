---
type: entity
title: SailorFog-QA
slug: sailorfog-qa
date: 2026-04-20
entity_type: dataset
aliases: [SailFog-QA]
tags: []
---

## Description

SailorFog-QA is the synthetic QA dataset introduced in [[li-2025-websailor-2507-02592]] to train web agents on high-uncertainty, hard-to-reduce information-seeking tasks.

## Key Contributions

- Builds questions from random-walk web graphs seeded by rare entities from [[wikidata]].
- Uses information obfuscation to increase ambiguity while keeping answers valid.
- Provides harder supervision than earlier open-source web-agent datasets according to the paper's pass@1 analysis.

## Related Concepts

- [[information-seeking]]
- [[information-obfuscation]]
- [[compositional-generalization]]

## Sources

- [[li-2025-websailor-2507-02592]]
