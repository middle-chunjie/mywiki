---
type: entity
title: TDI300K
slug: tdi300k
date: 2026-04-20
entity_type: dataset
aliases: [Tool Dependency Identification 300K]
tags: []
---

## Description

TDI300K is the tool dependency identification dataset introduced in [[unknown-nd-tool-2508-05152]] to train a three-class discriminator over ordered tool pairs. It combines synthetic pretraining data and manually labeled finetuning data.

## Key Contributions

- Supplies `92,000` pretraining examples per class for balanced learning.
- Adds an imbalanced finetuning split that better matches real tool collections.
- Enables the dependency discriminator that underpins TGR's graph construction.

## Related Concepts

- [[tool-dependency]]
- [[dependency-graph]]
- [[tool-retrieval]]

## Sources

- [[unknown-nd-tool-2508-05152]]
