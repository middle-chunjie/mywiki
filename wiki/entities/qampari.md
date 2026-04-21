---
type: entity
title: QAMParI
slug: qampari
date: 2026-04-20
entity_type: benchmark
aliases: [QAMPARI]
tags: []
---

## Description

QAMParI is an open-domain QA benchmark with many-answer outputs, used in [[he-2025-precise-2506-06589]] to evaluate a self-verification pipeline whose final generation step can be swapped to PIC-LM.

## Key Contributions

- Serves as a downstream factuality benchmark in the chain-of-verification application study.
- The paper reports PIC-LM improving `F_1@5` from `13.5%` to `22.6%`.
- Highlights whether stronger claim-level control transfers to multi-entity answer generation.

## Related Concepts

- [[chain-of-verification]]
- [[long-form-generation]]
- [[precise-information-control]]

## Sources

- [[he-2025-precise-2506-06589]]
