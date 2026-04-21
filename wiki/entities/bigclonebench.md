---
type: entity
title: BigCloneBench
slug: bigclonebench
date: 2026-04-20
entity_type: dataset
aliases: [Big Clone Bench]
tags: []
---

## Description

BigCloneBench is the Java clone-detection benchmark used in [[wang-2022-bridging-2112-02268]]. In this paper, the filtered dataset contains `9,134` Java code fragments and `901,028 / 415,416 / 415,416` train/validation/test pairs.

## Key Contributions

- Provides the main benchmark for evaluating whether the proposed augmentation and curriculum recipe improves code clone detection.
- Supports the result that `CodeBERT(10% data) + DA + CL` reaches `0.972` F1, slightly above the GraphCodeBERT baseline.

## Related Concepts

- [[code-clone-detection]]
- [[test-time-augmentation]]
- [[code-understanding]]

## Sources

- [[wang-2022-bridging-2112-02268]]
