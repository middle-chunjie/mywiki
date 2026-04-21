---
type: entity
title: Pathways
slug: pathways
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

Pathways is Google's distributed ML system for executing large JAX/XLA programs across multiple TPU pods. In [[chowdhery-2022-palm-2204-02311]], it is the systems layer that enables efficient 540B-parameter PaLM training on 6144 TPU v4 chips.

## Key Contributions

- Enables pod-level scaling without pipeline parallelism for PaLM 540B training.
- Coordinates cross-pod execution and gradient transfer for high-throughput LLM training.

## Related Concepts

- [[model-scaling]]
- [[large-language-model]]

## Sources

- [[chowdhery-2022-palm-2204-02311]]
