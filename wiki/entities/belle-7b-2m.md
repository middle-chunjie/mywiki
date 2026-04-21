---
type: entity
title: BELLE-7B-2M
slug: belle-7b-2m
date: 2026-04-20
entity_type: tool
aliases: [BELLE 7B 2M]
tags: []
---

## Description

BELLE-7B-2M is a Chinese-oriented open chat LLM baseline included in [[chen-2023-benchmarking-2309-01431]] as a lower-performing comparison point in RGB.

## Key Contributions

- Serves as a weak baseline showing that retrieval does not automatically fix model reliability, especially on rejection and multi-document integration.
- Reaches only `5.67%` exact rejection in English and `5.33%` in Chinese, with information-integration accuracy of `40%` / `49%` at noise ratio `0`.

## Related Concepts

- [[large-language-model]]
- [[negative-rejection]]
- [[information-integration]]

## Sources

- [[chen-2023-benchmarking-2309-01431]]
