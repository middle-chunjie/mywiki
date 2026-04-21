---
type: entity
title: CodeActInstruct
slug: codeactinstruct
date: 2026-04-20
entity_type: dataset
aliases: [CodeAct Instruct]
tags: []
---

## Description

CodeActInstruct is the instruction-tuning dataset introduced in [[wang-2024-executable-2402-01030]] for improving open-source LLM agents on executable code actions. It contains multi-turn agent-environment trajectories across several domains.

## Key Contributions

- Collects `7,139` curated trajectories and `10.58M` tokens for CodeAct-style interaction.
- Covers information seeking, software-package use, external memory, and robot planning.
- Filters for trajectories that demonstrate recovery from execution errors and iterative improvement.

## Related Concepts

- [[instruction-tuning]]
- [[agent-environment-interaction]]
- [[self-debugging]]

## Sources

- [[wang-2024-executable-2402-01030]]
