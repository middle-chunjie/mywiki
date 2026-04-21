---
type: entity
title: CMMLU
slug: cmmlu
date: 2026-04-20
entity_type: tool
aliases: [C-MMLU, Chinese MMLU, Chinese Massive Multitask Language Understanding]
tags: []
---

## Description

CMMLU is a benchmark for evaluating Chinese-language understanding across 67 subjects spanning STEM, humanities, and social sciences, designed as a Chinese counterpart to the English MMLU benchmark (Li et al., 2023).

## Key Contributions

- Used in KnowPAT to evaluate catastrophic forgetting after domain fine-tuning, testing general Chinese-language capability retention across five commonsense categories (history, clinical, politics, computer science, economics).
- Reveals that KnowPAT causes some degradation in clinical medicine ability but maintains or slightly improves performance in politics, history, and economics.

## Related Concepts

- [[large-language-model]]
- [[supervised-fine-tuning]]
- [[domain-specific-question-answering]]

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]
