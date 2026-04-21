---
type: entity
title: CodeAttack
slug: codeattack
date: 2026-04-20
entity_type: tool
aliases: [CodeAttack framework]
tags: []
---

## Description

CodeAttack is the black-box adversarial attack framework introduced in [[jha-2023-codeattack-2206-00052]] for pre-trained programming-language models. It uses vulnerable-token ranking, masked-code substitute generation, and code-specific constraints to craft adversarial code examples.

## Key Contributions

- Adapts adversarial attacks to code generation tasks in the natural channel of code.
- Combines masked [[codebert]] proposals with operator and token-class constraints.
- Demonstrates strong transferability across translation, repair, and summarization tasks.

## Related Concepts

- [[adversarial-attack]]
- [[black-box-attack]]
- [[greedy-search]]
- [[code-consistency]]

## Sources

- [[jha-2023-codeattack-2206-00052]]
