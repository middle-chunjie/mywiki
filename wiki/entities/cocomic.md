---
type: entity
title: CoCoMIC
slug: cocomic
date: 2026-04-20
entity_type: tool
aliases: [CoCoMIC Framework]
tags: [framework, code-model]
---

## Description

CoCoMIC is the repository-aware code completion framework proposed in [[ding-2023-cocomic-2212-10007]], built on top of an autoregressive code LM to jointly model in-file and cross-file context.

## Key Contributions

- Compresses each retrieved cross-file entity into one vector using a `[SUM]` token.
- Fuses in-file and cross-file context through joint attention at every Transformer layer.
- Improves exact-match code completion from `15.97` to `21.39` over the fine-tuned CodeGen baseline.

## Related Concepts

- [[code-completion]]
- [[cross-file-context]]
- [[autoregressive-language-model]]

## Sources

- [[ding-2023-cocomic-2212-10007]]
