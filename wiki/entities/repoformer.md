---
type: entity
title: REPOFORMER
slug: repoformer
date: 2026-04-20
entity_type: tool
aliases: [Repoformer]
tags: []
---

## Description

REPOFORMER is the selectively retrieving code language model introduced in [[wu-2024-repoformer-2403-10059]]. It extends StarCoderBase-style fill-in-the-middle completion with a learned retrieval trigger for repository-level code completion.

## Key Contributions

- Learns to emit `<cc>` after `<eof>` when cross-file retrieval is likely to improve code completion quality.
- Combines self-evaluation loss and generation loss in a single fine-tuning recipe over `1B`, `3B`, `7B`, and `16B` StarCoderBase backbones.
- Improves both completion quality and latency over always-retrieve RAG baselines across RepoEval and CrossCodeLongEval.

## Related Concepts

- [[selective-retrieval]]
- [[fill-in-the-middle]]
- [[repository-level-code-completion]]

## Sources

- [[wu-2024-repoformer-2403-10059]]
