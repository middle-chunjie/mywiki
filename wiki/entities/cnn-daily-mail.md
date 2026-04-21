---
type: entity
title: CNN/DailyMail
slug: cnn-daily-mail
date: 2026-04-20
entity_type: tool
aliases: [CNNDM, CNN Daily Mail, CNN-DailyMail]
tags: []
---

## Description

CNN/DailyMail is the summarization benchmark used in the paper's T5-XXL experiments to measure exact speculative-decoding speedups on long-form generation.

## Key Contributions

- Serves as the summarization testbed where T5-small drafts yield `3.1x` speedup at `temp = 0` and `2.3x` at `temp = 1`.
- Demonstrates that the method remains effective outside translation, albeit with slightly lower acceptance rates than EnDe.

## Related Concepts

- [[sequence-to-sequence]]
- [[speculative-decoding]]
- [[large-language-model]]

## Sources

- [[leviathan-2023-fast-2211-17192]]
