---
type: entity
title: UltraChat
slug: ultrachat
date: 2026-04-20
entity_type: tool
aliases: [UltraChat dataset]
tags: []
---

## Description

UltraChat is the short-context instruction-tuning dataset selected in [[gao-2024-how-2410-02660]] for the final supervised fine-tuning stage. The paper reports that UltraChat alone outperforms mixtures that include synthetic long instruction data in its setting.

## Key Contributions

- Supplies the final SFT data for ProLong over `1B` tokens.
- Outperforms Tulu-v2 and ShareGPT in the paper's SFT comparison.
- Supports the paper's claim that short instruction data can be sufficient for strong long-context downstream performance.

## Related Concepts

- [[supervised-fine-tuning]]
- [[instruction-tuning]]
- [[long-context-training]]

## Sources

- [[gao-2024-how-2410-02660]]
