---
type: concept
title: PII Redaction
slug: pii-redaction
date: 2026-04-20
updated: 2026-04-20
aliases: [personal identifiable information redaction, personally identifiable information redaction, PII masking, 个人身份信息脱敏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**PII Redaction** (个人身份信息脱敏) — the detection and masking of personally identifiable information in training data before model release or downstream use.

## Key Points

- The paper builds a dedicated redaction pipeline instead of relying only on regexes, motivated by the privacy risks of large-scale code and issue data collection.
- The authors annotate `12,000` files across `31` programming languages with `1,399` crowd-workers, covering names, emails, IPs, keys, passwords, usernames, and IDs.
- A `≈125M`-parameter StarEncoder is fine-tuned as a `6`-class NER model, then improved with pseudo-labels and post-processing heuristics for keys, names, and IP addresses.
- Pseudo-labeling improves key F1 from `56.66%` to `70.41%` and password F1 from `73.39%` to `81.57%`, which materially affects the final release pipeline.
- The redaction system is applied to programming-language data, GitHub issues, commits, and notebooks, but the paper still treats residual false positives and false negatives as an unresolved limitation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-starcoder-2305-06161]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-starcoder-2305-06161]].
