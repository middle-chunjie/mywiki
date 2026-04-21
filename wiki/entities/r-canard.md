---
type: entity
title: R-CANARD
slug: r-canard
date: 2026-04-20
entity_type: dataset
aliases: [Reverse CANARD]
tags: []
---

## Description

R-CANARD is the reverse rewriting dataset constructed in [[li-2023-sm-2312-16511]] from CANARD to train the conversational question rewriter.

## Key Contributions

- Reverses the original CANARD direction so the model rewrites a self-contained question into a follow-up conversational question.
- Supplies supervision for the paper's `p_q^{CQR}(q_t^c | C, H_{<t}, q_t)` rewriting model.

## Related Concepts

- [[question-rewriting]]
- [[conversational-question-answering]]

## Sources

- [[li-2023-sm-2312-16511]]
