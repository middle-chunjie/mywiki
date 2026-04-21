---
type: entity
title: PbNN
slug: pbnn
date: 2026-04-20
entity_type: tool
aliases: [Path-Based Neural Network]
tags: []
---

## Description

PbNN is one of the two baseline authorship attribution methods evaluated in the paper. It builds on code2vec-style path-context representations from program ASTs and predicts authorship with a fully connected classifier.

## Key Contributions

- Provides a path-based attribution baseline complementary to the token-based DL-CAIS model.
- Shows different robustness tradeoffs under the same RoPGen training procedure.

## Related Concepts

- [[source-code-authorship-attribution]]
- [[abstract-syntax-tree]]
- [[adversarial-training]]

## Sources

- [[li-2022-ropgen-2202-06043]]
