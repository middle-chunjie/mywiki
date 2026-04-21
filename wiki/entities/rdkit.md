---
type: entity
title: RDKit
slug: rdkit
date: 2026-04-20
entity_type: tool
aliases: [RDKit]
tags: []
---

## Description

RDKit is the cheminformatics toolkit used in [[friedman-2023-vendi-2210-02410]] to implement Morgan fingerprint similarity for molecular diversity evaluation. It operationalizes the paper's user-defined [[similarity-function]] choice in the molecule experiments.

## Key Contributions

- Implements the radius-`2` Morgan fingerprint similarity used for molecule kernels.
- Enables the molecular benchmark comparison between [[vendi-score]] and IntDiv.

## Related Concepts

- [[similarity-function]]
- [[vendi-score]]
- [[diversity-metric]]

## Sources

- [[friedman-2023-vendi-2210-02410]]
