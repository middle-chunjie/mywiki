---
type: entity
title: FORB
slug: forb
date: 2026-04-20
entity_type: tool
aliases: [Flat Object Retrieval Benchmark]
tags: []
---

## Description

FORB is the flat-object retrieval benchmark introduced by [[wu-2023-forb-2309-16249]]. It contains in-the-wild queries, canonical index images, distractors, and difficulty annotations for evaluating universal image embeddings.

## Key Contributions

- Introduces `8` flat-object domains with `13,901` queries, `4,585` index images, and `49,850` distractors.
- Adds query difficulty labels and the `t-mAP` evaluation protocol to expose margin-sensitive retrieval behavior.
- Serves as an OOD-style testbed for comparing low-, mid-, and high-level [[image-embedding]] strategies.

## Related Concepts

- [[benchmark-dataset]]
- [[image-retrieval]]
- [[thresholded-mean-average-precision]]

## Sources

- [[wu-2023-forb-2309-16249]]
