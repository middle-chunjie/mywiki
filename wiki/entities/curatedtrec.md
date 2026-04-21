---
type: entity
title: CuratedTrec
slug: curatedtrec
date: 2026-04-20
entity_type: dataset
aliases: [Curated TREC]
tags: []
---

## Description

CuratedTrec is an open-domain QA benchmark built from real user queries on search sites and evaluated in [[guu-2020-realm-2002-08909]]. The paper notes that answers are specified with regular expressions to account for multiple correct surface forms.

## Key Contributions

- Provides a `1k/1k` train/test benchmark for retrieval-based QA evaluation in the paper.
- Highlights a setting where generative baselines are less directly applicable because supervision is regex-based.
- REALM reaches `46.8` EM with Wikipedia pre-training, far above ORQA's `30.1`.

## Related Concepts

- [[open-domain-question-answering]]
- [[dense-retrieval]]
- [[retrieval-augmented-language-model]]

## Sources

- [[guu-2020-realm-2002-08909]]
