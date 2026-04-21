---
type: entity
title: ORQA
slug: orqa
date: 2026-04-20
entity_type: model
aliases: [Open-Retrieval Question Answering]
tags: []
---

## Description

ORQA is the dense-retrieval open-domain QA baseline from Lee et al. (2019) that serves as REALM's closest comparison point in [[guu-2020-realm-2002-08909]]. The paper reuses ORQA's fine-tuning setup and hyperparameters to isolate the effect of better pre-training.

## Key Contributions

- Provides the strongest directly comparable baseline for REALM.
- Uses latent retrieval and ICT initialization, but without REALM's retrieval-augmented language-model pre-training.
- Demonstrates that improved pre-training alone can raise test EM from `33.3 / 36.4 / 30.1` to `39.2 / 40.2 / 46.8`.

## Related Concepts

- [[dense-retrieval]]
- [[open-domain-question-answering]]
- [[inverse-cloze-task]]

## Sources

- [[guu-2020-realm-2002-08909]]
