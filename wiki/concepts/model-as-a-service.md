---
type: concept
title: Model-as-a-Service
slug: model-as-a-service
date: 2026-04-20
updated: 2026-04-20
aliases: [MaaS, model as a service, API-only model access, 模型即服务]
tags: [deployment, large-models, api, business-model]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Model-as-a-Service** (模型即服务) — a deployment paradigm in which a large pre-trained model is made available exclusively through an API, preventing end-users from accessing or modifying the model's internal parameters.

## Key Points

- Arises because training large-scale models requires enormous compute and often proprietary data; developers expose only an inference endpoint.
- Representative examples include text-to-image models (DALL-E, Imagen) and language model APIs (GPT-3, ChatGPT).
- Users cannot apply full fine-tuning or parameter-efficient methods (e.g., prompt tuning, adapters) under MaaS constraints.
- In-context learning and prompt retrieval become the primary adaptation mechanisms in this setting.
- MaaS is a key motivation for studying visual in-context learning: practitioners need reliable zero-update adaptation strategies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
