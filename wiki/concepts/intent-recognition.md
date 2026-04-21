---
type: concept
title: Intent Recognition
slug: intent-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [Intent Detection, Intent Classification, 意图识别]
tags: [dialogue, nlp, classification, nlu]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Intent Recognition** (意图识别) — the NLP task of identifying the communicative intent (e.g., "book a flight", "check balance") of a user utterance, typically as a classification problem over a predefined set of intent categories in a task-oriented dialogue system.

## Key Points

- Pre-trained language models greatly improve intent recognition accuracy, but their representations can be misaligned with intent semantics and benefit from task-specific fine-tuning on intent detection datasets.
- Few-shot and zero-shot intent recognition are active research directions given the cost of intent annotation.
- Standard benchmarks include CLINC150 (150 intents across 10 domains), BANKING (77 banking intents), and StackOverflow (20 question-type intents).
- NLI-based formulations (intent as sentence similarity) have been shown to improve representation quality for downstream intent classification.
- Intent recognition is a prerequisite for new intent discovery: the pre-trained representation from a supervised intent recognition model provides a strong initialization for discovering unknown intents.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-new-2205-12914]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-new-2205-12914]].
