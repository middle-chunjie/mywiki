---
type: entity
title: NovelEval
slug: novel-eval
date: 2026-04-20
entity_type: benchmark
aliases: [NovelEval-2306]
tags: []
---

## Description

NovelEval is the benchmark introduced in [[sun-2023-chatgpt-2304-09542]] to test passage re-ranking on recently emerging knowledge that frontier LLMs are unlikely to have memorized. Its first version contains `21` questions and `420` passages from `4` domains.

## Key Contributions

- Provides a contamination-aware evaluation set for re-ranking unknown knowledge rather than recycled benchmark questions.
- Supplies graded relevance labels `0/1/2` for `20` candidate passages per question.
- Shows GPT-4 can still rank unfamiliar information strongly, reaching `90.45` nDCG@10 in the reported setting.

## Related Concepts

- [[data-contamination]]
- [[information-retrieval]]
- [[zero-shot-generalization]]

## Sources

- [[sun-2023-chatgpt-2304-09542]]
