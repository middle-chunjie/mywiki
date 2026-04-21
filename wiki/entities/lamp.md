---
type: entity
title: LaMP
slug: lamp
date: 2026-04-20
entity_type: tool
aliases: ["Language Model Personalization", "LaMP benchmark"]
tags: []
---

## Description

LaMP is the benchmark used in [[salemi-2024-optimization]] to evaluate retrieval-augmented LLM personalization. It contains seven personalization tasks spanning classification and short-text generation with time-based splits.

## Key Contributions

- Supplies seven datasets: citation identification, movie tagging, product rating, news headline generation, scholarly title generation, email subject generation, and tweet paraphrasing.
- Provides task-specific evaluation metrics such as accuracy, F1, MAE, RMSE, ROUGE-1, and ROUGE-L.
- Enables analysis of retriever optimization and retriever selection under time-aware user profiles.

## Related Concepts

- [[personalization]]
- [[query-performance-prediction]]
- [[retriever-selection]]

## Sources

- [[salemi-2024-optimization]]
