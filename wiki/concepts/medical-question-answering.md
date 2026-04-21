---
type: concept
title: Medical Question Answering
slug: medical-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [Medical QA, MQA, biomedical QA, 医疗问答]
tags: [nlp, qa, medical, benchmark]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Medical Question Answering** (医疗问答) — the task of automatically answering clinical or biomedical questions, typically in multiple-choice format, using knowledge from medical corpora, textbooks, or external retrieval systems.

## Key Points

- Representative benchmarks include MMLU-Med (1,089 questions from multitask medical exams), MedQA-US (1,273 USMLE-style questions), MedMCQA (4,183 Indian medical entrance questions), PubMedQA (500 biomedical literature questions), and BioASQ-Y/N (618 yes/no questions from the BioASQ competition).
- Accuracy is typically measured with Exact Match (correct option selected) and is a primary metric for evaluating RAG systems in knowledge-intensive domains.
- The MIRAGE benchmark (Xiong et al., 2024) standardizes evaluation across the five datasets above, pairing them with the MedCorp retrieval corpus.
- Medical QA is sensitive to retrieval noise: correctly retrieved snippets occasionally mislead weaker LLMs when the backbone has limited parametric knowledge to reconcile conflicting information.
- Domain specificity (dense technical terminology, Latin nomenclature) makes naive chunking strategies suboptimal compared to adaptive granularity approaches.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2024-mixofgranularity-2406-00456]].
