---
type: entity
title: RTE
slug: rte
date: 2026-04-20
entity_type: dataset
aliases: [Recognizing Textual Entailment]
tags: []
---

## Description

RTE is the textual entailment benchmark used in [[lin-2023-batchprompt-2309-00384]] to measure how BatchPrompt trades off accuracy against batching efficiency on sentence-pair classification.

## Key Contributions

- Serves as the entailment task in the paper's three-dataset evaluation.
- Illustrates both the upside and downside of the method: GPT-4 BPE peaks at `92.9%`, while SEAS reduces some high-vote settings to about `91.7%` in the authors' discussion.

## Related Concepts

- [[batch-prompting]]
- [[few-shot-prompting]]
- [[early-stopping]]

## Sources

- [[lin-2023-batchprompt-2309-00384]]
