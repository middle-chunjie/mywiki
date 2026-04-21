---
type: source
subtype: paper
title: Deep code search
slug: gu-2018-deep
date: 2026-04-20
language: en
tags: [code-search, meta-learning, transfer-learning, codebert, software-engineering]
processed: true
raw_file: raw/papers/gu-2018-deep/paper.pdf
raw_md: raw/papers/gu-2018-deep/paper.md
bibtex_file: raw/papers/gu-2018-deep/paper.bib
possibly_outdated: true
authors:
  - Yitian Chai
  - Hongyu Zhang
  - Beijun Shen
  - Xiaodong Gu
year: 2022
venue: ICSE 2022
venue_type: conference
arxiv_id:
doi: 10.1145/3510003.3510125
url: https://doi.org/10.1145/3510003.3510125
citation_key: gu2018deep
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

The folder metadata is inconsistent: `paper.bib` names *Deep code search* (2018), while `paper.md` contains the ICSE 2022 paper *Cross-Domain Deep Code Search with Few-Shot Meta Learning*; the technical summary below follows the markdown content. The paper proposes CDCS, a cross-domain code search pipeline for low-resource domain-specific languages such as Solidity and SQL. It starts from a CodeBERT-style encoder, pre-trains on Python and Java, adds a MAML-based meta-learning phase to learn reusable initialization across source-language tasks, and then fine-tunes on target-language code search. Across both SQL and Solidity benchmarks, CDCS improves over direct cross-language CodeBERT fine-tuning, with the largest gains appearing when target-domain supervision is scarce.

## Problem & Motivation

Deep code search works well in common languages with abundant paired code-text data, but domain-specific languages often lack enough supervision to fine-tune robust semantic matching models. Direct transfer from multilingual pre-trained code models can still suffer from representation conflicts across source languages, especially when the target language is far from the pre-training mixture and the fine-tuning set is small. The paper therefore targets cross-domain code search under scarce data, asking whether pre-training can be combined with few-shot meta learning so that parameter initialization adapts faster and more reliably to a new domain.

## Method

- **Backbone**: CDCS uses a RoBERTa / CodeBERT-style bidirectional Transformer encoder with RoBERTa-base settings `H = 768`, `A = 12`, `L = 12`.
- **Pre-training input**: each example is serialized as `` `[CLS], w_1, ..., w_n, [SEP], c_1, ..., c_m, [EOS]` ``, where NL tokens precede code tokens in a joint sequence.
- **Pre-training objective**: masked language modeling only, with a `15%` mask ratio; the paper explicitly omits RTD because its gain was reported as marginal for this setting.
- **Meta-learning phase**: the training distribution is segmented into `k` tasks, each with its own `D_i^train` and `D_i^valid`, to simulate low-resource task adaptation across source languages.
- **MAML update**: local task parameters are updated with `` `theta_i = theta - alpha * grad_theta L_Ti(f_theta)` ``, and the global model is updated from validation gradients every `M = 100` steps using `` `theta <- theta - beta * grad_theta L_Ti(f_theta_i)` ``.
- **Optimization**: the implementation sets `` `alpha = 1e-5` ``, `` `beta = 1e-4` ``, batch size `` `64` ``, maximum sequence length `` `256` ``, and uses Adam with pre-training learning rate `` `5e-5` ``, `1000` warmup steps, and linear decay.
- **Fine-tuning task**: code search is cast as binary classification over positive and randomly corrupted negative `<NL, PL>` pairs, using the `[CLS]` hidden state and a fully connected classifier with binary cross-entropy.
- **Evaluation protocol**: source languages are Python and Java; target languages are Solidity and SQL; search quality is measured by `MRR`, `Acc@1`, `Acc@5`, and `Acc@10` over `1000` test queries.

## Key Results

- On SQL, CDCS reaches `Acc@1 = 0.746`, `Acc@5 = 0.952`, `Acc@10 = 0.972`, and `MRR = 0.8366`, outperforming cross-language CodeBERT at `0.675 / 0.920 / 0.960 / 0.7818`.
- On Solidity, CDCS reaches `0.658 / 0.829 / 0.879 / 0.7336`, compared with cross-language CodeBERT at `0.532 / 0.779 / 0.848 / 0.6436`.
- The gains are strongest in low-data regimes: the paper reports visibly larger improvements when the target-language training set drops below roughly `500` examples.
- The GPT-2 variant also benefits from meta learning: `CDCS_GPT-2` scores `MRR = 0.6464` on SQL and `0.6607` on Solidity, beating GPT-2 baselines in both domains.
- Pre-training corpora are substantial: Python contributes `412,178` positive pairs for pre-training and `824,342` pairs for meta learning; Java contributes `454,451` and `908,886`, respectively.

## Limitations

- The repository artifacts are inconsistent: `paper.bib` points to a different 2018 paper than the one stored in `paper.md`, so metadata should be manually reconciled before promotion into the real wiki.
- The paper itself notes that meta learning is adapted from classification to ranking-like code search, so the task fit is imperfect and may not address the root cause of cross-domain mismatch.
- Task construction for MAML is random batch splitting rather than linguistically or semantically grounded task design, leaving the transfer mechanism only partially explained.
- Meta learning adds notable overhead: the authors report around `50%` extra training time relative to baseline fine-tuning approaches.
- Empirical coverage is limited to two source languages and two target languages, so broader generalization remains uncertain.

## Concepts Extracted

- [[code-search]]
- [[few-shot-learning]]
- [[transfer-learning]]
- [[domain-adaptation]]
- [[model-agnostic-meta-learning]]
- [[masked-language-modeling]]
- [[domain-specific-language]]
- [[semantic-code-search]]
- [[triplet-ranking-loss]]

## Entities Extracted

- [[yitian-chai]]
- [[hongyu-zhang]]
- [[beijun-shen]]
- [[xiaodong-gu]]
- [[shanghai-jiao-tong-university]]
- [[university-of-newcastle]]
- [[codebert]]
- [[roberta]]
- [[spider]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
