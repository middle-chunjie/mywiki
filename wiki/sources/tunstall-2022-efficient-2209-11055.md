---
type: source
subtype: paper
title: Efficient Few-Shot Learning Without Prompts
slug: tunstall-2022-efficient-2209-11055
date: 2026-04-20
language: en
tags: [few-shot-learning, text-classification, sentence-transformers, multilingual, distillation]
processed: true
raw_file: raw/papers/tunstall-2022-efficient-2209-11055/paper.pdf
raw_md: raw/papers/tunstall-2022-efficient-2209-11055/paper.md
bibtex_file: raw/papers/tunstall-2022-efficient-2209-11055/paper.bib
possibly_outdated: true
authors:
  - Lewis Tunstall
  - Nils Reimers
  - Unso Eun Seo Jo
  - Luke Bates
  - Daniel Korat
  - Moshe Wasserblat
  - Oren Pereg
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2209.11055
doi:
url: http://arxiv.org/abs/2209.11055
citation_key: tunstall2022efficient
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature. SETFIT proposes a prompt-free few-shot text classification pipeline that replaces prompt engineering and very large encoder-decoder backbones with Sentence Transformer fine-tuning plus a lightweight classifier head. The method first builds positive and negative sentence pairs from a tiny labeled set, fine-tunes a Siamese encoder contrastively, and then trains logistic regression on the resulting embeddings. Across six English classification datasets, SETFIT is competitive with or better than ADAPET, PERFECT, and T-Few 3B while using much smaller models; it also transfers to multilingual Amazon reviews, supports student distillation, and reduces training/inference cost by roughly one to two orders of magnitude relative to prompt-based baselines.

## Problem & Motivation

Few-shot NLP methods in 2022 were increasingly dominated by prompt-based or PEFT-style approaches that depended on handcrafted templates, verbalizers, and billion-parameter models. The paper argues that this creates two practical barriers: high variance from prompt design and poor accessibility because training and deployment need expensive hardware. The authors target few-shot text classification specifically, asking whether prompt-free sentence embedding models can recover competitive accuracy from only a handful of labeled examples while remaining fast enough for practical use. A secondary motivation is robustness across languages and deployment settings where compact models and low latency matter more than absolute leaderboard performance.

## Method

- **Two-stage pipeline**: fine-tune a Sentence Transformer `ST` on generated sentence pairs, then train a classifier head `CH` over embeddings `Emb^{x_i} = ST(x_i)` and predict with `x_i^{pred} = CH(ST(x_i))`.
- **Pair generation**: from labeled data `D = {(x_i, y_i)}`, for each class `c in C` generate positives `T_p^c = {(x_i, x_j, 1): y_i = y_j = c}` and negatives `T_n^c = {(x_i, x_j, 0): y_i = c, y_j != c}`; concatenate them into `T`, where `|T| = 2R|C|` and the default is `R = 20`.
- **Data amplification intuition**: with `K` labeled examples in binary classification, the potential number of unique sentence pairs scales as `K(K - 1) / 2`, so contrastive pairing enlarges supervision beyond the raw labeled count.
- **Optimization details**: Sentence Transformer fine-tuning uses cosine-similarity loss with learning rate `1e-3`, batch size `16`, maximum sequence length `256`, and `1` epoch.
- **Classifier head**: the paper uses logistic regression throughout, trained on `T^{CH} = {(Emb^{x_i}, y_i)}` after encoder adaptation rather than end-to-end prompt tuning.
- **Model variants**: `SETFITROBERTA` uses `all-roberta-large-v1` (`355M` params), `SETFITMPNET` uses `paraphrase-mpnet-base-v2` (`110M`), and `SETFITMINILM` uses `paraphrase-MiniLM-L3-v2` (`15M`); multilingual experiments swap in `paraphrase-multilingual-mpnet-base-v2`.
- **Efficiency model**: computational cost is estimated as `C_inf = 2N * l_seq` and `C_train = 6N * l_seq * n_steps * n_batch` for encoder-only models, with `l_seq = 38`, `n_steps = 1000`, and `n_batch = 8` in the T-Few comparison.
- **Distillation variant**: a `110M` SETFIT teacher supervises a `15M` SETFIT student using teacher-induced sentence-pair similarities plus teacher-head logits, with the same split policy and fine-tuning hyperparameters as the main experiments.

## Key Results

- On six English test datasets with `|N| = 8` labeled examples per class, `SETFITMPNET` reaches `62.3 ± 4.9` average score, versus `58.3 ± 3.6` for ADAPET, `48.7 ± 6.0` for PERFECT, and `43.0 ± 5.2` for standard fine-tuning; it is essentially on par with `T-Few 3B` at `63.4 ± 1.9`.
- With `|N| = 64`, `SETFITMPNET` rises to `75.3 ± 1.3`, beating ADAPET (`73.8 ± 2.2`), PERFECT (`72.7 ± 1.9`), FINETUNE (`69.7 ± 7.8`), and `T-Few 3B` (`70.3 ± 1.5`).
- The paper reports that `SETFITMPNET` beats FINETUNE by `19.3` average points at `|N| = 8`, beats PERFECT by `13.6`, and beats ADAPET by `4.0`; at `|N| = 64` the gains narrow to `5.6`, `2.6`, and `1.5`, respectively.
- On RAFT, `SETFITROBERTA` scores `71.3`, above GPT-3 (`62.7`) and PET (`69.6`), and the paper notes it surpasses the human baseline on `7/11` tasks while remaining more than `30x` smaller than `T-Few 11B`; `SETFITMPNET` scores `66.9`.
- On multilingual MARC with `|N| = 8`, lower is better and SETFIT obtains average `MAE x 100` of `86.6` (`each`), `86.4` (`en`), and `88.3` (`all`), clearly outperforming FINETUNE (`121.9`, `117.7`, `117.2`) and ADAPET (`134.9`, `152.0`, `146.0`).
- In distillation, the SETFIT student beats the same-size baseline student by `24.8` accuracy on AG News, `25.1` on Emotion, and `8.9` on SST-5 when only `N = 8` unlabeled examples are available.
- Efficiency estimates show `SETFITMPNET` at `8.3e9` inference FLOPs and `2.0e14` training FLOPs, about `19x` faster than `T-Few 3B` (`1.6e11`, `3.9e15`); `SETFITMINILM` is `123x` faster and trains in about `30s` / `$0.025` per split versus `700s` / `$0.7` for `T-Few 3B`.

## Limitations

- The method is evaluated almost entirely on classification benchmarks; the paper does not test whether the prompt-free recipe transfers to generation, retrieval, structured prediction, or broader instruction-following tasks.
- Few-shot gains remain well below full-data fine-tuning in absolute terms, e.g. average `62.3` at `|N| = 8` and `75.3` at `|N| = 64` versus `84.8` for full-data fine-tuning.
- Pair construction expands supervision but can still become combinatorially large in principle, since the candidate pair space grows as `K(K - 1) / 2`.
- Comparisons to the strongest prompt-based systems are partly constrained by hardware: the authors could not run `T-Few 11B` in their direct experiments because it requires `80GB` A100 memory.
- The distillation advantage decreases as more unlabeled data becomes available, reaching parity around `N = 1K`, so the student setup is most compelling in the low-data regime.

## Concepts Extracted

- [[few-shot-learning]]
- [[sentence-transformer]]
- [[sentence-embedding]]
- [[contrastive-learning]]
- [[siamese-network]]
- [[parameter-efficient-fine-tuning]]
- [[prompt-engineering]]
- [[knowledge-distillation]]
- [[cloze-question]]

## Entities Extracted

- [[lewis-tunstall]]
- [[nils-reimers]]
- [[unso-eun-seo-jo]]
- [[luke-bates]]
- [[daniel-korat]]
- [[moshe-wasserblat]]
- [[oren-pereg]]
- [[hugging-face]]
- [[cohere]]
- [[technical-university-of-darmstadt]]
- [[intel-labs]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
