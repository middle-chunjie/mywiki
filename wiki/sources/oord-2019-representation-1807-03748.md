---
type: source
subtype: paper
title: Representation Learning with Contrastive Predictive Coding
slug: oord-2019-representation-1807-03748
date: 2026-04-20
language: en
tags: [contrastive-learning, self-supervised-learning, representation-learning, multimodal, predictive-coding]
processed: true
raw_file: raw/papers/oord-2019-representation-1807-03748/paper.pdf
raw_md: raw/papers/oord-2019-representation-1807-03748/paper.md
bibtex_file: raw/papers/oord-2019-representation-1807-03748/paper.bib
possibly_outdated: true
authors:
  - Aaron van den Oord
  - Yazhe Li
  - Oriol Vinyals
year: 2019
venue: arXiv
venue_type: preprint
arxiv_id: 1807.03748
doi:
url: http://arxiv.org/abs/1807.03748
citation_key: oord2019representation
paper_type: method
read_status: unread
domain: multimodal
---

## Summary

⚠ Possibly outdated: published 2019; re-verify against recent literature.

The paper introduces Contrastive Predictive Coding (CPC), a self-supervised representation learning framework that encodes observations into latent variables `z_t`, summarizes past latents into a context state `c_t`, and predicts future latent structure with a contrastive objective instead of reconstructing raw inputs. Its core claim is that maximizing a lower bound on mutual information between `c_t` and future samples extracts slow, high-level factors while discarding local noise. The same recipe is instantiated across speech, vision, natural language, and reinforcement learning with domain-specific encoders and autoregressive models. CPC reaches strong transfer or probing results on all four domains, including 64.6 phone classification accuracy on LibriSpeech, 48.7/73.6 top-1/top-5 on ImageNet, competitive sentence transfer scores, and reinforcement-learning gains on 4 of 5 DeepMind Lab tasks.

## Problem & Motivation

The paper targets generic unsupervised representation learning for high-dimensional data where direct reconstruction or full generative modeling is expensive and often wastes capacity on modality-specific detail. The authors argue that useful representations should preserve shared global structure across time or context, not every local pixel or waveform fluctuation. They therefore frame learning as future prediction in latent space: if a model can predict distant future content from current context, it must encode higher-level variables such as phonetic content, objects, sentence semantics, or environment state. This is positioned as a modality-agnostic route to transferable features.

## Method

- **Latent encoder + context model**: CPC maps each observation to a latent vector with `z_t = g_enc(x_t)` and summarizes history with an autoregressive model `c_t = g_ar(z_{\le t})`.
- **Density-ratio objective**: instead of modeling `p(x_{t+k} | c_t)` directly, CPC learns `f_k(x_{t+k}, c_t) \propto p(x_{t+k} | c_t) / p(x_{t+k})`, instantiated as the log-bilinear score `` `f_k(x_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)` `` with one predictor matrix `W_k` per horizon `k`.
- **InfoNCE loss**: for one positive and `N-1` negatives, CPC minimizes `` `L_N = -E_X log( f_k(x_{t+k}, c_t) / sum_{x_j in X} f_k(x_j, c_t) )` `` and uses the bound `` `I(x_{t+k}; c_t) >= log(N) - L_N` `` to connect the objective to mutual-information maximization.
- **Audio setup**: `g_enc` is a 5-layer strided CNN over 16 kHz waveform with strides `[5, 4, 2, 2, 2]`, filters `[10, 8, 4, 4, 4]`, `512` hidden units, total downsampling factor `160`, and a GRU context model with hidden size `256`; it predicts `12` future latent steps from windows of length `20480`, trains with Adam at `` `2e-4` ``, and uses `8` GPUs with minibatch `8` per GPU.
- **Vision setup**: CPC extracts a `7 x 7` grid of overlapping `64 x 64` crops from each `256 x 256` image, encodes each crop with an unpretrained ResNet-v2-101, mean-pools the third residual block to `1024` dimensions, then uses a PixelCNN-style autoregressive model to predict up to `5` future rows; training uses Adam at `` `2e-4` `` on `32` GPUs with batch size `16`.
- **Natural-language setup**: sentence-level CPC uses a `1D` convolution + ReLU + mean pooling encoder to produce a `2400`-dimensional sentence vector `z`, then a GRU with `2400` hidden units predicts up to `3` future sentence embeddings; training uses Adam at `` `2e-4` `` on `8` GPUs with batch size `64`.
- **Reinforcement-learning setup**: CPC is added as an auxiliary loss on top of a batched A2C agent with an existing convolutional encoder and temporal LSTM; the agent unrolls for `100` steps and predicts up to `30` future steps, with RMSProp hyperparameters chosen by random search.
- **Representation choice**: the paper explicitly allows downstream tasks to use either local latents `z_t` or contextualized states `c_t`, depending on whether the task benefits from past context.

## Key Results

- **Speech probing**: on LibriSpeech, CPC reaches `64.6` phone classification accuracy versus `39.7` with MFCCs and `27.6` from random initialization; speaker classification reaches `97.4` versus `17.6` with MFCCs and `98.5` for the supervised oracle.
- **Speech ablations**: predicting multiple future steps matters; phone accuracy rises from `28.5` at `2` steps to `64.6` at `12` steps, then slightly drops to `63.8` at `16` steps.
- **Vision transfer**: on ImageNet linear evaluation, CPC reaches `48.7` top-1 and `73.6` top-5 accuracy, improving over the previously reported best unsupervised results by about `9` and `4` absolute points respectively.
- **NLP transfer**: sentence CPC gets `76.9` on MR, `80.1` on CR, `91.2` on Subj, `87.7` on MPQA, and `96.8` on TREC, roughly matching or beating skip-thought on several tasks while using a simpler predictive objective.
- **RL auxiliary learning**: on `5` DeepMind Lab tasks, adding CPC improves performance on `4/5` tasks after training on `1` billion frames and is neutral on `lasertag_three_opponents_small`.

## Limitations

- The paper is pre-SimCLR / BYOL / masked-modeling era, so its empirical picture of self-supervised learning is historically important but no longer sufficient for current SOTA comparisons.
- The method is not truly architecture-invariant in practice: each modality uses a different encoder and often a different autoregressive backbone, so some gains may come from domain-specific design choices rather than CPC alone.
- NLP gains are only modestly above older baselines and mostly comparable to skip-thought-style transfer, so the textual evidence for broad superiority is weaker than the audio or vision evidence.
- Several reported improvements, especially in reinforcement learning, are summarized qualitatively or by figures rather than with a full numeric table in the paper.
- Contrastive prediction quality depends on negative sampling and horizon design; the paper does not provide a unified theory for how to tune these choices across domains.

## Concepts Extracted

- [[contrastive-predictive-coding]]
- [[contrastive-learning]]
- [[self-supervised-learning]]
- [[representation-learning]]
- [[mutual-information]]
- [[autoregressive-model]]
- [[predictive-coding]]
- [[infonce]]
- [[noise-contrastive-estimation]]
- [[negative-sampling]]
- [[future-prediction]]

## Entities Extracted

- [[aaron-van-den-oord]]
- [[yazhe-li]]
- [[oriol-vinyals]]
- [[deepmind]]
- [[librispeech]]
- [[imagenet]]
- [[bookcorpus]]
- [[deepmind-lab]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
