---
type: source
subtype: paper
title: "Adversarial Attacks on Deep Learning Models in Natural Language Processing: A Survey"
slug: zhang-2019-adversarial-1901-06796
date: 2026-04-20
language: en
tags: [adversarial-examples, nlp, deep-learning, robustness, survey]
processed: true
raw_file: raw/papers/zhang-2019-adversarial-1901-06796/paper.pdf
raw_md: raw/papers/zhang-2019-adversarial-1901-06796/paper.md
bibtex_file: raw/papers/zhang-2019-adversarial-1901-06796/paper.bib
possibly_outdated: true
authors:
  - Wei Emma Zhang
  - Quan Z. Sheng
  - Ahoud Alhazmi
  - Chenliang Li
year: 2019
venue: ACM Transactions on Intelligent Systems and Technology
venue_type: journal
arxiv_id: 1901.06796
doi:
url: http://arxiv.org/abs/1901.06796
citation_key: zhang2019adversarial
paper_type: survey
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2019; re-verify against recent literature.

The first comprehensive survey of adversarial attacks on deep learning models for NLP. The paper taxonomizes attack methods by model access (white-box vs. black-box), attack granularity (character, word, sentence), and target (untargeted vs. targeted). It covers seven white-box strategies (FGSM-based, JSMA-based, C&W-based, direction-based, attention-based, reprogramming, hybrid) and five black-box strategies (concatenation, edit, paraphrase-based, GAN-based, substitution), plus cross-modal attacks on OCR, captioning, VQA, and speech recognition. Defense methods—adversarial training (data augmentation, model regularization, robust optimization) and knowledge distillation—are reviewed. Open challenges around perceivability, transferability, automation, and new architectures are identified. Covers ~40 papers selected from top NLP and AI venues.

## Problem & Motivation

DNNs achieve strong NLP performance but are vulnerable to adversarial examples—inputs with imperceptible perturbations that cause wrong predictions. Unlike image adversarial attacks, textual perturbations are inherently discrete, perceptible, and must preserve grammar and semantics. There was no prior comprehensive survey of textual adversarial attacks; this work fills that gap to help researchers understand methods, challenges, and defenses in a unified framework.

## Method

This is a survey paper; the "method" is the taxonomy and analysis framework:

- **Threat model axes**: model access (white-box / black-box), attack target (untargeted / targeted), perturbation granularity (character / word / sentence), attacker motivation (attack / defense).
- **White-box attacks** (full model access):
  - FGSM-based: uses gradient sign or magnitude; `η = ε · sign(∇_x J(x, y))`
  - JSMA-based: saliency map via Jacobian `Jacb_F[i,j] = ∂F_i/∂x_j`; targets most sensitive input components
  - C&W-based: minimizes `||η||_p + λ·J(x+η, y')` with l_p-norm constraints
  - Direction-based (HotFlip): directional derivative of loss w.r.t. atomic character flip vectors
  - Attention-based: perturbs words with highest attention weights
  - Reprogramming: trains `g_θ` to re-purpose victim DNN via Gumbel-Softmax
  - Hybrid: applies image-domain methods (FGSM, DeepFool) on word embeddings, rounds to valid words using Word Mover's Distance
- **Black-box attacks** (input/output access only):
  - Concatenation adversaries: appends distracting but semantically neutral sentences
  - Edit adversaries: character/word-level swap, delete, insert, replace (e.g., DeepWordBug)
  - Paraphrase-based: syntactically controlled paraphrase networks (SCPNs), semantically equivalent adversaries (SEA)
  - GAN-based: searches adversarial latent space `z*` near `z'` using WGAN + inverter
  - Substitution: trains substitute RNN; white-box attacks applied on substitute
- **Perturbation constraints for text**: norm-based (L_p on embeddings), grammar/syntax checkers, perplexity, paraphrase validity, edit distance, Jaccard similarity, Word Mover's Distance, cosine similarity.
- **Defense**:
  - Adversarial training: data augmentation, model regularization `min(J(f(x),y) + λ·J(f(x'),y))`, robust optimization (min-max saddle point from Madry et al.)
  - Knowledge distillation: temperature-scaled softmax `q_i = exp(z_i/T)/Σ_k exp(z_k/T)` from original DNN to retrain student model

## Key Results

- Survey coverage: ~40 papers from top NLP/AI venues (ACL, NAACL, EMNLP, COLING, ICLR, AAAI, IJCAI) plus arXiv, 2017–2019.
- Jia & Liang [55] (concatenation adversary on SQuAD) drops F1 dramatically with simple distracting sentence appending; adversarial retraining only partially recovers robustness against unseen adversaries.
- HotFlip [31] achieves >99% attack success rate on character-level CNN/LSTM with minimal flips using directional derivatives.
- DeepWordBug [35] reduces accuracy by >50% on multiple text classification tasks with black-box character edits.
- SCPN [54] paraphrase attacks cause 60–75% accuracy drops on sentiment analysis and textual entailment models.
- Adversarial training (Miyato et al. [94]) consistently improves model robustness but does not fully mitigate transferable adversaries; distillation is generally less effective than adversarial training.
- Transferability: untargeted adversarial examples transfer better across models/datasets than targeted ones; cross-architecture transfer remains poor.

## Limitations

- Coverage limited to pre-2019 work; does not address large pretrained models (BERT, GPT) as attack targets or defenders.
- Most surveyed attacks cannot meet all three constraints simultaneously: unperceivability + grammar correctness + semantic preservation.
- Evaluation metrics are highly task-specific; no unified benchmark exists for comparing attack effectiveness across NLP tasks.
- The survey acknowledges that most black-box attacks are manual or semi-automatic; fully automated, practical black-box textual attacks remain an open problem.
- White-box attacks often generate embedding-space adversaries that do not map back to valid vocabulary tokens.

## Concepts Extracted

- [[adversarial-attack]]
- [[adversarial-training]]
- [[white-box-attack]]
- [[black-box-attack]]
- [[generative-adversarial-network]]
- [[word-mover-distance]]
- [[recurrent-neural-network]]
- [[convolutional-neural-network]]
- [[sequence-to-sequence]]
- [[attention-mechanism]]
- [[knowledge-distillation]]
- [[natural-language-inference]]
- [[machine-reading-comprehension]]
- [[text-classification]]
- [[neural-machine-translation]]

## Entities Extracted

- [[wei-emma-zhang]]
- [[quan-z-sheng]]
- [[ahoud-alhazmi]]
- [[chenliang-li]]
- [[macquarie-university]]
- [[wuhan-university]]
- [[squad]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
