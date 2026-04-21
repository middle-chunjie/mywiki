---
type: source
subtype: paper
title: Adversarial Robustness of Deep Code Comment Generation
slug: zhou-2022-adversarial
date: 2026-04-20
language: en
tags: [adversarial-attack, code-comment-generation, software-engineering, robustness, nlp]
processed: true
raw_file: raw/papers/zhou-2022-adversarial/paper.pdf
raw_md: raw/papers/zhou-2022-adversarial/paper.md
bibtex_file: raw/papers/zhou-2022-adversarial/paper.bib
possibly_outdated: true
authors:
  - Yu Zhou
  - Xiaoqing Zhang
  - Juanjuan Shen
  - Tingting Han
  - Taolue Chen
  - Harald Gall
year: 2022
venue: ACM Transactions on Software Engineering and Methodology
venue_type: journal
arxiv_id:
doi: 10.1145/3501256
url: https://dl.acm.org/doi/10.1145/3501256
citation_key: zhou2022adversarial
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify robustness claims against recent code LLM literature.

This paper proposes ACCENT (Adversarial Code Comment gENeraTor), a black-box identifier-substitution method that crafts adversarial code snippets for the code comment generation task. ACCENT extracts programmer-defined identifiers, selects semantically similar replacements via word2vec cosine similarity, ranks them by a saliency-weighted score, and iteratively substitutes up to `max` identifiers while preserving syntactic correctness and program semantics. To defend against such attacks, the paper introduces masked training, a lightweight adversarial training variant that mixes original and identifier-masked inputs under a combined loss. Evaluated on five seq2seq architectures (LSTM, Transformer, GNN, CSCG dual, Rencos) across Java and Python datasets, ACCENT achieves up to 79% relative BLEU degradation with 100% valid-code rate, outperforming Metropolis-Hastings and random-substitution baselines, and the masked training method recovers adversarial-example BLEU to near-clean levels with minimal clean-data accuracy loss.

## Problem & Motivation

Deep neural network models for code comment generation are vulnerable to adversarial examples: small, semantics-preserving perturbations to source code can cause the model to produce completely wrong comments. Unlike image or NLP adversarial attacks, source-code perturbations must satisfy strict syntactic (compilability) and semantic (functionality-preserving) constraints, making gradient-based continuous-optimization methods inapplicable. The discrete, grammar-bound nature of programs demands a dedicated attack strategy. Robustness is especially important since similar code should yield similar comments — inconsistency undermines program comprehension tooling.

## Method

**ACCENT — adversarial example generation:**

- **Step 1 — Identifier Extraction:** Use AST parsers (`javalang` for Java, `ast` lib for Python) to extract programmer-defined identifiers (method names, variable names) from all programs in the dataset; build a global candidate set `V`.
- **Step 2 — Candidate Selection:** Train `word2vec` skip-gram embeddings on code token sequences. For each identifier `w` in program `p`, compute `L_w = top_K(cos(w, V'))` where `V'` excludes identifiers already in `p`; default `K = 5`. Cosine similarity ensures replacements are semantically close and the resulting program remains functionality-preserving.
- **Step 3 — Best Candidate & Saliency Reranking:** For each candidate `w' ∈ L_w`, compute score change `Δscore_w* = score(p) − score(p[w←w'])` using BLEU as the score proxy. Compute identifier saliency `S(p, w) = cos(vec(w), vec(p))` where `vec(p)` is the output of an independent LSTM encoder. Rank identifiers by:
  `H(p, p*, w) = S(p,w)·Δscore_w*` (with fallback to single factor when one term is zero, using constants `α, β ∈ [0,1]`).
- **Step 4 — Substitution:** Replace the top-`max` identifiers (all occurrences) in descending `H` order; default `max ∈ {2, 3}`.

**Masked training — robustness defense:**

- Objective: `θ* = argmin_θ (λ·L_origin(p, com) + (1−λ)·L_masked(p', com))`
- `p'` is constructed by randomly replacing `k` programmer-defined identifiers with `<unk>` tokens.
- `L_origin` maintains clean-data performance; `L_masked` forces the model to generate correct comments even without identifier information, reducing reliance on non-robust features.
- Hyperparameter `λ` trades off between accuracy and robustness.

**Victim models evaluated:** LSTM-based seq2seq (2-layer BiLSTM, `embed_size=512`), Transformer (6 layers, 8 heads, `d_k=d_v=64`, `d_ff=2048`), GNN-based (structural + GRU textual encoder), CSCG Dual (dual-learning LSTM), Rencos (retrieval-augmented neural summarization).

**Datasets:** Java (69,708 train / 8,714 val / 8,714 test), Python (50,400 train / 13,248 val / 13,216 test).

## Key Results

- **Attack effectiveness (RQ1/RQ2, max=2):** ACCENT degrades BLEU by 63.12% (LSTM), 70.32% (Transformer), 58.28% (GNN), 79.12% (CSCG), 7.55% (Rencos) on Java; 40.64%, 43.08%, 37.80%, 61.39%, 6.96% on Python — all with 100% valid-code rate vs. ~30% for random substitution.
- **Baseline comparison:** ACCENT consistently outperforms MH-based algorithm on relative degradation (`r_d`); Mann-Whitney U test yields p < 0.05 for 15/20 experimental conditions.
- **Transferability (RQ3):** Adversarial examples generated for one model reduce BLEU by ~50% on Java and ~37% on Python when transferred to other models; ACCENT examples transfer better than MH-based ones across all models.
- **Masked training (RQ4):** Transformer masked-trained model recovers adversarial BLEU from 13.23 (normal) to 40.10 (max=2) and 39.24 (max=3), vs. data augmentation's 18.10 and 17.82 — a ~2× improvement on adversarial robustness with minimal clean accuracy loss (44.84 vs. 44.58 BLEU clean).
- **Human evaluation:** Masked training achieves mean similarity/naturalness/informativeness scores of 3.84/4.52/3.35 (Java) and 3.76/4.46/3.55 (Python), consistently above normal training (2.32/3.17/2.12) and data augmentation (2.88/4.23/2.48).
- **Rencos robustness:** The retrieval-augmented model is notably more robust than pure seq2seq models (only ~7–8% degradation), suggesting structural or retrieved context helps resist identifier-level perturbations.

## Limitations

- **Java and Python only:** The approach is language-agnostic in principle but evaluated only on two languages; generalizability to statically-typed languages (C++, Rust) or dynamically-typed scripting languages requires separate validation.
- **Identifier-level perturbation only:** ACCENT does not explore structural code transformations (e.g., loop refactoring, dead-code insertion), which may expose additional model weaknesses.
- **BLEU as proxy during attack:** Using BLEU as the score function in the attack loop may not align well with human judgments of comment quality; the attack's effectiveness under semantic similarity metrics is not independently verified.
- **`max` is small (2–3):** Practical attacks with more substitutions are not studied; adversarial stability with larger `max` is untested.
- **Masked training hyperparameter `λ` sensitivity:** The optimal balance between clean and masked loss is dataset- and model-dependent; the paper does not provide a principled selection strategy.
- **No evaluation against stronger defenses:** Masked training is compared only to data augmentation; certified robustness or detection-based defenses are not considered.

## Concepts Extracted

- [[adversarial-attack]]
- [[adversarial-training]]
- [[code-comment-generation]]
- [[identifier-substitution-attack]]
- [[masked-training]]
- [[sequence-to-sequence]]
- [[abstract-syntax-tree]]
- [[saliency-score]]
- [[adversarial-robustness]]
- [[transferability]]
- [[code-summarization]]
- [[source-code-modeling]]
- [[black-box-attack]]
- [[encoder-decoder-architecture]]
- [[graph-neural-network]]
- [[long-short-term-memory]]

## Entities Extracted

- [[yu-zhou]]
- [[xiaoqing-zhang]]
- [[juanjuan-shen]]
- [[tingting-han]]
- [[taolue-chen]]
- [[harald-gall]]
- [[nanjing-university-of-aeronautics-and-astronautics]]
- [[birkbeck-university-of-london]]
- [[university-of-zurich]]

## Contradictions

<!-- None yet; first source on code-comment-generation adversarial robustness. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
