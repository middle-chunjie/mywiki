---
type: source
subtype: paper
title: "Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering"
slug: zhang-2024-knowledgeable-2311-06503
date: 2026-04-20
language: en
tags: [preference-alignment, knowledge-graph, domain-specific-qa, llm, nlp]
processed: true
raw_file: raw/papers/zhang-2024-knowledgeable-2311-06503/paper.pdf
raw_md: raw/papers/zhang-2024-knowledgeable-2311-06503/paper.md
bibtex_file: raw/papers/zhang-2024-knowledgeable-2311-06503/paper.bib
possibly_outdated: true
authors:
  - Yichi Zhang
  - Zhuo Chen
  - Yin Fang
  - Yanxi Lu
  - Fangming Li
  - Wen Zhang
  - Huajun Chen
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: "2311.06503"
doi: ""
url: http://arxiv.org/abs/2311.06503
citation_key: zhang2024knowledgeable
paper_type: method
read_status: unread
domain: llm
---

> ⚠️ **Possibly outdated** (2024, volatile domain: LLM/NLP). Core claims may have been superseded by newer alignment and KG-enhanced LLM work.

## Summary

KnowPAT presents a preference-alignment pipeline for LLMs operating on domain-specific QA tasks that are backed by a domain knowledge graph (KG). The core insight is that two practical failure modes in KG-enhanced LLM QA — inappropriate answer style and erroneous or blind use of retrieved knowledge — both reduce to a preference misalignment problem. The paper proposes constructing two kinds of preference sets (style and knowledge) using multiple LLMs and varied retrieval groups, then training with a new pairwise alignment objective combined with adaptive per-candidate weighting. Experiments on a proprietary Huawei cloud-product QA corpus (CPKG: ~14 K entities, ~21 K triples; 8909 QA pairs) demonstrate consistent gains over 15 baselines across BLEU, ROUGE, BERTScore, and human evaluation.

## Problem & Motivation

Standard retrieve-then-fine-tune pipelines for domain-specific LLM QA face two intertwined problems: (1) generated answers may be stylistically inappropriate or low-quality (e.g., unfriendly, verbose, or irrelevant tone); (2) the retrieved KG triples are often noisy or irrelevant, and the model must learn to selectively use vs. ignore them. Vanilla fine-tuning (VFT) with autoregressive loss cannot solve either: it merely fits the model to gold answers without signaling what *not* to generate. The authors unify both requirements as model-preference problems requiring explicit alignment with human preference.

## Method

### Overall Pipeline

KnowPAT is a three-stage pipeline operating on a dataset $\mathcal{D}=\{(q_i, a_i)\}$ and a domain Cloud Product Knowledge Graph (CPKG) $\mathcal{G}=(\mathcal{E}, \mathcal{R}, \mathcal{T})$ (13,995 entities, 463 relations, 20,752 triples).

### Stage 1 — Unsupervised Triple Linking

A dense retriever $\mathcal{H}$ (BGE-base-zh-v1.5) encodes questions and triple text and selects top-$k$ triples by cosine similarity:
`sim(i,j) = Cosine(H(q_i), H(h_j, r_j, t_j))`

No labeled question-knowledge pairs are required; the retrieval is purely unsupervised and must generalize zero-shot to new questions.

### Stage 2 — Knowledgeable Preference Set Construction

For each QA pair $(q, a)$, two preference sets of size $l=4$ are built:

**Style Preference Set (SPS)** $\mathcal{P}_s$: Gold answer $b_1$ + answers from three LLMs of descending quality: ChatGPT ($b_2$) > ChatGLM-6B ($b_3$) > Vicuna-7B ($b_4$). Preference scores: $r_1 > r_2 > r_3 > r_4$.

**Knowledge Preference Set (KPS)** $\mathcal{P}_k$: Gold answer $c_1$ + answer generated with top-$k$ triples $\mathcal{K}_1$ ($c_2$) > answer with no retrieved knowledge $\mathcal{K}_2=\emptyset$ ($c_3$) > answer with borderline noisy triples $\mathcal{K}_3$ (ranks $k+1$ to $2k$) ($c_4$). This ordering encodes that misleading knowledge is worse than no knowledge.

The full preference dataset has $2N$ preference sets (two per QA pair, $N=7909$ training).

### Stage 3 — Fine-Tuning and Preference Alignment

Training combines vanilla fine-tuning loss $\mathcal{L}_{ft}$ (autoregressive on gold answers) with an alignment loss $\mathcal{L}_{align}$.

Model preference score for answer $a_i$:
`S_i = (1/|a_i|) Σ_j log P_M(a_{i,j} | I, q_i, a_{i,<j})`

The alignment objective contrasts preferred vs. unpreferred answers:
`L_align = Σ_{i=1}^{|P|-1} μ_i * (log(1 + e^{-S_i}) + Σ_{r_j < r_i} log(1 + e^{S_j}))`

Adaptive weight $\mu_i$ normalizes each candidate's influence by its relative model-score position:
`μ_i = (S_i - S_min) / (S_max - S_min)`

This differs from margin-rank losses used in RRHF/SLiC-HF: those only penalize when the model *already* inverts the preference order; KnowPAT's sigmoid-based loss continuously discourages unpreferred outputs regardless of the current ordering.

Final combined objective with coefficient $\lambda$ (searched over `{1, 0.1, 0.01, 0.001}`):
`L = L_ft + λ/(|P|-1) * L_align`

### Implementation

Backbone: Atom-7B-CP (Llama-2-based, Chinese vocabulary extended, first fine-tuned on CPKG product documents). Training: `bf16`, 3 epochs, gradient accumulation `8`, AdamW lr `3e-4`. Retriever: BGE-base-zh-v1.5.

## Key Results

- **Traditional metrics**: KnowPAT vs. best baseline (AFT-BC on BLEU-1; PRO on BLEU-3/4):
  - BLEU-1: `22.56` (+22.67% over `18.39`)
  - BLEU-4: `12.11` (+43.99% over `8.41`)
  - ROUGE-1: `20.28` (+12.10% over `18.09`)
  - CIDEr: `54.86` (+66.04% over `33.04`)
  - METEOR: `23.62` (+21.25% over `19.48`)
- **Model-based metrics**: BERTScore `69.34` (best; runner-up PRO `67.41`); PPL `29.93` (best; runner-up AFT `30.11`).
- **Human evaluation**: KnowPAT wins or ties vs. all baselines across 100-question pairwise comparisons; only rare losses observed.
- **Ablation**: Removing SPS degrades BLEU-1 by `4.99` pts; removing KPS by `6.44` pts; removing the knowledge graph entirely drops BLEU-1 to `15.09` (−7.47). Adaptive weights provide smaller but consistent gains.
- **General ability (CMMLU)**: Some degradation in clinical medicine; politics/history/economics maintained at backbone level or slightly improved.

## Limitations

- Proprietary dataset: CPKG and the QA corpus cannot be publicly released, limiting external reproducibility.
- Evaluation is confined to one domain (Huawei cloud products); generalizability to other domain QA settings is unverified.
- Preference scores in SPS rely on assumed LLM quality ranking (ChatGPT > ChatGLM > Vicuna), which may not hold across domains or evolve with newer model releases.
- The unsupervised KG retriever has no explicit negative supervision; performance degrades when KG coverage is low.
- Some catastrophic forgetting in biomedical capabilities after domain fine-tuning (CMMLU clinical medicine drop).
- No public benchmark comparison; all results are on an internal cloud-product QA test set.

## Concepts Extracted

- [[preference-alignment]]
- [[knowledge-graph]]
- [[supervised-fine-tuning]]
- [[reinforcement-learning-from-human-feedback]]
- [[direct-preference-optimization]]
- [[in-context-learning]]
- [[knowledge-graph-question-answering]]
- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[domain-specific-question-answering]]
- [[knowledge-preference-alignment]]
- [[style-preference-alignment]]
- [[contrastive-loss]]
- [[adaptive-weight]]

## Entities Extracted

- [[yichi-zhang]]
- [[zhuo-chen]]
- [[yin-fang]]
- [[yanxi-lu]]
- [[fangming-li]]
- [[wen-zhang]]
- [[huajun-chen]]
- [[zhejiang-university]]
- [[huawei]]
- [[chatgpt]]
- [[chatglm-6b]]
- [[vicuna]]
- [[llama-2]]
- [[atom-7b]]
- [[bge-base-zh]]
- [[knowpat]]
- [[zjukg]]
- [[cmmlu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
