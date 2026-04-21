---
type: source
subtype: paper
title: "Generative Retrieval via Term Set Generation"
slug: zhang-2024-generative-2305-13859
date: 2026-04-20
language: en
tags: [generative-retrieval, document-identifier, information-retrieval, autoregressive-search, term-set]
processed: true
raw_file: raw/papers/zhang-2024-generative-2305-13859/paper.pdf
raw_md: raw/papers/zhang-2024-generative-2305-13859/paper.md
bibtex_file: raw/papers/zhang-2024-generative-2305-13859/paper.bib
possibly_outdated: true
authors:
  - Peitian Zhang
  - Zheng Liu
  - Yujia Zhou
  - Zhicheng Dou
  - Fangchao Liu
  - Zhao Cao
year: 2024
venue: arXiv preprint
venue_type: preprint
arxiv_id: "2305.13859"
doi: ""
url: http://arxiv.org/abs/2305.13859
citation_key: zhang2024generative
paper_type: method
read_status: unread
domain: ir
---

## Summary

> ⚠️ **Possibly outdated**: Published 2023/2024 in a volatile domain (information retrieval / generative retrieval). Methods and benchmark SoTA may have been superseded.

AutoTSG proposes replacing the fixed sequential document identifier in auto-regressive search engines with an **unordered term-set identifier** and a matching **set-oriented generation pipeline**. The core insight is that sequential identifiers cause *false pruning*: one wrong decoding step irrecoverably drops the target document. By making any permutation of the term set a valid identifier, the model can recover through alternative orderings. Three techniques support this: (1) matching-oriented term selection using BERT + InfoNCE, (2) constrained greedy search that maintains validity via an inverted index, and (3) likelihood-adapted Seq2Seq learning that iteratively updates the training permutation. AutoTSG achieves notable improvements over all prior auto-regressive search engines on NQ320K and MS300K, with especially large gains on unseen documents.

## Problem & Motivation

Auto-regressive search engines (GENRE, DSI, SEAL, Ultron, NCI) require the Seq2Seq model to generate the *exact* predefined identifier for a target document. Because documents must be uniquely identifiable, identifier sequences are long; an incorrect prediction at any step causes **false pruning** — the document disappears from beam search candidates and cannot be recovered. Furthermore, the predefined ordering of terms in a sequential identifier may be unfavorable for a given query (terms that are easy to generate from the query may not appear early), creating a permutation-order dependency. Prior work ignores this; all identifiers are fixed before training. The gap between seen and unseen documents is another symptom: models memorize training-query-to-identifier mappings and fail to generalize when a document has no training queries.

## Method

**Document Identifier: Term-Set**
- Identifier `T(D) = {t_1, …, t_N}` — an unordered set of $N$ terms from document $D$ (default: $N=12$ for NQ320K).
- Retrieval: document $D$ is retrieved if *any* permutation of `T(D)` is generated. Relevance score: `Rel(Q, D) = max{ Π Pr(I_i | I_{<i}, Q; Θ) : I ∈ perms(T(D)) }`.

**Term Selection: Matching-Oriented (Section 3.1)**
- A BERT encoder maps each document token $t_i^D$ to a hidden state $e_i^D$; a linear projection $W ∈ ℝ^{d×1}$ + ReLU assigns scalar importance weight $w_i^D = σ(W^T e_i^D)$.
- Training objective: InfoNCE loss over query-document pairs, maximizing `Σ_{t_i^Q = t_j^D+} w_i^Q w_j^{D+} / τ` relative to negatives.
- Top-$N$ terms by weight form the identifier; the same weights initialize the permutation order for Seq2Seq training.

**Constrained Greedy Search (Section 3.2)**
- At decoding step $i$, the next term must satisfy two constraints: (1) not already generated (`I_i ∉ {I_1,…,I_{i-1}}`); (2) there exists at least one document whose identifier is a superset of `{I_1,…,I_i}` (validity constraint).
- Efficient implementation: an inverted index maps each prefix to the set of documents whose identifiers contain it; the feasible next-token set is `X = ∪{T(D') : I_{<i} ⊆ T(D')} \ I_{<i}`. Documents are pruned as the prefix grows; the index is very sparse after a few steps.
- Beam size `K = 100` throughout; greedily selects `argtop-K Π_{j≤i} Pr(I_j | I_{<j}, Q; Θ)` at each step.

**Likelihood-Adapted Seq2Seq Learning (Section 3.3)**
- No fixed ground-truth sequence exists; training uses the permutation of highest generation likelihood under the current model snapshot.
- Iterative procedure: given model $Θ^{t-1}$, sample candidate permutations by top-K sampling from difference set; select argmax-likelihood permutation $I^t$; update model to $Θ^t$ via standard Seq2Seq training.
- Initialization by importance ordering (Section 3.3); convergence in ≈2 iterations on validation accuracy.

**Backbone & Data Augmentation**
- Backbone: T5-base (default); T5-large also evaluated.
- Data augmentation: DocT5Query pseudo-query generation, consistent with prior work.

## Key Results

**NQ320K (Table 1 — full corpus)**
- AutoTSG: `MRR@10 = 0.757`, `MRR@100 = 0.760`, `Recall@1 = 0.690`, `Recall@10 = 0.875`, `Recall@100 = 0.932`
- Best prior (NCI): `MRR@100 = 0.731`, `Recall@10 = 0.852` — AutoTSG gains `+3.9%` MRR@100, `+2.4%` Recall@10

**MS300K (Table 2 — full corpus)**
- AutoTSG: `MRR@10 = 0.484`, `MRR@100 = 0.491`, `Recall@1 = 0.359`, `Recall@10 = 0.766`, `Recall@100 = 0.907`
- Best prior (DPR): `MRR@100 = 0.433`, `Recall@10 = 0.764` — AutoTSG gains `+13.4%` MRR@100; note DPR still leads on `Recall@100 = 0.948`

**Seen vs. Unseen Documents (Tables 3–4)**
- NQ320K Unseen (50%): AutoTSG `MRR@10 = 0.466`, `Recall@10 = 0.654` vs. NCI `0.050 / 0.159` and Ultron `0.300 / 0.383` — largest gain in generalization
- This confirms that term-set relaxation reduces dependence on memorizing query-to-identifier mappings

**Ablation (Table 5 — NQ320K)**
- Term-set vs. Sequence identifier: `+0.024` MRR@10, `+0.027` Recall@10
- Matching-oriented vs. Title selection: `+0.015` MRR@10
- Likelihood-adapted vs. Non-adaptive training: `+0.014` MRR@10
- Importance vs. Random initialization: `+0.034` MRR@10

**Efficiency (Table 6 — NQ320K)**
- Memory: `35 MB` (vs. SEAL 210 MB, GENRE/Ultron 27 MB); query latency `0.06s @ bs=10`, `0.69s @ bs=100` — marginal overhead vs. GENRE

## Limitations

- Evaluated only on NQ320K and MS300K (~320k documents each); scalability to million-scale or web-scale corpora is not established — false-pruning gains may shrink as corpus size grows, since $N$ must increase.
- Relies on DocT5Query augmentation for competitive results; without query generation, improvement is smaller (AutoTSG w.o. QG: `MRR@100 = 0.710`).
- DPR still leads on `Recall@100` on MS300K, indicating that dense retrieval retains a recall advantage at the deepest cut; auto-regressive search engines remain challenged on high-recall settings.
- The iterative likelihood-adapted training adds training complexity compared to simple fixed-objective Seq2Seq learning; convergence analysis is empirical rather than theoretical.
- Identifier collision guarantee depends on choosing $N$ large enough per corpus; the paper provides this for NQ320K/MS300K but does not give a principled formula for arbitrary corpora.

## Concepts Extracted

- [[generative-retrieval]]
- [[generative-information-retrieval]]
- [[document-identifier]]
- [[set-based-document-identifier]]
- [[set-oriented-generation]]
- [[matching-oriented-term-selection]]
- [[likelihood-adapted-sequence-to-sequence-learning]]
- [[false-pruning]]
- [[constrained-beam-search]]
- [[differentiable-search-index]]
- [[dense-retrieval]]
- [[sparse-retrieval]]
- [[infonce-loss]]
- [[sequence-to-sequence]]
- [[query-generation]]

## Entities Extracted

- [[peitian-zhang]]
- [[zheng-liu-huawei]]
- [[yujia-zhou]]
- [[zhicheng-dou]]
- [[fangchao-liu]]
- [[zhao-cao]]
- [[renmin-university-of-china]]
- [[huawei]]
- [[t5]]
- [[bert]]
- [[bm25]]
- [[dpr]]
- [[natural-questions]]
- [[nq320k]]
- [[ms-marco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
