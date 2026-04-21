---
type: source
subtype: paper
title: Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval
slug: alon-2022-neurosymbolic-2201-12431
date: 2026-04-20
language: en
tags: [retrieval, language-modeling, automata, neuro-symbolic, domain-adaptation]
processed: true

raw_file: raw/papers/alon-2022-neurosymbolic-2201-12431/paper.pdf
raw_md: raw/papers/alon-2022-neurosymbolic-2201-12431/paper.md
bibtex_file: raw/papers/alon-2022-neurosymbolic-2201-12431/paper.bib
possibly_outdated: true

authors:
  - Uri Alon
  - Frank F. Xu
  - Junxian He
  - Sudipta Sengupta
  - Dan Roth
  - Graham Neubig
year: 2022
venue: ICML 2022
venue_type: conference
arxiv_id: 2201.12431
doi:
url: https://arxiv.org/abs/2201.12431
citation_key: alon2022neurosymbolic
paper_type: method

read_status: unread

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

The paper proposes RETOMATON, an unsupervised retrieval automaton that reduces the search overhead of retrieval-based language models without discarding non-parametric evidence. Starting from a standard kNN-LM datastore, it stores successor pointers between consecutive entries and clusters nearby keys into automaton states, yielding a weighted finite automaton traversed alongside the base LM at inference time. The traversal approximates future nearest-neighbor results and only falls back to expensive kNN search when the active state set becomes too small. On WIKIText-103, RETOMATON matches kNN-LM while saving `81%` of searches and lowers perplexity from `16.65` to `16.08` at `FoSS = 0`; on Law-MT it reduces perplexity from `12.34` to `10.49`, and improves a fine-tuned model to `7.10`.

## Problem & Motivation

Retrieval-based language models improve prediction by consulting an external datastore at test time, but vanilla `kNN-LM` performs a nearest-neighbor lookup at every decoding step. That repeated search is much slower than the parametric LM forward pass and becomes the deployment bottleneck even when retrieval itself is highly beneficial.

The paper asks whether retrieval trajectories contain reusable symbolic structure. Its answer is that neighbors found at time `t` often imply plausible neighbors at time `t+1`, so the model should be able to continue retrieval by following stored transitions instead of restarting full search every step. This motivates a neuro-symbolic design that keeps the LM's dense representations while exposing a lightweight automaton over the datastore.

## Method

- **Base retrieval model**: start from `kNN-LM`, where the datastore is `(\mathcal{K}, \mathcal{V}) = {(f(c_i), w_i)}` and the non-parametric distribution is `p_{kNN}(w|c) \propto \sum_{(f(c_i), w_i) \in \mathcal{N}} 1_{w=w_i} \exp(-dist(f(c), f(c_i)))`.
- **Datastore augmentation**: each entry becomes a triple `(key, value, pointer)`, where the pointer links entry `i` to the next corpus entry `i+1`; this preserves sequential continuation signals that flat datastores discard.
- **Automaton construction**: cluster nearby key vectors into states `Q` and define a weighted finite automaton `A = <Q, \Sigma, q_0, \delta, \phi>`. States contain multiple datastore entries, and transitions are induced by the pointers of their members.
- **Traversal rule**: after emitting token `w^(t)`, continue with `\hat{\delta}(\mathcal{S}^{(t)}, w^(t))` if the resulting state set size is at least `\tau`; otherwise restart kNN search and union the new states with any surviving traversal states. Lower `\tau` means fewer searches and more aggressive reuse.
- **Transition scoring**: for active states `\mathcal{S}`, compute `\phi(q, c, w) = \sum_{(k_i, w_i, \cdot) \in \pi^{-1}(q)} 1_{w=w_i} \exp(-dist(f(c), k_i))`, then aggregate `p_{auto}(w|c, \mathcal{S}) \propto \sum_{q \in \mathcal{S}} \phi(q, c, w)`.
- **Final prediction**: interpolate automaton and base LM distributions as `p(w|c,\mathcal{S}) = \lambda p_{auto}(w|c,\mathcal{S}) + (1-\lambda)p_{LM}(w|c)`, preserving retrieval influence even when no fresh search is issued.
- **Implementation details**: reuse FAISS search from prior `kNN-LM`; retrieve `k_neigh = 1024` neighbors on full search; cap traversal evaluation to `max_knns = 1024` entries; store keys in `fp16`; use `k_clus = 1M` on WIKIText-103 (`103M` entries) and `k_clus = 200K` on Law-MT (`19M` entries), giving average cluster size around `100`.
- **Experimental setup**: in-domain experiments use a `16`-layer Transformer LM with `16` heads, `1024`-dimensional hidden states and `4096`-dimensional FFN (`247M` parameters) on WIKIText-103; domain-adaptation experiments use a `12`-layer, `1536`-dimensional Transformer with `42K` subword vocabulary (`656M` parameters) trained on WMT News Crawl and adapted with a Law-MT datastore.

## Key Results

- On WIKIText-103, RETOMATON matches `kNN-LM` while saving `81%` of nearest-neighbor searches; without clustering it still saves more than `60%`.
- At `FoSS = 0` on WIKIText-103, perplexity drops from `16.65` (`kNN-LM`) and `16.35` (ADAPTRET) to `16.08` with RETOMATON, a gain of `0.57` and `0.27` perplexity respectively.
- On Law-MT domain adaptation, perplexity falls from `12.34` (`kNN-LM`) and `12.01` (ADAPTRET) to `10.49` with RETOMATON when search is performed every step.
- For a Law-MT fine-tuned LM, RETOMATON improves perplexity from `8.61` to `7.10`, a relative reduction of `17.5%`; at `FoSS = 0.5`, it achieves `7.15`, still `17.0%` better than the fine-tuned LM.
- The wall-clock proxy analysis shows RETOMATON can save up to `83%` of searches; with GPU FAISS plus clustering, wall-clock benefit starts around `FoSS = 0.32`.
- Ablations show pointers drive most gains for low search-saving regimes (`FoSS < 0.4`), while clustering matters more once `FoSS` rises above roughly `0.7`.

## Limitations

- The method depends on repeated local structure in the datastore; the paper itself reports stronger gains on Law-MT than on WIKIText-103 because Law-MT has higher `n`-gram overlap between train and validation.
- It still needs an initial nearest-neighbor system and a large datastore, so preprocessing and memory costs remain substantial even if test-time searches are reduced.
- Clustering quality is a new sensitivity: coarse clustering (`k = 100K` on WIKIText-103) is too noisy, while overly fine clustering reduces long traversal benefits at high `FoSS`.
- Evaluation is limited to perplexity-based language modeling scenarios; the paper argues the approach may generalize to phrase- or chunk-level retrieval, but does not validate that claim experimentally.
- The interpolation factor `\lambda` stays fixed; the paper suggests learned dynamic interpolation as future work, indicating that confidence calibration is not solved.

## Concepts Extracted

- [[retrieval-based-language-model]]
- [[knn-language-model]]
- [[weighted-finite-automaton]]
- [[nearest-neighbor-search]]
- [[domain-adaptation]]
- [[clustering]]
- [[datastore]]
- [[fine-tuning]]
- [[transformer]]

## Entities Extracted

- [[uri-alon]]
- [[frank-f-xu]]
- [[junxian-he]]
- [[sudipta-sengupta]]
- [[dan-roth]]
- [[graham-neubig]]
- [[carnegie-mellon-university]]
- [[amazon-aws]]
- [[aws-ai-labs]]
- [[retomaton]]
- [[faiss]]
- [[wikitext-103]]
- [[law-mt]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
