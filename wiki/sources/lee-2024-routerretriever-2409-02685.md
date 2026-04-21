---
type: source
subtype: paper
title: "RouterRetriever: Exploring the Benefits of Routing over Multiple Expert Embedding Models"
slug: lee-2024-routerretriever-2409-02685
date: 2026-04-20
language: en
tags: [dense-retrieval, routing, mixture-of-experts, lora, beir]
processed: true
raw_file: raw/papers/lee-2024-routerretriever-2409-02685/paper.pdf
raw_md: raw/papers/lee-2024-routerretriever-2409-02685/paper.md
bibtex_file: raw/papers/lee-2024-routerretriever-2409-02685/paper.bib
possibly_outdated: false
authors:
  - Hyunji Lee
  - Luca Soldaini
  - Arman Cohan
  - Minjoon Seo
  - Kyle Lo
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2409.02685
doi:
url: http://arxiv.org/abs/2409.02685
citation_key: lee2024routerretriever
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper proposes RouterRetriever, a dense retrieval system that routes each query to one of several domain-specific expert embedding models instead of relying on a single general-purpose retriever. The architecture keeps a frozen [[contriever]] base encoder and adds one LoRA expert per domain, then performs training-free routing with a pilot embedding library built from centroid embeddings of instances best served by each expert. On BEIR, the method beats both a single MSMARCO-trained model and a single multi-task model while remaining lightweight and modular: new experts can be added or removed without retraining the router. The main empirical claim is that routing over specialized embedders is a stronger alternative to forcing one encoder to serve heterogeneous retrieval domains.

## Problem & Motivation

Dense retrieval systems are often trained as a single embedding model on large general-domain corpora such as [[msmarco]], which gives broad coverage but often underperforms domain-specialized retrievers on target domains like biomedical, scientific, or finance retrieval. Multi-task training is one response, but it still requires full retraining when new domains arrive and may degrade or destabilize performance across domains. The paper asks whether it is better to maintain multiple lightweight domain experts and route each query to the most suitable one. The motivation is both effectiveness and maintainability: preserve domain-specific strengths, reduce the cost of adding new domains, and avoid retraining a monolithic retriever whenever the expert set changes.

## Method

- **Backbone and experts**: use a frozen [[contriever]] dense encoder as the base model and train one LoRA expert per domain dataset `D_i`; the expert set is `\mathcal{E} = \{e_1, \dots, e_T\}`.
- **Expert training**: only the query encoder LoRA parameters are tuned in the main setup, with LoRA rank `r = 8`, scaling `\alpha = 32`, and about `1M` trainable parameters per expert (`~0.5%` of the model).
- **Pilot embedding library**: for each training instance `x_j` in a domain dataset, run all experts and assign `e_{\max} = \arg\max_{e_i \in \mathcal{E}} \mathrm{Performance}(e_i, x_j)`. Group instances by the winning expert and compute one centroid `\mathbf{c}_m = \mathrm{Centroid}(\mathrm{BaseEncoder}(\mathrm{Group}_m))` per non-empty group.
- **Library size and structure**: repeating the grouping-and-centroid procedure across `T` datasets yields at most `T^2` pilot embeddings. The appendix reports that `k = 1` centroid per group works better than larger `k`, because extra centroids become distractors.
- **Routing rule**: given a query, encode it once with the base encoder, compute similarities to all pilot embeddings, average similarities over the pilot embeddings attached to each expert, and select the expert with the highest mean score.
- **Final query embedding**: after routing, run a second forward pass through the base encoder plus the selected expert LoRA module to obtain the query embedding used for retrieval.
- **Training hyperparameters**: learning rate `1e-4`, batch size `256` with in-batch negatives, and up to `500` epochs with early stopping. The context encoder stays frozen in the main experiments.

## Key Results

- On the seven-domain BEIR setup, RouterRetriever without an MSMARCO expert reaches average `49.3` nDCG@10, beating a single [[msmarco]] model at `47.5` and a multi-task model at `46.4`.
- Adding an MSMARCO expert raises RouterRetriever to `49.6` average nDCG@10, still below `DatasetOracle` (`50.9`) and well below `InstanceOracle` (`57.6`), which quantifies remaining routing headroom.
- RouterRetriever outperforms routing baselines used in language modeling: `ExpertClassifierRouter = 46.4`, `ClassificationHeadRouter = 46.8`, and `DatasetRouter = 48.5`, showing the benefit of retrieval-specific routing by embedding similarity.
- Gains are especially strong on some specialized datasets, for example `SciFact: 76.0` vs `67.2` (MSMARCO) and `69.4` (Multi-Task), and `HotpotQA: 59.5` vs `57.6` and `52.1`.
- Zero-shot generalization also improves modestly on unseen BEIR domains without trained experts: average `31.9` versus `31.6` for MSMARCO and `31.2` for Multi-Task.

## Limitations

- Inference requires two query-side passes: one for routing and one for final embedding generation, so the method improves modularity at the cost of extra latency.
- Routing quality remains the main bottleneck: the gap from `49.6` to `57.6` against `InstanceOracle` shows that the current similarity-based router still misroutes many instances.
- Performance gains diminish as more experts are added, and the analysis shows routing becomes more distracted as the expert pool grows.
- The evaluation is centered on [[beir]], [[contriever]], and LoRA-based query experts, so it does not establish that the same routing design transfers unchanged to other retriever families or end-to-end learned dual encoders.
- The paper depends on access to sufficiently strong domain-specific training data; small domains were already filtered or augmented before the main study.

## Concepts Extracted

- [[information-retrieval]]
- [[dense-retrieval]]
- [[mixture-of-experts]]
- [[low-rank-adaptation]]
- [[parameter-efficient-fine-tuning]]
- [[routing-mechanism]]
- [[pilot-embedding]]
- [[multi-task-learning]]
- [[zero-shot-generalization]]
- [[domain-specific-retriever]]

## Entities Extracted

- [[hyunji-lee]]
- [[luca-soldaini]]
- [[arman-cohan]]
- [[minjoon-seo]]
- [[kyle-lo]]
- [[allen-institute-for-ai]]
- [[contriever]]
- [[beir]]
- [[msmarco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
