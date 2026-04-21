---
type: source
subtype: paper
title: "jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval"
slug: g-nther-2025-jinaembeddingsv-2506-18902
date: 2026-04-20
language: en
tags: [embeddings, multimodal-retrieval, multilingual, late-interaction, code-retrieval]
processed: true
raw_file: raw/papers/g-nther-2025-jinaembeddingsv-2506-18902/paper.pdf
raw_md: raw/papers/g-nther-2025-jinaembeddingsv-2506-18902/paper.md
bibtex_file: raw/papers/g-nther-2025-jinaembeddingsv-2506-18902/paper.bib
possibly_outdated: false
authors:
  - Michael Günther
  - Saba Sturua
  - Mohammad Kalim Akram
  - Isabelle Mohr
  - Andrei Ungureanu
  - Bo Wang
  - Sedigheh Eslami
  - Scott Martens
  - Maximilian Werk
  - Nan Wang
  - Han Xiao
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2506.18902
doi: 10.48550/arXiv.2506.18902
url: http://arxiv.org/abs/2506.18902
citation_key: gnther2025jinaembeddingsv
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper introduces `jina-embeddings-v4`, a `3.8B`-parameter multimodal multilingual embedding model built on Qwen2.5-VL-3B-Instruct. It unifies text and image inputs in one shared decoder-style pathway, then exposes both dense and late-interaction retrieval outputs: a `2048`-dimensional single vector truncatable to `128`, and `128`-dimensional token-level multi-vectors. The model freezes the backbone and trains only a projection head plus three `60M` task-specific LoRA adapters for asymmetric retrieval, semantic similarity, and code retrieval. Beyond the model itself, the paper introduces the Jina-VDR benchmark for visually rich multilingual document retrieval. Empirically, the system is especially strong on visually rich retrieval, where late interaction consistently outperforms dense mode and surpasses prior CLIP-style and OCR-based baselines.

## Problem & Motivation

Existing embedding systems are usually fragmented across modalities, tasks, and deployment tradeoffs: text embedders, image embedders, code embedders, and visually rich document retrievers are often separate models. Dual-encoder multimodal systems such as CLIP-style architectures also suffer from a modality gap, making text-image alignment weaker than same-modality similarity. The paper aims to build a single embedding model that supports multilingual text, images, visually rich documents, and code search while preserving operational flexibility across dense and late-interaction retrieval. A second motivation is practical deployment: a shared frozen backbone with lightweight task adapters is cheaper to maintain than separate full models for each use case.

## Method

- **Backbone and input path**: the model uses Qwen2.5-VL-3B-Instruct with `3.8e9` parameters as a unified multimodal backbone; text inputs are tokenized normally, while images are converted into image-token sequences and then processed through the same language-model pathway.
- **Input limits and representation sizes**: text length is up to `32768` tokens; images are resized to `20` megapixels; dense embeddings are `2048`-dimensional and truncatable to `128`; multi-vector outputs use `128` dimensions per token.
- **Dual output modes**: dense retrieval applies mean pooling to the final hidden states, while late interaction keeps unpooled token embeddings and projects them for ColBERT-style scoring.
- **Late-interaction scoring**: the base similarity is `s_late(q, p) = sum_i max_j q_i p_j^T`; during pair training the score is normalized as `s'_late(q_i, p_j) = s_late(q_i, p_j) / t`, where `t` is the query token count.
- **Parameter-efficient specialization**: three task-specific LoRA adapters are trained for asymmetric query-document retrieval, semantic similarity / symmetric retrieval, and code retrieval; each adapter has `60M` parameters, so keeping all three adds less than `2%` memory overhead.
- **Training regime**: weights are initialized from Qwen2.5-VL-3B-Instruct; the backbone is frozen, while the LoRA adapters and the multi-vector projection layer are trained in two phases: pair training and task-specific specialization.
- **Joint pair objective**: pair training uses text-text batches `B_text` and text-image batches `B_multi`, combining dense InfoNCE, late-interaction InfoNCE, and KL alignment terms in a weighted loss `L_joint = sum_{i=1}^6 w_i L_i`.
- **Retrieval adapter**: asymmetric retrieval uses prefixes to distinguish queries from documents and extends InfoNCE to `L_NCE+` with in-batch and curated hard negatives, including multimodal hard negatives.
- **Similarity adapter**: semantic similarity uses CoSENT ranking loss on labeled STS pairs and falls back to InfoNCE where graded similarity labels are unavailable.
- **Code adapter**: code retrieval uses triplet-based training on resources such as CodeSearchNet, CodeFeedback, APPS, and CornStack, with temperature fixed at `tau = 0.02`.
- **Truncation-aware dense embeddings**: Matryoshka Representation Learning is applied so that early embedding dimensions retain the most semantically useful information when truncating from `2048` down to smaller prefixes.
- **Benchmark contribution**: the paper also constructs Jina-VDR by extending ViDoRe with `30` additional retrieval tasks spanning visually rich documents, multilingual coverage, non-question queries, and synthetic plus manually annotated data.

## Key Results

- On the overview benchmark table, dense `jina-embeddings-v4` scores `73.98` on Jina-VDR, `84.11` on ViDoRe, `84.11` on CLIP Benchmark, `66.49` on MMTEB retrieval, `71.59` on MTEB-CoIR, and `85.89` on English STS; late interaction raises Jina-VDR to `80.55` and ViDoRe to `90.17`.
- On the detailed Jina-VDR benchmark, the dense model averages `75.47` nDCG@5 and the multi-vector model reaches `81.52`, beating BM25+OCR (`46.88`), jina-embeddings-v3+OCR (`48.97`), ColPali-v1.2 (`65.39`), and DSE-Qwen2-2B-MRL-V1 (`68.89`).
- On ViDoRe, dense mode reaches `84.11` average nDCG@10 and late interaction reaches `90.17`, surpassing voyage-multimodal-3 (`84.20`) and ColPali-v1.2 (`83.90`) on the reported average.
- On CLIP Benchmark text-to-image retrieval, `jina-embeddings-v4` scores `84.11` Recall@5 versus `81.12` for jina-clip-v2 and `83.19` for nllb-siglip-large.
- For multilingual and long-context text retrieval, the model improves over jina-embeddings-v3 on MMTEB retrieval (`66.5` vs `58.6`) and LongEmbed (`67.11` vs `55.66`).
- For code retrieval, it reaches `71.59` average nDCG@10 on MTEB-CoIR, clearly above jina-embeddings-v3 (`55.07`) but still below the specialized voyage-code-3 model (`77.33`).
- In cross-modal alignment analysis, the reported alignment scores are `0.71` on Flickr30K, `0.72` on MSCOCO, and `0.56` on CIFAR-100, compared with OpenAI CLIP at `0.15`, `0.14`, and `0.20`.

## Limitations

- Late interaction is more accurate but materially more expensive in storage and scoring because it keeps token-level vectors instead of a single dense representation.
- The backbone language coverage constrains some multilingual image-text performance; the paper notes that nllb-siglip-large is better on Crossmodal3600 because that benchmark includes low-resource languages not supported by Qwen2.5-VL-3B-Instruct.
- The code adapter is text-only in practice: the paper explicitly notes that code training does not affect the vision branch, so multimodal benefits do not transfer to code retrieval.
- Part of Jina-VDR relies on LLM-generated or synthetic queries, which broadens coverage but may introduce benchmark-construction bias relative to fully human-authored search intents.
- The training recipe freezes the backbone and optimizes only adapters and projection layers, so the paper does not test whether end-to-end multimodal fine-tuning would further improve alignment or retrieval.

## Concepts Extracted

- [[text-embedding]]
- [[multimodal-retrieval]]
- [[dense-retrieval]]
- [[contrastive-learning]]
- [[low-rank-adaptation]]
- [[parameter-efficient-fine-tuning]]
- [[matryoshka-representation-learning]]
- [[mean-pooling]]
- [[semantic-textual-similarity]]
- [[information-retrieval]]

## Entities Extracted

- [[michael-gunther]]
- [[saba-sturua]]
- [[mohammad-kalim-akram]]
- [[isabelle-mohr]]
- [[andrei-ungureanu]]
- [[bo-wang-jina]]
- [[sedigheh-eslami]]
- [[scott-martens]]
- [[maximilian-werk]]
- [[nan-wang]]
- [[han-xiao]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
