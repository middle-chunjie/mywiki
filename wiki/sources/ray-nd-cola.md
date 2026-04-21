---
type: source
subtype: paper
title: "Cola: A Benchmark for Compositional Text-to-image Retrieval"
slug: ray-nd-cola
date: 2026-04-20
language: en
tags: [vision-language, multimodal, retrieval, benchmark, compositionality]
processed: true
raw_file: raw/papers/ray-nd-cola/paper.pdf
raw_md: raw/papers/ray-nd-cola/paper.md
bibtex_file: raw/papers/ray-nd-cola/paper.bib
possibly_outdated: true
authors:
  - Arijit Ray
  - Filip Radenovic
  - Abhimanyu Dubey
  - Bryan A. Plummer
  - Ranjay Krishna
  - Kate Saenko
year: 2023
venue: NeurIPS 2023 Datasets and Benchmarks
venue_type: conference
arxiv_id: 2305.03689
doi:
url: https://arxiv.org/abs/2305.03689
citation_key: rayndcola
paper_type: benchmark
read_status: unread
domain: multimodal
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces `COLA`, a text-to-image retrieval benchmark for testing whether vision-language models bind attributes to the correct objects rather than to distractors with similar parts and properties. It covers both single-object and multi-object compositional queries and pairs them with hard distractor images, making retrieval depend on fine-grained attribute-object grounding instead of coarse object presence. Using `COLA` as a testbed, the authors compare six adaptation strategies on CLIP and FLAVA and show that a lightweight multimodal adapter that jointly attends over frozen image and text features is consistently stronger than prompt tuning, linear probing, and standard late-layer fine-tuning. The benchmark also exposes a large remaining gap between model performance and human accuracy.

## Problem & Motivation

Off-the-shelf vision-language models often match captions to images by recognizing the right objects and attributes somewhere in the scene, but fail to attach each attribute to the intended object. This weakness matters for embodied agents, retrieval systems, and assistive interfaces that must distinguish queries such as "round white table" from scenes where "round" and "white" belong to different objects. The paper argues that attribute-object binding is a simpler but still under-solved form of compositional reasoning, and that text-to-image retrieval over hard distractors is a stricter test than image-to-text settings used in contemporary benchmarks such as CREPE.

## Method

- **Benchmark construction**: `COLA` builds hard text-to-image retrieval tasks with about `1.2k` composed queries spanning `168` objects and `197` attributes over roughly `30k` images. It includes single-object queries and paired multi-object queries where distractors contain the same words in the wrong configuration.
- **Base models**: the study adapts frozen or partially trainable vision-language backbones, primarily CLIP ViT-B/32, and compares against FLAVA B-16 as a larger model with pretrained multimodal layers.
- **Multimodal adapter**: concatenate image patch features `I` and query token features `Q` as `M = [I; Q]`, run a multimodal transformer encoder `A = Att(M)`, then let a learned `[CLS]` token cross-attend to `A` through a transformer decoder to produce `out_MM`.
- **Scoring rule**: MM-Adapter aligns the multimodal output to frozen text features with `f(I, Q) = (1 / N_q) Σ_i out_MM · q_i`, where `q_i` are frozen query-token embeddings; MM-Pred instead feeds the multimodal `[CLS]` output to a learned scalar prediction head.
- **Baselines**: compare against linear probing, prompt tuning, full fine-tuning, and late-layer fine-tuning. The FT-Late baseline trains separate unimodal transformer encoders over `49` image patches and up to `77` text tokens, with `2` self-attention layers and `4` heads.
- **Optimization**: the multimodal modules on CLIP use `lr = 1e-5` and `weight_decay = 1e-4`; FLAVA adaptations use `lr = 1e-3` and `weight_decay = 1e-4`. The appendix states that image-text matching is trained with batchwise sigmoid binary cross-entropy because multiple image-query pairs in a batch may be valid.
- **Evaluation**: single-object retrieval uses `COLA mAP` over hard distractors, with seen/unseen splits by held-out attribute-object pairs; multi-object retrieval scores whether two captions are matched to the correct two images, where random accuracy is `25%`.

## Key Results

- Benchmark scale: about `1.2k` composed queries, `168` objects, `197` attributes, and about `30k` images; human accuracy on the cleaned multi-object validation set is about `83.9%`.
- Single-object `COLA` mAP on GQA: CLIP baseline `36.53`, CLIP + MM-Adapter `46.83`, and FLAVA + MM-Adapter `48.54`.
- Single-object `COLA` mAP on CLEVR: CLIP baseline `15.38`, CLIP + MM-Adapter `88.21`, and FLAVA + MM-Adapter `91.10`.
- Single-object `COLA` mAP on PACO: CLIP baseline `12.21`, CLIP + MM-Adapter `18.56`, and FLAVA + MM-Adapter `19.36`.
- Multi-object `COLA` text-to-image accuracy: CLIP baseline is below chance at `21.42`; the best reported adaptation reaches `41.42` with CLIP + MM-Pred, while CLIP + MM-Adapter reaches `40.95`.
- On CREPE, simpler methods are already competitive: CLIP + Linear reaches `87.35` image-to-text accuracy, while CLIP + MM-Adapter reaches `87.02`, supporting the claim that `COLA` is the harder benchmark.
- Standard mAP on all images also improves sharply, for example on GQA from `0.65` (CLIP) to `17.40` (CLIP + MM-Adapter), but the paper argues that the harder distractor-only `COLA mAP` is the more informative metric.

## Limitations

- The benchmark focuses narrowly on attribute-object binding and does not directly test broader compositional structures such as relationships, scene graphs, counting, or long-horizon multimodal reasoning.
- The analysis is centered on CLIP and FLAVA-era models; the paper explicitly notes that newer vision-language models should be re-evaluated on `COLA`.
- Some benchmark cases remain ambiguous even after human cleaning, especially around color under lighting changes, subjective size judgments, and spatial descriptions.
- Adaptation may trade off generic capabilities of the original model: the paper reports zero-shot CIFAR10 dropping from `87%` for pretrained CLIP to `83%` for the adapted model before quick task-specific recovery.
- Real-world datasets such as GQA have incomplete annotations, which can inject noise into standard retrieval metrics and complicate interpretation.

## Concepts Extracted

- [[attribute-binding]]
- [[compositional-reasoning]]
- [[compositional-generalization]]
- [[text-to-image-retrieval]]
- [[multimodal-retrieval]]
- [[vision-language-model]]
- [[cross-modal-alignment]]
- [[contrastive-learning]]
- [[multimodal-adaptation]]
- [[multimodal-fusion]]

## Entities Extracted

- [[arijit-ray]]
- [[filip-radenovic]]
- [[abhimanyu-dubey]]
- [[bryan-a-plummer]]
- [[ranjay-krishna]]
- [[kate-saenko]]
- [[clip]]
- [[flava]]
- [[crepe]]
- [[gqa]]
- [[clevr]]
- [[paco]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
