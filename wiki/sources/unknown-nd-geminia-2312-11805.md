---
type: source
subtype: paper
title: "Gemini: A Family of Highly Capable Multimodal Models"
slug: unknown-nd-geminia-2312-11805
date: 2026-04-20
language: en
tags: [gemini, multimodal, llm, reasoning, evaluation]
processed: true
raw_file: raw/papers/unknown-nd-geminia-2312-11805/paper.pdf
raw_md: raw/papers/unknown-nd-geminia-2312-11805/paper.md
bibtex_file: raw/papers/unknown-nd-geminia-2312-11805/paper.bib
possibly_outdated: true
authors:
  - Gemini Team
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2312.11805
doi:
url: https://arxiv.org/abs/2312.11805
citation_key: unknownndgeminia
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This report presents Gemini 1.0, a multimodal large language model family jointly trained on text, image, audio, and video and then post-trained into app-facing and API-facing variants. Architecturally, Gemini is a decoder-only Transformer with `32k` context and efficient attention such as multi-query attention, while the Nano line is distilled into `1.8B` and `3.25B` on-device models and quantized to `4-bit`. The report combines system, model, and evaluation perspectives: it describes TPUv4/TPUv5e training infrastructure, multimodal multilingual pre-training data curation, post-training with SFT and RLHF, and broad benchmarking. Gemini Ultra is reported as state of the art on 30 of 32 benchmarks, surpassing human-expert MMLU performance and improving multimodal reasoning, video understanding, speech tasks, and tool-augmented reasoning.

## Problem & Motivation

The paper targets a central 2023 question for frontier foundation models: whether one jointly trained model can achieve strong general-purpose performance across text, code, image, audio, and video rather than relying on modality-specific systems stitched together downstream. The authors also aim to cover multiple deployment regimes, from frontier reasoning workloads to latency- and memory-constrained on-device use cases.

Beyond raw capability, the report frames Gemini as a full-stack effort. It couples multimodal pre-training with post-training for instruction following, factuality, multilinguality, tool use, and safety, then evaluates both academic benchmarks and deployment-oriented behavior. The motivation is therefore not just higher benchmark scores, but a scalable family of models that can reason across modalities, support products and APIs, and remain operationally deployable on Google's infrastructure.

## Method

- **Base architecture**: Gemini is built on a decoder-only [[transformer]] with `L = 32768` context length and efficient attention mechanisms including [[multi-query-attention]]; the model is trained jointly across text, image, audio, and video from the start rather than attaching separate modality modules only at inference time.
- **Multimodal tokenization and encoding**: textual input is tokenized with SentencePiece trained on a large sample of the full corpus; visual input includes natural images, charts, screenshots, PDFs, and video frames; audio is ingested through Universal Speech Model features sampled at `f_audio = 16 kHz`.
- **Model family design**: Gemini 1.0 is exposed as Ultra, Pro, and Nano variants. Nano has two small-device instantiations with `N_nano1 = 1.8B` and `N_nano2 = 3.25B` parameters, distilled from larger Gemini models and quantized to `4-bit` for deployment.
- **Training infrastructure**: large-scale runs use TPUv4 and TPUv5e. Gemini Ultra training spans multiple datacenters, combining model parallelism within superpods and data parallelism across superpods; TPUv4 superpods contain `4096` chips and can reconfigure torus topologies in about `10 s`.
- **Reliability engineering**: instead of conventional checkpoint-heavy recovery, training uses redundant in-memory model-state replicas; with deterministic replay and hardware diagnostics, reported overall goodput increases from `85%` to `97%` at the largest scale.
- **Pre-training data mixture**: the corpus is multilingual and multimodal, built from web documents, books, code, images, audio, and video. The pipeline applies heuristic and model-based quality filters, safety filtering, decontamination against eval sets, and staged mixture reweighting toward domain-relevant data later in training.
- **Post-training stack**: after pre-training, Gemini undergoes [[post-training]] via [[supervised-fine-tuning]] and [[reinforcement-learning-from-human-feedback]] to improve instruction following, coding, factuality, multilinguality, vision behavior, and safety. The paper distinguishes Gemini Apps models from Gemini API models because downstream objectives differ.
- **Inference and prompting recipes**: several reported text benchmarks use explicit reasoning recipes rather than plain greedy decoding. For MMLU, the model samples chain-of-thought candidates with `k = 8` or `32`, accepts a consensus answer above a threshold, and otherwise falls back to greedy decoding; GSM8K additionally uses [[self-consistency]].
- **Tool-use formulation**: API and app variants model tool use as code generation, where tool invocations are emitted inside executable code blocks and interleaved with tool execution results in a control loop; this lets the model compose multiple tools and condition on returned evidence.

## Key Results

- Gemini Ultra is reported to achieve state of the art on `30 / 32` benchmarks in the report.
- On MMLU, Gemini Ultra reaches `90.04%`, exceeding the human-expert reference of `89.8%` and the cited prior SOTA `86.4%`.
- On GSM8K, Gemini Ultra reaches `94.4%`, above the cited previous best `92.0%` with comparable prompting.
- On MATH, Gemini Ultra scores `53.2%` with `4-shot` prompting; on harder AMC-derived problems it solves `32%` versus `30%` for GPT-4.
- On HumanEval and Natural2Code, Gemini Ultra scores `74.4%` and `74.9%`, respectively.
- On WMT23 average BLEURT, Gemini Ultra scores `74.4`, compared with GPT-4 `73.8` and PaLM 2-L `72.7`; on very low-resource translation it reaches average `chrF = 27.0` versus `25.3` for PaLM 2-L.
- On multilingual MGSM, Gemini Ultra reaches `79.0%` versus `74.7%` for PaLM 2-L.
- On the synthetic long-context retrieval test, Gemini Ultra achieves `98%` accuracy across the full `32k` window.
- On MMMU, Gemini Ultra reaches `62.4%` (`Maj@32`) and `59.4%` (`pass@1`), improving over GPT-4V `56.8%`.
- On image understanding, Gemini Ultra reports `90.9%` on DocVQA, `80.8%` on ChartQA, `80.3%` on InfographicVQA, `53.0%` on MathVista, and `79.5%` on AI2D.
- On video understanding, Gemini Ultra reports `62.7` CIDER on VATEX, `135.4` on YouCook2, `29.9` on NextQA, `52.2` on ActivityNet-QA, and `54.7%` on Perception Test MCQA.
- On speech tasks, Gemini Pro reports `4.9%` WER on YouTube English, `4.8%` on Multilingual LibriSpeech, `7.6%` on FLEURS, `9.1%` on VoxPopuli, and `40.1` BLEU on CoVoST 2.
- In tool-use evaluation, Gemini API Pro with tools improves over the without-tools counterpart from `69.7%` to `80.1%` on GSM8K, `30.7%` to `41.8%` on MATH, `59.0%` to `68.0%` on NQ, and `39.2%` to `70.8%` on realtime QA.

## Limitations

- The report withholds key scaling details such as Ultra/Pro parameter counts, exact training-token counts, and detailed data-mixture weights, which limits reproducibility and makes compute-efficiency comparisons incomplete.
- Several results depend on specialized prompting or post-training recipes, so some numbers are not pure measurements of the base pre-trained model.
- The authors explicitly acknowledge evaluation fragility and contamination risk, noting leaked-data analysis issues and withholding some benchmarks such as LAMBADA.
- Much of the deployment, safety, and product evaluation relies on internal benchmarks, internal raters, or product-specific settings that are difficult to independently audit.
- Despite broad multimodal coverage, the report still exposes uneven strengths: some tasks remain behind prior fine-tuned specialist systems, and full-response instruction-following accuracy remains only `54.1%` for Gemini Advanced in the cited internal benchmark.

## Concepts Extracted

- [[transformer]]
- [[decoder-only-transformer]]
- [[multi-modal-learning]]
- [[multimodal-reasoning]]
- [[multi-query-attention]]
- [[context-window]]
- [[chain-of-thought-prompting]]
- [[self-consistency]]
- [[quantization]]
- [[post-training]]
- [[instruction-following]]
- [[supervised-fine-tuning]]
- [[reinforcement-learning-from-human-feedback]]
- [[data-filtering]]

## Entities Extracted

- [[gemini-team]]
- [[google]]
- [[gemini-ultra]]
- [[gemini-pro]]
- [[gemini-nano]]
- [[mmlu]]
- [[mmmu]]
- [[alphacode-2]]
- [[google-ai-studio]]
- [[cloud-vertex-ai]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
