---
type: source
subtype: paper
title: "Chatting Makes Perfect: Chat-based Image Retrieval"
slug: levy-nd-chatting
date: 2026-04-20
language: en
tags: [image-retrieval, conversational-search, multimodal, vision-language, llm]
processed: true
raw_file: raw/papers/levy-nd-chatting/paper.pdf
raw_md: raw/papers/levy-nd-chatting/paper.md
bibtex_file: raw/papers/levy-nd-chatting/paper.bib
possibly_outdated: true
authors:
  - Matan Levy
  - Rami Ben-Ari
  - Nir Darshan
  - Dani Lischinski
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id: 2305.20062
doi:
url: https://arxiv.org/abs/2305.20062
citation_key: levyndchatting
paper_type: method
read_status: unread
domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces ChatIR, a chat-based image retrieval framework that turns single-shot text-to-image search into a multi-round dialog between a question generator and a user answerer. The system combines an LLM questioner `G` that asks follow-up questions from dialog history `D_i = (C, Q_1, A_1, ..., Q_i, A_i)` with a BLIP-based retriever `F` that maps the full dialog into a shared image-text embedding space for cosine ranking over a `50K` image corpus. Using VisDial as supervision and BLIP2 as a scalable answer simulator, ChatIR reaches `78.3%` Hit@10 after `5` rounds and `81.3%` after `10`, versus `64%` for single-shot retrieval, showing that dialog can materially clarify user intent in image search.

## Problem & Motivation

Existing image retrieval pipelines mainly assume a one-shot query, such as text-to-image retrieval or composed image retrieval, even though many user intents are underspecified in a single caption. The paper argues that retrieval systems should proactively acquire missing evidence instead of forcing users to repeatedly rewrite standalone queries. ChatIR is motivated by this gap: it treats retrieval as an interactive process where the system asks targeted questions, accumulates answers, and progressively sharpens the ranking of the target image.

## Method

- **Task formulation**: the dialog state at round `i` is `D_i = (C, Q_1, A_1, ..., Q_i, A_i)`, where `C` is the initial caption. The question generator predicts `G: D_i -> Q_{i+1}` and the retriever predicts `F: D_i -> R^d`.
- **Two-stage pipeline**: ChatIR alternates between Dialog Building and Image Search. Dialog Building uses an instructional LLM to ask the next question without access to the target image, while Image Search embeds the full dialog and ranks all corpus images by cosine similarity.
- **Retriever architecture**: `F` is built from BLIP text/image encoders fine-tuned for dialog-based retrieval. Dialog components are concatenated with `[SEP]` delimiters plus a `[CLS]` token whose final representation is projected into the shared visual embedding space.
- **Embedding setup**: each candidate image is pre-encoded into a feature vector `f in R^d`; retrieval ranks images by cosine similarity between `F(D_i)` and `f`. The image embedder is frozen during training.
- **Training data**: the model repurposes VisDial dialogs as retrieval supervision by pairing each dialog with its underlying image target. Partial dialogs of varying lengths are used so the retriever can work from round `0` onward.
- **Optimization**: `F` is trained with AdamW, initial learning rate `5e-5`, exponential decay `0.93` down to `1e-6`, batch size `512`, and `36` epochs. The objective is a differentiable Recall@K surrogate loss rather than plain contrastive cross-entropy.
- **Evaluation loop**: large-scale evaluation substitutes human answers with BLIP2, keeping the answerer fixed while comparing different questioners. Retrieval is counted as success once the target enters top-`10`.
- **Best training variant**: in ablations, masking `20%` of captions during training gives the strongest retriever, suggesting the model learns to rely more on multi-turn evidence instead of overfitting to the initial description.

## Key Results

- On VisDial with a `50K` image corpus, ChatIR reaches `78.3%` Hit@10 after `5` rounds and `81.3%` after `10`, versus `64%` for single-shot text-to-image retrieval.
- The best questioner already achieves `73.5%` Hit@10 after only `2` rounds on `50K` unseen images, about `10` points above the round-`0` baseline (`~63%`).
- In zero-shot transfer to COCO, the dialogue-trained retriever improves single-hop accuracy from `83%` to `87%` and exceeds `95%` after `10` dialog rounds.
- On VisDial, a caption-only fine-tuned BLIP baseline and ChatIR are nearly tied at round `0` (`63.66%` vs. `63.61%`), showing the gain comes from subsequent dialog rather than a stronger initial caption encoder.
- In the real human-in-the-loop setup, `ChatGPT^Q + Human^A` reaches `81%` Hit@10 after `5` rounds; fully human dialogs still outperform both ChatGPT variants beyond `5` rounds.
- Caption masking improves retrieval by roughly `2-3%` relative to training without masking, regardless of whether the questioner is ChatGPT or human.

## Limitations

- The training and evaluation data come from repurposed VisDial conversations rather than dialogs originally collected for retrieval, so user intents and question distributions may not fully match real search behavior.
- Large-scale evaluation depends on BLIP2 as a proxy answerer; the paper finds a small but real domain gap relative to human answers.
- The question generator only conditions on dialog history, not on the current retrieved candidates, which limits its ability to ask maximally discriminative follow-up questions.
- Benefits saturate in later rounds, and weaker questioners often repeat questions or fail to monotonically improve target rank.
- Experiments focus on image retrieval over moderate-scale corpora (`50K` images) and do not test broader multimodal or web-scale search settings.

## Concepts Extracted

- [[conversational-search]]
- [[image-retrieval]]
- [[interactive-retrieval]]
- [[multimodal-retrieval]]
- [[text-to-image-retrieval]]
- [[visual-dialog]]
- [[question-generation]]
- [[contrastive-learning]]
- [[cosine-similarity]]
- [[large-language-model]]

## Entities Extracted

- [[matan-levy]]
- [[rami-ben-ari]]
- [[nir-darshan]]
- [[dani-lischinski]]
- [[hebrew-university-of-jerusalem]]
- [[originai]]
- [[blip]]
- [[blip-2]]
- [[visdial]]
- [[chatgpt]]
- [[clip]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
