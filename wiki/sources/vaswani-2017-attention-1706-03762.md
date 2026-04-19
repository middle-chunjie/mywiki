---
type: source
subtype: paper
title: Attention Is All You Need
slug: vaswani-2017-attention-1706-03762
date: 2026-04-17
language: en
tags: [transformer, attention, nmt, seq2seq, nlp]
processed: true

raw_file: raw/papers/vaswani-2017-attention-1706-03762/paper.pdf
raw_md: raw/papers/vaswani-2017-attention-1706-03762/paper.md
bibtex_file: raw/papers/vaswani-2017-attention-1706-03762/paper.bib
raw_sha256: bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697
raw_md_sha256: 14ba86b34b276e759ba21621715e320a1edbbbc057db0577b9f224acb5610059
last_verified: 2026-04-17
possibly_outdated: true

authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Lukasz Kaiser
  - Illia Polosukhin
year: 2017
venue: NeurIPS 2017
venue_type: conference
arxiv_id: 1706.03762
doi:
url: https://arxiv.org/abs/1706.03762
citation_key: vaswani2017attention
paper_type: method

read_status: unread
read_date:
rating:

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2017; re-verify against recent literature.

The paper introduces the Transformer, a sequence transduction architecture that replaces recurrence and convolution with stacked self-attention and position-wise feed-forward layers. Queries, keys, and values are projected in parallel through `h = 8` heads of Scaled Dot-Product Attention, with the dot products divided by `√d_k` to control softmax saturation. Positions are injected through fixed sinusoidal encodings added to the embeddings. The model reaches 28.4 BLEU on WMT 2014 English–German and 41.8 BLEU on WMT 2014 English–French, surpassing previous state-of-the-art ensembles at roughly one quarter of their training cost, and generalizes to English constituency parsing (F1 91.3 / 92.7). The work is the architectural foundation for virtually all subsequent large language models.

## Problem & Motivation

Sequence transduction at the time was dominated by RNN/LSTM encoder-decoder models whose sequential computation precluded within-example parallelism and made long-range dependencies hard to learn. Convolutional alternatives (ByteNet, ConvS2S) improved parallelism but still grew the path length between distant positions linearly or logarithmically with sequence length. The authors aimed to keep the path length between any two positions constant while maximizing parallelism, trusting attention alone to model global dependencies.

## Method

- **Encoder**: stack of `N = 6` identical layers; each layer has multi-head self-attention followed by a position-wise feed-forward network. Both sub-layers wrapped by residual connection + layer normalization, yielding `LayerNorm(x + Sublayer(x))`. Output dimension `d_model = 512`.
- **Decoder**: also 6 layers; each layer adds a third sub-layer for multi-head attention over encoder outputs. Self-attention in the decoder is masked (future positions set to `-∞` before softmax) to preserve auto-regression.
- **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`. The `1/√d_k` rescaling prevents softmax saturation for large `d_k`.
- **Multi-Head Attention**: `h = 8` parallel heads with `d_k = d_v = d_model / h = 64`; outputs concatenated and projected by `Wᴼ`. Three uses: encoder self-attention, decoder masked self-attention, and encoder–decoder cross-attention.
- **Position-wise FFN**: two linear layers with ReLU, `d_ff = 2048`; equivalent to two kernel-1 convolutions.
- **Embeddings**: shared weight matrix between input, output and pre-softmax projections; embeddings scaled by `√d_model`.
- **Positional encoding**: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(...)`. Sinusoids were chosen over learned embeddings to allow extrapolation to longer sequences; empirical results were nearly identical.
- **Training**: Adam (β1 = 0.9, β2 = 0.98, ε = 1e-9) with learning rate schedule `d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`, `warmup_steps = 4000`.
- **Regularization**: residual dropout 0.1 (0.3 for big EN–DE), label smoothing `ε_ls = 0.1`.
- **Inference**: beam search, beam size 4, length penalty `α = 0.6`; base model averages last 5 checkpoints, big model last 20.

## Key Results

- WMT 2014 EN–DE: base 27.3 BLEU, big 28.4 BLEU (+2.0 over best prior ensemble) with training cost ~3.3·10¹⁸ / 2.3·10¹⁹ FLOPs.
- WMT 2014 EN–FR: big 41.8 BLEU single-model SOTA; training 3.5 days on 8×P100 — roughly ¼ of the previous SOTA cost.
- Ablations (Table 3): too few or too many heads hurt; smaller `d_k` hurts (compatibility function non-trivial); dropout is important; learned vs sinusoidal positional encoding yields nearly identical BLEU.
- Generalization: WSJ constituency parsing F1 91.3 (WSJ-only, 4-layer) / 92.7 (semi-supervised), competitive with specialized parsers despite minimal tuning.

## Limitations

- `O(n²·d)` self-attention cost is prohibitive for very long sequences; the paper suggests restricted attention with neighborhood `r` giving `O(n/r)` path length as future work.
- Reduced effective resolution due to attention averaging; mitigated by multi-head attention.
- Evaluated mainly on MT and a single parsing task; broader modality coverage (images, audio, video) is left as future work.
- Uses fixed sinusoidal encoding; does not address relative positional encoding, later a significant research direction.

## Concepts Extracted

- [[transformer]]
- [[self-attention]]
- [[scaled-dot-product-attention]]
- [[multi-head-attention]]
- [[positional-encoding]]
- [[encoder-decoder-architecture]]
- [[residual-connection]]
- [[layer-normalization]]
- [[position-wise-feed-forward-network]]
- [[label-smoothing]]
- [[learning-rate-warmup]]
- [[byte-pair-encoding]]
- [[beam-search]]

## Entities Extracted

- [[ashish-vaswani]]
- [[noam-shazeer]]
- [[niki-parmar]]
- [[jakob-uszkoreit]]
- [[llion-jones]]
- [[aidan-gomez]]
- [[lukasz-kaiser]]
- [[illia-polosukhin]]
- [[google-brain]]
- [[google-research]]
- [[tensor2tensor]]
- [[wmt-2014-en-de]]
- [[wmt-2014-en-fr]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
