---
type: source
subtype: paper
title: Hyper-Connections
slug: zhu-2025-hyperconnections-2409-19606
date: 2026-04-20
language: en
tags: [neural-architecture, residual-connections, language-model-pretraining, transformer, training-stability]
processed: true
raw_file: raw/papers/zhu-2025-hyperconnections-2409-19606/paper.pdf
raw_md: raw/papers/zhu-2025-hyperconnections-2409-19606/paper.md
bibtex_file: raw/papers/zhu-2025-hyperconnections-2409-19606/paper.bib
possibly_outdated: false
authors:
  - Defa Zhu
  - Hongzhi Huang
  - Zihao Huang
  - Yutao Zeng
  - Yunyao Mao
  - Banggu Wu
  - Qiyang Min
  - Xun Zhou
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: "2409.19606"
doi: 10.48550/arXiv.2409.19606
url: http://arxiv.org/abs/2409.19606
citation_key: zhu2025hyperconnections
paper_type: method
read_status: unread
domain: llm
---

## Summary

Hyper-connections (HC) is a drop-in replacement for residual connections in Transformers, addressing the well-known seesaw effect between gradient vanishing (afflicting Post-Norm) and representation collapse (afflicting Pre-Norm). The core idea is to replace the scalar residual weight with a learnable matrix `HC ∈ R^{(n+1)×(n+1)}` that encodes both depth-connections (how each hidden copy blends the layer output back in) and width-connections (information exchange among `n` parallel hidden vectors). A dynamic variant (DHC) further conditions these weights on the input via a lightweight linear + tanh transform. Evaluated on OLMo-1B, OLMo-7B, and OLMoE-1B-7B pre-training, DHC with expansion rate `n=4` consistently outperforms Pre-Norm residuals in loss, downstream accuracy, and training stability, with overhead under 0.2% in FLOPs and 0.03% in parameters.

## Problem & Motivation

Residual connections are ubiquitous in deep networks but lock in a fixed connection strength. Pre-Norm addresses gradient vanishing but causes representation collapse (adjacent layers become highly similar, reducing effective depth). Post-Norm reduces collapse but reintroduces vanishing gradients. These two failure modes are like two ends of a seesaw — choosing either variant forces a trade-off. The paper asks whether a network can autonomously learn the optimal connection strengths rather than relying on a manually chosen norm placement.

## Method

**Static Hyper-Connections (SHC):**
- The initial hidden vector `h^0 ∈ R^d` is replicated `n` times to form a hyper hidden matrix `H^0 ∈ R^{n×d}`.
- Each layer operates on `H^{k-1}` via an HC matrix `HC ∈ R^{(n+1)×(n+1)}`:
  ```
  HC = [[0, B],
        [A_m, A_r]]
  ```
  where `B ∈ R^{1×n}` weights the layer output into each of the `n` hidden copies; `A_m ∈ R^{n×1}` performs a weighted sum of the `n` hidden copies to form the single layer input; `A_r ∈ R^{n×n}` performs width-mixing among hidden copies.
- Forward pass: `h_0 = A_m^T H` (mix copies → single input); run layer `T(h_0)`; update `H' = B^T T(h_0) + A_r^T H`.
- Initialization mirrors Pre-Norm: `W_β, W_m, W_r = 0`; static matrices set so that DHC collapses to Pre-Norm at step 0 (Eq. 14).
- Parameter cost of SHC per layer module: `|θ_SHC| = n·(n+2)` scalars. For OLMo-1B-SHC×4: 768 extra parameters total.

**Dynamic Hyper-Connections (DHC):**
- HC weights become input-dependent via a small network: `norm(H) → linear → tanh → scaled offset on B, A_m, A_r`.
- Small learnable scaling factor (`s_β`, `s_α`) initialized near zero for training stability.
- Parameter cost of DHC: `|θ_DHC| = |θ_norm| + d_model·(n+2) + n·(n+2) + 2` per layer module. For OLMo-1B-DHC×4: ~394K extra parameters (0.033% overhead).
- FLOPs overhead at `n=4`: `+0.200%` for 1B models; `+0.147%` for 7B models.
- Memory overhead at `n=4`: `+26.1%` activation memory during training (can be reduced with recomputation).

**Sequential-parallel duality:** By analyzing the learned HC matrix, the authors show that specific HC configurations reproduce both sequential (standard residual) and parallel transformer block (PTB) arrangements. DHC learns a soft, per-token mixture of both.

**Pre-Norm and Post-Norm as degenerate HC:** Both are expressible as fixed `n=1` HC matrices (Eq. 15–16), confirming HC strictly generalizes residual connections.

## Key Results

- **OLMo-1B (1B dense, 500B tokens):** DHC×4 reduces V2 Eval Loss by `0.030` (2.781 vs 2.811) and V3 Eval Loss by `0.029` (2.515 vs 2.544); downstream avg accuracy `63.8` vs `62.5`.
- **OLMo-1B ablation:** `n=1` DHC underperforms baseline; `n=4` is the sweet spot; `n=8` offers marginal gains. Width-connections (`WC`) are the critical component — removing them degrades V2 loss by `0.021`.
- **OLMo-7B (7B dense, 500B tokens):** DHC×4 improves V2 loss by `0.022` (2.559 vs 2.581) and downstream avg accuracy `71.0` vs `70.1`. No training spikes observed, while baseline exhibits frequent spikes.
- **OLMoE-1B-7B (MoE, 500B tokens):** DHC×4 converges `1.8×` faster; improves ARC-Challenge by `+6` points (47.8 vs 41.8); MMLU Var `+1.2` (39.7 vs 38.5); BoolQ `+3.1` (68.5 vs 65.4).
- **Comparison vs. related methods:** AltUp×2 and ResiDual both fall below the Pre-Norm baseline at 500B tokens; DHC×2 outperforms both.
- **Representation collapse:** Pre-Norm OLMo-1B shows high cosine similarity between adjacent layers (near 1 in deeper layers); DHC models show significantly lower median similarity and wider spread, indicating richer layer-by-layer transformations.
- **Learned connection pattern:** OLMo-1B-DHC×4 exhibits a Λ-shaped dense connection pattern — short-range Post-Norm-style decay combined with Pre-Norm-style access to early layers. PTB-like parallel patterns also emerge spontaneously (e.g., layers 11 and 12).

## Limitations

- Memory overhead during training is non-trivial (+26% activation memory at `n=4`), though this can be mitigated with activation recomputation, reducing the persistent cost to `n·s·b·d_model`.
- The paper evaluates only up to 7B dense and 7B MoE models; scaling behavior beyond 7B is uncharacterized.
- Vision experiments (DiT, ViT on ImageNet) are relegated to the appendix and discussed less thoroughly.
- Dynamic HC requires the input-dependent computation of `A_m, A_r, B` at every layer; the overhead is small but non-zero and may interact with structured pruning or quantization techniques not studied here.
- The `n=1` expansion rate consistently underperforms the baseline, meaning HC provides no benefit in its minimal form — the width-connections require `n≥2`.

## Concepts Extracted

- [[hyper-connections]]
- [[residual-connection]]
- [[pre-norm]]
- [[post-norm]]
- [[representation-collapse]]
- [[gradient-vanishing]]
- [[parallel-transformer-block]]
- [[mixture-of-experts]]
- [[layer-normalization]]
- [[large-language-model]]
- [[transformer]]

## Entities Extracted

- [[defa-zhu]]
- [[hongzhi-huang]]
- [[zihao-huang]]
- [[yutao-zeng]]
- [[yunyao-mao]]
- [[banggu-wu]]
- [[qiyang-min]]
- [[xun-zhou]]
- [[bytedance]]
- [[olmoe-1b-7b]]

## Contradictions

<!-- None yet; first source on hyper-connections in this wiki. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
