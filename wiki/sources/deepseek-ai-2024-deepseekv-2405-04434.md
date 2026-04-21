---
type: source
subtype: paper
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
slug: deepseek-ai-2024-deepseekv-2405-04434
date: 2026-04-20
language: en
tags: [llm, moe, efficiency, long-context, alignment]
processed: true
raw_file: raw/papers/deepseek-ai-2024-deepseekv-2405-04434/paper.pdf
raw_md: raw/papers/deepseek-ai-2024-deepseekv-2405-04434/paper.md
bibtex_file: raw/papers/deepseek-ai-2024-deepseekv-2405-04434/paper.bib
possibly_outdated: false
authors:
  - DeepSeek-AI
  - Aixin Liu
  - Bei Feng
  - Bin Wang
  - Bingxuan Wang
  - Bo Liu
  - Chenggang Zhao
  - Chengqi Dengr
  - Chong Ruan
  - Damai Dai
  - Daya Guo
  - Dejian Yang
  - Deli Chen
  - Dongjie Ji
  - Erhang Li
  - Fangyun Lin
  - Fuli Luo
  - Guangbo Hao
  - Guanting Chen
  - Guowei Li
  - H. Zhang
  - Hanwei Xu
  - Hao Yang
  - Haowei Zhang
  - Honghui Ding
  - Huajian Xin
  - Huazuo Gao
  - Hui Li
  - Hui Qu
  - J. L. Cai
  - Jian Liang
  - Jianzhong Guo
  - Jiaqi Ni
  - Jiashi Li
  - Jin Chen
  - Jingyang Yuan
  - Junjie Qiu
  - Junxiao Song
  - Kai Dong
  - Kaige Gao
  - Kang Guan
  - Lean Wang
  - Lecong Zhang
  - Lei Xu
  - Leyi Xia
  - Liang Zhao
  - Liyue Zhang
  - Meng Li
  - Miaojun Wang
  - Mingchuan Zhang
  - Minghua Zhang
  - Minghui Tang
  - Mingming Li
  - Ning Tian
  - Panpan Huang
  - Peiyi Wang
  - Peng Zhang
  - Qihao Zhu
  - Qinyu Chen
  - Qiushi Du
  - R. J. Chen
  - R. L. Jin
  - Ruiqi Ge
  - Ruizhe Pan
  - Runxin Xu
  - Ruyi Chen
  - S. S. Li
  - Shanghao Lu
  - Shangyan Zhou
  - Shanhuang Chen
  - Shaoqing Wu
  - Shengfeng Ye
  - Shirong Ma
  - Shiyu Wang
  - Shuang Zhou
  - Shuiping Yu
  - Shunfeng Zhou
  - Size Zheng
  - T. Wang
  - Tian Pei
  - Tian Yuan
  - Tianyu Sun
  - W. L. Xiao
  - Wangding Zeng
  - Wei An
  - Wen Liu
  - Wenfeng Liang
  - Wenjun Gao
  - Wentao Zhang
  - X. Q. Li
  - Xiangyue Jin
  - Xianzu Wang
  - Xiao Bi
  - Xiaodong Liu
  - Xiaohan Wang
  - Xiaojin Shen
  - Xiaokang Chen
  - Xiaosha Chen
  - Xiaotao Nie
  - Xiaowen Sun
  - Xiaoxiang Wang
  - Xin Liu
  - Xin Xie
  - Xingkai Yu
  - Xinnan Song
  - Xinyi Zhou
  - Xinyu Yang
  - Xuan Lu
  - Xuecheng Su
  - Y. Wu
  - Y. K. Li
  - Y. X. Wei
  - Y. X. Zhu
  - Yanhong Xu
  - Yanping Huang
  - Yao Li
  - Yao Zhao
  - Yaofeng Sun
  - Yaohui Li
  - Yaohui Wang
  - Yi Zheng
  - Yichao Zhang
  - Yiliang Xiong
  - Yilong Zhao
  - Ying He
  - Ying Tang
  - Yishi Piao
  - Yixin Dong
  - Yixuan Tan
  - Yiyuan Liu
  - Yongji Wang
  - Yongqiang Guo
  - Yuchen Zhu
  - Yuduan Wang
  - Yuheng Zou
  - Yukun Zha
  - Yunxian Ma
  - Yuting Yan
  - Yuxiang You
  - Yuxuan Liu
  - Z. Z. Ren
  - Zehui Ren
  - Zhangli Sha
  - Zhe Fu
  - Zhen Huang
  - Zhen Zhang
  - Zhenda Xie
  - Zhewen Hao
  - Zhihong Shao
  - Zhiniu Wen
  - Zhipeng Xu
  - Zhongyu Zhang
  - Zhuoshu Li
  - Zihan Wang
  - Zihui Gu
  - Zilin Li
  - Ziwei Xie
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.04434
doi:
url: https://arxiv.org/abs/2405.04434
citation_key: deepseekai2024deepseekv
paper_type: method
read_status: unread
domain: llm
---

## Summary

DeepSeek-V2 presents a bilingual open-source Mixture-of-Experts language model that aims to improve the capability-to-cost tradeoff of large-scale LLM deployment. The model has `236B` total parameters but activates only `21B` per token, extends context from `4K` to `128K`, and combines two central ideas: Multi-Head Latent Attention (MLA), which compresses the attention KV state into a low-rank latent cache plus decoupled RoPE carriers, and DeepSeekMoE, which uses fine-grained routed experts, shared experts, device-limited routing, and explicit load-balance losses. After pretraining on `8.1T` tokens and applying supervised fine-tuning plus GRPO-based reinforcement learning, the paper reports top-tier open-source performance. Relative to DeepSeek 67B, it cuts training cost by `42.5%`, reduces KV cache by `93.3%`, and raises maximum generation throughput by `5.76x`.

## Problem & Motivation

The paper targets a core tension in frontier open-source LLM design: stronger dense models usually require prohibitive training compute and incur heavy inference-time memory costs, especially from the autoregressive key-value cache in standard multi-head attention. The authors want a model that remains competitive on English, Chinese, code, and math tasks while being cheaper to train and easier to serve at large batch sizes and long context windows. Their solution is to redesign both the attention path and the sparse FFN path so that deployment efficiency improves without paying the usual quality penalty of lightweight KV-cache reductions such as MQA or aggressively sparse routing without load control.

## Method

- **Backbone configuration**: DeepSeek-V2 keeps a Transformer backbone but scales it to `60` layers with hidden size `5120`, `236B` total parameters, and `21B` activated parameters per token.
- **Multi-Head Latent Attention (MLA)**: standard attention is replaced by low-rank latent compression with `c_t^KV = W^DKV h_t`, `k_t^C = W^UK c_t^KV`, and `v_t^C = W^UV c_t^KV`; queries are also compressed via `c_t^Q = W^DQ h_t`, `q_t^C = W^UQ c_t^Q`. The paper sets `n_h = 128`, `d_h = 128`, `d_c = 512`, and `d'_c = 1536`.
- **Decoupled RoPE for MLA**: to keep RoPE compatible with KV compression, the model carries positional information through separate terms `q_t^R = RoPE(W^QR c_t^Q)` and `k_t^R = RoPE(W^KR h_t)`, concatenated with the compressed content vectors. This yields cache size `(d_c + d_h^R) l` with `d_h^R = 64`, instead of `2 n_h d_h l` for MHA.
- **DeepSeekMoE FFN stack**: all FFNs except the first layer become MoE blocks with `h'_t = u_t + sum FFN_i^(s)(u_t) + sum g_{i,t} FFN_i^(r)(u_t)`. Each MoE layer uses `2` shared experts, `160` routed experts, routed expert width `1536`, and activates `K_r = 6` routed experts per token.
- **Routing and balance control**: expert parallelism places routed experts across `D = 8` devices, while device-limited routing constrains each token to at most `M = 3` devices. Training adds expert-, device-, and communication-balance losses with `alpha_1 = 0.003`, `alpha_2 = 0.05`, and `alpha_3 = 0.02`, plus device-level token dropping under capacity factor `1.0`.
- **Pretraining recipe**: the model uses a `100K` byte-level BPE tokenizer and an `8.1T` token corpus with Chinese tokens roughly `12%` more numerous than English tokens. Optimization uses `AdamW(beta_1 = 0.9, beta_2 = 0.95, weight_decay = 0.1)`, `lr_max = 2.4e-4`, `2K` warmup steps, gradient clip `1.0`, batch size ramp `2304 -> 9216`, and maximum sequence length `4K`.
- **Systems stack**: training runs on NVIDIA H800 clusters with `16`-way zero-bubble pipeline parallelism, `8`-way expert parallelism, ZeRO-1 data parallelism, overlapped expert communication, custom CUDA kernels, and an improved FlashAttention-2 implementation for MLA.
- **Long-context extension**: after base pretraining, YaRN extends the window from `4K` to `128K` by adapting the RoPE-carrying shared key with `s = 40`, `alpha = 1`, `beta = 32`, target length `160K`, and an additional `1000` training steps at `32K` sequence length.
- **Alignment pipeline**: instruction tuning uses `1.5M` conversations (`1.2M` helpfulness, `0.3M` safety), `2` epochs, and `lr = 5e-6`. Reinforcement learning then applies two-stage GRPO: a reasoning stage with code/math rewards and a preference stage combining helpful, safety, and rule-based reward models.

## Key Results

- **Base model quality**: with only `21B` activated parameters, DeepSeek-V2 reaches `78.5` MMLU, `43.6` MATH, `79.2` GSM8K, `48.8` HumanEval, `84.0` CMMLU, `92.7` CHID, and `93.1` CCPM.
- **Cost efficiency**: training cost per trillion tokens drops from `300.6K` GPU-hours for DeepSeek 67B to `172.8K` GPU-hours for DeepSeek-V2, a `42.5%` reduction.
- **Serving efficiency**: after FP8 conversion and `6`-bit-average KV-cache quantization, deployed DeepSeek-V2 exceeds `50K` generated tokens/s and `100K` prompt tokens/s on one `8 x H800` node, with `5.76x` higher maximum generation throughput than DeepSeek 67B.
- **Cache reduction**: MLA plus deployment optimizations reduce the KV cache requirement by `93.3%` relative to DeepSeek 67B while maintaining stronger benchmark performance.
- **Chat model gains**: DeepSeek-V2 Chat (RL) reaches `81.1` HumanEval, `72.0` MBPP, `92.2` GSM8K, `53.9` MATH, `8.97` MT-Bench, `38.9` AlpacaEval 2.0 length-controlled win rate, and `7.91` AlignBench overall score.
- **Long context**: the YaRN-extended model is reported to behave robustly up to `128K` context in Needle-In-A-Haystack evaluation.

## Limitations

- The model is text-only and primarily trained on Chinese and English, so the paper does not establish multimodal capability or robust performance in other languages.
- The long-context claim is supported mainly by YaRN adaptation and Needle-In-A-Haystack tests; it does not provide a broad evaluation of `128K` reasoning, retrieval, or tool-use workloads.
- The `8.1T` corpus composition and many systems optimizations are only described at a high level, limiting reproducibility for outside labs.
- Some comparisons rely on the authors' internal evaluation framework, so exact cross-paper comparability may be imperfect even when benchmark names match.
- RL improves open-ended alignment but introduces measurable alignment tax on some standard benchmarks, e.g. BBH drops from `81.3` to `79.7`, MMLU from `78.4` to `77.8`, and IFEval from `64.1` to `63.8`.
- Like other LLMs, the model still lacks post-training knowledge updates and may hallucinate or provide non-factual advice.

## Concepts Extracted

- [[large-language-model]]
- [[transformer]]
- [[mixture-of-experts]]
- [[multi-head-latent-attention]]
- [[grouped-query-attention]]
- [[multi-query-attention]]
- [[rotary-positional-embedding]]
- [[key-value-cache]]
- [[expert-parallelism]]
- [[byte-pair-encoding]]
- [[supervised-fine-tuning]]
- [[group-relative-policy-optimization]]
- [[long-context-training]]

## Entities Extracted

- [[deepseek-ai]]
- [[deepseek-v2]]
- [[deepseek-67b]]
- [[deepseekmoe]]
- [[hai-llm]]
- [[nvidia-h800]]
- [[flash-attention]]
- [[yarn]]
- [[vllm]]
- [[qwen1-5-72b]]
- [[llama-3-70b]]
- [[mixtral-8x22b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
