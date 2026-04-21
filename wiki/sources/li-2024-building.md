---
type: source
subtype: paper
title: Building A Coding Assistant via the Retrieval-Augmented Language Model
slug: li-2024-building
date: 2026-04-20
language: en
tags: [code-generation, dense-retrieval, retrieval-augmented-generation, code-search, llm]
processed: true

raw_file: raw/papers/li-2024-building/paper.pdf
raw_md: raw/papers/li-2024-building/paper.md
bibtex_file: raw/papers/li-2024-building/paper.bib
possibly_outdated: false

authors:
  - Xinze Li
  - Hanbin Wang
  - Zhenghao Liu
  - Shi Yu
  - Shuo Wang
  - Yukun Yan
  - Yukai Fu
  - Yu Gu
  - Ge Yu
year: 2024
venue: ACM Transactions on Information Systems
venue_type: journal
arxiv_id:
doi: 10.1145/3695868
url: https://dl.acm.org/doi/10.1145/3695868
citation_key: li2024building
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes CONAN, a unified coding assistant built from a retrieval stack (CONAN-R) and a retrieval-augmented generator (CONAN-G). CONAN-R continuously pretrains CodeT5 with Code-Documentation Alignment and Masked Entity Prediction so that code snippets and documentation are represented in a retrieval-friendly embedding space. CONAN-G then combines retrieved code snippets and documentation with a Fusion-in-Decoder architecture and a dual-view document representation that prepends documentation as a semantic gist for code. Across code generation, summarization, completion, and code retrieval benchmarks, the system improves over prior retrieval-augmented baselines, and it can also act as a denoising assistant for code LLMs by compressing retrieved evidence into shorter, higher-quality code knowledge before generation.

## Problem & Motivation

The paper targets the knowledge-boundary problem of code language models: even strong pretrained models often fail when the required API usage, repository pattern, or task-specific knowledge is not contained in the prompt or model parameters. Prior retrieval-augmented code systems already showed that external code snippets and documentation help, but they suffer from two bottlenecks: weak retrieval quality and noisy long retrieved contexts. CONAN is motivated by the way human developers search for code examples and documentation during programming, and it tries to improve both stages jointly: better retrieval via structure-aware pretraining, and better use of retrieved evidence via FID-style multi-document generation plus documentation-guided code understanding.

## Method

- **Retrieval formulation**: encode query and document with CodeT5 as `` `h^q = CodeT5(q)` `` and `` `h^d = CodeT5(d)` ``, then score with dot product `` `f(q,d) = sim(h^q, h^d)` `` and retrieve the top-`N` documents.
- **Retriever training objective (CONAN-R)**: fine-tune dense retrieval with in-batch negatives using `` `L_CONAN-R = -log e^{f(q,d+)}/(e^{f(q,d+)} + Σ_{d-∈D-} e^{f(q,d-)})` ``.
- **Code-Documentation Alignment (CDA)**: contrastively align matched documentation and code pairs with `` `L_CDA = -log e^{f(d_doc,d_code+)}/(e^{f(d_doc,d_code+)} + Σ e^{f(d_doc,d_code-)})` `` so documentation and code occupy a shared retrieval space.
- **Masked Entity Prediction (MEP)**: identify code entities such as variables, functions, methods, and libraries, replace identical entities with identical mask tokens, and train the model to reconstruct them with `` `L_MEP = Σ_j -log P(Y_d(t_j) | X_d^mask, Y_d(t_<j))` ``. Python entities are extracted with `BytesIO`; other languages use `tree-sitter`.
- **Generator architecture (CONAN-G)**: use Fusion-in-Decoder so that generation probability is `` `P(t_j | q, D, t_<j) = FID(q, D, t_<j)` `` with per-document encoding `` `Enc(q,D) = ⊕_i CodeT5-Encoder(d^i ⊕ q)` ``.
- **Dual-view code representation**: represent each retrieved document as documentation plus code, `` `CodeT5-Encoder(d_doc^i ⊕ d_code^i ⊕ q)` ``, so documentation acts as a semantic gist that helps the model focus on relevant code structure.
- **Task usage**: for code generation, retrieved documents are represented mainly by code snippets; for code summarization and completion, retrieval uses documentation-oriented evidence. In the LLM-assistant setting, CONAN first retrieves `` `D = CONAN-R(q, D~)` ``, then summarizes it into `` `d* = CONAN-G(q, D)` ``, and finally prompts the LLM with `` `LLM(d* ⊕ q)` ``.
- **Data and retrieval corpora**: pretraining uses CodeSearchNet with `429,596` Python, `514,127` PHP, `317,824` Go, `454,433` Java, `122,682` JavaScript, and `48,790` Ruby positive pairs. Retrieval databases include up to `1.6M` Java code snippets and `1.1M` documentation entries.
- **Implementation details**: CONAN-R starts from `CodeT5-base`; structure-aware pretraining uses learning rate `` `1e-4` `` for `10` epochs. Retrieval fine-tuning uses batch size `` `128` `` for `12` epochs, with learning rates `` `2e-5` `` or `` `1e-5` `` depending on dataset and one hard negative from top-100 results. CONAN-G also starts from `CodeT5-base`, uses top-`5` retrieved documents, learning rate `` `1e-4` `` on Concode and `` `5e-5` `` elsewhere, batch size `` `1` ``, max epoch `` `1` ``, AdamW, and `` `1000` `` warmup steps.

## Key Results

- **Concode + CsCSN**: CONAN reaches `42.8` BLEU / `45.1` CodeBLEU on Concode, `23.5` BLEU on CsCSN-Python, and `26.5` BLEU on CsCSN-Java, beating REDCODER-EXT (`42.5` / `43.4`, `20.9`, `22.9`).
- **CgCSN code generation**: CONAN improves Python generation to `14.6` EM, `32.9` BLEU, `37.3` CodeBLEU and Java generation to `17.2` EM, `37.7` BLEU, `45.4` CodeBLEU, clearly above REDCODER-EXT (`9.6/24.4/30.2` on Python; `10.2/29.0/33.2` on Java).
- **Code completion**: CONAN does not beat ReACC, but it improves over the authors' CodeT5 baseline, reaching `40.12` EM / `69.44` ES on PY150 and `26.02` EM / `62.86` ES on JavaCorpus versus CodeT5's `35.99/66.76` and `25.20/57.99`.
- **Retriever quality**: on Adv, fine-tuned CONAN-R reaches `47.3` MRR versus CodeT5's `39.3`; in zero-shot retrieval it gets overall `70.9` on CodeSearch and `46.1` on Adv, outperforming CodeRetriever's `69.1` and `34.7`.
- **LLM assistance**: adding CONAN to DeepSeek-Coder raises CgCSN-Python from `4.2` to `33.2` BLEU and Concode from `7.7` to `42.4` BLEU; for CodeQwen, CgCSN-Java rises from `9.8` to `38.1` BLEU and Concode from `8.5` to `43.1` BLEU.
- **HumanEval / MBPP**: as an assistant, CONAN yields competitive execution-based gains, e.g. CodeQwen improves from `77.9` to `79.3` on HumanEval and DeepSeek-Coder improves from `68.9` to `69.9` on MBPP, while top-5 raw retrieval remains slightly stronger in some settings.

## Limitations

- The method depends strongly on retrieval quality; the paper explicitly shows that answer-like retrieved documents correlate with successful generation, so poor retrieval can still inject harmful noise.
- CONAN is not state of the art on line-level code completion, where ReACC remains better; the authors attribute this partly to the T5-style backbone being less tailored to autoregressive completion.
- The dual-view mechanism is constrained by data coverage because only about `25%` of code snippets in the retrieval database have paired documentation, which limits gains on code summarization relative to code generation.
- Evaluation is concentrated on Java and Python generation/completion benchmarks plus code-search benchmarks, so generalization to repository-scale engineering tasks or other programming languages is not fully established.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[dense-retrieval]]
- [[structure-aware-pretraining]]
- [[code-documentation-alignment]]
- [[masked-entity-prediction]]
- [[fusion-in-decoder]]
- [[dual-view-code-representation]]
- [[code-generation]]
- [[code-summarization]]
- [[code-completion]]
- [[large-language-model]]

## Entities Extracted

- [[xinze-li]]
- [[hanbin-wang]]
- [[zhenghao-liu]]
- [[shi-yu]]
- [[shuo-wang]]
- [[yukun-yan]]
- [[yukai-fu]]
- [[yu-gu-neu]]
- [[ge-yu]]
- [[codet5]]
- [[codesearchnet]]
- [[humaneval]]
- [[mbpp]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
