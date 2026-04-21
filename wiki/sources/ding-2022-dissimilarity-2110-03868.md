---
type: source
subtype: paper
title: Towards Learning (Dis)-Similarity of Source Code from Program Contrasts
slug: ding-2022-dissimilarity-2110-03868
date: 2026-04-20
language: en
tags: [code-modeling, contrastive-learning, vulnerability-detection, clone-detection, self-supervised-learning]
processed: true

raw_file: raw/papers/ding-2022-dissimilarity-2110-03868/paper.pdf
raw_md: raw/papers/ding-2022-dissimilarity-2110-03868/paper.md
bibtex_file: raw/papers/ding-2022-dissimilarity-2110-03868/paper.bib
possibly_outdated: false

authors:
  - Yangruibo Ding
  - Luca Buratti
  - Saurabh Pujar
  - Alessandro Morari
  - Baishakhi Ray
  - Saikat Chakraborty
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2110.03868
doi:
url: http://arxiv.org/abs/2110.03868
citation_key: ding2022dissimilarity
paper_type: method

read_status: unread

domain: software-engineering
---

## Summary

The paper proposes DISCO, a self-supervised Transformer encoder for learning functional similarity and dissimilarity between code snippets, targeting vulnerability detection and clone detection. Instead of relying on very large randomly collected corpora, DISCO augments each program with a semantically equivalent positive sample and a syntactically similar but functionally altered hard negative produced by structure-guided program transformations. The model combines standard masked language modeling with a new AST-type reconstruction objective that uses token node type plus parent type, and a contrastive loss over `(x, x+, x-)` triplets. Despite pre-training on only `992 MB` Java and `865 MB` C code, roughly `5%` of the data used by some baselines, DISCO matches or exceeds stronger pretrained code models on REVEAL, CodeXGLUE, POJ-104, and BigCloneBench.

## Problem & Motivation

Existing pretrained code models are strong at token- and syntax-level pattern matching but often fail on functional similarity: they may separate semantically equivalent code with different surface forms, or miss subtle functional changes caused by a small edit such as an altered comparison operator or function argument. The authors argue that random large-scale pretraining data does not directly teach these distinctions. They therefore target a more task-aligned pretraining recipe that explicitly exposes the model to semantic clones and bug-injected near-misses, so the encoder learns code representations useful for vulnerability detection and clone detection with much less data.

## Method

- **Data augmentation**: for each program `x`, generate a positive sample `x+` via semantics-preserving transformations and a hard negative `x-` via bug injection; positives include variable renaming, function renaming, and statement permutation.
- **Bug injection heuristics**: six categories are used to create functionally different but token-close negatives, including misuse of data type, misuse of pointer, conditional changes, variable misuse, value misuse, and function-call changes.
- **Encoder architecture**: DISCO uses a `12`-layer Transformer encoder with `12` attention heads and hidden size `768`, initialized and trained as a code representation model rather than a seq2seq generator.
- **Token representation**: source code is flattened, tokenized with SentencePiece using vocabulary size `20,000`, then represented as `V = V^src + V^type`, where `V^type` encodes local AST information.
- **AST-aware input**: each subtoken receives a local AST type `t = tt#pt`, combining terminal node type and parent node type, so the encoder sees both lexical context and local tree structure.
- **Pretraining objectives**: optimize the joint loss `L(theta) = L_MLM + L_NT-MLM + L_CLR` where MLM reconstructs masked code tokens, NT-MLM reconstructs masked AST-type tokens, and CLR contrasts sequence embeddings from `[CLS]`.
- **Masking setup**: randomly choose `15%` of tokens; among them, replace with `[MASK]` for `80%`, random tokens for `10%`, and leave unchanged for `10%`, mirroring BERT-style masking for both code tokens and aligned AST types.
- **Contrastive learning**: use cosine similarity and temperature `tau = 0.05` over minibatch triplets `(h, h+, h-)`, where `h = h_[CLS]` is the sequence representation.
- **Training configuration**: small DISCO is pretrained for `24` hours on `2 x 32 GB` NVIDIA Tesla V100 GPUs, with batch size `128` at max length `256` and batch size `64` at max length `512`; optimizer is Adam with learning rate `1e-4`.
- **Data scale**: the main corpus contains `187K` Java files (`992 MB`) and `64K` C files (`865 MB`) from top-100 GitHub repositories per language, much smaller than `20 GB`-scale baselines such as CodeBERT and GraphCodeBERT.

## Key Results

- On REVEAL vulnerability detection, the best DISCO variant (`MLM+CLR±+NT-MLM`) reaches `48.3` precision, `44.6` recall, and `46.4` F1, beating GraphCodeBERT's `45.8` F1 and CodeBERT's `45.2` F1.
- On CodeXGLUE vulnerability detection, `MLM+CLR+` achieves the best DISCO accuracy at `64.4%`, outperforming GraphCodeBERT (`63.2%`), CodeBERT (`62.1%`), and the scratch Transformer (`62.0%`).
- On POJ-104 clone detection, DISCO obtains up to `82.77` MAP@R, slightly above CodeBERT's `82.67` and close to or above other baselines despite much less pretraining data.
- On BigCloneBench, `MLM+CLR±+NT-MLM` reaches `94.2` precision, `94.6` recall, and `94.4` F1; GraphCodeBERT reports `95.0` F1, but DISCO stays competitive with far smaller data.
- Scaling from DISCOsmall to DISCOmedium improves POJ-104 from `82.8` to `83.8` MAP@R, CodeXGLUE VD from `63.8` to `64.6` accuracy, and REVEAL F1 from `46.4` to `50.2`.
- The paper claims the small model uses about `5%` of the pretraining data size of large baselines, yet still outperforms them on several downstream tasks.

## Limitations

- The bug-injection heuristics do not guarantee that generated negatives are actual exploitable vulnerabilities; they only ensure functional deviation from the original program.
- Positive transformations do not cover all real-world clone patterns; the authors explicitly note this may explain why plain MLM performs best on POJ-104.
- Evaluation is limited to code understanding tasks in C/C++ and Java, mainly vulnerability and clone detection, so transfer to other software-engineering tasks remains unverified.
- DISCO's BigCloneBench recall still trails GraphCodeBERT, suggesting structure-aware pretraining with larger corpora may remain advantageous.
- The method depends on AST parsing and flow/dependency analyses, which may be harder to generalize across programming languages or noisy codebases.

## Concepts Extracted

- [[self-supervised-learning]]
- [[contrastive-learning]]
- [[masked-language-modeling]]
- [[data-augmentation]]
- [[hard-negative-sampling]]
- [[code-clone-detection]]
- [[vulnerability-detection]]
- [[code-understanding]]
- [[transformer]]

## Entities Extracted

- [[yangruibo-ding]]
- [[luca-buratti]]
- [[saurabh-pujar]]
- [[alessandro-morari]]
- [[baishakhi-ray]]
- [[saikat-chakraborty]]
- [[columbia-university]]
- [[ibm-research]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
