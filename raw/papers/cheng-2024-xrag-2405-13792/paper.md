xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token
====================================================================================

Xin Cheng1 Xun Wang2 Xingxing Zhang2 Tao Ge2  
Si-Qing Chen2 Furu Wei2 Huishuai Zhang1,3 Dongyan Zhao1,3  
  
1Peking University2Microsoft  
3National Key Laboratory of General Artificial Intelligence  
chengxin1998@stu.pku.edu.cnWork done during internship at Microsoft, corresponding to Xun Wang, Huishuai Zhang and Dongyan Zhao

###### Abstract

This paper introduces xRAG, a novel context compression method designed specifically for retrieval-augmented generation. xRAG redefines the use of document embeddings in dense retrieval—traditionally limited to retrieval purposes—by integrating them as features from the retrieval modality. Through a modality fusion approach, xRAG effectively merges these embeddings into the language model’s representation space, eliminating the need for their textual counterparts and achieving an extreme compression rate. In xRAG, the modality bridge is the only trainable component, while the retriever and language model remain frozen. This design choice allows for the reuse of offline-constructed document embeddings and preserves the plug-and-play nature of retrieval augmentation. Experimental results demonstrate that xRAG achieves an average improvement of over 10% across six knowledge-intensive tasks, compatible with various language model backbones, ranging from a dense 7B model to an 8x7B Mixture of Experts configuration. xRAG not only significantly outperforms previous context compression methods but also matches the performance of uncompressed models on several benchmarks, while reducing overall FLOPs by a factor of 3.53. This work pioneers new avenues in retrieval-augmented generation through multimodal fusion, potentially setting a groundwork for future developments in efficient and scalable retrieval systems.

<img src='x1.png' alt='Refer to caption' title='' width='830' height='449' />

*Figure 1: xRAG enables efficient retrieval augmentation by adding one document token [X].*

1 Introduction
--------------

Retrieval-Augmented Language Models (RALMs) *[[41], [21], [8], [13], [70]]* have shown exceptional performance in a variety of knowledge-intensive tasks. By retrieving domain-specific, long-tailed, and up-to-date knowledge from a non-parametric datastore, RALMs significantly extend the boundaries of parametric Large Language Models (LLMs). However, the integration of entire documents into prompts can significantly increase inference costs and may surpass the context limit of LLMs *[[28], [79]]*. As illustrated in Figure[1], while the inclusion of a relevant document enables the LLM to generate accurate responses, it does so at the expense of processing documents that expand the original query by more than tenfold.

How might we mitigate the costs associated with extended context while maintaining the benefits of retrieval augmentation? Recent research interest has converged on a promising direction: Context Compression. This concept is pursued through two primary strategies: soft-prompting methods, such as Gist *[[58]]*, AutoCompressor *[[14]]*, and ICAE *[[19]]*, which compress the context into dense memory slots, and hard-prompting methods, such as LLMLingua *[[28]]* and RECOMP *[[79]]*, where compression is applied on the surface form. These approaches, however, either require significant memory for storing LLM activations (e.g., 1.05 MB per token as reported by *[[58]]*) or suffer from relatively low compression rates. More critically, these methods overlook a crucial characteristic of RALMs: through large-scale contrastive learning with question-document pairs, modern dense retrieval systems already distill document content into a single high-dimensional embedding, and this embedding reveals (almost) as much information as the text *[[57], [43]]*.

In this paper, we pioneer an innovative approach to retrieval augmentation and context compression through the lens of modality fusion. Drawing from multimodal research, where text-only language models are taught to "perceive" and "listen," a pretrained modality encoder like CLIP *[[64]]* is typically used to extract modality features. These features are then integrated into language models using a modality fusion bridge *[[44], [51], [33]]*. Building on the conceptual overlap between the retriever encoder and modality encoder, we introduce xRAG. This model redefines document embeddings from dense retrieval—traditionally solely for retrieval purposes—as retrieval modality features. xRAG employs a modality fusion methodology to seamlessly integrate these embeddings into the language model’s representation space, thus obviating the need for textual counterparts and achieving significant context compression. In xRAG, the modality bridge is the only trainable component, while both the retriever and the LLM are kept frozen. This design decision facilitates the reuse of pre-constructed document embeddings and maintains the plug-and-play nature of retrieval augmentation—two essential factors for a functional RAG system.

To verify the effectiveness and versatility of our framework, we conducted comprehensive experiments with different LLM backbones, ranging from a dense 7B model to an 8x7B Mixture of Experts model. Our results reveal that adding just one document token could lead to over a 10% improvement across six knowledge-intensive tasks, significantly surpassing previous compression methods. xRAG also delivers results comparable to uncompressed models on several benchmarks. This is remarkable considering that the only trainable component constitutes less than 0.1% of the LLM’s parameters. In terms of efficiency, xRAG reduces total FLOPs by a factor of 3.53 compared to the uncompressed RAG model. We further provide detailed analyses of xRAG, examining various training strategies, data blends, and component selections for the retrieval system. We believe this research sets a strong foundation for the development of future efficient and scalable retrieval-augmented systems.

2 Related Work
--------------

#### Retrieval-augmented Generation

Equipping a parametric language model with a non-parametric datastore has proven effective for a range of NLP tasks, including language modeling *[[36], [56], [87]]*, open-domain question answering *[[24], [41], [70], [86]]*, domain adaptation *[[6]]* and machine translation *[[35], [11]]*, among others. Given the vast design space of this generation paradigm, numerous approaches with different focuses have been proposed. For instance, RETRO *[[8]]* and PlugLM *[[12]]* introduce architectural innovations for enhanced integration with the non-parametric datastore. REALM *[[21]]* pioneers an end-to-end approach for simultaneous optimization of the language model and retriever. REPLUG *[[70]]* and RA-DIT *[[50]]* improve retriever alignment using feedback from LLMs. DSP *[[37]]* and InteR *[[17]]* investigate complex interactions between the retriever and the language model. Selfmem *[[13]]* utilizes a reward model to refine retrieval and generation iteratively. Self-RAG *[[3]]* incorporates a self-reflection mechanism to enhance the quality and factuality of language model outputs. For a detailed overview, see *[[18], [4], [2]]*. Our contribution, xRAG, stands out by implementing a modality fusion approach to retrieval augmentation, creating an effective and efficient RAG system.

#### Context Compression

Context compression, aimed at reducing the input length for LLMs while retaining essential information, has recently attracted substantial interest *[[46]]*. Gist *[[58]]* achieves a compression rate of up to 26x by modifying the attention mask and caching soft gist token activations. ICAE *[[19]]*AutoCompressor *[[14]]*, and 500xCompressor*[[47]]* condense lengthy contexts into succinct, compact memory slots, which are directly utilizable by LLMs for diverse functions. LLMLingua *[[28], [29], [62]]* and CompAct*[[82]]* introduces a coarse-to-fine prompt compression technique based on perplexity scores and distilled token-level score. While these methods are generally applicable, others are tailored specifically for RAG systems, such as FilCo *[[76]]* and RECOMP *[[79]]*. A concurrent work directly employs passage embeddings for efficient listwise reranking *[[52]]*. For an in-depth comparison of these compression methods regarding memory efficiency, compression rates, and adaptability, refer to Appendix[A].

<img src='x2.png' alt='Refer to caption' title='' width='789' height='383' />

*Figure 2: Overview of xRAG (a) and RAG (b). For a given query, RAG typically concatenates the retrieved document with the query, significantly extending the context length. In contrast, xRAG addresses this issue through modality fusion by directly projecting the document embedding into the LLM’s representation space. This allows for efficient retrieval-augmentation with the addition of only one token.*

3 Methods
---------

#### Problem Formulation

In retrieval-augmented generation, a non-parametric datastore $\mathbb{D}\={(\textrm{E}_{i},\textrm{D}_{i})}_{i\=1}^{|\mathbb{D}|}$ consists of pairs where each $\textrm{D}_{i}$ represents a document chunk as a sequence of $L_{i}$ tokens $\textrm{D}_{i}\={d_{1}^{i},\ldots,d_{L_{i}}^{i}}$. Correspondingly, $\textrm{E}_{i}$ is the dense representation derived from a sentence embedding model $\mathbf{SE}_{\theta}(\cdot)$ with input $D_{i}$. For an input query $q$, its dense representation $\mathbf{SE}_{\theta}(q)$ is used to find the relevant documents by matching against the collection ${\textrm{E}_{i}}_{i\=1}^{|\mathbb{D}|}$ with certain similarity search algorithm such as MIPS. After retrieval, the system selects a relevant pair $(\textrm{E},\textrm{D})$ from $\mathbb{D}$, concatenates the chosen document D with $q$, and processes the combined input with a language model $\bm{\mathcal{F}_{\phi}}(\textrm{D}\oplus q)$. Optionally, a context compression module $\bm{\mathcal{C}}$ can be integrated to reduce the length of D from $L$ to a more concise $l$, achieving a compression ratio of $\frac{L}{l}$.

### 3.1 xRAG Architecture

Traditional methods for document compression typically focus on surface form of the document *[[28], [76], [79]]*. In contrast xRAG tackle the problem from a modality fusion view. Concretely, we introduce a modality projector ${{\bf W}}$, which is trained to directly project the retrieval features E into the LLM representation space. Our proposed framework is visually contrasted with the traditional RAG system in Figure[2]. In the standard RAG, the input to the LLM comprises the embeddings $\texttt{Emb}(\textrm{D}\oplus q)$ of length $|\textrm{D}|+|q|$, where Emb signifies the embedding layer of the LLM. Conversely, with xRAG, the modified input is represented as ${{\bf W}}(\textrm{E})\oplus\texttt{Emb}(q)$, which yields a substantially reduced length of $1+|q|$. In this framework, the challenges come from the modality fusion: How can a text-only language model understand features from retrieval modality? To achieve this, we explore a two-stage training strategy: Paraphrase Pretraining followed by Context-aware Instruction Tuning.

<img src='x3.png' alt='Refer to caption' title='' width='830' height='296' />

*Figure 3: Two-stage training strategy of xRAG including (a) Paraphrase Pre-training on unlabeled corpus and (b) Context-aware Instruction Tuning optimized with labeled data and self-distillation.*

### 3.2 Paraphrase Pretraining

Similar to the pretraining strategies employed in vision-language models that use image-captioning data to align two modalities *[[51], [15], [53]]*, the primary objective of our paraphrase pretraining is to build a compatible representation between the extracted retrieval feature and the corresponding document.
Illustrated in Figure [3](a), for each pair $(\textrm{E},\textrm{D})$ in a retrieval corpus $\mathbb{D}$, we employ a natural language instruction ${{\bf X}}_{\texttt{instruct}}$ to prompt the LLM to undertake a paraphrasing task (e.g. "[X] The above text could be paraphrased as: [D]", where [X] and [D] are placeholders for ${{\bf W}}(\textrm{E})$ and document D)111To maintain diversity, we sample from an instruction pool, which could be found in Appendix[B].. In this setup, the model learns to connect ${{\bf W}}(\textrm{E})$ and D by recovering D on the condition of ${{\bf W}}(\textrm{E})$ and the model is optimized by:

|  | $\mathcal{L}_{\text{nll}}\=-\sum_{i\=1}\text{log}\ p_{\phi}(d_{i}|{{\bf W}}(% \textrm{E}),{{\bf X}}_{\texttt{instruct}},d_{<i})$ |  | (1) |
| --- | --- | --- | --- |

where $p_{\phi}$ is given by the softmax distribution of LLM $\bm{\mathcal{F}_{\phi}}$, and $d_{<i}$ denotes the document token before current prediction token $d_{i}$, achieved by casual attention mask in auto-regressive LMs.

### 3.3 Context-aware Instruction Tuning

After the pretraining phase, although the language model $\bm{\mathcal{F}_{\phi}}$ has developed an internally compatible representation, it has never been explicitly trained to utilize these features for downstream tasks. To address this gap, we proceed to instruct the model in harnessing the fused feature ${{\bf W}}(\textrm{E})$ by continually training the model on data where the answer is closely associated with the given context, including reading comprehension, summarization, and open domain question answering data.
We constructed an mixed dataset, containing approximately 1 million entries from open-source datasets, as detailed in Appendix[C]. For each triplet in the dataset, $({{\bf X}}_{\texttt{context}},{{\bf X}}_{\texttt{question}},{{\bf X}}_{\texttt%
{answer}})$, we initially obtain the sentence representation for ${{\bf X}}_{\texttt{context}}$ via the embedding model $\textrm{E}_{\texttt{context}}\=\mathbf{SE}_{\theta}({{\bf X}}_{\texttt{context}})$. Subsequently, we refine the optimization of on two directions:

#### Optimization I: Language Modeling.

Aligned with established instruction tuning methodologies *[[75], [23], [50], [42]]*, our objective is to finetune the model so that it generates the correct output when provided with a specific instruction, conditioned upon the given context information. Unlike traditional models that utilize the textual context ${{\bf X}}_{\texttt{context}}$, our method employs a dense feature $\textrm{E}_{\texttt{context}}$ to encapsulate the context information:

|  | $\mathcal{L}_{\text{nll}}\=-\sum_{i\=1}\text{log}\ p_{\phi}({{\bf X}}_{\texttt{% answer},i}|{{\bf W}}(\textrm{E}_{\texttt{context}}),{{\bf X}}_{\texttt{% question}},{{\bf X}}_{\texttt{answer},<i})$ |  | (2) |
| --- | --- | --- | --- |

#### Optimization II: Self-Distillation.

The second trajectory of optimization aims to guide the xRAG in the effective utilization of contextual information, drawing from the principles of self-distillation *[[1], [71]]* and imitation learning *[[61], [22]]*. By considering the RAG model as a "teacher" and xRAG as a "student", we endeavor to distill the knowledge from RAG, thereby enabling xRAG to emulate the RAG model’s proficiency in handling the full, uncompressed documents. This approach enhances xRAG’s resilience in scenarios where it encounters noisy or irrelevant context that may not directly lead to the correct answer, detailedly discussed in $\S$[6.1]. Concretely, for a language model $\bm{\mathcal{F}_{\phi}}$ using either ${{\bf X}}_{\texttt{context}}$ or $\textrm{E}_{\texttt{context}}$ as the source of context, our objective is to minimize the divergence between the two resulting output distributions. This discrepancy is measured using the Kullback-Leibler (KL) divergence:

|  | $\mathcal{L}_{\text{kl}}\=D_{\textrm{KL}}(p_{\phi}({{\bf X}}_{\texttt{answer}}|{% {\bf X}}_{\texttt{context}},\cdot)\ ||\ p_{\phi}({{\bf X}}_{\texttt{answer}}|{% {\bf W}}(\textrm{E}_{\texttt{context}}),\cdot))$ |  | (3) |
| --- | --- | --- | --- |

Here ${{\bf X}}_{\texttt{question}}$ is omitted for brevity and the final loss is the linear combination controlled by a hyperparameter: $\mathcal{L}_{nll}+\alpha\mathcal{L}_{kl}$.

### 3.4 Design Principle

In designing the projector $\mathbf{W}$, our primary objective is to maintain the simplicity of the framework. We therefore opted for a two-layer MLP while other more sophisticated module such as Q-Former *[[44]]* could also be considered. Notice that the projector is the only trainable component, accounting for only 0.46% of the total parameters in the Mistral-7b model and 0.07% in the Mixtral-8x7b model.
Such a design choice departs from previous studies that necessitated full-parameter tuning to adapt LLMs for compressed contexts*[[58], [76], [14]]*. We believe this approach will likely be more accessible and practical because, fundamentally, the RAG itself functions as a plug-and-play module for LLMs, and so should its compressed version. This design also avoid the risk of compromising other core capabilities of LLM during full-parameter tuning, as observed in*[[54], [55]]*.

Moreover, in contrast to other compression methods that necessitate storing LLM activations for each compressed token *[[58], [19], [14]]*—an impractical strategy in the RAG setting, given the millions of documents involved—our method introduces no additional memory overhead. Instead, it leverages offline-constructed document embeddings, originally designed for retrieval. To summarize, xRAG not only simplifies the integration process but also avoids unnecessary computational or memory expenses.

4 Experimental Setup
--------------------

### 4.1 Evaluation Dataset

We evaluated the performance of xRAG primarily on knowledge-intensive tasks which encompass a range of challenges: (1) three Open Domain Question Answering datasets that address questions on a wide array of topics: Natural Questions *[[40]]*, TriviaQA *[[32]]*, and Web Questions *[[7]]*. (2) one Multihop Question Answering dataset, HotpotQA *[[81]]*, which necessitates multi-step reasoning to generate answers.
(3) one Long-form Question Answering dataset, TruthfulQA *[[49]]*, that requires the generation of long-form and truthful responses.
(4) one fact-checking dataset, FactKG *[[38]]*, which challenges the model to use complex reasoning to determine the factual accuracy of given claims.

In line with the KILT *[[63]]* and GenRead *[[84]]*, we assessed three ODQA datasets and HotpotQA using the Exact Match (EM) metric, FactKG with Accuracy, and for the long-form QA, we used both the F1 score and Rouge-L (R-L) score. These tasks demand a broad spectrum of world knowledge and have been extensively explored in the retrieval-augmentation literature *[[41], [21], [34], [9], [24], [70]]*.

### 4.2 Implementation Details

To demonstrate the versatility of our framework, we choose two backbones, differing in scale and architecture: Mistral-7b *[[26]]* and Mixtral-8x7b *[[27]]*. For the retrieval corpus, we utilized the Wikipedia dump from December 2021, which was pre-processed into passages following the methodology described in *[[25]]*. This resulted in approximately 37 million passages, each averaging 180 tokens in length. Our default retrieval model is the SFR*[[60]]*, which, at the time of writing this paper, holds the leading position on the MTEB leaderboard *[[59]]*. We use top-1 ranked document for inclusion in our instruction-tuning dataset and for the evaluation of downstream tasks. More details are provided in Appendix[D]. Code is available at: <https://github.com/Hannibal046/xRAG>.

### 4.3 Baselines

In determining appropriate baselines for comparison, we adhered to a fundamental principle: the selected compression methods must support the general plug-and-play capability of retrieval augmentation. This entails that they should function effectively without the need for dataset-specific tuning *[[79], [76]]*, or any alteration to the parameters of LLMs *[[76], [58]]*. Furthermore, given the extensive volume of the retrieval corpus, it is essential that these compression methods demonstrate memory efficiency, specifically by not requiring the storage of LLM activations for each individual token *[[58], [19], [14]]*. With these criteria in mind, our evaluation compares xRAG to the following baselines: (I) Primarily, we consider two variants of LLMs: one that operates without retrieval augmentation and another that includes it. These serve as the lower and upper performance bounds for our study of compression techniques, respectively. (II) Additionally, our comparisons extend to LLMLingua *[[28]]*, a plug-and-play approach for context compression. (III) Taking inspiration from *[[58]]*, we incorporate a method of discrete compression using TF-IDF. This approach yields compression rates comparable to those achieved by xRAG and serves as the lower bound for discrete compression.

*Table 1: Experimental results on six downstream tasks. The best results are in bold and the second best are with underscore. Percentage in the brackets denotes the relative improvement over non-retrieval setting. LLMs are frozen during the experiments and retrieved documents are set the same for different compression methods. ‡ and † denotes different compression ratio.*

|  | NQ | TriviaQA | WebQA | HotpotQA | TrutufulQA | | FactKG | Average | # DocLength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task Type | | Open-Domain QA | | --- | | (EM) | | | |
| Mistral-7b | | | | | | | | | |
| w/o retrieval | 30.25 | 57.08 | 34.89 | 27.02 | 26.23 | 25.51 | 54.78 | 36.54 (0.0%) | 0 |
| w retrieval | 42.71 | 65.88 | 37.84 | 38.79 | 26.50 | 25.92 | 67.76 | 43.63 (19.4%) | 175.1 |
| *with Compression | | | | | | | | | |
| LLMLingua † | 30.64 | 57.94 | 32.63 | 29.91 | 25.70 | 25.10 | 64.17 | 38.01 (4.0%) | 98.6 |
| LLMLingua ‡ | 28.81 | 57.09 | 32.33 | 29.13 | 26.10 | 25.39 | 63.57 | 37.48 (2.5%) | 61.1 |
| TF-IDF | 30.25 | 58.49 | 35.43 | 26.62 | 26.33 | 25.83 | 59.56 | 37.49 (2.6%) | 1 |
| xRAG | 39.10 | 65.77 | 39.40 | 34.05 | 28.10 | 27.71 | 63.08 | 42.46 (16.2%) | 1 |
| Mixtral-8x7b | | | | | | | | | |
| w/o retrieval | 41.99 | 71.10 | 40.31 | 32.87 | 25.60 | 24.90 | 62.64 | 42.76 (0.0%) | 0 |
| w retrieval | 45.15 | 70.34 | 41.26 | 43.46 | 27.10 | 25.80 | 70.42 | 46.22 (8.0%) | 175.1 |
| *with Compression | | | | | | | | | |
| LLMLingua† | 37.65 | 67.70 | 36.02 | 35.66 | 25.99 | 25.39 | 67.98 | 42.32 (-1.0%) | 96.6 |
| LLMLingua‡ | 37.81 | 67.81 | 35.78 | 35.27 | 25.68 | 25.00 | 68.03 | 44.17 (-1.3%) | 61.1 |
| TF-IDF | 41.19 | 69.94 | 41.63 | 32.05 | 26.80 | 26.00 | 66.17 | 43.41 (1.4%) | 1 |
| xRAG | 47.28 | 74.14 | 44.50 | 39.66 | 27.80 | 26.64 | 68.20 | 46.91 (9.7%) | 1 |

5 Experimental Results
----------------------

### 5.1 Knowledge Intensive Tasks

In Table[1], we present our main results. Across both Mistral-7b and Mixtral-8x7b configurations, we observe a consistent and significant improvement when retrieval augmentation is applied (p-value < 0.05), although the gains are more modest for the larger model configurations. This trend aligns with observations reported by *[[70], [50]]*.
Further analysis on the efficacy of various compression techniques reveals that xRAG outperforms other approaches by a large margin. Remarkably, xRAG not only reduces the token count drastically—from 175.1 to a single token—but also maintains robust performance levels. In some instances, xRAG’s performance is comparable to, or even exceeds, that of the uncompressed models. Specifically, in the Mistral-7b configuration, xRAG achieves nearly the same performance improvement as the uncompressed model (16.6% compared to 19.4%), and in the Mixtral-8x7b configuration, it surpasses the uncompressed model (9.7% compared to 8.0%). One possible reason lies in the vulnerability of current RAG system when the irrelevant or misleading documents are presented, a topic detailed discussed in §[6.1].
We also observe that xRAG performs well in tasks that require document understanding, such as TriviaQA. However, in tasks that demand reasoning over document, like HotpotQA and FactKG, xRAG lags behind by a considerable gap.

*Table 2: Comparison of RAG and xRAG performance in CUDA Time and GFLOPS.*

|  | CUDA Time (ms) | | | GFLOPs | | |
| --- | --- | --- | --- | --- | --- | --- |
|  | RAG | xRAG | Improvement | RAG | xRAG | Improvement |
| FactKG | 431.5 | 215.6 | x2.01 | 4683.8 | 1289.5 | x3.63 |
| NQ | 918.7 | 611.3 | x1.51 | 1338.6 | 384.0 | x3.48 |
| TriviaQA | 807.1 | 512.1 | x1.57 | 1667.2 | 492.3 | x3.38 |
| WebQA | 872.6 | 577.3 | x1.51 | 1405.1 | 386.8 | x3.63 |
| Average |  |  | x1.64 |  |  | x3.53 |

### 5.2 Computational Efficiency

In this section, we conduct a thorough assessment of our framework’s computational efficiency and memory management.
To rigorously evaluate our model, we employed Torch Profiler222[https://pytorch.org/docs/stable/profiler.html#module-torch.profiler](https://pytorch.org/docs/stable/profiler.html#module-torch.profiler "") to measure the CUDA Time (milliseconds) and Giga FLOPs of both the RAG and xRAG models across four real-world datasets. In these evaluations, the Mistral-7b, operating in bfloat16 inference mode, served as the base LLM. CUDA Time and GFLOPs were calculated on an average per batch basis with a fixed batch size, and GFLOPs were normalized by the number of generated tokens. These experiments were performed on the same computational hardware, specifically an Nvidia A100 and an AMD EPYC 7V12 64-Core Processor. As depicted in Table[2], despite variations in prompt and generation lengths across the datasets, xRAG significantly outpaced the RAG model, achieving a x1.64 increase in CUDA Time efficiency and a x3.53 reduction in GFLOPs.

6 Analysis
----------

### 6.1 Evaluation Beyond the Overall Score

Although retrieval augmentation generally boosts performance as shown by aggregate metrics, it may not uniformly benefit all instances. In certain cases, the retrieval system might provide irrelevant or even misleading information, leading to incorrect answers that were previously correct*[[83], [76], [4]]*. To enable a more fine-grained evaluation, we introduce two novel metrics: the Resilience Rate and the Boost Rate. The resilience rate quantifies the percentage of instances in which the system’s responses remain correct both before and after retrieval augmentation, highlighting the system’s stability and robustness. Conversely, the boost rate measures the percentage of instances that were initially answered incorrectly but were rectified following the introduction of a retrieved document, thereby assessing the efficacy of retrieval augmentation. An ideal RAG system should have both high resilience rate and boost rate.

In Figure[4], we display these metrics for the uncompressed RAG and two compression methods: LLMLingua and xRAG. Surprisingly, although retrieval augmentation generally enhances performance, the resilience rate for RAG averages only 75.2%, indicating that retrieval can adversely affect about one-quarter of previously correct answers. In contrast, xRAG demonstrates considerable robustness across all evaluated datasets. This robustness largely stems from xRAG’s ability to maintain an unbiased stance toward the internal knowledge representation of the LLM, especially when confronted with noisy retrieval content. Similar trends are noted in *[[50], [54]]*, where search-augmented instruction learning is shown to bolster the robustness of language models. However, xRAG still lags behind RAG in boost rate, particularly in multi-hop reasoning tasks. It is crucial to note that a high resilience rate does not necessarily mean that the LLM disregards the provided information, which could potentially lead to a reduced boost rate. A comparative analysis with LLMLingua indicates that xRAG is not only more robust but also more effective.

<img src='x4.png' alt='Refer to caption' title='' width='788' height='382' />

*Figure 4: Resilience rate and boost rate of three augmentation methods: LLMLinuga, xRAG and RAG over a Mixtral-8x7b baseline without retrieval augmentation.*

### 6.2 What makes xRAG effective?

*Table 3: Ablation on different training strategy for xRAG.*

|  | NQ | TriviaQA | WebQA | HotpotQA | Averaged | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | Performance | Resilience | Boost |
| Mistral-7b | | | | | | | |
| xRAG | 39.10 | 65.77 | 39.40 | 34.05 | 44.58 | 82.3% | 22.2% |
| w/o finetune | 30.14 | 59.48 | 35.19 | 26.70 | 37.87 | 66.6% | 20.8% |
| w/o pretrain | 31.25 | 59.07 | 41.19 | 24.32 | 38.95 | 79.8% | 14.1% |
| w/o nll | 35.46 | 65.27 | 39.57 | 31.80 | 43.02 | 83.7% | 19.4% |
| w/o self-kd | 34.99 | 64.33 | 39.22 | 27.45 | 41.49 | 76.2% | 20.8% |
| w LoRA | 35.71 | 60.14 | 40.45 | 22.91 | 39.80 | 76.0% | 18.0% |
| Mixtral-8x7b | | | | | | | |
| xRAG | 47.48 | 74.14 | 44.50 | 39.66 | 51.45 | 84.9% | 20.0% |
| w/o finetune | 34.46 | 64.08 | 34.89 | 30.43 | 40.96 | 65.9% | 17.8% |
| w/o pretrain | 42.54 | 71.17 | 47.44 | 31.23 | 48.09 | 85.0% | 14.2% |
| w/o nll | 45.10 | 72.85 | 45.03 | 37.11 | 50.02 | 84.8% | 18.9% |
| w/o self-kd | 42.38 | 72.26 | 44.73 | 32.41 | 47.94 | 79.8% | 18.9% |

This section delves into a thorough evaluation of various elements that contribute to xRAG’s overall performance, focusing on its training strategy, the blend of datasets used and the effect of different embedding models. Due to the space limit, we present the last factor in Appendix[E].

<img src='x5.png' alt='Refer to caption' title='' width='830' height='472' />

*Figure 5: Given the misleading document, RAG model tend to generate a wrong answer based on the document, while xRAG demonstrate its robustness by leveraging the internal knowledge of LLM.*

1. Training Strategy We carefully ablate four optimization choices: pretraining, instruction tuning, and two optimization objectives—language modeling (nll) and self-distillation (self-kd). We also train a Mistral-7b with LoRA on our instruction tuning dataset to rule out the possibility that our improvement simply comes from tuning on more data.
The outcomes are presented in Table[3]. Our analysis reveals that the interplay of different training strategies significantly contributes to the performance of our framework. In the case of Mistral-7b, pretraining and finetuning phases are of equal significance to the end results. However, for Mixtral-8x7b, the impact of pretraining is notably diminished, likely due to the larger model’s enhanced capability to incorporate multi-modality information. Furthermore, we find that during finetuning, self-distillation is more important than language modeling. The primary advantage of self-distillation lies in bolstering the resilience rate of the xRAG system. Optimization with nll loss tends to cause an overreliance on context information, rendering the system more vulnerable when the retriever fails to fetch relevant documents.

*Table 4: Abaltion results on different data selection strategy.*

|  | # Train | NQ | TriviaQA | WebQA | HotpotQA | Average | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | Performance | Resilience | Boost |
| xRAG (Mistral-7b) | 955k | 39.10 | 65.77 | 39.40 | 34.05 | 44.58 | 82.3% | 22.2% |
| w RC only | 488k | 36.98 | 65.77 | 41.39 | 32.82 | 44.24 | 81.9% | 22.4% |
| w QA only | 385k | 36.45 | 65.57 | 41.14 | 31.80 | 43.74 | 80.5% | 22.1% |
| w Summ only | 81k | 36.37 | 64.95 | 40.40 | 31.98 | 43.42 | 78.8% | 22.8% |

II. Instruction-tuning Dataset Blend As discussed in $\S$[3.3], our instruction-tuning dataset primarily comprises three categories: reading comprehension, open-domain QA, and text summarization. To explore the effects of different data blends, we instruction-tune three xRAG model variants, each using data from these distinct categories. The results are shown in Table[4].
Our analysis reveals that among the dataset blends, reading comprehension data most significantly enhances the xRAG model’s performance, as evidenced by both high resilience and boost rates. Intriguingly, when tuned solely with summarization data, xRAG still manages to deliver strong performance on QA datasets it has never been exposed to. This finding underscores that the advantages of instruction tuning for xRAG are not rooted in task-specific knowledge. Instead, they derive from the model’s improved ability to utilize projected context information effectively.

### 6.3 Case Study

In Figure [5], we show one interesting case about the robustness of xRAG. When retrieval system provide misleading content, standard RAG would overly rely on the document and generate answer that are faithful to the document while not factually true. Our xRAG model opt to rely on the internal knowledge of LLM and being robust to the misleading content. In Appendix[H], we include more cases about xRAG including several error analysis.

7 Conclusion
------------

In this work, we present xRAG, an innovative context compression method tailored for retrieval-augmented generation. For knowledge-intensive tasks, xRAG can be significantly faster than RAG while maintaining comparable performance. We are excited about the future of this modality-based retrieval-augmented system and plan to further improve its performance in the areas of reasoning over embedding, handling multiple documents, and combining with multi-vector retrieval.

8 Acknowledgement
-----------------

We would like to express our sincere gratitude to the anonymous reviewers for their thorough review, insightful comments, and constructive suggestions, which have significantly improved the quality of this manuscript. This work paper is supported (in part) by the State Key Laboratory of General Artificial Intelligence.

References
----------

* [1]Zeyuan Allen-Zhu and Yuanzhi Li.Towards understanding ensemble, knowledge distillation and self-distillation in deep learning.arXiv preprint arXiv:2012.09816, 2020.
* [2]Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen.Retrieval-based language models and applications.In Yun-Nung (Vivian) Chen, Margot Margot, and Siva Reddy, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts), pages 41–46, Toronto, Canada, July 2023. Association for Computational Linguistics.
* [3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023.
* [4]Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and Wen tau Yih.Reliable, adaptable, and attributable language models with retrieval, 2024.
* [5]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang.Ms marco: A human generated machine reading comprehension dataset, 2018.
* [6]David Beauchemin, Zachary Gagnon, and Ricahrd Khoury.Quebec automobile insurance question-answering with retrieval-augmented generation.arXiv preprint arXiv:2410.09623, 2024.
* [7]Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang.Semantic parsing on freebase from question-answer pairs.In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1533–1544, 2013.
* [8]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre.Improving language models by retrieving from trillions of tokens, 2022.
* [9]Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes.Reading wikipedia to answer open-domain questions, 2017.
* [10]Yulong Chen, Yang Liu, Liang Chen, and Yue Zhang.Dialogsum: A real-life scenario dialogue summarization dataset, 2021.
* [11]Xin Cheng, Shen Gao, Lemao Liu, Dongyan Zhao, and Rui Yan.Neural machine translation with contrastive translation memories, 2022.
* [12]Xin Cheng, Yankai Lin, Xiuying Chen, Dongyan Zhao, and Rui Yan.Decouple knowledge from parameters for plug-and-play language modeling, 2023.
* [13]Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan.Lift yourself up: Retrieval-augmented text generation with self memory, 2023.
* [14]Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen.Adapting language models to compress contexts, 2023.
* [15]Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi.Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
* [16]Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner.Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs, 2019.
* [17]Jiazhan Feng, Chongyang Tao, Xiubo Geng, Tao Shen, Can Xu, Guodong Long, Dongyan Zhao, and Daxin Jiang.Synergistic interplay between search and large language models for information retrieval, 2023.
* [18]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang.Retrieval-augmented generation for large language models: A survey, 2024.
* [19]Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei.In-context autoencoder for context compression in a large language model, 2023.
* [20]Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer.Samsum corpus: A human-annotated dialogue dataset for abstractive summarization.In Proceedings of the 2nd Workshop on New Frontiers in Summarization. Association for Computational Linguistics, 2019.
* [21]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.Realm: Retrieval-augmented language model pre-training, 2020.
* [22]Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan, and Chrisina Jayne.Imitation learning: A survey of learning methods.ACM Computing Surveys (CSUR), 50(2):1–35, 2017.
* [23]Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi.Camels in a changing climate: Enhancing lm adaptation with tulu 2, 2023.
* [24]Gautier Izacard and Edouard Grave.Leveraging passage retrieval with generative models for open domain question answering, 2021.
* [25]Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.Atlas: Few-shot learning with retrieval augmented language models, 2022.
* [26]Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed.Mistral 7b, 2023.
* [27]Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed.Mixtral of experts, 2024.
* [28]Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.Llmlingua: Compressing prompts for accelerated inference of large language models, 2023.
* [29]Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression, 2023.
* [30]Kelvin Jiang, Dekun Wu, and Hui Jiang.FreebaseQA: A new factoid QA data set matching trivia-style question-answer pairs with Freebase.In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 318–323, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
* [31]Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.PubMedQA: A dataset for biomedical research question answering.In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics.
* [32]Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer.Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.
* [33]Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh.Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024.
* [34]Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih.Dense passage retrieval for open-domain question answering, 2020.
* [35]Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis.Nearest neighbor machine translation, 2021.
* [36]Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis.Generalization through memorization: Nearest neighbor language models, 2020.
* [37]Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia.Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp, 2023.
* [38]Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, and Edward Choi.Factkg: Fact verification via reasoning on knowledge graphs.arXiv preprint arXiv:2305.06590, 2023.
* [39]Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette.The narrativeqa reading comprehension challenge, 2017.
* [40]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.Natural questions: A benchmark for question answering research.Transactions of the Association for Computational Linguistics, 7, 2019.
* [41]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021.
* [42]Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, and Furu Wei.Synthetic data (almost) from scratch: Generalized instruction tuning for language models, 2024.
* [43]Haoran Li, Mingshi Xu, and Yangqiu Song.Sentence embedding leaks more information than you expect: Generative embedding inversion attack to recover the whole sentence, 2023.
* [44]Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023.
* [45]Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.Compressing context to enhance inference efficiency of large language models.In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6342–6353, Singapore, December 2023. Association for Computational Linguistics.
* [46]Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Collier.Prompt compression for large language models: A survey, 2024.
* [47]Zongqian Li, Yixuan Su, and Nigel Collier.500xcompressor: Generalized prompt compression for large language models, 2024.
* [48]Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen tau Yih, and Xilun Chen.How to train your dragon: Diverse augmentation towards generalizable dense retrieval, 2023.
* [49]Stephanie Lin, Jacob Hilton, and Owain Evans.Truthfulqa: Measuring how models mimic human falsehoods, 2022.
* [50]Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Yih.Ra-dit: Retrieval-augmented dual instruction tuning, 2023.
* [51]Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.Visual instruction tuning.Advances in neural information processing systems, 36, 2024.
* [52]Qi Liu, Bo Wang, Nan Wang, and Jiaxin Mao.Leveraging passage embeddings for efficient listwise reranking with large language models, 2024.
* [53]Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan.Deepseek-vl: Towards real-world vision-language understanding, 2024.
* [54]Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass.Sail: Search-augmented instruction learning, 2023.
* [55]Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang.An empirical study of catastrophic forgetting in large language models during continual fine-tuning, 2024.
* [56]Sewon Min, Weijia Shi, Mike Lewis, Xilun Chen, Wen tau Yih, Hannaneh Hajishirzi, and Luke Zettlemoyer.Nonparametric masked language modeling, 2023.
* [57]John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander M. Rush.Text embeddings reveal (almost) as much as text, 2023.
* [58]Jesse Mu, Xiang Lisa Li, and Noah Goodman.Learning to compress prompts with gist tokens, 2024.
* [59]Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers.Mteb: Massive text embedding benchmark, 2023.
* [60]Xuan-Phi Nguyen, Shrey Pandit, Senthil Purushwalkam, Austin Xu, Hailin Chen, Yifei Ming, Zixuan Ke, Silvio Savarese, Caiming Xong, and Shafiq Joty.Sfr-rag: Towards contextually faithful llms.arXiv preprint arXiv:2409.09916, 2024.
* [61]Junhyuk Oh, Yijie Guo, Satinder Singh, and Honglak Lee.Self-imitation learning.In International conference on machine learning, pages 3878–3887. PMLR, 2018.
* [62]Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang.Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression, 2024.
* [63]Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel.Kilt: a benchmark for knowledge intensive language tasks, 2021.
* [64]Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.Learning transferable visual models from natural language supervision, 2021.
* [65]Pranav Rajpurkar, Robin Jia, and Percy Liang.Know what you don’t know: Unanswerable questions for squad, 2018.
* [66]Siva Reddy, Danqi Chen, and Christopher D. Manning.Coqa: A conversational question answering challenge, 2019.
* [67]Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky.Getting closer to ai complete question answering: A set of prerequisite real tasks.In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 8722–8731, 2020.
* [68]Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.Colbertv2: Effective and efficient retrieval via lightweight late interaction, 2022.
* [69]Abigail See, Peter J. Liu, and Christopher D. Manning.Get to the point: Summarization with pointer-generator networks, 2017.
* [70]Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih.Replug: Retrieval-augmented black-box language models, 2023.
* [71]Charlie Snell, Dan Klein, and Ruiqi Zhong.Learning by distilling context, 2022.
* [72]Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.Commonsenseqa: A question answering challenge targeting commonsense knowledge, 2019.
* [73]Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei.Text embeddings by weakly-supervised contrastive pre-training, 2024.
* [74]Shuohang Wang, Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu, Chenguang Zhu, and Michael Zeng.Training data is more valuable than you think: A simple and effective method by retrieving from training data, 2022.
* [75]Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi.How far can camels go? exploring the state of instruction tuning on open resources, 2023.
* [76]Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig.Learning to filter context for retrieval-augmented generation, 2023.
* [77]Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le.Finetuned language models are zero-shot learners, 2022.
* [78]Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff.C-pack: Packaged resources to advance general chinese embedding, 2023.
* [79]Fangyuan Xu, Weijia Shi, and Eunsol Choi.Recomp: Improving retrieval-augmented lms with compression and selective augmentation, 2023.
* [80]Yi Yang, Wen-tau Yih, and Christopher Meek.WikiQA: A challenge dataset for open-domain question answering.In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 2013–2018, Lisbon, Portugal, September 2015. Association for Computational Linguistics.
* [81]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018.
* [82]Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Minbyul Jeong, and Jaewoo Kang.Compact: Compressing retrieved documents actively for question answering, 2024.
* [83]Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant.Making retrieval-augmented language models robust to irrelevant context, 2023.
* [84]Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang.Generate rather than retrieve: Large language models are strong context generators, 2023.
* [85]Jungmin Yun, Mihyeon Kim, and Youngbin Kim.Focus on the core: Efficient attention via pruned token compression for document classification.In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13617–13628, Singapore, December 2023. Association for Computational Linguistics.
* [86]Xinping Zhao, Yan Zhong, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Dongfang Li, Baotian Hu, and Min Zhang.Funnelrag: A coarse-to-fine progressive retrieval paradigm for rag.arXiv preprint arXiv:2410.10293, 2024.
* [87]Zexuan Zhong, Tao Lei, and Danqi Chen.Training language models with memory augmentation, 2022.

Appendix A Comparison between different Context Compression Models
------------------------------------------------------------------

In Table[5], we present a detailed comparison of various context compression models, emphasizing their real-world applicability. This comparison focuses on two key aspects: (1) Plug-and-play capability, which assesses whether dataset-specific tuning is necessary for new, unseen data; (2) Memory efficiency, which evaluates if additional memory space is required to store the compressed information, such as high-dimensional vectors typically used in soft prompting methods.

| Model | | Specifically | | --- | | designed for RAG | | | Maximun | | --- | | Compression Rate | | Approach | Plug-and-Play | Memory Efficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AutoCompressor [[14]] | ✗ | x15 | Soft Prompting | ✗ | ✗ |
| Gist [[58]] | ✗ | x26 | Soft Prompting | ✗ | ✗ |
| ICAE [[19]] | ✗ | x8 | Soft Prompting | ✓ | ✗ |
| LLMLingua [[28]] | ✗ | x20 | Prompt Editing | ✓ | ✓ |
| Selective Context [[45]] | ✗ | x5 | Prompt Editing | ✓ | ✓ |
| Token Elimination [[85]] | ✓ | x10 | Attention Filtering | ✓ | ✓ |
| FilCo [[74]] | ✓ | x2 | Prompt Editing | ✗ | ✓ |
| RECOMP [[79]] | ✓ | x16.6 | Prompt Editing | ✓ | ✓ |
| xRAG | ✓ | x178 | Modality Fusion | ✓ | ✓ |

*Table 5: Comparison between different compression methods from their setting to design principle.*

Appendix B Instructions for Paraphrase Pretraining
--------------------------------------------------

The list of instructions for Paraphrase Pretraining is shown in Table[6]. They present the same meaning with natural language variance.


*Table 6: Instructions used for Paraphrase Pretraining where [X] and [D] are placeholders for projected retrieval feature W(E) and document D.*

Appendix C Details for Instruction Tuning Dataset
-------------------------------------------------

After collecting the raw data from different categories, we use templates333[https://github.com/google-research/FLAN/blob/main/flan/templates.py](https://github.com/google-research/FLAN/blob/main/flan/templates.py "") from FLAN *[[77]]* to construct instruction tuning dataset. In Table[7] we list an overview and in Table[8] we list the detailed information for each subtask of our dataset. For QA tasks that lack an explicit context, we perform a retrieval operation within the corpus $\mathbb{D}$ to identify the most relevant document to serve as context. This approach is akin to the retrieval-augmented instruction tuning depicted in *[[54], [50]]*.

| Task Type | # Involved datasets | # Train | # Prompt | # Label |
| --- | --- | --- | --- | --- |
| Reading Comprehension | 7 | 488,344 | 447.62 | 30.34 |
| Summarization | 3 | 81,821 | 483.49 | 53.29 |
| Open Domain QA | 7 | 385,173 | 203.55 | 20.09 |

*Table 7: Overall statistics of Instruction Tuning dataset.*

| Task Type | Dataset | # Train | # Prompt Len | # Label Len |
| --- | --- | --- | --- | --- |
| ReadingComprehension | CoQA [[66]] | 7101 | 617.98 | 77.75 |
| | DROP [[16]] | 76098 | 356.06 | 3.86 |
| NarrativeQA [[39]] | 32747 | 702.39 | 7.86 |
| PubMedQA [[31]] | 1000 | 397.91 | 65.4 |
| QuAIL [[67]] | 10246 | 512.9 | 2.0 |
| SQuAD v2 [[65]] | 130319 | 214.54 | 6.87 |
| PwC [[19]] | 241564 | 571.35 | 53.07 |
| Open DomainQA | NQ [[40]] | 87925 | 203.62 | 5.976 |
| | TriviaQA [[32]] | 78785 | 216.1 | 6.49 |
| CommonsenseQA [[72]] | 9741 | 223.64 | 2.0 |
| WikiQA [[80]] | 1040 | 192.89 | 40.79 |
| YahooQA444<https://huggingface.co/datasets/yahoo_answers_qa> | 87358 | 196.56 | 56.7 |
| FreebaseQA [[30]] | 20353 | 218.49 | 4.87 |
| MSMarco [[5]] | 99994 | 194.82 | 15.91 |
| Summarization | CNN/DM [[69]] | 100000 | 616.99 | 63.37 |
| | SamSum [[20]] | 14731 | 187.87 | 29.12 |
| DialogSum [[10]] | 12460 | 247 | 37.61 |

*Table 8: Detailed data statistics for our Context-aware Instruction Tuning Dataset.*

Appendix D Implementation Details
---------------------------------

For the language models we use, Mixtral-8x7b is approximately 6.5 times larger in scale compared to Mistral-7b and features a divergent architectural approach—specifically, a dense versus mixture-of-experts design. For our assessments, we employed the instruction-tuned variants of these models.

Owing to efficiency constraints, we opted not to perform on-the-fly retrieval. Instead, we pre-constructed a retrieval index using the efficient and robust multi-vector retriever, ColBERT-v2 *[[68]]*, from which we retrieved the top-1 ranked document for inclusion in our instruction-tuning dataset and for the evaluation of downstream tasks. Subsequently, we re-encoded these documents using the embedding model of interest (e.g., SFR). This strategy allows us to iterate data-centric experiments quickly. All experiments are conducted on the a setup of 8xNvidia A100 GPUs.

In Table[9] and Table[10], we list the hyperparameters for Paraphrase Pretraining and Context-aware Instruction Tuning.

| Hyperparameter | Assignment |
| --- | --- |
| optimizer | AdamW |
| learning rate | 6e-3 |
| lr scheduler type | linear |
| warmup ratio | 0.03 |
| weight dacay | 0.0 |
| epochs | 1 |
| flash attention | True |
| batch size | 12 |
| gradient accumulation steps | 4 |
| num GPUs | 8 |
| max sequence length | 336 |
| max train samples | 2,000,000 |

*Table 9: Hyperparameters for Paraphrase Pretraining.*

| Hyperparameter | Assignment |
| --- | --- |
| optimizer | AdamW |
| learning rate | 2e-5 |
| lr scheduler type | linear |
| warmup ratio | 0.03 |
| weight dacay | 0.0 |
| epochs | 1 |
| KL $\alpha$ | 2.0 |
| KL temperature | 1.0 |
| flash attention | True |
| batch size | 4 |
| gradient accumulation steps | 2 |
| num GPUs | 8 |
| max sequence length | 1024 |
| max train samples | 955,338 |

*Table 10: Hyperparameters for Context-aware Instruction Tuning.*

Appendix E About different Embedding Models
-------------------------------------------

In our primary experiments, we use the SFR model as our default sentence embedding model. This section delves into the effects of different embedding models. We examine four universal text embedding models: E5-Mistral and E5-Large *[[73]]* alongside BGE-Large and BGE-Base *[[78]]*. Additionally, we assess two retrieval-specific models: Dragon *[[48]]* and DPR *[[34]]*. The configurations of the different retrievers and their MTEB scores555<https://huggingface.co/spaces/mteb/leaderboard>—a metric indicating their general sentence representation capability—are listed in Table[E]. To isolate the impact of potentially different retrieved documents, we ensure that all models utilize the same top-1 document. The performance is averaged over four question answering datasets. A general pattern is that embedding models with stronger sentence representation capabilities tend to further enhance the downstream performance. Remarkably, the Dragon model, despite being a BERT-base-sized retrieval-specific model, outperforms general text embedding models that are twice its size (BGE-Large).

| Model | | ModelSize | EmbeddingDim | UniversalEmbedding | MTEBScore | Average | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Performance | | | | | Resilience | Boost |
| Mistral-7b |  |  |  |  |  |  |  |  |
| w/o retrieval | |  |  |  |  | 37.2 | - | - |
| w retrieval | |  |  |  |  | 46.3 | 74.1% | 29.5% |
| \hdashline xRAG | | | w SFR | 7B | 4096 | ✓ | 67.56 | 44.5 | 82.3% | 22.2% |
|  | w E5-Mistral | 7B | 4096 | ✓ | 66.63 | 44.0 | 84.0% | 20.6% |
|  | w E5-Large | 335M | 1024 | ✓ | 62.25 | 42.1 | 80.2% | 19.6% |
|  | w BGE-Large | 335M | 1024 | ✓ | 64.23 | 41.6 | 78.2% | 19.8% |
|  | w BGE-Base | 109M | 768 | ✓ | 63.55 | 41.2 | 78.7% | 18.9% |
|  | w Dragon | 109M | 768 | ✗ | - | 42.1 | 84.2% | 16.9% |
|  | w DPR | 109M | 768 | ✗ | - | 40.5 | 77.4% | 18.2% |

*Table 11: Ablation on different sentence embedding models.*

Appendix F Analysis on Mistral-7b
----------------------------------

In Figure[6], we list the Resilience rate and Boost Rate on Mistral-7b model, which exhibit same pattern with Mixtral-8x7b model.

<img src='x6.png' alt='Refer to caption' title='' width='788' height='392' />

*Figure 6: Robustness and effectiveness analysis on 4 QA datasets with Mistral-7b model.*

Appendix G Limitations
----------------------

We discuss the limitations of our framework as follows:

* •

    In this work, we only consider the most commonly used retrieval system—single dense vector retrieval, while sparse retrieval methods such as BM25 or multi-vector retrieval methods like ColBERT are not included. We believe that combining these methods would be a promising direction for xRAG, as sparse vectors could complement dense vectors, and multi-vector retrieval would provide xRAG with more flexibility by not condensing all information into one token.

* •

    Currently, xRAG delivers decent performance when a relevant document is fetched; however, it lags behind RAG by a considerable margin in tasks that require reasoning (such as HotpotQA and FactKG). One possible reason is that during the training phase of xRAG, reasoning-relevant data is not provided. How to make xRAG a better reasoner remains our future work.

* •

    We only consider the Top-1 retrieval setting, while ensembling multiple relevant documents has been shown to be effective for RAG systems due to the complementary information contained in Top-K documents. We believe there is potential advantage for xRAG to scale to multi-document settings, as the input length of xRAG for multi-documents scales by a factor of 1, while for RAG, it scales by the document length factor.

Appendix H More interesting cases
---------------------------------

* •

    In Figure [7], we report a failure case of xRAG. In this case, retrieval alone is not enough to derive the final answer and the LLM is required to perform reasoning over retrieved document (the listed universities are all located in Switzerland).

* •

    An interesting example is shown in Figure [8], when the retrieved document is a list of a characters in the book Discworld, the RAG model would respond with a fictional character, while xRAG generate the right answer by focusing on the relevant part of the document.

* •

    In Figure[9], even when the retrieved document is relevant, RAG would still hallucinate while xRAG could generate the right answer based on the document.

* •

    In Figure[10], the retriever mistakenly fetch the wrong document (Phantom of the Opera of interest is a music rather than a file) and RAG would be misled while xRAG remain robust to generate the correct answer.

<img src='x7.png' alt='Refer to caption' title='' width='830' height='426' />

*Figure 7: Failure case of xRAG when reasoning is required to derive the final answer.*

<img src='x8.png' alt='Refer to caption' title='' width='830' height='458' />

*Figure 8: xRAG correctly locates the relevant part in a long document by selecting Diogenes of Sinope as the answer rather than Didactylos, a fictional character in the book Discworld.*

<img src='x9.png' alt='Refer to caption' title='' width='830' height='458' />

*Figure 9: xRAG correctly locates the relevant part in a long document while RAG would still hallucinate the wrong answer.*

<img src='x10.png' alt='Refer to caption' title='' width='830' height='458' />

*Figure 10: xRAG correctly locates the relevant part in a long document while RAG would still hallucinate the wrong answer.*

NeurIPS Paper Checklist
-----------------------

1. 1.

    Claims

2. Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?
3. Answer: [Yes]
4. Justification: Abstract and Section 1
5. Guidelines:

    * •
            The answer NA means that the abstract and introduction do not include the claims made in the paper.

        * •
            The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

        * •
            The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

        * •
            It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

6. 2.

    Limitations

7. Question: Does the paper discuss the limitations of the work performed by the authors?
8. Answer: [Yes]
9. Justification: Section[5.1] and Appendix[G]
10. Guidelines:

    * •
            The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

        * •
            The authors are encouraged to create a separate "Limitations" section in their paper.

        * •
            The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

        * •
            The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

        * •
            The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

        * •
            The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

        * •
            If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

        * •
            While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

11. 3.

    Theory Assumptions and Proofs

12. Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
13. Answer: [N/A]
14. Justification: the paper does not include theoretical results.
15. Guidelines:

    * •
            The answer NA means that the paper does not include theoretical results.

        * •
            All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

        * •
            All assumptions should be clearly stated or referenced in the statement of any theorems.

        * •
            The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

        * •
            Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

        * •
            Theorems and Lemmas that the proof relies upon should be properly referenced.

16. 4.

    Experimental Result Reproducibility

17. Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
18. Answer: [Yes]
19. Justification: in Appendix[D] we provide full details about our training process
20. Guidelines:

    * •
            The answer NA means that the paper does not include experiments.

        * •
            If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

        * •
            If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

        * •
            Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

        * •
            While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

            1. (a)
                    If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

                2. (b)
                    If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

                3. (c)
                    If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

                4. (d)
                    We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

21. 5.

    Open access to data and code

22. Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
23. Answer: [Yes]
24. Justification: all the models and data we use are publicly available and we carefully cite each paper
25. Guidelines:

    * •
            The answer NA means that paper does not include experiments requiring code.

        * •
            Please see the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.

        * •
            While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

        * •
            The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.

        * •
            The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

        * •
            The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

        * •
            At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

        * •
            Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

26. 6.

    Experimental Setting/Details

27. Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
28. Answer: [Yes]
29. Justification: Section[4.1]
30. Guidelines:

    * •
            The answer NA means that the paper does not include experiments.

        * •
            The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

        * •
            The full details can be provided either with the code, in appendix, or as supplemental material.

31. 7.

    Experiment Statistical Significance

32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
33. Answer: [Yes]
34. Justification: in Section[5.1]
35. Guidelines:

    * •
            The answer NA means that the paper does not include experiments.

        * •
            The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

        * •
            The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

        * •
            The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

        * •
            The assumptions made should be given (e.g., Normally distributed errors).

        * •
            It should be clear whether the error bar is the standard deviation or the standard error of the mean.

        * •
            It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

        * •
            For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

        * •
            If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

36. 8.

    Experiments Compute Resources

37. Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
38. Answer: [Yes]
39. Justification: in Appendix[D]
40. Guidelines:

    * •
            The answer NA means that the paper does not include experiments.

        * •
            The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

        * •
            The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

        * •
            The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

41. 9.

    Code Of Ethics

42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?
43. Answer: [Yes]
44. Justification: reviewed and confirmed
45. Guidelines:

    * •
            The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

        * •
            If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

        * •
            The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

46. 10.

    Broader Impacts

47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
48. Answer: [Yes]
49. Justification: in section[5.2]
50. Guidelines:

    * •
            The answer NA means that there is no societal impact of the work performed.

        * •
            If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

        * •
            Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

        * •
            The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

        * •
            The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

        * •
            If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

51. 11.

    Safeguards

52. Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
53. Answer: [N/A]
54. Justification: all the data and model we use is publicly available
55. Guidelines:

    * •
            The answer NA means that the paper poses no such risks.

        * •
            Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

        * •
            Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

        * •
            We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

56. 12.

    Licenses for existing assets

57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
58. Answer:[N/A]
59. Justification: in Section[C]
60. Guidelines:

    * •
            The answer NA means that the paper does not use existing assets.

        * •
            The authors should cite the original paper that produced the code package or dataset.

        * •
            The authors should state which version of the asset is used and, if possible, include a URL.

        * •
            The name of the license (e.g., CC-BY 4.0) should be included for each asset.

        * •
            For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

        * •
            If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

        * •
            For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

        * •
            If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

61. 13.

    New Assets

62. Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
63. Answer: [N/A]
64. Justification: n/a
65. Guidelines:

    * •
            The answer NA means that the paper does not release new assets.

        * •
            Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

        * •
            The paper should discuss whether and how consent was obtained from people whose asset is used.

        * •
            At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

66. 14.

    Crowdsourcing and Research with Human Subjects

67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
68. Answer: [N/A]
69. Justification: n/a
70. Guidelines:

    * •
            The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

        * •
            Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

        * •
            According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

71. 15.

    Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
73. Answer: [N/A]
74. Justification: n/a
75. Guidelines:

    * •
            The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

        * •
            Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

        * •
            We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

        * •
            For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
