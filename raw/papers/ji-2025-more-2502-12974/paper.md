Learning Refined Document Representations for Dense Retrieval via Deliberate Thinking
=====================================================================================

Yifan JiNortheastern UniversityShenyangChina[bigtailwolf001@gmail.com](mailto:bigtailwolf001@gmail.com), Zhipeng XuNortheastern UniversityShenyangChina[xuzhipeng@stumail.neu.edu.cn](mailto:xuzhipeng@stumail.neu.edu.cn), Zhenghao LiuNortheastern UniversityShenyangChina[liuzhenghao@mail.neu.edu.cn](mailto:liuzhenghao@mail.neu.edu.cn), Yukun YanTsinghua UniversityBeijingChina[yanyk.thu@gmail.com](mailto:yanyk.thu@gmail.com), Shi YuTsinghua UniversityBeijingChina[yushi17@foxmail.com](mailto:yushi17@foxmail.com), Yishan LiModelBest Inc.BeijingChina[liyishanthu@gmail.com](mailto:liyishanthu@gmail.com), Zhiyuan LiuTsinghua UniversityBeijingChina[liuzy@tsinghua.edu.cn](mailto:liuzy@tsinghua.edu.cn), Yu GuNortheastern UniversityShenyangChina[guyu@mail.neu.edu.cn](mailto:guyu@mail.neu.edu.cn), Ge YuNortheastern UniversityShenyangChina[yuge@mail.neu.edu.cn](mailto:yuge@mail.neu.edu.cn) and Maosong SunTsinghua UniversityBeijingChina[sms@tsinghua.edu.cn](mailto:sms@tsinghua.edu.cn)

(2025)

###### Abstract.

Recent dense retrievers increasingly leverage the robust text understanding capabilities of Large Language Models (LLMs), encoding queries and documents into a shared embedding space for effective retrieval.
However, most existing methods represent each document with a single embedding, which is less effective at capturing its multifaceted semantics and thereby limits matching accuracy. In this paper, we propose Deliberate Thinking based Dense Retriever (Debater), a novel approach that enhances document representations by incorporating a step-by-step thinking process. Debater introduces a Chain-of-Deliberation mechanism, which iteratively refines document embeddings through a continuous chain-of-thought. To integrate information from various thinking steps, Debater further employs a Self Distillation mechanism that identifies and fuses the most informative steps into a unified embedding. Experimental results show that Debater significantly outperforms existing methods across several retrieval benchmarks, demonstrating superior accuracy and robustness. All codes and datasets are available at https://github.com/OpenBMB/DEBATER.

Dense Retrieval, Large Language Models, Deliberate Thinking, Knowledge Distillation

††journalyear: 2025††ccs: Information systems Information retrieval

1. Introduction
----------------

Dense retrieval models encode both queries and documents into a dense embedding space and measure their similarity to retrieve relevant documents*(Karpukhin et al., [2020]; Zhao et al., [2024]; Xiong et al., [2021])*, demonstrating strong effectiveness in various downstream NLP tasks, such as open-domain question answering*(Chen and Yih, [2020])*, fact verification*(Liu et al., [2020])*, and web search*(Chen et al., [2024])*. However, recent findings have shown that dense retrievers suffer from significant performance degradation when applied to new tasks or domains*(Su et al., [2023])*, raising concerns about their versatility*(Luo et al., [2024]; Khramtsova et al., [2024])*.

<img src='x1.png' alt='Refer to caption' title='' width='789' height='1065' />

*Figure 1. The Illustration of Our Deliberate Thinking based Dense Retriever (Debater). Debater leverages the reasoning capability of LLM to conduct fine-grained document representations for retrieval.*

Large Language Models, such as ChatGPT*(Achiam et al., [2023])* and LLaMA*(Touvron et al., [2023])*, have demonstrated extraordinary emergent capabilities*(Wei et al., [2022a]; Zhao et al., [2023])*, inspiring researchers to leverage them to enhance the task and domain generalization of dense retrievers*(Zhu et al., [2023]; Khramtsova et al., [2024])*. In particular, existing work has focused on prompting LLMs to generate dense representations for retrieval*(Zhuang et al., [2024])*. These methods typically use task-specific instructions or in-context demonstrations to guide LLMs in generating task- and domain-aware embeddings. To learn more tailored representations for dense retrieval, researchers further focus on optimizing LLM-based retrievers using relevance labels*(Ma et al., [2024]; Neelakantan et al., [2022]; Li et al., [2025])*. These methods exploit the superior reasoning abilities of LLMs, achieving impressive performance across various retrieval tasks*(Wang et al., [2024a]; Zhu et al., [2023]; Luo et al., [2024])*. Recent studies suggest that LLMs pose strong reasoning capability, particularly implemented by their step-by-step thinking*(Kudo et al., [2024]; Wei et al., [2022b])*. LLM-based retrievers typically rely on the hidden state of the end-of-sequence token as both query and document representations. Nevertheless, only relying on one embedding usually shows less effectiveness in representing documents from different views that can match queries*(Zhang et al., [2022]; Khattab and Zaharia, [2020])*.

In this paper, we propose a Deliberate Thinking based Dense Retriever (Debater) model to learn more effective document representations through deliberately thinking step-by-step before retrieval. As shown in Figure[1], our method stimulates LLMs to conduct the reasoning process, enabling them to generate more fine-grained document representations for retrieval. Specifically, Debater introduces the Chain-of-Deliberation mechanism to encourage LLMs to conduct deliberate thinking by autoregressively decoding the document representations. Then Debater utilizes the Self Distillation mechanisms to gather all information from previous steps and compress them into the document embedding at the last step.

Our experiments demonstrate that Debater achieves retrieval performance comparable to, or even surpassing, that of baseline methods implemented by larger-scale LLMs, demonstrating its effectiveness. Further analysis reveals that Chain-of-Deliberation and Self-Distillation play complementary roles in Debater, and that increasing the number of reasoning steps appropriately can benefit LLM-based dense retrieval models. The document representations produced by Debater are progressively refined through iterative thinking, where each step autoregressively generates intermediate representations. By incorporating Self-Distillation, the model is able to extract different salient information at different reasoning steps and integrate them into a more comprehensive and semantically rich final representation.

2. Related Work
----------------

Dense retrieval*(Karpukhin et al., [2020]; Xiong et al., [2021]; Su et al., [2023])* has proven effective in various NLP downstream tasks*(Liu et al., [2020]; Chen et al., [2024]; Guu et al., [2020])*. However, the versatility of dense retrievers remains a challenge that hinders their progress*(Luo et al., [2024]; Lee et al., [2024])*, particularly their inability to generate task- and domain-specific embeddings and return suitable results*(Su et al., [2023]; Luo et al., [2024]; Tao et al., [2024])*. To address this limitation, prior work has focused on conducting fine-grained data curation to fine-tune dense retrievers with multi-task instructions*(Su et al., [2023]; Asai et al., [2023])*. However, obtaining high-quality relevance labels can be difficult for training dense retrievers*(Yu et al., [2022]; Gao et al., [2023]; Wang et al., [2024a])*.

Recent research has shifted towards using LLMs as the backbone for dense retrievers*(Tao et al., [2024])*, thriving on their strong emergence capabilities. Some studies attempt to directly prompt LLMs to generate embeddings for retrieval*(Zhuang et al., [2024])*. However, prompt-based approaches cannot leverage pre-existing retrieval signals, limiting their effectiveness*(Zhu et al., [2023])*. In contrast, recent efforts have focused on fine-tuning LLMs for dense retrieval tasks*(Wang et al., [2024a]; Ma et al., [2024]; Li et al., [2024])*, or designing additional pretraining tasks to transform LLMs into dense retrievers*(BehnamGhader et al., [2024])*, achieving strong retrieval performance and generalization capabilities. However, existing methods typically extract the last hidden state of the end-of-sequence token as the dense representation*(Ma et al., [2024]; Luo et al., [2024])*, which is not always effective for fully representing documents from different perspectives to match queries*(Zhang et al., [2022]; Khattab and Zaharia, [2020])*. The exploration of different document representations, such as leveraging the reasoning ability of LLMs, remains an underexplored area.

<img src='x2.png' alt='Refer to caption' title='' width='789' height='393' />

*Figure 2. The Architecture of Deliberate Thinking based Dense Retriever (Debater).*

To enhance the reasoning capability of LLMs, one approach is to generate intermediate reasoning steps using Chain-of-Thought (CoT)*(Wei et al., [2022b])* or its variants*(Chen et al., [2023]; Zhang et al., [2024])*. CoT allows LLMs to delay final answers by engaging in reasoning*(Kudo et al., [2024])*, improving response accuracy*(Wei et al., [2022b]; Chu et al., [2024])*. However, these approaches operate within the language space and often require generating tens or even hundreds of additional tokens, which can hinder their ability to meet the latency requirements of dense retrievers. Current research is exploring the integration of CoT reasoning into a continuous latent space*(Hao et al., [2024]; Xie et al., [2024])* to enhance computational efficiency. Building on these advancements, our Debater focuses on latent reasoning chains, encouraging LLM-based retrievers to think step-by-step to enhance the dense representations of documents.

3. Methodology
---------------

In this section, we introduce our Deliberate Thinking based Dense Retriever (Debater).
We first introduce the preliminary of LLM-based dense retrieval (Sec.[3.1]). Then we describe our deliberation thinking based embedding learning method used by Debater(Sec.[3.2]).

### 3.1. Preliminary of Dense Retrieval with Large Language Models as Foundations

Given a query $q$ and a document collection $\mathcal{D}$, the goal of the retrieval task is to identify a subset of documents most relevant to the query.

LLM-based dense retrievers typically map both the query $q$ and document $d$ into a shared latent space for retrieval, where the query embedding $h^{q}$ and document embedding $h^{d}$ are defined as:

| (1) |  | $\displaystyle h^{q}$ | $\displaystyle\=\text{LLM}(q,\text{{</s>}})[-1],$ |  |
| --- | --- | --- | --- | --- |
| | | $\displaystyle h^{d}$ | $\displaystyle\=\text{LLM}(d,\text{{</s>}})[-1].$ | |

The ranking score $f(q,d)$ between the query embedding $h^{q}$ and the document embedding $h^{d}$ is calculated as:

| (2) |  | $f(q,d)\=sim(h^{q},h^{d}),$ |  |
| --- | --- | --- | --- |

where sim denotes the similarity function. In Debater, we use cosine similarity to measure the similarity between queries and documents, which is also employed in previous works*(Wang et al., [2024a]; BehnamGhader et al., [2024])*. Subsequently, we contrastively train the LLM to maximize the probability of retrieving the positive document $d^{+}$ over the negative document $d^{-}$:

| (3) |  | $\displaystyle p(d^{+}|q,{d^{+}}\cup\mathcal{D}^{-})\=\frac{e^{f(q,d^{+})}}{e^{f(q,d^{+})}+\sum_{d^{-}\in\mathcal{D}^{-}}e^{f(q,d^{-})}},$ |  |
| --- | --- | --- | --- |

where $\mathcal{D}^{-}$ denotes the set of negative documents, typically obtained via in-batch negative sampling*(Karpukhin et al., [2020])*.

Current LLM-based dense retrievers typically use the last hidden state corresponding to the end-of-sequence token (</s>) as the dense representation. However, they do not fully exploit the reasoning capabilities of LLMs, which helps conduct more effective representations by learning information from diverse views of documents.

### 3.2. Enhancing Dense Retriever through Deliberate Thinking

In this subsection, we introduce the Deliberate Thinking based Dense Retriever (Debater), which aims to unleash the reasoning ability of LLMs and generate more fine-grained document representations. As shown in Figure[2], Debater consists of two modules to enhance the LLM-based dense retriever: Chain-of-Deliberation (CoD) and Self Distillation (SD).

Chain-of-Deliberation. To enhance these LLM-based dense retrievers, Debater introduces the Chain-of-Deliberation (CoD) approach, which delays the computation of document embeddings by performing several steps of reasoning.

Specifically, CoD incorporates a sequence of prompt tokens ${t_{1},t_{2},\dots,t_{m}}$ to stimulate the reasoning capability of LLMs when representing the document $d$. These tokens ${t_{1},t_{2},\dots,t_{m-1}}$ serve as intermediate thinking steps, encouraging the model to think step-by-step before producing the final document embedding at the $m$-th step:

| (4) |  | $h_{m}^{d}\=\text{LLM}(X,t_{1},t_{2},\dots,t_{m-1},t_{m}),$ |  |
| --- | --- | --- | --- |

where $m$ is a hyperparameter to control the thinking depth. An appropriate choice of $m$ is crucial to avoid overthinking or under-optimization.

During training, we first calculate the similarity score between query representation $h^{q}$ and the document representation $h^{d}_{i}$ at the $i$-th thinking step:

| (5) |  | $f(q,d(t_{i}))\=sim(h^{q},h^{d}_{i}).$ |  |
| --- | --- | --- | --- |

Next, we gather all similarity scores using the decoded document representations ${h^{d}_{1},...,h^{d}_{m}}$. We then select the most useful thinking step from CoD and use the corresponding embedding as the document representation to compute the training loss. The relevance scores $f_{\text{max}}(q,d)$ between the query and the document are computed as:

| (6) |  | $f_{\text{max}}(q,d)\=\max_{1\leq i\leq m}sim(h^{q},h^{d}_{i}),$ |  |
| --- | --- | --- | --- |

The LLM is optimized by minimizing the contrastive training loss:

| (7) |  | $\mathcal{L}_{c}\=-\text{log}\frac{e^{f_{\text{max}}(q,{d^{+}})}}{e^{f_{\text{max}}(q,{d^{+}})}+\sum_{d^{-}\in\mathcal{D}^{-}}e^{f_{\text{max}}(q,{d^{-}})}}.$ |  |
| --- | --- | --- | --- |

Self Distillation. Although the final token of the Chain-of-Deliberation aggregates information from all thinking steps through autoregressive decoding, it may overlook crucial reasoning cues presented in embeddings decoded at earlier steps.

To address this, we introduce Self Distillation (SD), a strategy for distilling knowledge from different thinking steps into the final document representation $h^{d}_{m}$. Specifically, we use the most informative thinking step as the teacher to guide the representation learning of the final token in CoD, thereby enhancing the document representation.

For the query $q$, we compute the ranking probability of the $i$-th document $d_{i}$ in the document collection $\tilde{\mathcal{D}}\={d^{+}}\cup\mathcal{D}^{-}$ as:

| (8) |  | $P(d_{i}|q)\=\frac{e^{f_{\text{max}}(q,d_{i})}}{\sum_{d_{j}\in\tilde{\mathcal{D}}}e^{f_{\text{max}}(q,d_{j})}},$ |  |
| --- | --- | --- | --- |

where $|\tilde{\mathcal{D}}|\=k$. This yields a probability distribution $P(\tilde{\mathcal{D}}|q)$ over the $k$ documents:

| (9) |  | $P(\tilde{\mathcal{D}}|q)\=\left[P(d_{1}|q),P(d_{2}|q),\dots,P(d_{k}|q)\right].$ |  |
| --- | --- | --- | --- |

Each value $P(d_{i}|q)$ represents the ranking probability of the $i$-th document $d_{i}$ using the document representations from all thinking steps ${h^{d}_{1},...,h^{d}_{m}}$ of CoD that yield the highest similarity with the query. Concurrently, we compute the rank probability of $d_{i}$ using the last-token embedding $h^{d}_{m}$ from CoD:

| (10) |  | $Q(d_{i}(t_{m})|q)\=\frac{e^{f(q,d_{i}(t_{m}))}}{\sum_{d_{j}\in{d^{+}}\cup\mathcal{D}^{-}}e^{f(q,d_{j}(t_{m}))}}.$ |  |
| --- | --- | --- | --- |

Then we can obtain the ranking probability distribution $Q(\tilde{\mathcal{D}}|q)$ as well:

| (11) |  | $Q(\tilde{\mathcal{D}}|q)\=\left[Q(d_{1}|q),Q(d_{2}|q),\dots,Q(d_{k}|q)\right].$ |  |
| --- | --- | --- | --- |

We then minimize the Kullback-Leibler (KL) divergence between two probability distributions $P(\tilde{\mathcal{D}}|q)$ and $Q(\tilde{\mathcal{D}}|q)$:

| (12) |  | $\mathcal{L}_{t}\={P(\tilde{\mathcal{D}}|q)\cdot\log\frac{P(\tilde{\mathcal{D}}|q)}{Q(\tilde{\mathcal{D}}|q)}},$ |  |
| --- | --- | --- | --- |

where the Self Distillation loss $\mathcal{L}_{t}$ optimizes the document representation $h_{m}^{d}$ by capturing more crucial matching signals from all thinking steps.

Training. Finally, we train our Debater models by minimizing the following loss $\mathcal{L}$:

| (13) |  | $\mathcal{L}\=\mathcal{L}_{c}+\mathcal{L}_{t},$ |  |
| --- | --- | --- | --- |

where $\mathcal{L}_{c}$ optimizes the CoD, and $\mathcal{L}_{t}$ is used to distill crucial information from the thinking steps into the final dense representation of the document. This combined loss allows Debater to leverage both thinking depth and self-knowledge distillation to improve retrieval performance.

4. Experimental Methodology
----------------------------

In this section, we describe the datasets, evaluation metrics, baselines, and implementation details for our experiments.

Datasets. We train all Debater models using the public portion of the E5 dataset*(Wang et al., [2024a]; Springer et al., [2025])*, a carefully curated collection of approximately 1.5M publicly available samples. Table[1] presents its statistics, and the full dataset is accessible on their website111[https://github.com/jakespringer/echo-embeddings](https://github.com/jakespringer/echo-embeddings "").

*Table 1. Data Statistics of E5 Dataset. We show the composition and distribution of E5 Dataset.*

| Dataset | #Samples | Proportion |
| --- | --- | --- |
| ELI5 (Fan et al., [2019]) | 32,547 | 2.16% |
| HotpotQA (Yang et al., [2018]) | 90,447 | 5.99% |
| FEVER (Thorne et al., [2018]) | 101,578 | 6.73% |
| MIRACL (Zhang et al., [2023]) | 32,561 | 2.16% |
| MSMARCO Passage Ranking (Bajaj et al., [2016]) | 249,592 | 16.53% |
| MSMARCO Document Ranking (Bajaj et al., [2016]) | 73,400 | 4.86% |
| NQ (Kwiatkowski et al., [2019]) | 100,231 | 6.64% |
| NLI (Gao et al., [2021]) | 277,230 | 18.36% |
| SQuAD (Rajpurkar et al., [2016]) | 87,599 | 5.80% |
| TriviaQA (Joshi et al., [2017]) | 73,346 | 4.86% |
| Quora Duplicate Questions (DataCanary et al., [2017]) | 101,762 | 6.74% |
| Mr-TyDi (Zhang et al., [2021]) | 48,715 | 3.23% |
| DuReader (He et al., [2018]) | 86,395 | 5.72% |
| T2Ranking (Xie et al., [2023]) | 154,294 | 10.22% |

*Table 2. Data Statistics of the BEIR Benchmark. We show the task type, along with the number of queries and passages for each dataset.*

| Dataset | Task | #Query | #Corpus |
| --- | --- | --- | --- |
| TREC-COVID | Bio-Medical | 50 | 171,332 |
| NFCorpus | Information | 323 | 3,633 |
| NQ | Question | 3,452 | 2,681,468 |
| HotpotQA | Answering | 7,405 | 5,233,329 |
| FiQA-2018 | (QA) | 648 | 57,638 |
| ArguAna | Argument | 1,406 | 8,674 |
| Touché-2020 | Retrieval | 49 | 382,545 |
| CQADupStack | Duplicate-Question | 13,145 | 457,199 |
| Quora | Retrieval | 10,000 | 522,931 |
| DBPedia | Entity-Retrieval | 400 | 4,635,922 |
| SCIDOCS | Citation-Prediction | 1,000 | 25,657 |
| FEVER | Fact Checking | 6,666 | 5,416,568 |
| Climate-FEVER | Wikipedia | 1,535 | 5,416,593 |
| SciFact | Scientific | 300 | 5,183 |

The retrieval effectiveness of Debater is evaluated on the BEIR benchmark*(Thakur et al., [2021])*, which includes 18 datasets that span a variety of domains.
Our evaluation focuses on the 14 publicly available datasets used for the zero-shot retrieval task. The statistics for these datasets are provided in Table[2].

Evaluation Metrics. To evaluate the retrieval effectiveness of Debater, we use nDCG@10, the standard metric for the BEIR benchmark. The metric implementation follows the pytrec-eval toolkit*(Gysel and de Rijke, [2018])*, which is consistent with prior work*(Zhu et al., [2023])*.

Baselines. We compare Debater with several baseline retrievers implemented with different language models. GTR*(Ni et al., [2022])* employs large dual encoder-only models to build a dense retriever, while SGPT*(Muennighoff, [2022])* trains dense retrieval models using decoder-only architectures. Emb-V3222[https://cohere.com/blog/introducing-embed-v3](https://cohere.com/blog/introducing-embed-v3 "") is a commercial text retrieval model provided by Cohere. PromptReps*(Zhuang et al., [2024])* directly prompts LLMs to generate dense representations without supervision. RepLLaMA*(Ma et al., [2024])* and E5-Mistral*(Wang et al., [2024b])* fine-tune LLMs as dense retrievers, using the hidden state of an additional end-of-sequence token to represent the input context. Notably, E5-Mistral is trained on the same dataset as Debater but leverages a larger foundational model.

*Table 3. Overall Retrieval Performances on BEIR Benchmark. † indicates the 11 most representative BEIR tasks used in CPT*(Neelakantan et al., [2022])* and Avg CPT Sub reflects the average performance across these tasks.*

| Method ($\rightarrow$) | BM25 | GTR | SGPT | PromptReps | RepLLaMA | Emb-V3 | E5-Mistral | Debater | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model Size ($\rightarrow$) | / | 4.8B | 5.8B | 8B | 7B | / | 7B | 2.4B | 4B |
| TREC-COVID† | 0.656 | 0.501 | 0.873 | 0.693 | 0.847 | 0.794 | 0.708 | 0.795 | 0.836 |
| NFCorpus† | 0.325 | 0.342 | 0.362 | 0.330 | 0.378 | 0.336 | 0.353 | 0.378 | 0.399 |
| NQ | 0.329 | 0.568 | 0.524 | 0.431 | 0.624 | 0.580 | 0.482 | 0.560 | 0.561 |
| HotpotQA† | 0.603 | 0.599 | 0.593 | 0.471 | 0.685 | 0.668 | 0.756 | 0.678 | 0.678 |
| FiQA† | 0.236 | 0.467 | 0.372 | 0.324 | 0.458 | 0.388 | 0.545 | 0.434 | 0.462 |
| ArguAna† | 0.414 | 0.540 | 0.514 | 0.330 | 0.486 | 0.508 | 0.625 | 0.567 | 0.562 |
| Touché-2020† | 0.367 | 0.256 | 0.254 | 0.218 | 0.305 | 0.319 | 0.191 | 0.211 | 0.250 |
| Quora† | 0.789 | 0.892 | 0.846 | 0.805 | 0.868 | 0.881 | 0.895 | 0.886 | 0.886 |
| DBPedia† | 0.313 | 0.408 | 0.399 | 0.377 | 0.437 | 0.410 | 0.477 | 0.430 | 0.432 |
| SCIDOCS | 0.158 | 0.161 | 0.197 | 0.176 | 0.181 | 0.181 | 0.190 | 0.197 | 0.212 |
| FEVER† | 0.753 | 0.740 | 0.783 | 0.711 | 0.834 | 0.876 | 0.731 | 0.859 | 0.857 |
| Climate-FEVER† | 0.213 | 0.267 | 0.305 | 0.214 | 0.310 | 0.289 | 0.252 | 0.303 | 0.294 |
| SciFact† | 0.665 | 0.662 | 0.747 | 0.657 | 0.756 | 0.667 | 0.744 | 0.735 | 0.743 |
| CQADupStack | 0.299 | 0.399 | 0.381 | / | / | 0.389 | / | 0.431 | 0.428 |
| Avg CPT sub† | 0.485 | 0.516 | 0.550 | 0.466 | 0.579 | 0.558 | 0.571 | 0.571 | 0.582 |
| Avg | 0.437 | 0.486 | 0.511 | / | / | 0.520 | / | 0.533 | 0.543 |

Implementation Details. We initialize the Debater models with MiniCPM-2.4B and MiniCPM-4B*(Hu et al., [2024])*. All Debater models are trained for 1,000 steps using the AdamW optimizer with a batch size of 256. The learning rate follows a cosine decay schedule, with a warm-up phase covering the first 3% of the total iterations, peaking at 2e-4. We train Debater using hybrid negatives, including one hard negative from the E5 dataset and seven in-batch negatives. The CoD length for all Debater models is set to 8. Debater is implemented using the OpenMatch toolkit*(Yu et al., [2023])*, with flash-attention*(Dao et al., [2022])* and LoRA*(Hu et al., [2022])* enabled to mitigate memory constraints and improve computational efficiency.

5. Evaluation Results
----------------------

In this section, we first evaluate the retrieval effectiveness of Debater and then conduct ablation studies to show the roles of different modules in Debater. Then we analyze the characteristics of learned embeddings during thinking step by step.

### 5.1. Overall Performance

The overall performance of Debater and the baseline retrievers is shown in Table[3].

Overall, Debater outperforms all baseline retrievers in terms of average retrieval accuracy on BEIR, achieving more than a 2% improvement. This highlights its effectiveness in enhancing the representation capability of LLMs for retrieval. Compared to the prompt-based method PromptReps, these fine-tuned LLM-based methods consistently show improvements, indicating that LLMs also benefit from supervised training to learn more tailored embeddings for retrieval.
When compared to E5-Mistral-7B, which is trained on the same E5 corpus as Debater, Debater significantly improves retrieval performance on TREC-COVID, NQ, and FEVER, demonstrating its capability across diverse question-answering scenarios.
Notably, when implemented with MiniCPM-2.4B, Debater achieves retrieval performance comparable to that of larger 7B-scale LLM-based dense retrievers while utilizing only 35% of the parameters. This demonstrates that Debater can enhance the representation learning capabilities of smaller-scale LLMs, rather than relying on larger foundational LLMs.
Furthermore, when implemented with MiniCPM-4B, the retrieval performance of Debater is improved by 1%, demonstrating that larger models effectively enhance the retrieval capabilities of Debater.

### 5.2. Ablation Study

As shown in Table[4], we conduct ablation studies to further investigate the roles of Chain-of-Deliberation (CoD) and Self Distillation (SD) modules in Debater.

*Table 4. Ablation Study of Deliberate Thinking based Dense Retriever (Debater). We train three Debater variations: MiniCPM w/ SD, MiniCPM w/ CoD and vanilla MiniCPM. ${\dagger}$, ${\ddagger}$, and ${\mathsection}$ indicate statistically significant improvements over MiniCPM w/ SD†, MiniCPM w/ CoD‡ and vanilla MiniCPM§.*

| Method | MiniCPM-2.4B | | | | MiniCPM-4B | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Vanilla | w/ SD | w/ CoD | Debater | Vanilla | w/ SD | w/ CoD | Debater |
| TREC-COVID | 0.728 | 0.822 | 0.805 | 0.795 | 0.747 | 0.742 | 0.791 | 0.836 |
| NFCorpus | 0.368 | 0.368 | 0.371 | 0.378 | 0.379 | 0.388 | 0.378 | 0.399 |
| NQ | 0.545 | 0.531 | 0.568 | 0.560 | 0.533 | 0.544 | 0.508 | 0.561 |
| HotpotQA | 0.670 | 0.656 | 0.669 | 0.678 | 0.564 | 0.597 | 0.631 | 0.678 |
| FiQA-2018 | 0.406 | 0.409 | 0.430 | 0.434 | 0.428 | 0.428 | 0.413 | 0.462 |
| ArguAna | 0.561 | 0.526 | 0.547 | 0.560 | 0.569 | 0.575 | 0.497 | 0.562 |
| Touché-2020 | 0.202 | 0.250 | 0.219 | 0.211 | 0.195 | 0.208 | 0.237 | 0.250 |
| Quora | 0.880 | 0.788 | 0.882 | 0.886 | 0.886 | 0.890 | 0.883 | 0.886 |
| SCIDOCS | 0.191 | 0.194 | 0.195 | 0.197 | 0.210 | 0.214 | 0.198 | 0.212 |
| Climate-FEVER | 0.277 | 0.310 | 0.258 | 0.303 | 0.211 | 0.189 | 0.184 | 0.294 |
| SciFact | 0.715 | 0.720 | 0.733 | 0.735 | 0.731 | 0.737 | 0.730 | 0.743 |
| Avg | 0.504 | 0.507 | 0.516 | 0.522†§ | 0.496 | 0.501 | 0.495 | 0.535†‡§ |

We compare our Debater with three variations, using MiniCPM-2.4B and MiniCPM-4B as the foundations for building dense retrievers. Both vanilla LLM and MiniCPM w/ CoD models represent documents using the hidden state of the last token and train query and document representations using contrastive training. The key difference between them lies in that MiniCPM w/ CoD performs additional CoD steps before obtaining the document representation. Besides, MiniCPM w/ SD is identical to Debater but removes the CoD steps when generating the document representation.

Compared to vanilla LLM, MiniCPM w/ SD shows almost identical retrieval performance, indicating that relying solely on a few last tokens in the input sequence does not effectively enhance the document representations. This suggests that the special tokens used in CoD serve as prompts that stimulate LLMs to produce more meaningful embeddings.
On the other hand, MiniCPM w/ CoD still yields a limited improvement over the vanilla LLM, demonstrating that directly incorporating CoD in representing documents fails to enhance the representation ability of LLMs. After incorporating the Self Distillation mechanism, MiniCPM w/ CoD achieves further improvements, demonstrating its importance in capturing semantics from the different deliberative steps of CoD to optimize the last token as the document representation.
Additionally, when using contrastive training to optimize LLMs, the 4B-scale retrieval model performs worse than the 2.4B-scale model. Notably, Debater not only mitigates this performance degradation but also leads to an additional 1.3% improvement, highlighting the effectiveness and robustness of Debater.

### 5.3. Effectiveness of Chain-of-Deliberation with Different Thinking Depths

In this subsection, we explore how thinking depth affects the effectiveness of Debater. Specifically, we vary the length of the Chain-of-Deliberation (CoD) to train several Debater-2.4B and Debater-4B models and evaluate their retrieval performance on TREC-COVID and FiQA.

<img src='image/TC.png' alt='Refer to caption' title='' width='287' height='272' />

*(a) TREC-COVID.*

<img src='image/FQ.png' alt='Refer to caption' title='' width='287' height='272' />

*(b) FiQA.*

<img src='image/NF.png' alt='Refer to caption' title='' width='287' height='272' />

*(c) NFCorpus.*

<img src='image/SciDocs.png' alt='Refer to caption' title='' width='287' height='272' />

*(d) SCIDOCS.*

*Figure 3. Retrieval Performance of Debater with Different Thinking Depths. We set the length of the CoD to train different Debater models and evaluate them on different subsets of BEIR.*

As illustrated in Figure[3], both Debater-2.4B and Debater-4B exhibit significant and consistent improvements in retrieval performance as the thinking depth increases to 4.
This indicates that an appropriate thinking depth effectively activates the reasoning capabilities of LLM-based retrievers, enabling them to generate finer-grained representations of documents.
When the thinking depth is further extended to 8, Debater-2.4B reaches a plateau, indicating that it may be nearing its capacity to process more complex or prolonged deliberations.
In contrast, Debater-4B continues to show incremental improvements when the length of CoD extends to 8, indicating that larger models benefit more from extended reasoning due to their stronger ability to integrate and retain detailed intermediate steps.
Nonetheless, further increasing the CoD beyond a certain point (e.g., 12) may lead to overthinking and result in performance degradation for both model sizes. These observations demonstrate that while moderate depths effectively boost retrieval accuracy, excessively long chains can dilute the benefits and introduce unnecessary computational overhead. Overall, these findings underscore the importance of carefully tuning the thinking depth for LLM-based retrievers.

<img src='image/Treccovid-step.png' alt='Refer to caption' title='' width='290' height='274' />

*(a) TREC-COVID.*

<img src='image/FIQA-step.png' alt='Refer to caption' title='' width='290' height='274' />

*(b) FiQA.*

<img src='image/NF-step.png' alt='Refer to caption' title='' width='290' height='274' />

*(c) NFCorpus.*

<img src='image/SCIDOCS-step.png' alt='Refer to caption' title='' width='290' height='274' />

*(d) SCIDOCS.*

*Figure 4. Performance of Debater at Different Thinking Steps. We collect all documents from each thinking step to demonstrate the retrieval performance of Debater across different stages of reasoning.*

### 5.4. Retrieval Performance of CoD-Generated Document Representations

In this subsection, we investigate how the Chain-of-Deliberation (CoD) enhances the representation capability of LLM-based retrievers. Specifically, we evaluate the quality of embeddings produced at different thinking steps in CoD and assess their retrieval performance individually.

As shown in Figure[4], early steps (e.g., 1–2) produce relatively weak results, and performance may even drop.This suggests that initial embeddings, based on minimal deliberation, may lack the nuanced understanding required for effective retrieval. However, as the number of thinking steps increases, performance generally improves, indicating that more deliberation leads to more refined embeddings for retrieval. These results demonstrate that Debater leverages the CoD mechanism to refine document representations step by step by reading information from previous steps. On the other hand, the gains eventually plateau, suggesting that once embeddings become sufficiently fine-grained, further deliberation provides limited benefits while increasing computational cost.

### 5.5. Characteristics of the Embeddings Generated by Debater

In this subsection, we analyze the embeddings learned by Debater from CoD. Specifically, we compute the average cosine similarity scores of embeddings generated at different positions in FiQA to understand how embeddings at various stages affect the final representation used for retrieval.

<img src='image/hot-wosdd.png' alt='Refer to caption' title='' width='287' height='295' />

*(a) Debater w/o SD.*

<img src='image/hot.png' alt='Refer to caption' title='' width='287' height='295' />

*(b) Debater.*

*Figure 5. Similarity Relationship Between Adjacent Position Embeddings. Darker blue indicates a higher similarity score.*

<img src='image/Sim-Emb7-wosdd.png' alt='Refer to caption' title='' width='291' height='270' />

*(a) Debater w/o SD.*

<img src='image/Sim-Emb7.png' alt='Refer to caption' title='' width='291' height='268' />

*(b) Debater.*

*Figure 6. Similarity Scores Between the First Seven Embeddings and the Last Embedding. The last embedding is used as the representation of documents for retrieval.*

*Table 5. Retrieval Performance of Debater across Different Model Scales.*

| Method | SmolLM2-135M | | SmolLM2-360M | | MiniCPM-1B | | Qwen2.5-7B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Vanilla | Debater | Vanilla | Debater | Vanilla | Debater | Vanilla | Debater |
| TREC-COVID | 0.681 | 0.696 | 0.721 | 0.646 | 0.678 | 0.778 | 0.770 | 0.829 |
| NFCorpus | 0.286 | 0.299 | 0.317 | 0.341 | 0.359 | 0.367 | 0.386 | 0.398 |
| FiQA-2018 | 0.223 | 0.230 | 0.292 | 0.378 | 0.400 | 0.402 | 0.483 | 0.487 |
| ArguAna | 0.455 | 0.428 | 0.480 | 0.481 | 0.548 | 0.534 | 0.561 | 0.577 |
| Touché-2020 | 0.199 | 0.248 | 0.187 | 0.208 | 0.162 | 0.227 | 0.242 | 0.235 |
| Quora | 0.841 | 0.821 | 0.869 | 0.874 | 0.884 | 0.883 | 0.891 | 0.893 |
| SCIDOCS | 0.135 | 0.135 | 0.163 | 0.167 | 0.182 | 0.192 | 0.227 | 0.235 |
| SciFact | 0.530 | 0.554 | 0.576 | 0.674 | 0.704 | 0.705 | 0.763 | 0.767 |
| Avg | 0.419 | 0.426 | 0.451 | 0.471 | 0.490 | 0.511 | 0.540 | 0.553 |

*Table 6. Case Studies. We present two cases from FiQA and TREC-COVID, and show the top 1 passage retrieved from MiniCPM and Debater. Different colors are used to annotate important content: Blue denotes critical information from the query, while Orange highlights supporting details from the passage retrieved by Debater.*

| Case #1 in FiQA |  |
| --- | --- |
| Query | In the US, is it a good idea to hire a tax consultant for doing taxes? |
| MiniCPM: (Less Relevant) | This may not exactly answer your question but, as a small business owner, I would highly recommend having a professional handle your taxes … I would recommend this especially if this is how you make your primary income, you can always write it off as a business expense. |
| Debater: (Most Relevant) | Whether you do decide to go with a tax advisor or not, be sure to do some research on your own. When we moved to the US about 5 years ago, I did find the taxes here pretty complicated and confusing ··· After all, they are also humans prone to mistakes and your taxes are your liability in the end. My suggestion is to start with a good tool that supports tax filing for non-residents. Most of them provide a step-by-step QA based tool. As you go through the steps, Google each question you don’t understand. It may take more time than hiring a tax advisor directly but in the end it will all be worth it. |
| Case #2 in TREC-COVID |  |
| Query | What is the mechanism of inflammatory response and pathogenesis of COVID-19 cases? |
| MiniCPM: (Less Relevant) | The novel coronavirus disease (COVID-19) pandemic is placing significant strains on health systems… In this context, the worlds scientific biomedical establishment is unleashing an unprecedented response to the COVID-19 pandemic. In this commentary, based on a very recent research report, we intend to highlight how a new mechanism describing the RAGE transactivation produced by Ang II-mediated ATR1 activation can run continuously and thus, reinforcing a sustained inflammation in lungs, due to the SARS-Cov-2-mediated imbalance of the ACE/And II/ATR1 pathway. |
| Debater: (Most Relevant) | The evidence on the pathophysiology of the novel coronavirus SARS-CoV-2 infection is rapidly growing … The answer to this question would allow rationalizing the fear surrounding this pandemic. Understanding of the pathophysiology of COVID-19 relies on an understanding of interplaying mechanisms, including SARS-CoV-2 virulence, human immune response, and complex inflammatory reactions with coagulation playing a major role … More importantly, a comprehensive understanding of pathological mechanisms of COVID-19 will increase the efficacy of therapy and decrease mortality. Herewith, presented is the current state of knowledge on COVID-19: beginning from the virus, its transmission, and mechanisms of entry into the human body, through the pathological effects on the cellular level, up to immunological reaction, systemic and organ presentation. Last but not least, currently available and possible future therapeutic and diagnostic options are briefly commented on. |

Learning Patterns of CoD. As shown in Figure[5(b)], we present the average similarity scores among the first five embeddings generated by Debater to explore how Debater refines document representations step by step during CoD.

The results reveal a clear pattern in the similarity relationships: each embedding is most similar to its immediate neighbors, with similarity gradually decreasing as the distance between embeddings increases. This indicates that each embedding heavily relies on the previously decoded representations to generate more refined embeddings, which likely results from the autoregressive decoding mechanism of LLMs. Comparing this with the Debater w/o SD model (Figure[5(a)]), we observe that the Debater model shows higher similarity scores with representations from more recent steps during CoD. This suggests that our Self Distillation method effectively encourages LLMs to learn more diverse representations at different thinking steps and to gather more relevant information from nearby steps, which leads to finer-grained document representations.

Contributions of CoD Steps to Document Representations.

Figure[6] illustrates the similarity relationship between embeddings at intermediate thinking steps of CoD and the final document representation generated at the last thinking step. This helps us explore the contributions of different thinking steps to the final document representations.

In general, both Debater w/o SD and Debater models exhibit a trend of gradually increasing similarity to the final embedding as the thinking steps progress. As shown in Figure[6(a)], the Debater w/o SD model tends to produce similar similarity scores with the final step, showing that relying solely on CoD may degrade the performance of Debater. It may lie in that all CoD generated embeddings are supervised with the same training loss and optimized to match the same query, making them become homogeneous. In contrast, Debater (Figure[6(b)]) shows a more significant increase in similarity, indicating that these thinking steps contribute more variably to the final document representation. Notably, the information generated at each CoD step is gradually compressed into the last embedding, which further demonstrates the effectiveness of Debater in leveraging the thinking capacity of LLMs to generate more effective document representations for retrieval.

### 5.6. Effectiveness of Debater across Different Model Scales

In this subsection, we examine whether the proposed Debater remains effective across language models of varying scales.

We evaluate Debater on a range of language models with varying parameter scales, including SmolLM2-135M, SmolLM2-360M*(Allal et al., [2025])*, MiniCPM-1B, and Qwen2.5-7B-Instruct*(Yang et al., [2025])*. These models span from lightweight architectures suitable for edge deployment to more advanced large language models. Due to limited computational resources, our experiments were conducted on a subset of low-resource datasets from the BEIR benchmark, which still provide a diverse set of retrieval tasks for evaluation.

As shown in Table[5], the results demonstrate that Debater consistently improves average performance across diverse retrieval tasks for all model sizes. Notably, even with the smallest model, SmolLM2-135M, Debater delivers performance gains on most tasks, indicating its effectiveness under limited capacity. Moreover, as model size increases, the performance improvements become more pronounced, reflecting Debater’s ability to better leverage the increased thinking capacity and representational power of larger models. These findings highlight the robustness and scalability of Debater across LLMs of different scales.

### 5.7. Case Studies

In this subsection, we present two case studies on FiQA and TREC-COVID to illustrate the effectiveness of Debater. Table[6] shows the top-1 retrieved documents from MiniCPM and Debater.

For the FiQA query “In the US, is it a good idea to hire a tax consultant for doing taxes?”, MiniCPM retrieves a generic document suggesting hiring a professional but ignores the explicit “US” context. In contrast, Debater retrieves a document that fully aligns with the query, including the US-specific aspect. This shows that Debater enables finer-grained representations for contextually appropriate retrieval.

For the TREC-COVID query “What is the mechanism of inflammatory response and pathogenesis of COVID-19 cases?”, MiniCPM returns documents with many pathological terms but lacking clear explanations of the mechanisms, reflecting limited fine-grained understanding. Debater, however, retrieves results that directly address the mechanisms, demonstrating the benefit of deliberate reasoning before retrieval.

6. Conclusion
--------------

This paper proposed the Deliberate Thinking based Dense Retriever (Debater), a novel method designed to enhance the reasoning capabilities of LLM-based dense retrievers via deliberation-augmented embedding.
Through the integration of Chain-of-Deliberation (CoD) and Self Distillation (SD), Debater significantly improves retrieval performance by capturing different views of documents before generating final embeddings.
Our experimental results demonstrate that Debater outperforms existing dense retrievers by implementing with the LLM of a smaller scale.

###### Acknowledgements.

This work is partly supported by the National Natural Science Foundation of China (No. 62206042) and the Fundamental Research Funds for the Central Universities (No. N25ZLL045). This work is also supported by the AI9Stars community.

References
----------

* (1)
* Achiam et al. (2023)Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023.Gpt-4 technical report.*ArXiv preprint* abs/2303.08774 (2023).[https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774 "")
* Allal et al. (2025)Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, et al. 2025.SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model.*ArXiv preprint* abs/2502.02737 (2025).[https://arxiv.org/abs/2502.02737](https://arxiv.org/abs/2502.02737 "")
* Asai et al. (2023)Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2023.Task-aware retrieval with instructions. In *Findings of ACL*. 3650–3675.[doi:10.18653/v1/2023.findings-acl.225](https://doi.org/10.18653/v1/2023.findings-acl.225 "")
* Bajaj et al. (2016)Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2016.MS MARCO: A human generated machine reading comprehension dataset.*ArXiv preprint* abs/1611.09268 (2016).[https://arxiv.org/abs/1611.09268](https://arxiv.org/abs/1611.09268 "")
* BehnamGhader et al. (2024)Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024.LLM2Vec: Large language models are secretly powerful text encoders. In *Proceedings of COLM*.[https://openreview.net/forum?id\=IW1PR7vEBf](https://openreview.net/forum?id=IW1PR7vEBf "")
* Chen and Yih (2020)Danqi Chen and Wen-tau Yih. 2020.Open-domain question answering. In *Proceedings of ACL*. 34–37.[https://aclanthology.org/2020.acl-tutorials.8](https://aclanthology.org/2020.acl-tutorials.8 "")
* Chen et al. (2024)Qi Chen, Xiubo Geng, Corby Rosset, Carolyn Buractaon, Jingwen Lu, Tao Shen, Kun Zhou, Chenyan Xiong, Yeyun Gong, Paul Bennett, et al. 2024.MS MARCO web search: A large-scale information-rich web dataset with millions of real click labels. In *Proceedings of WWW*. 292–301.[https://doi.org/10.1145/3589335.3648327](https://doi.org/10.1145/3589335.3648327 "")
* Chen et al. (2023)Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023.Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.*Transactions on Machine Learning Research* (2023).[https://openreview.net/forum?id\=YfZ4ZPt8zd](https://openreview.net/forum?id=YfZ4ZPt8zd "")
* Chu et al. (2024)Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. 2024.Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future. In *Proceedings of ACL*. 1173–1203.[doi:10.18653/v1/2024.acl-long.65](https://doi.org/10.18653/v1/2024.acl-long.65 "")
* Dao et al. (2022)Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.FlashAttention: Fast and memory-efficient exact attention with io-awareness. In *Proceedings of NeurIPS*. 16344–16359.[https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf "")
* DataCanary et al. (2017)DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, and tomtung. 2017.Quora question pairs.[https://kaggle.com/competitions/quora-question-pairs](https://kaggle.com/competitions/quora-question-pairs "").Kaggle.
* Fan et al. (2019)Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019.ELI5: Long form question answering. In *Proceedings of ACL*. 3558–3567.[https://aclanthology.org/P19-1346](https://aclanthology.org/P19-1346 "")
* Gao et al. (2023)Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023.Precise zero-shot dense retrieval without relevance labels. In *Proceedings of ACL*. 1762–1777.[doi:10.18653/v1/2023.acl-long.99](https://doi.org/10.18653/v1/2023.acl-long.99 "")
* Gao et al. (2021)Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.SimCSE: Simple contrastive learning of sentence embeddings. In *Proceedings of EMNLP*. 6894–6910.[https://aclanthology.org/2021.emnlp-main.552/](https://aclanthology.org/2021.emnlp-main.552/ "")
* Guu et al. (2020)Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.Retrieval augmented language model pre-training. In *Proceedings of ICML*. 3929–3938.<https://proceedings.mlr.press/v119/guu20a.html>
* Gysel and de Rijke (2018)Christophe Van Gysel and Maarten de Rijke. 2018.Pytrec_eval: An extremely fast python interface to trec_eval. In *Proceedings of SIGIR*. 873–876.[https://doi.org/10.1145/3209978.3210065](https://doi.org/10.1145/3209978.3210065 "")
* Hao et al. (2024)Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024.Training large language models to reason in a continuous latent space.*ArXiv preprint* abs/2412.06769 (2024).[https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769 "")
* He et al. (2018)Wei He, Kai Liu, Jing Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, Xuan Liu, Tian Wu, and Haifeng Wang. 2018.DuReader: A chinese machine reading comprehension dataset from real-world applications. In *Proceedings of the Workshop on Machine Reading for Question Answering*. 37–46.[https://aclanthology.org/W18-2605](https://aclanthology.org/W18-2605 "")
* Hu et al. (2022)Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022.LoRA: Low-rank adaptation of large language models. In *Proceedings of ICLR*.[https://openreview.net/forum?id\=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9 "")
* Hu et al. (2024)Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Xinrong Zhang, Zhen Leng Thai, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, dahai li, Zhiyuan Liu, and Maosong Sun. 2024.MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. In *Proceedings of COLM*.[https://openreview.net/forum?id\=3X2L2TFr0f](https://openreview.net/forum?id=3X2L2TFr0f "")
* Joshi et al. (2017)Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017.TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of ACL*. 1601–1611.[https://aclanthology.org/P17-1147](https://aclanthology.org/P17-1147 "")
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.Dense passage retrieval for open-domain question answering. In *Proceedings of EMNLP*. 6769–6781.[https://aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550 "")
* Khattab and Zaharia (2020)Omar Khattab and Matei Zaharia. 2020.Colbert: Efficient and effective passage search via contextualized late interaction over bert. In *Proceedings of SIGIR*. 39–48.[https://doi.org/10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075 "")
* Khramtsova et al. (2024)Ekaterina Khramtsova, Shengyao Zhuang, Mahsa Baktashmotlagh, and Guido Zuccon. 2024.Leveraging llms for unsupervised dense retriever ranking. In *Proceedings of SIGIR*. 1307–1317.[https://doi.org/10.1145/3626772.3657798](https://doi.org/10.1145/3626772.3657798 "")
* Kudo et al. (2024)Keito Kudo, Yoichi Aoki, Tatsuki Kuribayashi, Shusaku Sone, Masaya Taniguchi, Ana Brassard, Keisuke Sakaguchi, and Kentaro Inui. 2024.Think-to-talk or talk-to-think? When llms come up with an answer in multi-step reasoning.*ArXiv preprint* abs/2412.01113 (2024).[https://arxiv.org/abs/2412.01113](https://arxiv.org/abs/2412.01113 "")
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.Natural Questions: A benchmark for question answering research.*Transactions of the Association for Computational Linguistics* (2019), 452–466.[doi:10.1162/tacl_a_00276](https://doi.org/10.1162/tacl_a_00276 "")
* Lee et al. (2024)Jinhyuk Lee, Zhuyun Dai, Xiaoqi Ren, Blair Chen, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Boratko, Rajvi Kapadia, Wen Ding, et al. 2024.Gecko: Versatile text embeddings distilled from large language models.*ArXiv preprint* abs/2403.20327 (2024).[https://arxiv.org/abs/2403.20327](https://arxiv.org/abs/2403.20327 "")
* Li et al. (2024)Chaofan Li, Zheng Liu, Shitao Xiao, Yingxia Shao, and Defu Lian. 2024.Llama2vec: Unsupervised adaptation of large language models for dense retrieval. In *Proceedings of ACL*. 3490–3500.[https://aclanthology.org/2024.acl-long.191/](https://aclanthology.org/2024.acl-long.191/ "")
* Li et al. (2025)Chaofan Li, Minghao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Yingxia Shao, Defu Lian, and Zheng Liu. 2025.Making text embedders few-shot learners. In *Proceedings of ICLR*.[https://openreview.net/forum?id\=wfLuiDjQ0u](https://openreview.net/forum?id=wfLuiDjQ0u "")
* Liu et al. (2020)Zhenghao Liu, Chenyan Xiong, Maosong Sun, and Zhiyuan Liu. 2020.Fine-grained fact verification with kernel graph attention network. In *Proceedings of ACL*. 7342–7351.[https://aclanthology.org/2020.acl-main.655](https://aclanthology.org/2020.acl-main.655 "")
* Luo et al. (2024)Kun Luo, Minghao Qin, Zheng Liu, Shitao Xiao, Jun Zhao, and Kang Liu. 2024.Large language models as foundations for next-gen dense retrieval: A comprehensive empirical assessment. In *Proceedings of EMNLP*. 1354–1365.[doi:10.18653/v1/2024.emnlp-main.80](https://doi.org/10.18653/v1/2024.emnlp-main.80 "")
* Ma et al. (2024)Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024.Fine-tuning llama for multi-stage text retrieval. In *Proceedings of SIGIR*. 2421–2425.[https://doi.org/10.1145/3626772.3657951](https://doi.org/10.1145/3626772.3657951 "")
* Muennighoff (2022)Niklas Muennighoff. 2022.Sgpt: Gpt sentence embeddings for semantic search.*ArXiv preprint* abs/2202.08904 (2022).[https://arxiv.org/abs/2202.08904](https://arxiv.org/abs/2202.08904 "")
* Neelakantan et al. (2022)Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022.Text and code embeddings by contrastive pre-training.*ArXiv preprint* abs/2201.10005 (2022).[https://arxiv.org/abs/2201.10005](https://arxiv.org/abs/2201.10005 "")
* Ni et al. (2022)Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022.Large dual encoders are generalizable retrievers. In *Proceedings of EMNLP*. 9844–9855.[doi:10.18653/v1/2022.emnlp-main.669](https://doi.org/10.18653/v1/2022.emnlp-main.669 "")
* Rajpurkar et al. (2016)Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of EMNLP*. 2383–2392.[https://aclanthology.org/D16-1264](https://aclanthology.org/D16-1264 "")
* Springer et al. (2025)Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. 2025.Repetition improves language model embeddings. In *Proceedings of ICLR*.[https://openreview.net/forum?id\=Ahlrf2HGJR](https://openreview.net/forum?id=Ahlrf2HGJR "")
* Su et al. (2023)Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2023.One embedder, any task: Instruction-finetuned text embeddings. In *Findings of ACL*. 1102–1121.[doi:10.18653/v1/2023.findings-acl.71](https://doi.org/10.18653/v1/2023.findings-acl.71 "")
* Tao et al. (2024)Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang, Zhen Li, Zhengwei Tao, and Shuai Ma. 2024.LLMs are also effective embedding models: An in-depth overview.*ArXiv preprint* abs/2412.12591 (2024).[https://arxiv.org/abs/2412.12591](https://arxiv.org/abs/2412.12591 "")
* Thakur et al. (2021)Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021.BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *Proceedings of NeurIPS*.[https://openreview.net/forum?id\=wCu6T5xFjeJ](https://openreview.net/forum?id=wCu6T5xFjeJ "")
* Thorne et al. (2018)James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018.FEVER: A large-scale dataset for fact extraction and verification. In *Proceedings of NAACL-HLT*. 809–819.[https://aclanthology.org/N18-1074](https://aclanthology.org/N18-1074 "")
* Touvron et al. (2023)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.Llama: Open and efficient foundation language models.*ArXiv preprint* abs/2302.13971 (2023).[https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971 "")
* Wang et al. (2024a)Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024a.Improving text embeddings with large language models. In *Proceedings of ACL*. 11897–11916.[doi:10.18653/v1/2024.acl-long.642](https://doi.org/10.18653/v1/2024.acl-long.642 "")
* Wang et al. (2024b)Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024b.Multilingual e5 text embeddings: A technical report.*ArXiv preprint* abs/2402.05672 (2024).[https://arxiv.org/abs/2402.05672](https://arxiv.org/abs/2402.05672 "")
* Wei et al. (2022a)Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a.Emergent abilities of large language models.*ArXiv preprint* abs/2206.07682 (2022).[https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682 "")
* Wei et al. (2022b)Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b.Chain-of-thought prompting elicits reasoning in large language models.*Advances in neural information processing systems* 35 (2022), 24824–24837.[https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf "")
* Xie et al. (2023)Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, and Jin Ma. 2023.T2Ranking: A large-scale chinese benchmark for passage ranking. In *Proceedings of SIGIR*. 2681–2690.[https://doi.org/10.1145/3539618.3591874](https://doi.org/10.1145/3539618.3591874 "")
* Xie et al. (2024)Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. 2024.Self-evaluation guided beam search for reasoning.*Advances in Neural Information Processing Systems* 36 (2024).[https://proceedings.neurips.cc/paper_files/paper/2023/file/81fde95c4dc79188a69ce5b24d63010b-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/81fde95c4dc79188a69ce5b24d63010b-Paper-Conference.pdf "")
* Xiong et al. (2021)Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021.Approximate nearest neighbor negative contrastive learning for dense text retrieval. In *Proceedings of ICLR*.[https://openreview.net/forum?id\=zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln "")
* Yang et al. (2025)An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, et al. 2025.Qwen2.5 Technical Report.*ArXiv preprint* abs/2412.15115 (2025).[https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115 "")
* Yang et al. (2018)Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018.HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of EMNLP*. 2369–2380.[https://aclanthology.org/D18-1259](https://aclanthology.org/D18-1259 "")
* Yu et al. (2023)Shi Yu, Zhenghao Liu, Chenyan Xiong, and Zhiyuan Liu. 2023.OpenMatch-v2: An all-in-one multi-modality plm-based information retrieval toolkit. In *Proceedings of the SIGIR*. 3160–3164.[https://doi.org/10.1145/3539618.3591813](https://doi.org/10.1145/3539618.3591813 "")
* Yu et al. (2022)Yue Yu, Chenyan Xiong, Si Sun, Chao Zhang, and Arnold Overwijk. 2022.COCO-DR: Combating the distribution shift in zero-shot dense retrieval with contrastive and distributionally robust learning. In *Proceedings of EMNLP*. 1462–1479.[doi:10.18653/v1/2022.emnlp-main.95](https://doi.org/10.18653/v1/2022.emnlp-main.95 "")
* Zhang et al. (2022)Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. 2022.Multi-view document representation learning for open-domain dense retrieval. In *Proceedings of ACL*. 5990–6000.[doi:10.18653/v1/2022.acl-long.414](https://doi.org/10.18653/v1/2022.acl-long.414 "")
* Zhang et al. (2021)Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. 2021.Mr. TyDi: A multi-lingual benchmark for dense retrieval. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*. 127–137.[https://aclanthology.org/2021.mrl-1.12/](https://aclanthology.org/2021.mrl-1.12/ "")
* Zhang et al. (2023)Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2023.MIRACL: A multilingual retrieval dataset covering 18 diverse languages.*Transactions of the Association for Computational Linguistics* 11 (2023), 1114–1131.[https://aclanthology.org/2023.tacl-1.63/](https://aclanthology.org/2023.tacl-1.63/ "")
* Zhang et al. (2024)Yongheng Zhang, Qiguang Chen, Jingxuan Zhou, Peng Wang, Jiasheng Si, Jin Wang, Wenpeng Lu, and Libo Qin. 2024.Wrong-of-thought: An integrated reasoning framework with multi-perspective verification and wrong information. In *Findings of EMNLP*. 6644–6653.[doi:10.18653/v1/2024.findings-emnlp.388](https://doi.org/10.18653/v1/2024.findings-emnlp.388 "")
* Zhao et al. (2024)Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. 2024.Dense text retrieval based on pretrained language models: A survey.*ACM Transactions on Information Systems* 4 (2024), 1–60.[https://doi.org/10.1145/3637870](https://doi.org/10.1145/3637870 "")
* Zhao et al. (2023)Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023.A survey of large language models.*ArXiv preprint* abs/2303.18223 (2023).[https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223 "")
* Zhu et al. (2023)Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2023.Large language models for information retrieval: A survey.*ArXiv preprint* abs/2308.07107 (2023).[https://arxiv.org/abs/2308.07107](https://arxiv.org/abs/2308.07107 "")
* Zhuang et al. (2024)Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon. 2024.PromptReps: Prompting large language models to generate dense and sparse representations for zero-shot document retrieval. In *Proceedings of EMNLP*. 4375–4391.[doi:10.18653/v1/2024.emnlp-main.250](https://doi.org/10.18653/v1/2024.emnlp-main.250 "")
