BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence
===========================================================================================================

Jiajie Jin Yutao Zhu Yujia Zhou Zhicheng Dou  
Gaoling School of Artificial Intelligence, Renmin University of China  
{jinjiajie,zhouyujia,dou}@ruc.edu.cn, yutaozhu94@gmail.com*Corresponding author

###### Abstract

Retrieval-augmented large language models (LLMs) have demonstrated efficacy in knowledge-intensive tasks such as open-domain QA, addressing inherent challenges in knowledge update and factual inadequacy.
However, inconsistencies between retrieval knowledge and the necessary knowledge for LLMs, leading to a decline in LLM’s answer quality.
This paper introduces BIDER, an approach that refines retrieval documents into Key Supporting Evidence (KSE) through knowledge synthesis, supervised fine-tuning (SFT), and preference alignment.
We train BIDER by learning from crafting KSE, while maximizing its output to align with LLM’s information acquisition preferences through reinforcement learning.
Evaluations across five datasets show BIDER boosts LLMs’ answer quality by 7% while reducing input content length in retrieval documents by 80%, outperforming existing methods. The proposed KSE simulation effectively equips LLMs with essential information for accurate question answering.

1 Introduction
--------------

Large language models (LLMs) are currently developing rapidly and showing tremendous capabilities*OpenAI et al. ([2023](#bib.bib28 "")); Touvron et al. ([2023](#bib.bib42 ""))*.
Nevertheless, they face challenges in knowledge updates and furnishing factual responses*Bang et al. ([2023](#bib.bib3 ""))*,
especially in knowledge-intensive tasks like open-domain QA*Jiang et al. ([2023b](#bib.bib14 ""))*.
To address these issues, retrieval-augmented generation (RAG) has emerged as a promising approach*Lewis et al. ([2020](#bib.bib21 "")); Guu et al. ([2020](#bib.bib11 ""))*.
Retrieval-augmented methodologies serve to mitigate the drawbacks of LLMs by incorporating external knowledge, thereby enhancing the quality and reliability of generated answers*Izacard et al. ([2022](#bib.bib12 "")); Shi et al. ([2023b](#bib.bib38 "")); Press et al. ([2023](#bib.bib32 ""))*.

The standard RAG procedure involves retrieving pertinent documents related to a given question and subsequently inputting these documents as auxiliary information directly into the prompt. This strategic utilization enables the model to capitalize on its advanced text comprehension skills, facilitating the generation of precise and contextually appropriate answers.

<img src='x1.png' alt='Refer to caption' title='' width='368' height='281' />

*Figure 1: Key supporting evidence in RAG framework.*

However, retrieval-augmented LLMs are not always beneficial. Due to imperfections in the retrieval system and the inaccessibility of LLM’s self-knowledge*Wang et al. ([2023b](#bib.bib45 ""))*, the retrieved documents provided to LLM are frequently lengthy and noisy, which can detrimentally affect generation quality*Petroni et al. ([2020](#bib.bib30 "")); Shi et al. ([2023a](#bib.bib37 ""))*.

Recognizing this decline, recent researches have made strides in optimizing retrieved documents. These efforts aim to mitigate noise in retrieved documents by employing sorting mechanisms to retain the most pertinent sentences*Xu et al. ([2023](#bib.bib48 "")); Arefeen et al. ([2023](#bib.bib1 ""))*, summarizing the retrieved text*Xu et al. ([2023](#bib.bib48 ""))*, and eliminating content that contributes minimally to the model’s understanding*Li ([2023](#bib.bib22 "")); Jiang et al. ([2023a](#bib.bib13 ""))* or hinders effective generation*Yang et al. ([2023](#bib.bib49 ""))*.

While prior methods have shown progress in enhancing the quality of retrieved documents, they often rely excessively on feedback from the generator, overlooking the essential knowledge required for addressing the questions themselves.
This over-reliance on LLM’s feedback is not only insufficient but also susceptible to the instability of LLM feedback, potentially resulting in the loss of crucial information and the retention of noisy elements.
We argue that this limitation might stem from the neglect of knowledge inconsistency between the retrieved results and the knowledge truly required by the model for answering the question. We term this essential knowledge as Key Supporting Evidence (KSE).
As shown in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence"), due to the imperfections of the retrieval system and the inaccessibility of LLM’s self-knowledge*Wang et al. ([2023b](#bib.bib45 ""))*, retrieved results often contain numerous noise elements beyond key supporting evidence.

To address the aforementioned knowledge inconsistency issue, we propose BIDER(BrIDging knowledge inconsistency for efficient Retrieval-augmented LLMs), a method designed to refine retrieval documents into KSE. The overall training process of BIDER consists of three stages, integrating the strengths of both supervised and reinforcement learning, as shown in Figure[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence").
In the knowledge synthesizing stage, we employ a meticulous three-step process to synthesize authentic KSE. In the supervised fine-tuning stage, we construct a seq2seq model to learn the mapping from retrieved documents to KSE. Finally, in the preference alignment stage, we leverage reinforcement learning techniques to align the developed model with the preferences of the downstream LLM. This alignment ensures that the refined retrieval documents contain coherent and easily digestible key information, which is crucial for the LLM to generate accurate and informative responses.

We evaluate the effectiveness of our method on five datasets from three types of knowledge-intensive tasks, i.e., NQ, TQA, and HotpotQA for open-domain QA, WoW for dialogue generation, and FEVER for fact verification. Results show that our method achieves better generation performance while reducing the input information length by 80%, effectively condensing retrieved documents, and outperforming existing methods.
We also validate the advantages of our proposed KSE data construction process and investigate the impact of the preference alignment stage on the final results.
Furthermore, we validate the robustness of our approach under various text retrieval quality conditions.

The main contributions of this work are:
(1) We propose a three-step knowledge synthesis method to generate oracle KSE.
(2) We introduce a method to refine retrieval documents into KSE, thereby bridging knowledge inconsistencies between retrieval documents and the knowledge required by LLMs for answering.
(3) We train the refiner model using supervised distillation and preference alignment techniques, efficiently enhancing RAG performance during inference by reducing input length and improving answer quality.

<img src='x2.png' alt='Refer to caption' title='' width='461' height='257' />

*Figure 2: The overall architecture of BIDER. The first two lines represent the training process, which consists of three stages, and the last line represents the inference process of RAG with BIDER.*

2 Related Work
--------------

### 2.1 RAG for LLMs

In knowledge-intensive tasks*Petroni et al. ([2021](#bib.bib31 ""))*, RAG*Lewis et al. ([2020](#bib.bib21 ""))* has been introduced to enhance generative outcomes by incorporating external knowledge sources.
In previous work, the retriever and generator are usually jointly trained end-to-end*Guu et al. ([2020](#bib.bib11 "")); Lewis et al. ([2020](#bib.bib21 "")); Borgeaud et al. ([2022](#bib.bib4 ""))*. With the advent of LLMs*OpenAI et al. ([2023](#bib.bib28 "")); Touvron et al. ([2023](#bib.bib42 ""))*, most works now directly use them as generators due to their strong text comprehension ability, without the need for additional training*Jiang et al. ([2023b](#bib.bib14 "")); Yao et al. ([2023](#bib.bib51 "")); Shinn et al. ([2023](#bib.bib39 ""))*.
While this approach demonstrates efficiency, it introduces new challenges, including susceptibility to interference from irrelevant content*Shi et al. ([2023a](#bib.bib37 "")); Bai et al. ([2023](#bib.bib2 "")); Mallen et al. ([2022](#bib.bib27 ""))*, insufficient attention to middle positions*Liu et al. ([2023](#bib.bib24 ""))*, and increased inference costs*Dettmers et al. ([2022](#bib.bib6 ""))*.
Our method refines retrieved documents to eliminate noise, significantly reducing the input required for inference. By learning information retrieval preferences from LLM feedback, it provides the LLM with text that is more informative and easily captures relevant information, offering a substantial solution to the aforementioned issues.

### 2.2 Knowledge Refinement for RAG

Recent works leverage the capabilities of LLMs to identify pertinent information from various perspectives.
Some approaches directly task the LLM with summarizing retrieval documents to identify pertinent information*Laskar et al. ([2023](#bib.bib18 "")); Chen et al. ([2023](#bib.bib5 "")); Gilbert et al. ([2023](#bib.bib9 "")); Xu et al. ([2023](#bib.bib48 ""))*.
Moreover, certain methods employ smaller models to calculate perplexity as an importance indicator for filtering low-information text*Li ([2023](#bib.bib22 "")); Jiang et al. ([2023a](#bib.bib13 ""))*. *Xu et al. ([2023](#bib.bib48 ""))* employ the LLM to assess the utility of each sentence in retrieval documents, using this information as labels to train a ranking model.
Other works leverage LLM feedback for training; for instance, *Arefeen et al. ([2023](#bib.bib1 ""))* train a ranking model using reinforcement learning to retain top-ranked sentences, and *Yang et al. ([2023](#bib.bib49 ""))* design a reward mechanism to train a refinement model for retrieval documents. While these methods are effective, they are constrained by the instability of LLM feedback, providing limited guidance on specific information deemed valuable, which results in inefficient and suboptimal training outcomes. In contrast, our method employs well-designed key supporting evidence as a training objective, allowing the refiner to learn knowledge comprehensively before reinforcement learning, ensuring the provision of knowledge that better aligns with the LLM’s needs.

3 BIDER: a Knowledge Refiner for RAG
------------------------------------

Our objective is to furnish the necessary knowledge for a generator, specifically the KSE as defined earlier, to answer a question. Since authentic KSE is unattainable, we employ a synthesize-and-learn paradigm.
We design a method for synthesizing authentic KSE, training the refiner to learn the map from retrieval documents to constructed KSE, and adapting the model’s information acquisition preferences based on the generator’s feedback.

The overall framework of BIDER is illustrated in Figure[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence").
In this section, we first formulate the research problem (§[3.1](#S3.SS1 "3.1 Problem Formulation ‣ 3 BIDER: a Knowledge Refiner for RAG ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence")). Then, we introduce the details of three training stages of BIDER, including Knowledge Synthesis (§[3.2](#S3.SS2 "3.2 Knowledge Synthesis Stage ‣ 3 BIDER: a Knowledge Refiner for RAG ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence")), Supervised Distillation (§[3.3](#S3.SS3 "3.3 Supervised Distillation Stage ‣ 3 BIDER: a Knowledge Refiner for RAG ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence")), and Preference Alignment (§[3.4](#S3.SS4 "3.4 Preference Alignment Stage ‣ 3 BIDER: a Knowledge Refiner for RAG ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence")).

### 3.1 Problem Formulation

In this problem, we assume that a document collection $\mathcal{C}$, a fixed retriever $\mathcal{R}$, and a fixed generator $\mathcal{G}$ are provided. For a given question $q$ and its corresponding golden answer $o$, we assume that $K$ documents are retrieved by retriever $\mathcal{R}$, denoted as $\mathcal{D}_{q}\={{d_{i}}}_{i\=1}^{K}$. In the naive RAG framework, $\mathcal{D}_{q}$ is directly incorporated into the generator’s input to obtain the output answer. We aim to find the optimal mapping function $\mathcal{F}^{*}$ for the retrieved documents $\mathcal{D}_{q}$, in order for the generator $\mathcal{G}$ to use $\mathcal{F}^{*}(\mathcal{D}_{q})$ and achieve the best output results. We design BIDER to act as the mapping function, refining retrieval documents to make them more suitable for the input preferences of the generator.

### 3.2 Knowledge Synthesis Stage

We design a three-step method to gradually synthesize authentic KSE.

(1) Nugget Extraction. We initially narrow down the scope of knowledge helpful for answering by extracting nuggets from the retrieval documents. Here a nugget can be a sentence, a passage, or even a key phrase. In this paper, we use sentences as nuggets because using sentences already yields robust and consistent results. We will explore approaches with different nugget granularities in our future work.
For each input question $q$ and its corresponding golden answer $o$, we first formulate them into a fact $f\=\text{concat}(q,o)$ to ensure comprehensive semantic representation.111For FEVER and HotpotQA where the answers have no actual semantic meaning, we use annotated evidence as a golden answer to ensure more accurate semantic information. Then, we use $f$ as the query to perform sentence-level nugget retrieval in the retrieved documents $\mathcal{D}_{q}$ to remove noise and retain helpful sentences.
In nugget retrieval, $\mathcal{D}_{q}$ is split into nuggets and transformed into vectors, while the query is vectorized.
Based on the similarity between the query vector and nugget vectors, we obtain a positive nugget set $\mathcal{S}$ including retrieved top $K$ nuggets:

|  | $\mathcal{S}\=\mathop{\text{TopK}}\limits_{s\in\mathcal{D}_{q}}(\text{sim}(s,f)).$ |  | (1) |
| --- | --- | --- | --- |

Here, $\text{sim}(\cdot,\cdot)$ represents the function for calculating semantic similarity by the E5 model*Wang et al. ([2022](#bib.bib43 ""))*, and $K$ is a hyperparameter. A larger $K$ can improve the recall of relevant information in $\mathcal{D}_{q}$, but it also raises the risk of including more irrelevant information.

(2) Nugget Refinement. While the extraction step effectively reduces noise in retrieved documents, there may be redundancy in $\mathcal{S}$.
Therefore, we further design an iterative selection method to retain the minimal nugget subset necessary for answering the question.

Initially, we set up a candidate pool $\mathcal{P}$.
In each round, we calculate a gain score for each nugget in $\mathcal{S}$, which represents the degree of assistance in answering the question. The gain score is defined as follows:

|  | $\kappa_{i}\=\text{sim}(s_{i},f)-\frac{1}{|\mathcal{P}|}\sum_{s_{j}\in\mathcal{P}}\text{sim}(s_{i},s_{j}).$ |  | (2) |
| --- | --- | --- | --- |

This takes into account the importance of the nugget itself as well as its duplication with the already-selected nuggets.
Then, we select the sentence with the highest $\kappa_{i}$ from $\mathcal{S}$ and move it to the candidate pool $\mathcal{P}$. After moving, we use an NLI model to measure to what extent the candidate pool $\mathcal{P}$ can support answering $q$, i.e., yielding a support degree $\eta_{k}$ in $k$-th nugget selection.

The iterative selecting process will terminate in two cases: (1) when the support degree $\eta_{k}$ exceeds $\lambda_{\text{max}}$; (2)when the difference in support degree between two rounds, $\eta_{k}-\eta_{k-1}$, is less than $\lambda_{\text{min}}$, where $\lambda_{\text{max}}$ and $\lambda_{\text{min}}$ are predefined thresholds.
This aims to avoid introducing redundant information, especially in scenarios where retrieval documents fail to furnish adequate information, such as instances where the retriever’s quality is subpar or when the posed question is challenging.

(3) Nugget Cleaning. The candidate pool $\mathcal{P}$ from the previous stage serves as the minimal subset of information necessary for answering. However, we have yet to consider the knowledge intrinsic to the generator itself, which encompasses information either known by LLM or detrimental to its generation.
To mitigate conflicts arising from the disparity between external and internal knowledge, we conduct nugget cleaning in the candidate pool.
In our experiments, we observe that the first nugget within the candidate pool is usually important.
To avoid unintentionally removing vital information, we retain the first nugget directly and perform nugget cleaning for the left set.

For each nugget $s_{i}(i\geq 2)$ in $\mathcal{P}$, we assess its influence on the generator by determining whether it contributes to the model’s output improvement when utilized as input. Specifically, we calculate the change in the log probability of generating the correct answer $o$ between the model’s output before and after the inclusion of the nugget. This score for each nugget $s_{i}$ is denoted as

|  | $\tau_{i}^{\prime}\=\log\frac{\mathcal{G}(o|q\oplus s_{i})}{\mathcal{G}(o|q)},i\geq 2.$ |  | (3) |
| --- | --- | --- | --- |

where $\mathcal{G}$ represents the generator.

Subsequently, we normalize all scores within the candidate pool to derive the final score $\tau_{i}$:

|  | $\tau_{i}\=\frac{\tau_{i}^{\prime}}{\sum_{j\=2}^{|\mathcal{P}|}\tau_{j}^{\prime}},i\geq 2.$ |  | (4) |
| --- | --- | --- | --- |

Nuggets with scores below $\lambda_{\text{lm}}$ are deemed unhelpful or potentially detrimental to the generator’s response and are consequently excluded from the candidate pool. The surviving nuggets in the filtered candidate pool represent the ultimate oracle KSE, which correspond to the distillation results for each sample triplet $(q,o,\mathcal{D}_{q})$.

### 3.3 Supervised Distillation Stage

In this stage, we aim to develop BIDER to acquire the ability to comprehend the relationship between retrieval documents and oracle KSE. This enhancement will enable BIDER to effectively refine its output during inference, particularly when provided only with the question.

A common approach is to consider this as a ranking task*(Xu et al., [2023](#bib.bib48 ""); Liu, [2019](#bib.bib25 ""))*, using the nuggets extracted in the previous section as positive examples and other nuggets as negative examples for training the ranker. Although this method can relatively stably filter information, it is not able to effectively generate content that can adapt to the input of the generation model, as the refined content can only come from the original text.

We model the task as a seq2seq task, which is similar to the idea of pointer network*(See et al., [2017](#bib.bib36 ""); Gu et al., [2016](#bib.bib10 ""))*. This method ensures the flexibility of refinement while enhancing the potential of the generation model in expression. Meanwhile, this serialization modeling approach makes it easier for the model to capture the generated sentences during generation. In Section[5.1](#S5.SS1 "5.1 Main Results ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence"), we will compare the two methods and demonstrate the effectiveness of our approach.

We use a pre-trained seq2seq model as the backbone model. For each sample triplet $(q,o,\mathcal{D}_{q})$, the refiner model’s input is the concatenation of the question and the original retrieval document: $q\oplus\mathcal{D}_{q}$. For ease of processing, we add separators between each document in $\mathcal{D}_{q}$ and merge them into one string. The target output of the model is the $\mathcal{P}$ extracted in the Knowledge Synthesis Stage, where each nugget is merged into one string in order. The training loss function of the model is the cross-entropy loss between the model output and the target output.

### 3.4 Preference Alignment Stage

Inspired by the RLHF technology*(Ziegler et al., [2019](#bib.bib53 ""); Stiennon et al., [2020](#bib.bib40 ""); Ouyang et al., [2022](#bib.bib29 ""))*, we further enhance the adaptability of BIDER by incorporating feedback from a downstream LLM.

Specifically, we model the optimization problem of the model as a RL problem, with the objective to generate content that conforms to the LLM’s information acquisition preferences without losing its original information capturing ability. The refiner model
$\mathcal{M}$ to be optimized acts as a policy, where its action space encompasses all tokens in the vocabulary. We use the CLIP version of the PPO algorithm*(Schulman et al., [2017](#bib.bib35 ""))* for optimization, which uses CLIP to control the magnitude of model updates.
The loss function consists of three parts:

|  | $L_{t}^{\text{ALL}}\=\mathrm{E}_{t}[L_{t}^{\text{CLIP}}-L_{t}^{\text{VF}}+L_{t}^{\text{BONUS}}].$ |  | (5) |
| --- | --- | --- | --- |

$L_{t}^{\text{CLIP}}$ is the primary objective function for optimizing the policy at step $t$, expressed as:

|  | $\displaystyle L_{t}^{\text{CLIP}}\=\text{min}(r_{t}\hat{A_{t}},\text{clip}(r_{t},1-\epsilon,1+\epsilon)\hat{A_{t}}$ |  | (6) |
| --- | --- | --- | --- |
|  | $\displaystyle r_{t}\=\frac{\pi_{\theta}(y|x)}{\pi_{\text{old}}(y|x)},$ |  | (7) |
| --- | --- | --- | --- |

where $\epsilon$ is a hyperparameter to control the policy update magnitude, $r_{t}$ represents the conditional generation probability ratio between the new policy and the old policy, and $A_{t}$ denotes the estimated value of the advantage function at step $t$, calculated from Generalized Advantage Estimation (GAE)*(Schulman et al., [2016](#bib.bib34 ""))*:

|  | $\hat{A_{t}}\=\sum_{l\=0}^{T-t+1}(\gamma\lambda)^{l}\left(R_{t}+\gamma V(s_{t+1})-V\left(s_{t}\right)\right),$ |  |
| --- | --- | --- |

where $\gamma$ and $\lambda$ are hyperparameters. $V$ represents the critic network used to estimate expected rewards, and $R_{t}$ indicates the reward at step $t$.
$L_{t}^{VF}(\theta)$ is the squared error between the predicted reward and the actual reward output by the critic network, used to fit the critic network:

|  | $L_{T}^{\text{VF}}(\theta)\=(V_{\theta}(s_{t})-R_{t})^{2}.$ |  | (8) |
| --- | --- | --- | --- |

$L_{t}^{\text{VF}}(\theta)$ is an entropy bonus designed to ensure the model can explore sufficiently.

To calculate the above loss, we need a well-defined reward function $R_{t}$. Considering that the downstream LLM is highly sensitive to the overall information density and the position of key information, providing rewards before all sentences are generated could lead to inaccurate guidance. Thus, we design a segmented reward function:

|  | $R_{t}\=\begin{cases}0,\&s_{t}\neq\langle\text{EOF}\rangle,\\ \text{F}_{1}(a_{\text{pred}},o)-\text{F}_{1}(a_{\text{ori}},o),\&s_{t}\=\langle\text{EOF}\rangle.\end{cases}$ |  |
| --- | --- | --- |

where $a_{\text{pred}}$ and $a_{\text{ori}}$ represent the answers generated by LLM based on the refiner result and original retrieval result respectively
, $\langle\text{EOF}\rangle$ represents the end-of-sentence symbol. We generate answers from the LLM using the original document and refined results separately as references, and evaluate the quality of the refiner’s distillation of the retrieved document by comparing the token-level $\text{F}_{1}$ scores of these two types of answers with the golden answer.

| Methods | NQ | | TQA | | Fever | | HotPotQA | | Wow | | Avg | Avg tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | EM | # tok | EM | # tok | Acc | # tok | F1 | # tok | F1 | # tok | | |
| Without refinement | | | | | | | | | | | | |
| Original Prompt | 0.356 | 725 | 0.480 | 787 | 0.517 | 805 | 0.376 | 770 | 0.086 | 710 | 0.363 | 759 |
| Zero-shot | 0.189 | 0 | 0.456 | 0 | 0.517 | 0 | 0.268 | 0 | 0.085 | 0 | 0.303 | 0 |
| Extractive refinement | | | | | | | | | | | | |
| BM25 | 0.295 | 163 | 0.479 | 181 | 0.520 | 193 | 0.356 | 186 | 0.085 | 177 | 0.347 | 180 |
| SBERT | 0.339 | 162 | 0.512 | 183 | 0.521 | 192 | 0.36 | 187 | 0.086 | 178 | 0.364 | 180 |
| LLM-Embedder | 0.357 | 161 | 0.503 | 179 | 0.522 | 192 | 0.352 | 186 | 0.118 | 179 | 0.370 | 179 |
| $\text{Bge-Reranker}^{\ast}$ | 0.380 | 164 | 0.504 | 181 | 0.522 | 194 | 0.384 | 186 | 0.117 | 180 | 0.381 | 181 |
| Abstractive refinement | | | | | | | | | | | | |
| BART-Summarizer | 0.326 | 185 | 0.507 | 204 | 0.518 | 215 | 0.369 | 254 | 0.085 | 194 | 0.361 | 210 |
| Selective-Context | 0.263 | 203 | 0.439 | 225 | 0.522 | 236 | 0.332 | 234 | 0.081 | 220 | 0.327 | 224 |
| LongLLMLingua | 0.221 | 251 | 0.433 | 175 | 0.551 | 111 | 0.302 | 222 | 0.077 | 124 | 0.317 | 177 |
| BIDER(ours) | 0.403 | 77 | 0.523 | 98 | 0.524 | 93 | 0.386 | 113 | 0.122 | 69 | 0.390 | 90 |

*Table 1: Evaluation results on five knowledge-intensive datasets. The best results are in bold and second best results are underlined. The method marked with ∗ have undergone additional training.*

4 Experimental Setup
--------------------

### 4.1 Datasets and Metrics

We experiment on five datasets of three knowledge-intensive tasks in the KILT benchmark *(Petroni et al., [2021](#bib.bib31 ""))*: (1) Open-domain QA, including NaturalQuestions (NQ) *(Kwiatkowski et al., [2019](#bib.bib17 ""))*, TriviaQA (TQA) *Joshi et al. ([2017](#bib.bib15 ""))*,and HotpotQA *(Yang et al., [2018](#bib.bib50 ""))*;
(2) Dialog Generation, including the Wizard of Wikipedia (WoW)*(Dinan et al., [2019](#bib.bib7 ""))*, where the generator is tasked with continuing the dialogue based on the preceding conversation history;
(3) Fact-checking, including FEVER*(Thorne et al., [2018](#bib.bib41 ""))* that classifies a given claim as "SUPPORTS" or "REFUTES".

| Dataset | Task | Metric | # Train / #Test |
| --- | --- | --- | --- |
| NQ | Open-domain QA | EM | 79.1k / 2.6k |
| TQA | Open-domain QA | EM | 78.7k / 11.3k |
| HoPo | Open-domain QA | F1 | 88.8k / 5.6k |
| WoW | Dialogue | F1 | 63.7k / 3.0k |
| FEVER | Fact checking | Acc | 104.9k / 10.4k |

*Table 2: Statistics and task metrics for five datasets.*

We use Exact Match (EM) as the evaluation metric for NQ and TQA, use accuracy for FEVER, and use token-level $\mathrm{F}_{1}$*(Jiang et al., [2023b](#bib.bib14 ""))* for HotpotQA and Wow.
Evaluation is conducted on top 1000 samples in the test set of NQ, TQA, and HotpotQA, while on the development set of FEVER and WoW. Table[2](#S4.T2 "Table 2 ‣ 4.1 Datasets and Metrics ‣ 4 Experimental Setup ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence") provides detailed sample sizes and the evaluation metrics used for each dataset.

### 4.2 Baselines

We compare with two types of baselines.

#### Extractive Methods:

We employ three retrieval methods to extract sentences from retrieval documents, including BM25*(Xu et al., [2023](#bib.bib48 ""))*, Sentence-BERT*(Reimers and Gurevych, [2019](#bib.bib33 ""))*, and LLM-Embedder*(Zhang et al., [2023](#bib.bib52 ""))* which is trained with contrastive learning and feedback from LLM.
To demonstrate the superiority of modeling the task as a seq2seq problem in the supervised distillation stage, we fine-tuned the bge-reranker-large*(Xiao et al., [2023](#bib.bib47 ""))* for comparison.

#### Abstractive Methods:

BART-Large is utilized for summarization*(Lewis et al., [2019a](#bib.bib19 ""))*, along with two state-of-the-art perplexity-based prompt refinement models: Selective Context*(Li, [2023](#bib.bib22 ""))* and LongLLMLingua*(Jiang et al., [2023a](#bib.bib13 ""))*.

### 4.3 Implementation Details

The size of the positive nugget set $K$ is set to 7. We utilize a T5-XXL model as the NLI model with the threshold $\lambda_{\text{max}}$ set to 0.5, $\lambda_{\text{min}}$ set to 0.01,$\lambda_{\text{lm}}$ set to 0.05.222<https://huggingface.co/google/t5_xxl_true_nli_mixture> We utilized BART-Large*(Lewis et al., [2019b](#bib.bib20 ""))* as the base model for BIDER. During training, we utilize AdamW*(Loshchilov and Hutter, [2019](#bib.bib26 ""))* as the optimizer with a learning rate of 5e-5, and a batch size of 32. In the preference alignment stage, $top_{k}$ is set to 10, and $top_{p}$ is set to 0.95. The training is implemented with HuggingFace Transformers*(Wolf et al., [2020](#bib.bib46 ""))* and PFRL*(Fujita et al., [2021](#bib.bib8 ""))*.
We use the December 2018 Wikipedia dump*(Karpukhin et al., [2020](#bib.bib16 ""); Lin et al., [2021](#bib.bib23 ""))* as retrieval corpus, BM25 as retriever, and SimLM as reranker*(Wang et al., [2023a](#bib.bib44 ""))* for the top 100 documents returned by the retriever.
LLAMA2-7B*(Touvron et al., [2023](#bib.bib42 ""))* is utilized as the generator to provide answers.

5 Experimental Results
----------------------

### 5.1 Main Results

Table[1](#S3.T1 "Table 1 ‣ 3.4 Preference Alignment Stage ‣ 3 BIDER: a Knowledge Refiner for RAG ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence") reports the results of our approach alongside baseline methods on five knowledge-intensive datasets. It can be observed that
our method outperforms the baseline on all datasets except FEVER, showcasing a notable performance advantage over other existing approaches, demonstrating around a 10% performance increase on all datasets. We also observe a relatively small performance gap on the FEVER dataset, indicating a potential weakness in the model’s ability to leverage retrieval documents for text-based verification tasks.
Compared to the original prompt, our method refines the retrieval documents to 20% of their original length, achieving an average improvement of approximately 8%. Notably, on the WoW dataset, the improvement approaches nearly 40%.

Comparison with extractive methods. The overall performance of extractive methods is quite satisfactory.
Fine-tuning bge-reranker with our KSE extraction yielded the best results, indicating the effectiveness of our extracted KSE. However, there still exists a discernible gap between this approach and ours, possibly highlighting the influence of the preference alignment stage and model structure.

Comparison with abstractive methods. Abstractive refinement methods like Selective-Context and LongLLMLingua show a significant performance gap compared to our approach, particularly in QA tasks. This may result from their reliance on perplexity-based computations, posing a risk of losing essential entity information crucial for answering questions during refinement. In contrast, our method minimizes the risk of token-level information loss by employing sentence-level processing in data construction.

### 5.2 Evaluation on Knowledge Synthesis Stage

<img src='x3.png' alt='Refer to caption' title='' width='461' height='230' />

*Figure 3: Performance of generator responses with different reference contents. ‘Nugget Extraction’, ‘Nugget Refinement’, and ‘Nugget Cleaning’ correspond to the two intermediate products and the final output in knowledge synthesis stage, respectively.‘Extract by evidence’ involves extracting the top 3 sentences based on the similarity between the golden target (answer or evidence) and sentences in the retrieved documents.*

To explore the necessity and effectiveness of the three steps in the knowledge synthesis stage, we use the results of each step as reference inputs for generating answers.
For a comprehensive comparison, we incorporate results from the extraction based on the similarity between golden evidence and sentences in the retrieved documents.

As illustrated in Figure[3](#S5.F3 "Figure 3 ‣ 5.2 Evaluation on Knowledge Synthesis Stage ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence"), with the further refinement of the retrieved text in the knowledge synthesis stage, the length of the input to the generator significantly decreases. However, there is a notable improvement in the quality of the LLM’s responses. This observation indicates the effectiveness of our approach in reducing noise in the retrieved text, providing the generator with more easily exploitable information.
Simultaneously, it is observed that directly using golden evidence as the target for information extraction results in an inferior performance. Overall, the effectiveness is somewhat lower compared to our second step. This suggests that relying solely on the relationship between the text and the question/answer for data extraction is insufficient, and it’s necessary to consider the knowledge of the model itself when constructing the data.

### 5.3 Ablation Study

| Method | NQ | | TQA | |
| --- | --- | --- | --- | --- |
| | EM | # tok | EM | # tok |
| BIDER | 0.403 | 77 | 0.523 | 98 |
| w/o preference alig. | 0.373 | 94 | 0.518 | 85 |
| w/o knowledge syn. | 0.340 | 118 | 0.504 | 133 |
| Original retrieval results | 0.356 | 725 | 0.480 | 787 |

*Table 3: Ablation study on NQ and TQA.*

To assess the impact of BIDER’s key components, we performed ablation experiments on NQ and TQA. Two variants were introduced for study: i) BIDER w/o preference alignment, using models without reinforcement learning, and ii) BIDER w/o knowledge synthesis, replacing knowledge synthesis method with a naive sentence-level retrieval method as the training target for SFT.

Table[3](#S5.T3 "Table 3 ‣ 5.3 Ablation Study ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence") displays the results, emphasizing a decline in performance when either component is removed. This underscores the indispensability of both components.
Particularly, the impact on performance due to the absence of the knowledge synthesis method is more significant than that of the preference alignment part. This implies that the construction of the training target in the initial phase is more crucial than preference alignment. Hence, emphasizing the construction of training data in the first phase should be a priority, rather than relying solely on LLM feedback for learning.

### 5.4 Impact of Preference Alignment

| Dataset | Align | EM | % Gold in Out. | Avg Gold Pos. |
| --- | --- | --- | --- | --- |
| NQ | ✕ | 0.373 | 48.1 | 1.33 |
| | $\checkmark$ | 0.403 | 51.1 | 1.19 |
| TQA | ✕ | 0.518 | 56.4 | 1.14 |
| | $\checkmark$ | 0.523 | 61.1 | 1.10 |

*Table 4: The comparison of refiner results with and without preference alignment on NQ and TQA. ‘Avg Gold Pos.’ represents the average position of sentences containing the golden answer (calculated only on samples that include the golden answer).*

To further investigate the impact of preference alignment, we analyzed model output before and after this stage, specifically focusing on effective information content and its optimal sequence. We measured the proportion of golden answers in the generated results and their average position on a sentence level for models trained through supervised learning and those additionally trained with preference alignment.

Table[4](#S5.T4 "Table 4 ‣ 5.4 Impact of Preference Alignment ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence") presents the results, indicating that after preference alignment training, the proportion of golden answers in the model output increased by 3%-4%, and their position in the output text moved closer to the beginning. This improvement suggests a dual effect: an augmentation in information content and a repositioning of crucial information towards the text’s forefront.

### 5.5 Impact of Retrieval Quality

| Method | BM25 | | BM25+SimLM | |
| --- | --- | --- | --- | --- |
| | EM | # tok | EM | # tok |
| Original Prompt | 0.257 | 716 | 0.356 | 725 |
| LLM-Embedder | 0.278 | 165 | 0.357 | 161 |
| BAET-Summarizer | 0.269 | 183 | 0.326 | 185 |
| BIDER(ours) | 0.325 | 80 | 0.403 | 77 |

*Table 5: Experiments with different retrievers on NQ.*

We directly utilize top 5 retrieval documents from BM25(without reranker) to demonstrate generalizability on weaker retrievers.
As depicted in Table[5](#S5.T5 "Table 5 ‣ 5.5 Impact of Retrieval Quality ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence"), our approach performs well under different quality retrievers, surpassing other methods. And it can be observed that our method brings more improvements when the retriever quality is worse, indicating the effectiveness of our refinement method.

### 5.6 Inference Latency

| Method | BIDER | Generator | Total |
| --- | --- | --- | --- |
| End-to-End w/o BIDER | \ | 1.33 | 1.33 |
| End-to-End w/ BIDER | 0.10 | 1.08 | 1.18 |

*Table 6: Inference latency (seconds/query) on NQ.*

Table[6](#S5.T6 "Table 6 ‣ 5.6 Inference Latency ‣ 5 Experimental Results ‣ BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence") shows the inference latency of various components within the system on a V100-32G GPU. It is observed that the time required for text refinement using BIDER is notably short, facilitating effective support for applications in the RAG scenario. Additionally, as the refined input to the generator is shorter, the time taken by the generator to produce responses has also decreased. Consequently, there is a 10% enhancement in the overall end-to-end speed.

6 Conclusion
------------

We present BIDER, a method to refine retrieved documents into KSE, addressing inconsistencies between retrieved results and the knowledge needed by the generator.
We designed a three-step process to synthesize authentic key supporting evidence to enhance the effectiveness of supervised learning, while utilizing LLM’s feedback for further alignment. Through a well-structured training process, BIDER effectively provides the generator with the necessary information to answer questions based on the original retrieval text, achieving a significant improvement in answer quality while reducing input length by 80%.

Limitations
-----------

Our approach has some limitations. It performs less effectively in complex datasets like HotpotQA compared with NQ and TQA, suggesting that additional factors need to be considered for complex tasks.
Also, our method requires separate training for each dataset and generator, limiting its use across different tasks and generators.
Lastly, our datasets are based solely on Wikipedia, while real-world RAG applications involve diverse sources with varied writing styles. Optimizing for this diversity may require further refinement.

References
----------

* Arefeen et al. (2023)Md Adnan Arefeen, Biplob Debnath, and Srimat Chakradhar. 2023.Leancontext: Cost-efficient domain-specific question answering using llms.*arXiv preprint arXiv:2309.00841*.
* Bai et al. (2023)Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023.Longbench: A bilingual, multitask benchmark for long context understanding.*arXiv preprint arXiv:2308.14508*.
* Bang et al. (2023)Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023.[A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity](http://arxiv.org/abs/2302.04023 "").
* Borgeaud et al. (2022)Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.Improving language models by retrieving from trillions of tokens.In *International Conference on Machine Learning*, pages 2206–2240. PMLR.
* Chen et al. (2023)Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. 2023.Walking down the memory maze: Beyond context limit through interactive reading.*arXiv preprint arXiv:2310.05029*.
* Dettmers et al. (2022)Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022.[Gpt3.int8(): 8-bit matrix multiplication for transformers at scale](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf "").In *Advances in Neural Information Processing Systems*, volume 35, pages 30318–30332. Curran Associates, Inc.
* Dinan et al. (2019)Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019.[Wizard of Wikipedia: Knowledge-powered conversational agents](https://openreview.net/forum?id=r1l73iRqKm "").In *International Conference on Learning Representations*.
* Fujita et al. (2021)Yasuhiro Fujita, Prabhat Nagarajan, Toshiki Kataoka, and Takahiro Ishikawa. 2021.[Chainerrl: A deep reinforcement learning library](http://jmlr.org/papers/v22/20-376.html "").*Journal of Machine Learning Research*, 22(77):1–14.
* Gilbert et al. (2023)Henry Gilbert, Michael Sandborn, Douglas C Schmidt, Jesse Spencer-Smith, and Jules White. 2023.Semantic compression with large language models.*arXiv preprint arXiv:2304.12512*.
* Gu et al. (2016)Jiatao Gu, Zhengdong Lu, Hang Li, and Victor OK Li. 2016.Incorporating copying mechanism in sequence-to-sequence learning.*arXiv preprint arXiv:1603.06393*.
* Guu et al. (2020)Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020.[REALM: Retrieval-augmented language model pre-training](https://dl.acm.org/doi/abs/10.5555/3524938.3525306 "").In *International Conference on Machine Learning*. JMLR.org.
* Izacard et al. (2022)Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022.[Atlas: Few-shot learning with retrieval augmented language models](http://arxiv.org/abs/2208.03299 "").
* Jiang et al. (2023a)Huiqiang Jiang, Qianhui Wu, , Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a.[LongLLMLingua: Accelerating and enhancing llms in long context scenarios via prompt compression](https://arxiv.org/abs/2310.06839 "").*ArXiv preprint*, abs/2310.06839.
* Jiang et al. (2023b)Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023b.[Active retrieval augmented generation](https://doi.org/10.18653/v1/2023.emnlp-main.495 "").In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 7969–7992, Singapore. Association for Computational Linguistics.
* Joshi et al. (2017)Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017.[TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension](https://doi.org/10.18653/v1/P17-1147 "").In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.[Dense passage retrieval for open-domain question answering](https://doi.org/10.18653/v1/2020.emnlp-main.550 "").In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online. Association for Computational Linguistics.
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.[Natural questions: A benchmark for question answering research](https://doi.org/10.1162/tacl_a_00276 "").*Transactions of the Association for Computational Linguistics*, 7:452–466.
* Laskar et al. (2023)Md Tahmid Rahman Laskar, Mizanur Rahman, Israt Jahan, Enamul Hoque, and Jimmy Huang. 2023.Cqsumdp: A chatgpt-annotated resource for query-focused abstractive summarization based on debatepedia.*arXiv preprint arXiv:2305.06147*.
* Lewis et al. (2019a)Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019a.[BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension](http://arxiv.org/abs/1910.13461 "").*CoRR*, abs/1910.13461.
* Lewis et al. (2019b)Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019b.[BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension](http://arxiv.org/abs/1910.13461 "").*CoRR*, abs/1910.13461.
* Lewis et al. (2020)Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020.[Retrieval-Augmented Generation for knowledge-intensive NLP tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf "").In *Advances in Neural Information Processing Systems*, volume 33, pages 9459–9474.
* Li (2023)Yucheng Li. 2023.Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering.*arXiv preprint arXiv:2304.12102*.
* Lin et al. (2021)Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021.Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations.In *Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)*, pages 2356–2362.
* Liu et al. (2023)Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023.Lost in the middle: How language models use long contexts.*arXiv preprint arXiv:2307.03172*.
* Liu (2019)Yang Liu. 2019.Fine-tune bert for extractive summarization.*arXiv preprint arXiv:1903.10318*.
* Loshchilov and Hutter (2019)Ilya Loshchilov and Frank Hutter. 2019.[Decoupled weight decay regularization](https://openreview.net/forum?id=Bkg6RiCqY7 "").In *International Conference on Learning Representations*.
* Mallen et al. (2022)Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. 2022.When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories.*arXiv preprint*.
* OpenAI et al. (2023)OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao,
Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe,
Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder,
Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao
Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2023.[Gpt-4 technical report](http://arxiv.org/abs/2303.08774 "").
* Ouyang et al. (2022)Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022.[Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf "").In *Advances in Neural Information Processing Systems*, volume 35, pages 27730–27744. Curran Associates, Inc.
* Petroni et al. (2020)Fabio Petroni, Patrick S. H. Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. 2020.[How context affects language models’ factual predictions](http://arxiv.org/abs/2005.04611 "").*CoRR*, abs/2005.04611.
* Petroni et al. (2021)Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021.[KILT: a benchmark for knowledge intensive language tasks](https://doi.org/10.18653/v1/2021.naacl-main.200 "").In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2523–2544, Online. Association for Computational Linguistics.
* Press et al. (2023)Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. 2023.[Measuring and narrowing the compositionality gap in language models](https://doi.org/10.18653/v1/2023.findings-emnlp.378 "").In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 5687–5711, Singapore. Association for Computational Linguistics.
* Reimers and Gurevych (2019)Nils Reimers and Iryna Gurevych. 2019.[Sentence-bert: Sentence embeddings using siamese bert-networks](https://arxiv.org/abs/1908.10084 "").In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics.
* Schulman et al. (2016)John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. 2016.[High-dimensional continuous control using generalized advantage estimation](http://arxiv.org/abs/1506.02438 "").In *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*.
* Schulman et al. (2017)John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017.[Proximal policy optimization algorithms](https://api.semanticscholar.org/CorpusID:28695052 "").*ArXiv*, abs/1707.06347.
* See et al. (2017)Abigail See, Peter J. Liu, and Christopher D. Manning. 2017.[Get to the point: Summarization with pointer-generator networks](https://doi.org/10.18653/v1/P17-1099 "").In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1073–1083, Vancouver, Canada. Association for Computational Linguistics.
* Shi et al. (2023a)Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. 2023a.[Large language models can be easily distracted by irrelevant context](https://arxiv.org/pdf/2302.00093 "").*arXiv preprint arXiv:2302.00093*.
* Shi et al. (2023b)Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023b.Replug: Retrieval-augmented black-box language models.*arXiv preprint arXiv:2301.12652*.
* Shinn et al. (2023)Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023.[Reflexion: Language agents with verbal reinforcement learning](http://arxiv.org/abs/2303.11366 "").
* Stiennon et al. (2020)Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020.[Learning to summarize with human feedback](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf "").In *Advances in Neural Information Processing Systems*, volume 33, pages 3008–3021. Curran Associates, Inc.
* Thorne et al. (2018)James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018.[FEVER: a large-scale dataset for fact extraction and VERification](https://doi.org/10.18653/v1/N18-1074 "").In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.[Llama 2: Open foundation and fine-tuned chat models](https://arxiv.org/pdf/2307.09288 "").*arXiv preprint arXiv:2307.09288*.
* Wang et al. (2022)Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022.Text embeddings by weakly-supervised contrastive pre-training.*arXiv preprint arXiv:2212.03533*.
* Wang et al. (2023a)Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2023a.[SimLM: Pre-training with representation bottleneck for dense passage retrieval](https://doi.org/10.18653/v1/2023.acl-long.125 "").In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2244–2258, Toronto, Canada. Association for Computational Linguistics.
* Wang et al. (2023b)Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023b.[Self-knowledge guided retrieval augmentation for large language models](https://doi.org/10.18653/v1/2023.findings-emnlp.691 "").In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 10303–10315, Singapore. Association for Computational Linguistics.
* Wolf et al. (2020)Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020.[Transformers: State-of-the-art natural language processing](https://doi.org/10.18653/v1/2020.emnlp-demos.6 "").In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics.
* Xiao et al. (2023)Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023.[C-pack: Packaged resources to advance general chinese embedding](http://arxiv.org/abs/2309.07597 "").
* Xu et al. (2023)Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023.Recomp: Improving retrieval-augmented lms with compression and selective augmentation.*arXiv preprint arXiv:2310.04408*.
* Yang et al. (2023)Haoyan Yang, Zhitao Li, Yong Zhang, Jianzong Wang, Ning Cheng, Ming Li, and Jing Xiao. 2023.[PRCA: Fitting black-box large language models for retrieval question answering via pluggable reward-driven contextual adapter](https://doi.org/10.18653/v1/2023.emnlp-main.326 "").In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 5364–5375, Singapore. Association for Computational Linguistics.
* Yang et al. (2018)Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018.[HotpotQA: A dataset for diverse, explainable multi-hop question answering](https://doi.org/10.18653/v1/D18-1259 "").In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.
* Yao et al. (2023)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023.ReAct: Synergizing reasoning and acting in language models.In *International Conference on Learning Representations*.
* Zhang et al. (2023)Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023.[Retrieve anything to augment large language models](http://arxiv.org/abs/2310.07554 "").
* Ziegler et al. (2019)Daniel M. Ziegler, Nisan Stiennon, Jeff Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019.[Fine-tuning language models from human preferences](https://api.semanticscholar.org/CorpusID:202660943 "").*ArXiv*, abs/1909.08593.
