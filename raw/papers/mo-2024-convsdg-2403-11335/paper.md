# ConvSDG: Session Data Generation for Conversational Search

Fengran Mo

RALI, Université de Montréal

Montréal, Québec, Canada

fengran.mo@umontreal.ca

Chen Qu

University of Massachusetts Amherst

Amherst, MA, USA

mail@cqu.org

Bole Yi

RALI, Université de Montréal

Montréal, Québec, Canada

bole.yi@umontreal.ca

Kaiyu Huang

Beijing Jiaotong University

Beijing, China

kyhuang@bjtu.edu.cn

Kelong Mao

Renmin University of China

Beijing, China

mkl@ruc.edu.cn

Jian-Yun Nie*

RALI, Université de Montréal

Montréal, Québec, Canada

nie@iro.umontreal.ca

# ABSTRACT

Conversational search provides a more convenient interface for users to search by allowing multi-turn interaction with the search engine. However, the effectiveness of the conversational dense retrieval methods is limited by the scarcity of training data required for their fine-tuning. Thus, generating more training conversational sessions with relevant labels could potentially improve search performance. Based on the promising capabilities of large language models (LLMs) on text generation, we propose ConvSDG, a simple yet effective framework to explore the feasibility of boosting conversational search by using LLM for session data generation. Within this framework, we design dialogue/session-level and query-level data generation with unsupervised and semi-supervised learning, according to the availability of relevance judgments. The generated data are used to fine-tune the conversational dense retriever. Extensive experiments on four widely used datasets demonstrate the effectiveness and broad applicability of our ConvSDG framework compared with several strong baselines.

# CCS CONCEPTS

- Computing methodologies  $\rightarrow$  Discourse, dialogue and pragmatics; - Information systems  $\rightarrow$  Information retrieval.

# KEYWORDS

Conversational Search, Session Data Generation, Large Language Model

# ACM Reference Format:

Fengran Mo, Bole Yi, Kelong Mao, Chen Qu, Kaiyu Huang, and Jian-Yun Nie. 2024. ConvSDG: Session Data Generation for Conversational Search. In Companion Proceedings of the ACM Web Conference 2024 (WWW '24 Companion), May 13-17, 2024, Singapore, Singapore. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3589335.3651940

# 1 INTRODUCTION

Conversational search is an emerging area within information retrieval that is poised to become the future of search engines [13]. The systems empower users to engage in interactive, multi-turn conversations when searching for information, simplifying the process of addressing intricate information needs. The central hurdle lies in accurately understanding users' genuine search intent, given that their queries are context-dependent and prone to linguistic issues like omission, coreference, and ambiguity [9].

When tackling conversational search, an intuitive approach is to use conversational query rewriting (CQR). This method involves breaking down the task by employing rewrite models to convert the query of the current turn into a de-contextualized one and then conducting an ad-hoc search using the rewritten query. This approach allows for the use of existing retrievers for the search process. However, it is challenging to directly optimize the rewriting towards search [21, 27, 32, 48]. Another approach, known as conversational dense retrieval (CDR), focuses on training a conversational dense retriever to grasp the search intent by implicitly learning the latent representations of encoded queries and passages. Unlike CQR, the conversational dense retriever has the ability to naturally learn from the relevance signals between queries and passages.

Nonetheless, the current CQR and CDR techniques, which are trained on limited data, struggle to deliver satisfactory performance due to the prevalence of the long-tail phenomenon [28, 33] within conversational sessions and the scarcity of available samples in existing datasets [8-10]. Creating conversational search datasets manually is a costly and difficult endeavor, so an intuitive approach is to automatically enrich the session data for model training. While some prior studies [6, 29] have demonstrated its feasibility, the additional session data generated in this manner often lack the necessary power for continuous model improvement, and the generation process remains complex. Furthermore, these approaches still rely on the assumption that there is a substantial amount of relevant in-domain data available to build data augmentation models. The recent success of large language models (LLMs) [4, 40, 46, 53], which excel in generating texts, has brought notable advancements to the field of information retrieval [31]. These LLMs are now being applied to support various techniques within the field, such as query expansion [44], query generation [3, 7], and document prediction [14, 25]. Motivated by these developments, our research

explores the potential of harnessing LLMs for the automatic generation of session data, adapted appropriately to enhance conversational search performance. In essence, we aim to address the following research questions related to the utilization of LLMs:

RQ1: Is it feasible to exploit automatic session data generation for conversational search models to achieve comparable or better performance than relying on manually constructed datasets?  
RQ2: Can we improve conversational search performance by augmenting the diversity of session queries via query rewriting and existing annotations?

In order to address these inquiries, we introduce ConvSDG, a data augmentation framework aimed at employing Large Language Models (LLMs) to accomplish Session Data Generation for Conversational search. Our framework leverages the robust text generation capabilities of LLMs to automatically generate session data that can adapt effectively to the demands of conversational search scenarios. With well-defined supervision signals, this approach mitigates the problem of limited data availability and enhances the performance of conversational dense retrieval. Specifically, we designed two different prompts in our framework, each tailored to specific scenarios, enabling LLMs to generate dialogue-level session data and query-level augmented session data. Subsequently, we create or assign appropriate supervision signals for each query turn, catering to both unsupervised and semi-supervised learning settings. Finally, the generated session data, along with the annotations, are used to fine-tune the conversational dense retriever. We carry out comprehensive experiments using four widely used conversational search datasets and compare ConvSDG against several strong baselines, demonstrating its superior performance.

Our contributions are summarized as follows:

(1) We propose a simple yet effective framework to automatically generate session data for conversational search, showing the feasibility of using automatic data to train the models.  
(2) We develop two approaches for instructing the LLM to produce session data, at both dialogue and query levels. Additionally, we generate the necessary supervision signals to facilitate conversational dense retrieval with different learning manners.  
(3) We demonstrate the effectiveness of ConvSDG by achieving better results compared to models trained on manually curated data across four datasets and under two distinct settings. The analysis offers additional insights into the potential of automatic data generation to enhance conversational search.

# 2 RELATED WORK

# 2.1 Conversational Search

Conversational query rewriting (CQR) and conversational dense retrieval (CDR) represent the two primary approaches to conversational search. CQR focuses on transforming context-dependent queries within a search session into stand-alone queries. Common methods involve selecting relevant tokens from the search session [23, 35, 43] and training a generative rewriter model using human rewritten queries paired with their respective sessions [22, 26, 41, 50]. Some research efforts incorporate reinforcement learning [5, 48] or ranking signals [27, 32] to align the generation process with the downstream search task. In contrast, CDR utilizes conversational search session data to perform end-to-end dense

retrieval. To enhance conversational search performance, advanced techniques like context denoising [20, 28, 30, 33, 36, 51], data augmentation [6, 21, 29], and the mining of challenging negative examples [19, 34] have been explored.

# 2.2 Large Language Models for Data Generation

The quantity and quality of data hold significant value across various research domains within natural language processing (NLP) and information retrieval (IR). The emergence of pre-trained language models [11, 37], and more recently, large language models [4, 40, 46], has opened up new opportunities for automatic text generation. For example, some studies utilize these language models to generate data for a wide array of NLP tasks, including text classification [45], acquiring commonsense knowledge [47], natural language inference [24], open-domain dialogue generation [54], and sequence labeling [12]. The enhanced performance observed in downstream tasks through these approaches validates the effectiveness of employing language models for data generation.

# 2.3 LLM-based Data Generation for IR

There are also several other approaches using large language models (LLMs) to generate data for ad-hoc IR: to generate queries from a document [3, 7], to generate a document from a query [14, 25, 44], etc. Different from them, our work concentrates on investigating how LLMs can be harnessed to create conversational search data, an underexplored area in the existing literature. A recent work - CONVERSER [15] focuses on few-shot query generation with in-context learning. It relies on two-stage generation and the needed generated samples are quite large, while we only require one-stage generation with much less generated samples (higher efficiency). Besides, this work is narrower in its scope compared to ours, and its effectiveness, evaluated on the CAsT-19 dataset only, is much lower than our method and other state-of-the-art baselines.

# 3 METHODOLOGY

# 3.1 Task Definition

Conversational search tries to identify relevant passages  $p^*$  from a vast collection  $C$  in response to the current query ( $n$ -th)  $q_n$ . This process is based on the context provided by the ongoing conversation session  $S = \{q_i\}_{i=1}^{n-1}$ . Each query turn within a session depends on the preceding context, necessitating the conversational retriever to possess the capability to comprehend the user's search intent. Thus, having access to high-quality and adequate conversational search session data is important. The goal of this paper is to generate new session data  $S' = \{q_i'\}_{i=1}^n$  based on LLMs and produce a series of query-passage pairs  $\{(q_i', p_i')\}_{i=1}^n$  for fine-tuning conversational dense retrieval.

# 3.2 Overview

The construction of existing conversational search datasets [8-10] heavily relies on human effort, resulting in insufficient data samples to adequately support fine-tuning of end-to-end conversational dense retrievers. Drawing inspiration from recent successes in harnessing Large Language Models (LLMs) for data generation in various downstream tasks, we introduce the ConvSDG framework. This

Figure 1: Overview of ConvSDG. Three parts are included: (1) Two prompts for session data generation at different levels, (2) Produce supervision signals for generated data, PRF for session generation and existing annotations for query augmented, (3) Conduct conversational dense retrieval fine-tuning with the generated data.

framework explores the potent generative capabilities of LLMs to create high-quality conversational session data for conversational search, regardless of whether relevant judgments are available or not. In the case where relevant judgments are unavailable, we employ LLMs to efficiently generate the entire conversational sessions at the dialogue-level, using only the topic description. We then apply pseudo-relevance feedback as supervision signals. In contrast, when relevant judgments exist in the dataset, we perform query-level augmentation by rephrasing the query formulation for each turn, recognizing that queries with the same search intent can be expressed in multiple ways.

The overview of our ConvSDG is depicted in Fig. 1. It consists of three main steps: (1) Guiding the LLM to generate session data at two different levels, (2) Generating supervision signals for each generated query turn, and (3) Conducting fine-tuning of conversational dense retriever based on the generated data.

# 3.3 Dialogue-level Session Generation

A conversational session typically centers around a particular topic [1], like "animals", and each turn explores different aspects of that topic, such as "habits" or "various breeds". To mimic the real-world scenario, it is essential to consider specific conversational phenomena, such as ensuring coherent transitions between turns, handling co-references, and addressing instances of omission, when constructing these conversational sessions [52]. Based on our initial experiments and insights gleaned from existing research [18, 54], we have found that generating a conversation session one turn at a time using generative models does not yield high-quality results.

Additionally, we have noticed that maintaining turn-to-turn coherence for Large Language Models (LLMs) solely through prompt instructions is challenging. As a solution, we opt for dialogue-level session data generation, which involves creating the entire conversation session in one go by providing a specific topic description. This approach helps avoid the generation of inconsistent query turns.

In detail, we begin by sampling the topic description for a session from existing datasets<sup>1</sup>, which we then use to create a prompt instruction. Our prompt instruction template is structured as [Instruction, Topic Information], enabling the LLM to generate an entire session in one go. A comprehensive illustration is shown in Fig. 2.

Once we have the generated session data, we require relevance judgments that establish the connection between query turns and passages for the fine-tuning of the conversational dense retriever. In practice, obtaining such annotations is significantly more costly compared to acquiring conversation session data [8]. To enhance the flexibility of our framework, we attempt to train models in an unsupervised manner, i.e. we do not rely on relevance judgments provided by human experts. Instead, we adopt the idea of pseudo-relevance feedback [39] to create pseudo supervision signals for each query turn. It is worth noting that there is not a single fixed method for this purpose, and we leave further exploration of this area for future research.

Concretely, we perform off-the-shelf retrieval on each query turn, selecting three passages from top-5 at random as pseudo-relevant documents for that specific turn. However, it is important to consider the format of the input query used for this off-the-shelf

# Dialogue-Level Session Generation

Topic: The Neolithic Revolution

Prompt: Generate a long conversation between human and system. In this conversation, human asks and system answers, the conversation topic would be + Topic.

Query_1: Can you tell me about the Neolithic Revolution?

Answer_1: The Neolithic Revolution was a major turning point in human history that occurred ...

Query_2: Why was this transition so important?

Answer_2: Some of the most common goat breeds are Alpine, ...

Query_3: Can you tell me about Alpine goats?

Answer_3: The adoption of agriculture led to the development of permanent settlements ...

中

Query_8: What are some examples of societies that went through the Neolithic Revolution?

Answer_8: There were many societies that underwent a Neolithic transition, including ancient Egypt ...

# Query-Level Augmented Generation

Prompt: Transform one query + Input 查询 + into another query with same meaning. Only print the transformed query.

Input 查询: Tell me about the history of linguistics as a field.

Rewrite_1: What is the background and evolution of the field of linguistics?

Rewrite_2: What is the historical background and development of linguistics as a discipline?

Input 查询: What kind of problems can I expect?

Rewrite_1: What sort of difficulties should I anticipate?

Rewrite_2: What types of challenges should I prepare for?

中

Input 查询: How do I prepare for it?

Rewrite_1: What steps should I take to get ready for it?

Rewrite_2: What actions must I undertake to be ready for it?

Figure 2: An example to illustrate the conversational session data generation for both dialogue-level (left) and query-level (right).

retrieval because the original queries in the conversational session are not stand-alone and always rely on the context of the ongoing conversation. To prevent topic shifts and find an optimal solution, we explore four query forms that take contextual information into account: (1)  $q + a$ : which concatenates the current query and the corresponding hypothetical answer generated by LLM, (2)  $q + a + topic$ : which combines the current query, its answer, and the session topic information, (3)  $convq + topic$ : which concatenates the current query, all previous queries, and the session topic information, (4)  $convq + conva + topic$ : which combines the current query, all previous queries and their corresponding answers, along with the session topic information. Ultimately, we select the  $q + a + topic$  format for reformulating the input query due to its demonstrated effectiveness, as discussed later in Sec. 5.2.

# 3.4 Query-level Augmented Generation

While utilizing pseudo-relevance feedback to create supervision signals is efficient and often yields comparable results to fully supervised methods, there is still a potential downside of introducing false positive signals that could misguide model training. To mitigate this risk, we can leverage the limited relevance judgments provided by human experts in existing conversational search datasets to carry out query-level augmented generation. Our objective is to generate additional conversational search session data based on the original annotations, specifically for each query turn. The underlying assumption here is that the sequence of conversation should not be unique. In other words, the same user search intent can be expressed in different natural language forms, leading to various conversational sessions on the same topic. This variability is common in real-world scenarios.

Our method operates on the principle of generating new data by making adjustments to existing data points, guided by prior knowledge about the problems' underlying structure. The augmented data points, derived from labeled data within our framework, can be directly applied in semi-supervised learning through consistency regularization. "Semi-supervised" means we combine the original data points with manual labels and the generated data points without manual labels for model training. In practice, we prompt the LLM to rewrite each query, providing an alternative natural language expression with the same meaning. This instruction template adheres to the format [Instruction, Input Query], as illustrated in Fig. 2. By repeating this process  $t$  times, we generate  $t$  augmented queries for each original query turn, effectively expanding the initial dataset by a factor of  $t$ . Subsequently, the original relevance judgments for each query turn are associated with each corresponding augmented query, serving as supervision signals.

# 3.5 Conversational Dense Retrieval

We conduct fine-tuning for conversational dense retrieval using the session data we have generated and the associated supervision signals. For this task, we employ a widely used ad-hoc search retriever, ANCE [49], which serves as both the query and passage encoder, denoted as  $\mathcal{F}_Q$  and  $\mathcal{F}_P$ , respectively. In this process, we consider all preceding queries within the same session to reformulate the current query turn  $q_{n}^{ref}$ , expressed as

$$
q _ {n} ^ {\text {r e f}} = q _ {1} \circ \dots q _ {i} \dots \circ q _ {n - 1} \circ q _ {n}, \quad i \in [ 1, n - 1 ] \tag {1}
$$

where  $\circ$  denotes concatenation. Then, a similarity function  $S$  based on dot product is applied to score a candidate passage  $p$  as:

$$
\mathcal {S} (q _ {n} ^ {r e f}, p) = \mathcal {F} _ {Q} (q _ {n} ^ {r e f}) ^ {T} \cdot \mathcal {F} _ {P} (p) \tag {2}
$$

During the training phase, we update only the query encoder, while the passage encoder is frozen. The training objective follows the widely used contrastive learning loss:

$$
\mathcal {L} = \frac {e ^ {\mathcal {S} \left(q _ {n} ^ {r e f} , p ^ {+}\right)}}{e ^ {\mathcal {S} \left(q _ {n} ^ {r e f} , p ^ {+}\right)} + \sum_ {p _ {n} ^ {-} \in C ^ {-}} e ^ {\mathcal {S} \left(q _ {n} ^ {r e f} , p ^ {-}\right)}} \tag {3}
$$

where the  $p^+$  and  $p^-$  denote the positive and negative samples for each query turn. During the inference phase, we perform the Approximate Nearest Neighbors (ANN) search based on the dense index using Faisss [16].

# 4 EXPERIMENTAL SETTINGS

# 4.1 Datasets and Evaluation Metrics

We carry out extensive experiments on four widely used conversational search datasets: CAsT-19 [8], CAsT-20 [9], CAsT-21 [10], and TopiOCQA [1]. The three CAsT datasets are curated by the experts of the TREC Conversational Assistance Track (CAsT) and each dataset comprises information-seeking conversations encompassing hundreds of turns in total. The TopiOCQA dataset addresses the novel challenge of topic switching, a common phenomenon in real-world scenarios. Table 1 provides an overview of the fundamental statistics of the datasets. Following previous studies [9, 10, 20, 26], we employ Mean Reciprocal Rank (MRR), NDCG, and Recall as our evaluation metrics, computed with the pytrec_eval tool [42].

# 4.2 Baseline Models

We compare ConvSDG with the following models: (1) BM25 [38]: A widely used unsupervised sparse retrieval model. (2) ConvDR [51]: A conversational dense retrieval model based on ANCE retriever, containing both zero-shot and few-shot manner, which is supervised by both conversational search data and manual rewritten queries. (3) ZeCo [20]: A zero-shot conversational search method that matches only the contextualized terms of the current query with passages based on ColBERT [17]. (4) LLMQR: A conversational query rewriting method based on the large language model ChatGPT (gpt-turbo-3.5-4k) without supervision signals to directly rewrite the current turn with the given conversation session. (5) CONVERSER [15]: A few-shot two-stage conversational query generation method based on the large language model for training conversational dense retrievers. (6) T5QR [22]: A conversational query rewriting method based on the T5 model using human rewritten queries in the QReCC dataset [2]. (7) ConvGQR [32]: A conversational query reformulation method by combining query rewrite and expansion, which leverages human rewritten queries and gold answers in QReCC dataset as generation objectives. (8) w/o Aug. [30]: A conversational dense retriever that fine-tunes ANCE with the original (non-augmented) conversational search data. We use it as the baseline of our methods to see the impact of data augmentation. The fine-tuning process is only based on the CAsT-19 training set.

Table 1: Statistics of the three CAsT and TopiOCQA datasets.  

<table><tr><td rowspan="2">Dataset</td><td colspan="2">CAsT-19</td><td>CAsT-20</td><td>CAsT-21</td><td colspan="2">TopiOCQA</td></tr><tr><td>Train</td><td>Test</td><td>Test</td><td>Test</td><td>Train</td><td>Test</td></tr><tr><td># Conversations</td><td>30</td><td>20</td><td>25</td><td>18</td><td>3,509</td><td>205</td></tr><tr><td># Turns (Queries)</td><td>108</td><td>173</td><td>208</td><td>157</td><td>45,450</td><td>2,514</td></tr><tr><td># Passages/Docs</td><td colspan="2">38M</td><td>38M</td><td>40M</td><td colspan="2">25M</td></tr></table>

# 4.3 Implementation Details

We utilize OpenAI's ChatGPT (gpt-turbo-3.5-4k) API for both dialogue-level session generation and query-level augmented generation with the default hyper-parameters. For the pseudo relevance supervision signals, we randomly select three passages from the top-5 retrieved passages using PRF. The ANCE system serves as the backbone model for fine-tuning conversational dense retrieval, with maximum input lengths truncated at 64 for queries, 384 for passages, and 512 for concatenated sessions. Model training employs a batch size of 16 with 5 epochs. Further details can be found in our released code repository<sup>3</sup>.

# 4.4 Experimental Scenarios

We evaluate our method on the following two training scenarios, with and without the relevance judgment, and compare with the suitable baseline models:

Dialogue-level augmentation w/o relevance judgment: We utilize solely the topic information from the CAsT-19 and TopiOCQA training sets for ConvSDG to perform dialogue-level session generation. Subsequently, we assess its performance on the respective test sets of all three CAsT datasets and TopiOCQA. Consequently, in the absence of existing relevance judgments, conversational dense retrieval fine-tuning is carried out following an unsupervised learning approach. The methods we compare include unsupervised and zero-shot methods, as well as the direct use of LLM.

Query-level augmentation w/ relevance judgment: We employ ConvSDG for query-level augmented generation and use the limited relevance judgments available in the CAsT-19 training set. The evaluation is then carried out on three CAsT datasets. As a result, conversational dense retrieval fine-tuning takes place in a semi-supervised learning manner, with the compared methods being supervised and trained using the same data samples.

# 5 EXPERIMENTAL RESULTS

# 5.1 Main Results

The overall performance comparisons on CAsT and TopiOCQA datasets with different settings are presented in Table 2 and Table 3.

In the absence of relevance judgments with dialogue-level augmentation, as shown in Table 2, we observe that ConvSDG, with unsupervised training, outperforms the compared methods across most evaluation metrics for both the CAsT and TopiOCQA datasets. In particular, it exhibits a significant relative improvement of  $19.2\%$  in MRR and  $17.7\%$  in NDCG@3 over the second-best results on the

Table 2: Performance of two different settings on CAsT datasets.  $\dagger$  denotes significant improvements with t-test at  $p < 0.05$  over all compared methods and bold indicates the best results in corresponding settings. The turns of CONVERSER are quoted from the original paper and the turns of our ConvSDG with relevance judgment are expanded two times and combined with the original 745 turns in the original training set.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Turn</td><td colspan="3">CAsT-19</td><td colspan="3">CAsT-20</td><td colspan="3">CAsT-21</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@100</td><td>MRR</td><td>NDCG@3</td><td>Recall@100</td><td>MRR</td><td>NDCG@3</td><td>Recall@100</td></tr><tr><td colspan="11">Dialogue-level augmentation w/o relevance judgement</td></tr><tr><td>BM25</td><td>-</td><td>39.7</td><td>18.0</td><td>20.1</td><td>13.9</td><td>7.2</td><td>11.5</td><td>30.3</td><td>16.6</td><td>24.9</td></tr><tr><td>ZeCo</td><td>-</td><td>-</td><td>23.8</td><td>21.6</td><td>-</td><td>17.6</td><td>20.0</td><td>-</td><td>23.4</td><td>26.7</td></tr><tr><td>ConvDR</td><td>-</td><td>42.0</td><td>24.7</td><td>18.3</td><td>23.4</td><td>15.0</td><td>15.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLMQR</td><td>-</td><td>57.8</td><td>35.0</td><td>27.7</td><td>36.8</td><td>24.5</td><td>28.2</td><td>42.1</td><td>28.2</td><td>31.3</td></tr><tr><td>CONVERSER</td><td>230k</td><td>35.8</td><td>21.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ConvSDG (Ours)</td><td>1,704</td><td>59.5†</td><td>32.1</td><td>33.5†</td><td>37.9†</td><td>25.3†</td><td>34.9†</td><td>50.2†</td><td>33.2†</td><td>37.5†</td></tr><tr><td colspan="11">Query-level augmentation w/ relevance judgement</td></tr><tr><td>T5QR</td><td>2,235</td><td>52.8</td><td>31.3</td><td>25.3</td><td>29.7</td><td>19.0</td><td>24.2</td><td>30.3</td><td>20.5</td><td>20.6</td></tr><tr><td>ConvGQR</td><td>2,235</td><td>61.0</td><td>34.6</td><td>30.3</td><td>35.1</td><td>24.3</td><td>30.3</td><td>37.6</td><td>24.6</td><td>28.4</td></tr><tr><td>ConvDR</td><td>2,235</td><td>62.1</td><td>35.0</td><td>29.7</td><td>34.6</td><td>23.6</td><td>28.8</td><td>37.6</td><td>25.2</td><td>31.4</td></tr><tr><td>w/o Aug.</td><td>745</td><td>56.8</td><td>31.5</td><td>29.2</td><td>34.2</td><td>22.6</td><td>32.6</td><td>45.6</td><td>29.8</td><td>35.2</td></tr><tr><td>ConvSDG (Ours)</td><td>2,235</td><td>60.6</td><td>35.3</td><td>32.1†</td><td>36.5†</td><td>24.2</td><td>34.2†</td><td>47.2†</td><td>30.8†</td><td>36.8†</td></tr></table>

Table 3: Performance on TopiOCQA dataset.  $\dagger$  denotes significant improvements with t-test at  $p < 0.05$  over all compared methods and bold indicates the best results (except for the results for reference).  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Turn</td><td colspan="4">TopiOCQA</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@10</td><td>Recall@100</td></tr><tr><td colspan="6">Dialogue-level augmentation w/o relevance judgement</td></tr><tr><td>BM25</td><td>-</td><td>10.7</td><td>9.7</td><td>11.2</td><td>26.7</td></tr><tr><td>ConvDR</td><td>-</td><td>10.3</td><td>9.1</td><td>15.7</td><td>35.4</td></tr><tr><td>ConvSDG (Ours)</td><td>5,231</td><td>21.4†</td><td>19.9†</td><td>37.8†</td><td>58.0†</td></tr><tr><td colspan="6">Query-level augmentation w/ relevance judgement</td></tr><tr><td>T5QR</td><td>5,231</td><td>18.4</td><td>17.6</td><td>30.8</td><td>45.3</td></tr><tr><td>ConvGQR</td><td>5,231</td><td>8.0</td><td>7.3</td><td>14.3</td><td>25.5</td></tr><tr><td colspan="6">For Reference</td></tr><tr><td>T5QR</td><td>63,501</td><td>23.0</td><td>22.2</td><td>37.6</td><td>54.4</td></tr><tr><td>ConvGQR</td><td>63,501</td><td>25.6</td><td>24.3</td><td>41.8</td><td>58.8</td></tr></table>

challenging CAsT-21 dataset. Besides, with fewer training samples, it even outperforms models trained with manual relevance judgments on CAsT-20 and CAsT-21. These findings confirm the quality and usefulness of our automatically generated data. Moreover, our approach outperforms all compared methods by applying the same augmented samples and is on par with supervised training methods with full original training samples on the TopiOCQA dataset (as shown in Table 3). These results emphasize the importance of conversational dense retrieval fine-tuning, especially when compared to zero-shot methods like ZeCo and ConvDR. Overall, our approach addresses the data scarcity challenge effectively through

the automatic generation of conversational search data, validating our motivation.

When considering the scenario with query-level augmentation relevance judgments in the training data, as presented in Table 2, ConvSDG continues to outperform the compared methods across most evaluation metrics with the same training samples on manual datasets. Specifically, it achieves substantial improvements by a relative boost of  $25.5\%$  in MRR,  $22.0\%$  in NDCG@3, and  $17.2\%$  in Recall@100 over the second-best results on the more challenging CAsT-21 dataset. These enhancements, facilitated by the augmented query turns, show the effectiveness of our approach in rewriting the queries in the original datasets while retaining the same search

Figure 3: Effectiveness of using generated supervision signals by different query forms based on ANCE dense retriever.

Figure 4: Effectiveness of different sizes of generated data used for conversational fine-tuning with unsupervised (left) and semi-supervised (right) learning manner.


intent. However, compared to dialogue-level ConvSDG without relevance judgment, ConvSDG with query-level augmentation is not more effective, even though it leverages human relevance judgments. This result suggests that there is still room for further improvement in system performance by fully harnessing the existing relevance annotations and enhancing the diversity of the conversational sessions. It is also worth noting that the compared methods do not perform well on the more challenging CAsT-21 dataset. This discrepancy could be attributed to the fact that these methods depend on human rewritten queries, and our generated augmented session queries may not align perfectly with these original annotations. This observation implies that CDR methods might be more suitable than CQR when being trained on the augmented queries.

# 5.2 Effectiveness of Supervision Signals

We present the retrieval performance achieved using four different query input forms, as discussed in Sec. 3.3, for generating pseudo-relevance feedback (PRF) based on the ANCE dense retriever in Fig. 3. In this setup, the top results obtained from PRF are employed as the pseudo supervision signals for the corresponding session query turns generated to fine-tune the conversational dense retriever. Our findings indicate that using only the information from the current turn, i.e., the current turn's query and the corresponding hypothetical answer generated by LLM, the method tends to yield better results compared to incorporating the entire conversation context, such as concatenating with queries or answers from previous turns. This observation can be attributed to the fact that ad-hoc search retrievers lack the capability to represent an entire

conversation session effectively, and the underlying search intent within the current turn query is context-dependent. Moreover, we observe that the inclusion of topic information in each conversation session proves beneficial. Indeed, the topic information will help the generation process of augmented data to produce more relevant and topic-related data, which is better than that produced without the topic information. Although topic information may not always be available during the inference stage, we can still leverage it to construct training datasets.

# 5.3 Effectiveness of Generated Data Size

We present an analysis of the effectiveness of employing varying sizes of generated data for conversational dense retrieval fine-tuning in two different scenarios across three CAsT datasets, as depicted in Fig. 4. The percentage counted by the whole generated query turns for both unsupervised and semi-supervised settings. For unsupervised learning, we observe a notable improvement in system performance as the volume of utilized data increases. This observation demonstrates the pivotal role of augmented data in mitigating the data scarcity issue, and it suggests that there is further potential for enhancing model performance by expanding the dataset even more. On the other hand, for semi-supervised learning, we note that the fine-tuned models do not exhibit improved performance consistently compared to models trained solely on the original training set until a sufficient amount of generated data is added. This indicates that the generated data points might alter the data distribution, and it implies the need for appropriate filtering mechanisms. Nonetheless, the performance enhancement achieved with the addition of full-sized data underlines the effectiveness of our generated data for model training.

# 6 CONCLUSION

In this paper, we introduce ConvSDG, a novel framework for generating session data with LLMs for conversational search. By harnessing the robust text generation capabilities of LLMs, we are able to fine-tune conversational dense retrieval using the session data generated through unsupervised or semi-supervised learning methods. Our experimental findings, based on four public datasets, demonstrate the remarkable effectiveness of our approach, as it outperforms existing comparable methods and even surpasses fully supervised models. Additionally, we analyze some crucial impacts of automatically constructing conversational search session data, offering insights for future exploration in this domain. Our study

shows the effectiveness of using an LLM to generate additional training data for fine-tuning a dense retriever. It enriches the already extensive body of studies trying to exploit LLMs for search. More research is still required to find the best way to leverage LLMs for enhancing conversational search.

# REFERENCES

[1] Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries, and Siva Reddy. 2022. TopiOCQA: Open-domain Conversational Question Answering with Topic Switching. Transactions of the Association for Computational Linguistics 10 (2022), 468-483.  
[2] Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, and Srinivas Chappidi. 2021. Open-Domain Question Answering Goes Conversational via Question Rewriting. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 520-534.  
[3] Luiz Henrique Bonifacio, Hugo Abonizio, Marzieh Fadaee, and Rodrigo Frassetto Nogueira. 2022. InPars: Data Augmentation for Information Retrieval using Large Language Models. CoRR abs/2202.05144 (2022). arXiv:2202.05144 https://arxiv.org/abs/2202.05144  
[4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurlIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/ hash/1457cd6bfcb4967418bbf8ac142f64a-Abstraction.html  
[5] Zhiyu Chen, Jie Zhao, Anjie Fang, Besnik Fetahu, Rokhlenko Oleg, and Shervin Malmasi. 2022. Reinforced Question Rewriting for Conversational Question Answering. (2022).  
[6] Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Y Zhao, Aida Amini, Qazi Mamunur Rashid, Mike Green, and Kelvin Guu. 2022. Dialog Inpainting: Turning Documents into Dialogs. In International Conference on Machine Learning. PMLR, 4558-4586.  
[7] Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. 2023. Promptagator: Few-shot Dense Retrieval From 8 Examples. In 11th International Conference on Learning Representations, ICLR 2023.  
[8] Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2020. TREC CAsT 2019: The conversational assistance track overview. In In Proceedings of TREC.  
[9] Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2021. CAsT 2020: The Conversational Assistance Track Overview.. In In Proceedings of TREC.  
[10] Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2022. TREC CAsT 2021: The conversational assistance track overview. In In Proceedings of TREC.  
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 4171-4186.  
[12] Bosheng Ding, Chengwei Qin, Linlin Liu, Lidong Bing, Shafiq Joty, and Boyang Li. 2023. Is gpt-3 a good data annotator?. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. 11173-11195.  
[13] Jianfeng Gao, Chenyan Xiong, Paul Bennett, and Nick Craswell. 2022. Neural approaches to conversational information retrieval. arXiv preprint arXiv:2201.05176 (2022).  
[14] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Precise Zero-Shot Dense Retrieval without Relevance Labels. CoRR abs/2212.10496 (2022).  
[15] Chao-Wei Huang, Chen-Yu Hsu, Tsu-Yuan Hsu, Chen-An Li, and Yun-Nung Chen. 2023. CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation. arXiv preprint arXiv:2309.06748 (2023).  
[16] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data 7, 3 (2019), 535-547.  
[17] Omar Khattab and Matei Zaharia, 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval (SIGIR). ACM, 39-48.  
[18] Minju Kim, Chaehyeong Kim, Yong Ho Song, Seung-won Hwang, and Jinyoung Yeo. 2022. BotsTalk: Machine-sourced Framework for Automatic Curation of Large-scale Multi-skill Dialogue Datasets. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 5149-5170.

[19] Sungdong Kim and Gangwoo Kim. 2022. Saving dense retriever from shortcut dependency in conversational search. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 10278-10287.  
[20] Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot Query Contextualization for Conversational Search. In Proceedings of the 45th International ACM SIGIR conference on research and development in Information Retrieval (SIGIR).  
[21] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. Contextualized Query Embeddings for Conversational Search. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 1004-1015.  
[22] Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. 2020. Conversational question reformulation via sequence-to-sequence architectures and pretrained language models. arXiv preprint arXiv:2004.01909 (2020).  
[23] Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. 2021. Multi-stage conversational passage retrieval: An approach to fusing term importance estimation and neural query rewriting. ACM Transactions on Information Systems (TOIS) 39, 4 (2021), 1-29.  
[24] Alisa Liu, Swabha Swayamdipta, Noah A Smith, and Yejin Choi. 2022. WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation. In Findings of the Association for Computational Linguistics: EMNLP 2022. 6826-6847.  
[25] Iain Mackie, Shubham Chatterjee, and Jeffrey Dalton. 2023. Generative Relevance Feedback with Large Language Models. CoRR abs/2304.13157 (2023). https://doi.org/10.48550/arXiv.2304.13157 arXiv:2304.13157  
[26] Kelong Mao, Zhicheng Dou, Haonan Chen, Fengran Mo, and Hongjin Qian. 2023. Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search. arXiv preprint arXiv:2303.06573 (2023).  
[27] Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian, Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao Cao. 2023. Search-Oriented Conversational Query Editing. In Findings of the Association for Computational Linguistics: ACL 2023. 4160-4172.  
[28] Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022. Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval. In Proceedings of the 45th International ACM SIGIR conference on research and development in Information Retrieval (SIGIR).  
[29] Kelong Mao, Zhicheng Dou, Hongjin Qian, Fengran Mo, Xiaohua Cheng, and Zhao Cao. 2022. ConvTrans: Transforming Web Search Sessions for Conversational Dense Retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2935-2946.  
[30] Kelong Mao, Hongjin Qian, Fengran Mo, Zhicheng Dou, Bang Liu, Xiaohua Cheng, and Zhao Cao. 2023. Learning Denoised and Interpretable Session Representation for Conversational Search. In Proceedings of the ACM Web Conference 2023. 3193-3202.  
[31] Kelong Mao, Xi Xiao, Jieming Zhu, Biao Lu, Ruiming Tang, and Xiuqiang He. 2020. Item tagging for information retrieval: A tripartite graph neural network based approach. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2327-2336.  
[32] Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, and Jian-Yun Nie. 2023. ConvGQR: Generative Query Reformulation for Conversational Search. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. 4998-5012.  
[33] Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. 2023. Learning to Relate to Previous Turns in Conversational Search. In 29th ACM SIGKDD Conference On Knowledge Discover and Data Mining (SIGKDD).  
[34] Fengran Mo, Chen Qu, Kelong Mao, Tianyu Zhu, Zhan Su, Kaiyu Huang, and Jian-Yun Nie. 2024. History-Aware Conversational Dense Retrieval. arXiv preprint arXiv:2401.16659 (2024).  
[35] Hongjin Qian and Zhicheng Dou. 2022. Explicit Query Rewriting for Conversational Dense Retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 4725-4737.  
[36] Chen Qu, Liu Yang, Cen Chen, Minghui Qiu, W Bruce Croft, and Mohit Iyyer. 2020. Open-retrieval conversational question answering. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 539-548.  
[37] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8 (2019), 9.  
[38] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval 3, 4 (2009), 333-389.  
[39] Tao Tao and ChengXiang Zhai. 2007. An exploration of proximity measures in information retrieval. In Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 295-302.  
[40] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen,

Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Kathleen S. Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Agiera y Arcas, Claire Cui, Marian Croak, Ed H. Chi, and Quoc Le. 2022. LaMDA: Language Models for Dialog Applications. CoRR abs/2201.08239 (2022). arXiv:2201.08239 https://arxiv.org/abs/2201.08239  
[41] Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu, and Raviteja Anantha. 2021. Question rewriting for conversational question answering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 355-363.  
[42] Christophe Van Gysel and Maarten de Rijke. 2018. Pytrec_eval: An Extremely Fast Python Interface to trec_eval. In SIGIR. ACM.  
[43] Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, and Maarten de Rijke. 2020. Query resolution for conversational search with limited supervision. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 921-930.  
[44] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query Expansion with Large Language Models. CoRR abs/2303.07678 (2023). https://doi.org/10.48550/arXiv:2303.07678 arXiv:2303.07678  
[45] Zirui Wang, Adams Wei Yu, Orhan First, and Yuan Cao. 2021. Towards zero-label language learning. arXiv preprint arXiv:2109.09193 (2021).  
[46] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned Language Models are Zero-Shot Learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. https://openreview.net/forum?id=gEZrGCozdqR

[47] Peter West, Chandra Bhagavatula, Jack Hessel, Jena Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. 2022. Symbolic Knowledge Distillation: from General Language Models to Commonsense Models. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 4602-4625.  
[48] Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reitter, and Gaurav Singh Tomar. 2022. CONQRR: Conversational Query Rewriting for Retrieval with Reinforcement Learning. (2022).  
[49] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations.  
[50] Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul Bennett, Jianfeng Gao, and Zhiyuan Liu. 2020. Few-shot generative conversational query rewriting. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 1933-1936.  
[51] Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, and Zhiyuan Liu. 2021. Few-shot conversational dense retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 829-838.  
[52] Hamed Zamani, Johanne R Trippas, Jeff Dalton, Filip Radlinski, et al. 2023. Conversational information seeking. Foundations and Trends® in Information Retrieval 17, 3-4 (2023), 244-456.  
[53] Le Zhang, Yihong Wu, Fengran Mo, Jian-Yun Nie, and Aishwarya Agrawal. 2023. MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model. In Findings of the Association for Computational Linguistics: EMNLP 2023. 1195-1210.  
[54] Chujie Zheng, Sahand Sabour, Jiaxin Wen, Zheng Zhang, and Minlie Huang. 2023. Augesc: Dialogue augmentation with large language models for emotional support conversation. In Findings of the Association for Computational Linguistics: ACL 2023. 1552-1568.

# Footnotes:

Page 0: *Corresponding author Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. WW'24 Companion, May 13-17, 2024, Singapore, Singapore © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0172-6/24/05...$15.00 https://doi.org/10.1145/3589335.3651940 
Page 2: <sup>1</sup>Topic descriptions can be sampled in any suitable ways. 
Page 4: 2These statistics consider only the turns that possess relevance labels. <sup>3</sup>https://github.com/fengranMark/ConvSDG 
