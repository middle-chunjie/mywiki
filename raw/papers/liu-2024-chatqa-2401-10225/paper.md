ChatQA: Building GPT-4 Level Conversational QA Models
======================================================

Zihan LiuWei PingRajarshi RoyPeng XuChankyu LeeMohammad ShoeybiBryan Catanzaro

###### Abstract

In this work, we introduce ChatQA, a family of conversational question answering (QA) models that obtain GPT-4 level accuracies.
Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs).
To handle retrieval-augmented generation in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost.
Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.

Machine Learning, ICML

1 Introduction
--------------

Most recently, ChatGPT*(OpenAI, [2022](#bib.bib49 ""))* and its follow ups*(OpenAI, [2023](#bib.bib50 ""); Anthropic, [2023b](#bib.bib5 ""); Google, [2023](#bib.bib27 ""))* have led to the paradigm shift of building question answering (QA) models in production and research community.
In particular, the following aspects of the QA models are preferred in real-world applications: *i)*The users can interact with the QA models in a conversational way, thus one can easily raise follow-up questions; *ii)*The generalist models can generate answers in zero-shot manner without dataset-specific fine-tuning, while matching the accuracies of fine-tuned expert models; *iii)*The QA models are capable of integrating retrieved chunks of evidence in both open-domain or long document settings, where the provided context is much longer than the context window of LLM*(e.g., Anthropic, [2023a](#bib.bib4 ""); Xu et al., [2023b](#bib.bib74 ""))*.
To this end, we focus on the conversational QA covering these three aspects.

However, building a conversational QA model that can match the accuracy of the state-of-the-art black-box model, i.e., GPT-4*(OpenAI, [2023](#bib.bib50 ""))*, is still a grand challenge for the research community.
In this work, we introduce ChatQA-70B, a white-box conversational QA model with a detailed specification and GPT-4 level accuracy, through the proposed two-stage instruction tuning recipe, an enhanced retriever for retrieval-augmented generation (RAG) in conversational QA, and a detailed data curation process.

Specifically, we make the following contributions:

1. 1.

    We propose a two-stage instruction tuning method and a dataset curation recipe that can largely enhance LLM’s capability of integrating user provided or retrieved context for zero-shot conversational QA tasks.
    We demonstrate that our method significantly outperforms regular instruction tuning or RLHF-based recipes (e.g., Llama2-Chat).

2. 2.

    For RAG in conversational QA, we show that fine-tuning the state-of-the-art single-turn query retriever on both human-annotated or synthetic multi-turn QA dataset works as well as utilizing the state-of-the-art LLM-based query rewriting model, i.e., GPT-3.5-turbo*(OpenAI, [2022](#bib.bib49 ""))*.

3. 3.

    We build a family of ChatQA models based on Llama2-7B, Llama2-13B, Llama2-70B*(Touvron et al., [2023](#bib.bib60 ""))*, and in-house GPT-8B, GPT-22B models.
    We conduct comprehensive study on 10 conversational QA datasets, including 5 datasets with long documents that need retrieval and 3 datasets with tables.
    In terms of average score, our ChatQA-70B model (54.14) can outperform GPT-3.5-turbo (50.37) and GPT-4 (53.90) without utilizing any synthetic data from ChatGPT models.

4. 4.

    We study the “unanswerable” scenario, where the desired answer is not included in the provided or retrieved context, thus the LLM needs to generate “cannot answer” to avoid hallucination.
    Our ChatQA-70B outperforms GPT-3.5-turbo in this regard, while still has a slight gap compared to GPT-4 (around 3.5%).

We organize the rest of the paper as follows.
We discuss related work in §[2](#S2 "2 Related Work ‣ ChatQA: Building GPT-4 Level Conversational QA Models").
We introduce the two-stage instruction tuning method and data curation for ChatQA in §[3](#S3 "3 ChatQA ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), and study retrieval in conversational QA in §[4](#S4 "4 Retrieval for Multi-Turn QA ‣ ChatQA: Building GPT-4 Level Conversational QA Models").
We present the experimental setup in §[5](#S5 "5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), results in §[6](#S6 "6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), and conclude the paper in §[7](#S7 "7 Conclusion ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

<img src='x1.png' alt='Refer to caption' title='' width='392' height='82' />

*Figure 1: Two-stage instruction tuning framework for ChatQA.*

2 Related Work
--------------

### 2.1 Conversational QA

Question answering in a conversational way naturally improves user experiences by addressing follow-up questions. The model can also raise clarification questions for users if necessary, which can reduce hallucination.
Thus, it becomes the default format of deploying QA models in production*(e.g. OpenAI, [2022](#bib.bib49 ""); Google, [2023](#bib.bib27 ""); Anthropic, [2023b](#bib.bib5 ""))*.

In recent years, many conversational QA datasets have been introduced, where the models are asked to answer questions based on provided context or documents.
The provided context or documents can be: *i*) text-only documents from various domains*(Feng et al., [2020](#bib.bib23 ""); Anantha et al., [2021](#bib.bib3 ""); Saeidi et al., [2018](#bib.bib58 ""); Adlakha et al., [2022](#bib.bib1 ""); Aliannejadi et al., [2021](#bib.bib2 ""); Reddy et al., [2019](#bib.bib57 ""); Qu et al., [2020](#bib.bib53 ""); Wu et al., [2023](#bib.bib72 ""); Deng et al., [2022](#bib.bib18 ""); Guo et al., [2021](#bib.bib28 ""); Choi et al., [2018](#bib.bib10 ""); Campos et al., [2020](#bib.bib7 ""))*, or *ii*) documents comprising plain text along with tables*(Pasupat \& Liang, [2015](#bib.bib52 ""); Nakamura et al., [2022](#bib.bib47 ""); Chen et al., [2022a](#bib.bib8 ""))*.

In contrast to the latest LLM-based generalist solution*(e.g., OpenAI, [2022](#bib.bib49 ""))*, most of the previous studies focus on fine-tuned expert models on specific domains or datasets*(Feng et al., [2020](#bib.bib23 ""); Izacard \& Grave, [2021](#bib.bib32 ""); Chen et al., [2022a](#bib.bib8 ""); Gao et al., [2022](#bib.bib25 ""); Nakamura et al., [2022](#bib.bib47 ""); Adlakha et al., [2022](#bib.bib1 ""); Wu et al., [2023](#bib.bib72 ""))*.

### 2.2 Retrieval for Multi-Turn QA

Conversational QA involves retrieval-augmented generation (RAG) in open-domain setting, or when the provided documents are longer than the context window of LLM.
The dense retrievers are usually trained to retrieve the top-*k* relevant chunks given a single question*(e.g., Lin et al., [2023a](#bib.bib39 ""); Wang et al., [2022a](#bib.bib65 ""); Izacard et al., [2022](#bib.bib33 ""))*.
In conversational QA, the follow-up questions (e.g., with pronouns referring to entities mentioned in the previous conversation) may have insufficient information for retrieval, while feeding them along with all of the dialogue history can be redundant, thus leading to sub-optimal results.

#### 2.2.1 Conversational Query Rewriting

Most of the previous solutions are query rewriting methods.
The latest turn of question is rewritten to be a standalone query without additional information from previous dialogue history*(Vakulenko et al., [2021a](#bib.bib62 ""); Ye et al., [2023](#bib.bib75 ""); Mo et al., [2023](#bib.bib45 ""))*, so it can be directly used by retrieval model to retrieve relevant context*(Vakulenko et al., [2021b](#bib.bib63 ""); Mele et al., [2021](#bib.bib43 ""); Raposo et al., [2022](#bib.bib56 ""); Mo et al., [2023](#bib.bib45 ""))*.
Many datasets have been collected to facilitate this line of research*(Elgohary et al., [2019](#bib.bib21 ""); Chu et al., [2020](#bib.bib11 ""); Qu et al., [2020](#bib.bib53 ""); Anantha et al., [2021](#bib.bib3 ""); Brabant et al., [2022](#bib.bib6 ""))*, alongside multiple proposed query rewriting methods*(Ishii et al., [2022](#bib.bib30 ""); Yu et al., [2020](#bib.bib76 ""); Wu et al., [2022](#bib.bib71 ""); Del Tredici et al., [2021](#bib.bib17 ""); Chen et al., [2022b](#bib.bib9 ""); Galimzhanova et al., [2023](#bib.bib24 ""))*.
For example, *Wu et al. ([2022](#bib.bib71 ""))* and *Chen et al. ([2022b](#bib.bib9 ""))* proposed to use reinforcement learning methods for the query rewriting. *Yu et al. ([2020](#bib.bib76 ""))* investigated few-shot generative models like GPT-2 for query rewriting. *Galimzhanova et al. ([2023](#bib.bib24 ""))* studied instruction tuned GPT-3.5-turbo and showed that it achieved state-of-the-art results for conversational query rewriting.

#### 2.2.2 Fine-tuning Retriever for multi-turn QA

Some previous work fine-tune a single-turn query retriever on in-domain conversational query and context pairs*(Feng et al., [2020](#bib.bib23 ""); Gao et al., [2022](#bib.bib25 ""); Adlakha et al., [2022](#bib.bib1 ""); Wu et al., [2023](#bib.bib72 ""))*, so it can directly take a concatenation of dialog history and current query as input.
In this work, we focus on the zero-shot evaluation.
We fine-tune a single-turn query retriever on a high-quality multi-turn dataset. Then, we evaluate zero-shot capability of the fine-tuned retriever on five benchmark datasets.
Surprisingly, we find this simple approach can obtain comparable zero-shot results as the state-of-the-art query rewriting model, i.e., GPT-3.5-turbo.

### 2.3 Instruction Tuning

The goal of instruction tuning is to equip LLMs with the capability to follow natural language instructions*(Wei et al., [2022a](#bib.bib69 ""); Sanh et al., [2022](#bib.bib59 ""); Mishra et al., [2022](#bib.bib44 ""); Iyer et al., [2022](#bib.bib31 ""); Du et al., [2022](#bib.bib19 ""); Ouyang et al., [2022](#bib.bib51 ""); Wang et al., [2023b](#bib.bib68 ""); Zhang et al., [2023b](#bib.bib78 ""); Gao et al., [2023](#bib.bib26 ""); Chung et al., [2022](#bib.bib12 ""); Muennighoff et al., [2022](#bib.bib46 ""); Xu et al., [2023a](#bib.bib73 ""); Wang et al., [2022c](#bib.bib67 ""); Zhou et al., [2023](#bib.bib79 ""))*.
There has been a surge in the development of high-quality instruction tuning datasets, including FLAN*(Chung et al., [2022](#bib.bib12 ""))*, Self-Instruct*(Wang et al., [2022b](#bib.bib66 ""))*, unnatural Instructions*(Honovich et al., [2022](#bib.bib29 ""))*, Dolly*(Conover et al., [2023b](#bib.bib14 ""))*, and OpenAssistant *(Köpf et al., [2023](#bib.bib37 ""))*.

Although numerous research on instruction tuning has been conducted, a few works focused on improving RAG or context awareness generation for QA. *Lin et al. ([2023b](#bib.bib40 ""))* introduced a retrieval-augmented instruction tuning method, which appends top-*k* retrieved chunks for LLM fine-tuning. *Wang et al. ([2023a](#bib.bib64 ""))* applied instruction tuning after retrieval-augmented pretraining.
In contrast, we propose a two-stage instruction tuning method to improve generation with retrieval or provided context. We find that appending top-*k* retrieved chunks for LLM fine-tuning does not help for a wide range of conversation QA tasks (see §[6.3](#S6.SS3 "6.3 Top-k Chunks for Stage-2 Instruction Tuning ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models") for details).

Similar to the latest work*(Zhang et al., [2023a](#bib.bib77 ""))*, we demonstrate that adding a small amount of “unanswerable” samples in instruction tuning can steer the model to generate “cannot answer” output when it is necessary, thus significantly reduce hallucination.

After the release of ChatGPT*(OpenAI, [2022](#bib.bib49 ""))*, instruction tuning became an indispensable ingredient to build the state-of-the-art dialogue agent which has astonishing zero-shot capability to a wide range of tasks.
The conversational QA capability plays a crucial role in a dialogue agent, yet there has been limited research dedicated to this vital aspect.

3 ChatQA
--------

In this section, we propose a two-stage instruction tuning method for ChatQA. See Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ ChatQA: Building GPT-4 Level Conversational QA Models") for an illustration.
Our method starts with a pretrained LLM foundation model.
At stage-1, we apply supervised fine-tuning (SFT) as in*Ouyang et al. ([2022](#bib.bib51 ""))* on a blend of instruction-following and dialog datasets.
After that, our model exhibits good capability to follow instructions as a conversational agent. However, its capability for contextualized or RAG-based QA remains limited. Hence, we introduce a subsequent stage, called context-enhanced instruction tuning, which is designed specifically for enhancing our model’s capability for context-aware or retrieval-augmented generation in conversational QA.

### 3.1 Stage-1: Supervised Fine-tuning

To construct a large and comprehensive supervised fine-tuning (SFT) dataset, we follow*Xu et al. ([2023b](#bib.bib74 ""))*, *Wang et al. ([2023a](#bib.bib64 ""))* and gather a combined set of 128K SFT samples from high-quality instruction tuning datasets. It consists of 1) a social dialogue dataset Soda*(Kim et al., [2022](#bib.bib34 ""))*, 2) a long-form QA dataset ELI5 containing elaborate answers*(Fan et al., [2019](#bib.bib22 ""))*, 3) FLAN and chain-of-thought datasets*(Wei et al., [2022b](#bib.bib70 ""); Chung et al., [2022](#bib.bib12 ""); Longpre et al., [2023](#bib.bib42 ""))*, 4) LLM synthetic instruction tuning datasets, including Self-Instruct*(Wang et al., [2022b](#bib.bib66 ""))* and Unnatural Instructions*(Honovich et al., [2022](#bib.bib29 ""))*, and 5) a private crowd-sourced conversational dataset, as well as two public human-written conversation datasets: OpenAssistant*(Köpf et al., [2023](#bib.bib36 ""))*, and Dolly*(Conover et al., [2023a](#bib.bib13 ""))*.

We unify the structure of all the SFT data in a conversational format. We first add a “System” role at the beginning to set up a general instruction guiding LLM to provide polite and helpful answers.
We also add “User” and “Assistant” roles to incorporate instruction and response pairs from the instruction tuning datasets.
We apply fine-tuning using this unified format on an LLM foundation model.

### 3.2 Stage-2: Context-Enhanced Instruction Tuning

To further enhance the model’s conversational QA capability over a given context, we conduct a second stage instruction tuning, which integrates contextualized QA datasets into the instruction tuning blend.
Specifically, the stage-2 instruction tuning datasets consist of a blend of contextualized single-turn QA and conversational QA datasets.
We present further details for the stage-2 instruction tuning datasets below.

#### 3.2.1 Human Annotated Data

In addition to the public available datasets, one of the key elements for stage-2 is to obtain a high-quality document-grounded conversational QA dataset.
We create a human-annotated conversational QA (called HumanAnnotatedConvQA) dataset only consisting of 7k dialogues. To build this dataset, we first collected 7k documents covering diverse topics from the Internet. Then, we instruct the annotator to act both as a curious user asking questions (and follow-up questions) about the document and as an agent to give the answers. We create a multi-turn conversation for each document, resulting in a total of 7k conversational QA dialogues with an average of 5 user-agent turns per dialogue. Details of data collection guideline can be found in Appendix[G.2](#A7.SS2 "G.2 Guidelines for Conversational QA Data Collection ‣ Appendix G Human Annotated Conversational QA Data ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

To reduce hallucinated answers in unanswerable cases, we aim to empower our model to explicitly indicate it when the answer cannot be found within the given context.
To obtain these unanswerable data samples, we requested annotators to identify all parts of the context locations that are relevant to the user’s question. Hence, it enabled us to construct unanswerable scenarios by deleting the text from the corresponding locations in the context.
After deleting the relevant text to the question, we use a sentence, “Sorry. I cannot find the answer based on the context”, as the response for the unanswerable questions.
Finally, we construct another 1.5k user-agent turns with unanswerable annotations, which provides a good trade-off of answerable and unanswerable cases (see §[6.5](#S6.SS5 "6.5 Evaluation of Unanswerable Case ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models") for details).

#### 3.2.2 Synthetic Data Generation

To validate the quality of the HumanAnnotatedConvQA, we leverage GPT-3.5-turbo to generate a synthetic conversational QA dataset given its powerful instruction-following and text generation capability.
Note that large-scale synthetic data for conversational QA has also been explored in *Dai et al. ([2022](#bib.bib15 ""))*.
In this work, we focus on midsize high-quality synthetic data for LLM fine-tuning.

The instruction for GPT-3.5-turbo comprises three parts: 1) system role guiding the model to provide helpful answers, 2) examples of conversational QA indicating the required data types, and 3) a document that directs the model to generate conversational QA based on its content.
We collect 7k documents (average $\sim$1k words per document) from common crawl, which cover a wide range of domains. Each document is used for generation of a single conversational QA sample, which leads to a total of 7k multi-turn QA dialogues with an average of 4.4 user-agent turns per dialogue (called SyntheticConvQA).

Similar to the HumanAnnotatedConvQA, we construct another 1.5k user-agent turns with unanswerable annotations in this synthetic dataset.
Since there are no annotations of the context location for the agent’s answer, we construct synthetic unanswerable samples from SyntheticConvQA.
Specifically, we first cut the document (for each dialogue) into different chunks. Then, we consider it as a valid unanswerable sample only when there are chunks that have “high overlaps” with the agent’s answer to be removed and the rest of the chunks show “low overlaps” with the agent’s answer. We use the 4-gram recall score between each chunk and the agent’s answer (which measures the ratios of the answer’s 4-gram phrases are within each chunk) as the metric to measure their overlaps, and consider it higher than 0.5 as “high overlaps” and lower than 0.1 as “low overlaps”.

#### 3.2.3 Training Blends

In this part, we introduce the details of training blends for stage-2 instruction tuning.
To boost the QA capability in handling tabular documents and arithmetic calculation, we add the TAT-QA dataset*(Zhu et al., [2021](#bib.bib80 ""))* which contains both elements. In addition, we integrate contextualized single-turn QA datasets to further strengthen the QA capability of our model. We also retain the stage-1 SFT dataset in the training blend to maintain the model’s instruction-following capability.

Finally, the training blend for stage-2 consists of:
1) A conversational QA dataset: HumanAnnotatedConvQA or SyntheticConvQA,111Unless specified otherwise, the experiments use HumanAnnotatedConvQA as the default setting. 2) single-turn QA datasets: DROP*(Dua et al., [2019](#bib.bib20 ""))*, NarrativeQA*(Kočiskỳ et al., [2018](#bib.bib35 ""))*, Quoref*(Dasigi et al., [2019](#bib.bib16 ""))*, ROPES*(Lin et al., [2019](#bib.bib38 ""))*, SQuAD1.1*(Rajpurkar et al., [2016](#bib.bib54 ""))*, SQuAD2.0*(Rajpurkar et al., [2018](#bib.bib55 ""))*, NewsQA*(Trischler et al., [2017](#bib.bib61 ""))*, TAT-QA*(Zhu et al., [2021](#bib.bib80 ""))*, and 3) all of SFT datasets from stage-1.

We follow the similar template as in stage-1 to unify all single-turn QA and conversational QA datasets. The differences are in two parts: 1) Following the system role, we append relevant context for the single-turn question or the multi-turn conversation, and 2) Just before the single-turn question or multi-turn conversation, we integrate further instruction based on the answer types of different QA datasets (e.g., short answer, long answer, arithmetic calculation). We use the format for SFT dataset from stage-1.222Details of the templates for both stage-1 and stage-2 instruction tuning as well as the synthetic data generation can be found in the Appendix[A](#A1 "Appendix A ChatQA Instruction Tuning ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

ModelsAverageDoc2DialQuACQReCCTopiOCQAINSCITtop-1top-5top-1top-5top-1top-5top-1top-5top-5*top-20*top-5*top-20**Adlakha et al. ([2022](#bib.bib1 ""))*---------70.40△--*Wu et al. ([2023](#bib.bib72 ""))*-----------71.10△E5-unsupervised† *(Wang et al., [2022a](#bib.bib65 ""))*31.5659.2223.0255.3343.4977.6844.7184.9926.2537.6720.3240.44E5-unsupervised + Rewrite‡33.2361.0225.5658.0046.0080.0145.5085.8927.5839.1521.5342.04E5-unsupervised + Fine-tune†47.7975.0045.2880.9646.5280.7453.3789.9141.0151.0752.7972.31Dragon† *(Lin et al., [2023a](#bib.bib39 ""))*46.2973.0943.3375.6156.8082.8646.1781.9657.6878.8027.4946.22Dragon + Rewrite‡54.4680.1347.6080.6047.1077.1551.7385.7873.0788.1952.7968.92Dragon + Fine-tune†52.7280.6748.9483.0152.6481.9550.7387.1767.8686.2843.4364.94- SyntheticConvQA♢52.9881.1548.6483.4754.7583.2349.6386.7064.4885.2447.4167.13

*Table 1: Retrieval results across five multi-turn QA datasets with the average top-1 and top-5 recall scores.
Compared to rewriting, fine-tuning performs much better on E5-unsupervised and is comparable on Dragon.
*Since the average context length in TopiOCQA and INSCIT is smaller than in other datasets, we report top-5 and top-20 to roughly match the context lengths of top-1 and top-5, respectively, in those datasets. †The inputs for these two models are a concatenation of the dialogue history and the current query. ‡The input for this model is the rewritten query. ♢denotes that the HumanAnnotatedConvQA dataset is replaced with the SyntheticConvQA for fine-tuning. △The numbers are not apple-to-apple comparison (e.g., they use the training set for fine-tuning).*

<img src='x2.png' alt='Refer to caption' title='' width='457' height='208' />

*Figure 2: Illustration of fine-tuning retriever for multi-turn QA.*

4 Retrieval for Multi-Turn QA
------------------------------

In conversational QA tasks, when a document becomes too lengthy to feed directly into LLMs, a retriever that can handle conversational queries becomes essential.
This conversational retriever encodes the concatenation of the dialogue history and the current query, and then retrieve relevant context from documents. After that, only the relevant context will be used as inputs for LLMs.
The state-of-the-art retrievers, e.g., Dragon*(Lin et al., [2023a](#bib.bib39 ""))*, are optimized for single-turn queries, resulting in a limited generalization capability for multi-turn conversational queries.
In Figure[2](#S3.F2 "Figure 2 ‣ 3.2.3 Training Blends ‣ 3.2 Stage-2: Context-Enhanced Instruction Tuning ‣ 3 ChatQA ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we depict our retriever fine-tuning method to alleviate this issue. We propose to use conversational query and context pairs for further fine-tuning a single-turn retriever to better cope with conversational inputs.

An alternative solution is conversational query rewriting method which uses a query rewriter to rewrite the current question based on the conversational history.
The rewritten query is then directly used as the input to a single-turn query retriever for retrieving relevant context.
In addition to the embedding and search cost, the query rewriting model introduces a large amount of extra computational expense to generate the rewritten query.

### 4.1 Fine-tuning Retriever for Multi-turn QA

To build a high-quality fine-tuning dataset, we leverage the conversational QA dataset from either the HumanAnnotatedConvQA or the SyntheticConvQA to construct conversational query and context pairs.

For the HumanAnnotatedConvQA, we directly take the annotations of the conversational query and context pairs, and use them to further fine-tune a single-turn query retriever.
For the SyntheticConvQA, we first cut each document in the conversational QA dataset into different chunks. Then, we calculate the 4-gram recall score between agent’s answer and each chunk. After that, we consider the chunk that has the highest recall score as the gold chunk for the current user’s question. Finally, the constructed conversational query and context pairs are used to fine-tune a single-turn query retriever.

### 4.2 Conversational Query Rewriting

To build powerful conversational query rewriting model, we take GPT-3.5-turbo as the rewriter given that *Galimzhanova et al. ([2023](#bib.bib24 ""))* demonstrated the state-of-the-art query rewriting results using GPT-3.5-turbo.
Similar to*Galimzhanova et al. ([2023](#bib.bib24 ""))*, we not only provide GPT-3.5-turbo with the rewriting task instruction, but also give it few-shot rewriting examples to enhance the quality of rewriting results.
More details can be found in Appendix[B.1](#A2.SS1 "B.1 Query Rewriting Prompts for GPT-3.5-turbo ‣ Appendix B More Details and Results for Retrieval in Conversational QA ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

### 4.3 Comparisons

In Table[1](#S3.T1 "Table 1 ‣ 3.2.3 Training Blends ‣ 3.2 Stage-2: Context-Enhanced Instruction Tuning ‣ 3 ChatQA ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we compare the query rewriting and fine-tuning methods across five datasets in the zero-shot setting. More details about these datasets can be found in §[5.2.1](#S5.SS2.SSS1 "5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"). We conduct experiments on a state-of-the-art retriever, Dragon*(Lin et al., [2023a](#bib.bib39 ""))*, and a strong unsupervised retriever, E5-unsupervised*(Wang et al., [2022a](#bib.bib65 ""))*, which is not finetuned on MS MACRO*(Nguyen et al., [2016](#bib.bib48 ""))*.
In terms of the experiments on Dragon, we find that fine-tuning performs marginally worse than query rewriting in average top-1 recall by 1.74%, while it achieves better results on average top-5 recall by 0.54%.
It demonstrates the effectiveness of the fine-tuning approach for the conversational retrieval.
In addition, we observe that the results are comparable between using HumanAnnotatedConvQA and SyntheticConvQA for fine-tuning. This highlights that our human-annotated dataset is in high-quality, and we *do not rely on* ChatGPT models for building the state-of-the-art multi-turn query retriever.

Surprisingly, fine-tuning performs significantly better than rewriting on E5-unsupervised. We conjecture that since E5-unsupervised does not use human-annotated query and context pairs in the pre-training stage, it leads to weak generalization for the high-quality rewritten query. In contrast, using a high-quality dataset to fine-tune E5-unsupervised brings a giant boost, with more than a 15% improvement on both average top-1 and top-5 recall scores.

Therefore, fine-tuning a good single-turn retriever on high-quality conversational query context pairs performs on par with leveraging the state-of-the-art rewriter. However, rewriting method requires extra computational time for autoregressive generation process and probably also API cost for using powerful models like GPT-3.5-turbo. In contrast, our proposed multi-turn fine-tuning bypasses these issues.
For the QA evaluations across these five datasets, we consistently use the retrieved top-5 results from the fine-tuning approach for all the QA models. We put more results on comparisons between rewriting and fine-tuning methods in the Appendix[B.2](#A2.SS2 "B.2 More Results for Retrieval in Conversational QA ‣ Appendix B More Details and Results for Retrieval in Conversational QA ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

5 Experimental Setup
--------------------

In this section, we present the details of our experimental setup for the conversational question answering task.

### 5.1 Baselines

We conduct experiments on different model sizes. First, to show the effectiveness of stage-2 context-enhanced instruction tuning, we compare against the Llama2-SFT-{7B, 13B, 70B}, which is the Llama2-7B/13B/70B foundation model after the stage-1 supervised fine-tuning (SFT). Second, we compare against Llama2-Chat-{7B, 13B, 70B} since Llama2-Chat models are shown to possess strong instruction following and conversational QA capabilities*(Touvron et al., [2023](#bib.bib60 ""))*.
Aside from Llama2 models, we also conduct experiments on our in-house GPT-{8B, 22B} foundation models, which are pretrained with 3.5 trillion tokens, and we compare against their stage-1 SFT baselines GPT-SFT-{8B, 22B}.
Finally, we compare against two very strong OpenAI models: GPT-3.5-turbo-0613 (4k) and GPT-4-0613 (8k).
For fair comparison,
when retrieval is needed, we use the same top-*k* retrieved chunks from our best retriever as the context for all baselines and our ChatQA models.
Note that we have carefully tuned the instructions for all the baselines to ensure they achieve as good as possible results.333The prompts for these baselines can be found in Appendix[C](#A3 "Appendix C Conversational QA Benchmarks ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

### 5.2 Evaluation Benchmarks

#### 5.2.1 Long Document Datasets

We collect five conversational QA datasets with long documents which cannot be directly fitted into LLMs with a sequence length of 4K tokens. Hence, we run our best multi-turn query retriever
to get top-5 relevant chunks as the inputs (experiments can be found in §[4.3](#S4.SS3 "4.3 Comparisons ‣ 4 Retrieval for Multi-Turn QA ‣ ChatQA: Building GPT-4 Level Conversational QA Models")).

* •

    Doc2Dial*(Feng et al., [2020](#bib.bib23 ""))* is a document-grounded conversational QA dataset covering four domains: DMV, SSA, VA, and Student Aid. Each sample comprises a dialogue where a user poses queries regarding the document, and an agent responds those questions. The average document length is around 101K words.

* •

    QuAC*(Choi et al., [2018](#bib.bib10 ""))* is based on Wikipedia documents. Originally, the document is short. Since each dialogue is linked to multiple Wikipedia URLs, we extract text from these links to increase the document size to approximately an average of 15K words. It contains unanswerable cases where answers cannot be found within the given context.

* •

    QReCC*(Anantha et al., [2021](#bib.bib3 ""))* is an open-domain conversational QA dataset across multiple sources.
    Similar to QuAC, each dialogue also has corresponding URLs. We extract text from those URLs to construct the documents. In the end, the average document size is around 5K words, with a maximum document size of 20K words.

* •

    TopiOCQA*(Adlakha et al., [2022](#bib.bib1 ""))* is grounded on the whole Wikipedia. It incorporates topic switching and requires the agent to search the entire Wikipedia for answers to user questions.

* •

    INSCIT*(Wu et al., [2023](#bib.bib72 ""))* is also grounded on the whole Wikipedia. It studies the case where user questions are under-specified and require clarification.

For Doc2Dial, QuAC, and QReCC, we segment documents into around 300-word chunks, and we retrieve top-5 relevant chunks as context for each user question. For TopioCQA and INSCIT, we follow their original segmentation, resulting in smaller chunks. Hence, we retrieved top-20 chunks to obtain similar context length to the first three datasets.

ModelsAverageDoc2DialQuACQReCCCoQADoQAConvFinQASQATopiOCQAHybridDialINSCITLlama2-7B-SFT34.8130.2619.2137.5562.7521.7634.4332.1832.8848.9628.16Llama2-7B-Chat38.8633.2725.8346.0272.2833.1536.5826.1436.6847.0231.67ChatQA-7B47.7137.8829.6946.9776.6141.5751.6161.8745.4554.5130.96GPT-8B-SFT34.4631.0320.0737.6959.2421.7215.4440.0638.1752.2928.86ChatQA-8B49.3636.7633.9545.5477.9044.6561.6860.7447.0353.8131.50Llama2-13B-SFT37.6930.6821.5938.2569.5221.7041.1437.8535.2652.2228.73Llama2-13B-Chat40.3434.7427.8947.1972.5032.6041.5425.3939.2549.8232.52ChatQA-13B50.8638.0534.2848.0677.2343.3165.4466.4148.8856.1930.79GPT-22B-SFT38.8332.9822.8339.9268.1922.8330.5447.0138.0754.2231.66ChatQA-22B53.5339.7136.8950.4780.3244.8872.0869.0848.8358.2034.11Llama2-70B-SFT43.2234.4225.6541.8873.0428.2146.6458.9037.2055.5230.71Llama2-70B-Chat45.2136.8732.4749.4080.4138.9746.8537.6244.3150.3534.88ChatQA-70B54.1438.9041.8248.0578.5751.9473.6969.1450.9856.4431.90- SyntheticConvQA♢54.0839.1938.3348.7379.8348.6576.4468.6351.3055.6833.98- w/o stage-1†52.1838.4337.5246.0873.5149.4272.1572.0851.2850.7430.56- w/o single-turn*52.2538.3037.8947.0876.7446.4372.4267.4149.8553.1633.18- w/o ConvQAData△48.9735.4728.1941.6875.8031.6873.8368.0545.9058.4030.70GPT-3.5-turbo (4k)50.3734.8337.1750.4679.3341.1173.1560.6344.3047.4235.27GPT-4 (8k)53.9034.1640.2952.0177.4243.3981.2879.2145.0949.8136.34

*Table 2: Zero-shot conversational QA results across 10 datasets. ♢ denotes that the HumanAnnotatedConvQA is replaced with the SyntheticConvQA. † denotes that the stage-1 (SFT) is removed and only ChatQA stage-2 tuning is applied. *denotes that the single-turn QA datasets are removed from the ChatQA stage-2 training blends. In both “w/o stage-1” and “w/o single-turn” settings, the SyntheticConvQA data is used. In terms of average scores, our ChatQA models greatly surpass SFT and Chat counterparts. Our best model, ChatQA-70B, slightly outperforms GPT-4. △ denotes that the HumanAnnotatedConvQA is removed from the ChatQA stage-2 training blends.*

#### 5.2.2 Short Document Datasets

To increase the diversity of document lengths, we collect five conversational QA datasets with short documents (less than 1.5K words). On average, 1 word will be tokenized into 1.5 tokens. Hence, the document can be directly fitted into LLMs with a sequence length of 4K tokens.

* •

    CoQA*(Reddy et al., [2019](#bib.bib57 ""))* is a conversational QA dataset with each dialogue grounded on a short passage. The answers are generally short, and the passages cover a wide range of domains like children’s stories, literature, mid/high school exams, news, Wikipedia.

* •

    DoQA*(Campos et al., [2020](#bib.bib7 ""))* covers three domains: cooking, travel, and movies collected from active Stack Exchange444<https://stackexchange.com/> forums. The dataset contains unanswerable cases where answers cannot be found within the given document.

* •

    ConvFinQA*(Chen et al., [2022a](#bib.bib8 ""))* is based on the Financial domain. Each document contains a single financial report table along with relevant text surrounding the table. This dataset involves arithmetic calculation and complex numerical reasoning.

* •

    SQA*(Pasupat \& Liang, [2015](#bib.bib52 ""))* is grounded on documents which contain only a single Table without any surrounding text. The documents are collected from Wikipedia, and the questions are highly compositional, which requires the model with robust table comprehension ability to give correct answers.

* •

    HybridDial*(Nakamura et al., [2022](#bib.bib47 ""))* is a conversational QA dataset grounded on documents containing both Wikipedia tabular and textual data. The questions are complex which requires reasoning over the documents.

Over all the 10 datasets, ConvFinQA, SQA, and HybridDial datasets contain tabular data in the documents, while documents of the remaining datasets are text-only.555Details of these benchmark datasets are in the Appendix[C](#A3 "Appendix C Conversational QA Benchmarks ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

### 5.3 Evaluation Metrics

Given that F1 score is the most commonly used automatic metric to assess QA models, we use it for all datasets except for ConvFinQA. In ConvFinQA, we follow*Chen et al. ([2022a](#bib.bib8 ""))* to use exact match metric since the answers in ConvFinQA are about extracting numbers from documents as well as arithmetic calculations. Hence, the answer only makes sense when it is exactly the same as the answer. When models generate the arithmetic formula, we will calculate its final result based on a calculator and compare it with the gold answer.

In addition, we also conduct human evaluations to assess the correctness of generated answers between our best model and GPT-4.

6 Results
---------

### 6.1 Main Results

#### 6.1.1 overview

In Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we compare different model variants and OpenAI models across 10 conversational QA datasets.

We find that our ChatQA method greatly improves the model’s conversational QA capability.
In terms of averaged scores, Llama2-Chat models surpass SFT model counterparts by a small margin, while our ChatQA models achieve around or over 10 point of absolute improvement over SFT and Chat counterparts. For example, ChatQA-13B improves on Llama2-13B-SFT and Llama2-13B-Chat by 13.17 (from 37.69 to 50.86) and 10.52 (from 40.34 to 50.86), respectively. This is because context-enhanced instruction fine-tuning enables the model to learn how to effectively extract useful information from retrieved or relevant context.

Compared to OpenAI models, our best model ChatQA-70B surpasses GPT-3.5-turbo by 3.77 average score and can outperform GPT-4 by 0.24 average score. In addition, a much smaller size of ChatQA-13B is able to marginally outperform GPT-3.5-turbo by an average score of 0.49.

Notably, our proposed stage-2 instruction tuning brings a significant boost to GPT-22B-SFT by 14.70 average score. As a result, ChatQA-22B is only behind GPT-4 by a small gap (53.53 vs. 53.90), despite having a considerably smaller model size than Llama2-70B.
This highlights the efficiency of the proposed instruction tuning method.

#### 6.1.2 Importance of Stage-1 SFT

In Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we conduct an ablation study on the importance of stage-1 SFT, which enhances the instruction-following capability of ChatQA. We remove the stage-1 SFT from the fine-tuning stages and only apply stage-2 context-enhanced instruction tuning on top of foundation LLM. We find that the average score drops 1.9 (from 54.08 to 52.18). Aside from the SQA, removing stage-1 makes the model consistently perform worse on other datasets. Results indicate that stage-1 still plays an important role, even through all of SFT datasets of stage-1 are blended in stage-2 instruction tuning as well.
We conclude that building instruction-following capability first is beneficial for stage-2 tuning.

#### 6.1.3 Effectiveness of Single-Turn Data

To investigate how single-turn QA datasets affect model’s multi-turn QA capability, we conduct an ablation study by removing them from the ChatQA-70B training blends in stage-2.
As shown in Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), incorporating single-turn QA datasets in the stage-2 training blends (ChatQA-70B) generally make the scores increase across all benchmark datasets, leading to an average improvement of 1.83 score. Interestingly, we observe improvement in ConvFinQA, SQA, and HybridDial (table-based datasets), despite the added single-turn QA datasets not having tabular data in the documents.
These results align with our intuitions. Adding single-turn datasets improves the model’s capability to extract answers from context, resulting in better scores in conversational QA datasets.

#### 6.1.4 Effectiveness of Conversational QA Data

We further explore the how conversational QA data affect the model’s multi-turn QA capability by removing HumanAnnotatedConvQA data from the ChatQA stage-2 training blends.
As illustrated in Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), “w/o ConvQAData” makes the results significantly worse than ChatQA-70B (average scores degrading from 54.08 to 48.97). We observe large degradation in datasets with text-based documents, such as QuAC, QReCC, and DoQA. However, the degradation in datasets with table-based documents (e.g., ConvFinQA, SQA) are small, thanks to having TAT-QA*(Zhu et al., [2021](#bib.bib80 ""))* in the training blends.

Interestingly, “w/o ConvQAData” can still outperform Llama2-70B-SFT (48.97 vs. 43.22) and Llama2-70B-Chat (48.97 vs. 45.21). This further demonstrates the effectiveness of our proposed stage-2 tuning, which enhances the model’s capability to handle the contextualized QA task.

#### 6.1.5 Human Annotated Data vs. GPT-3.5-Turbo Synthetic Data

In Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we also compare our ChatQA models using the 7k GPT-3.5-Turbo synthetic dataset (SyntheticConvQA) and our collected 7k human-annotated dataset (HumanAnnotatedConvQA). First, we find that both achieve comparable results in terms of average scores, which suggests that we do not need to rely on synthetic data from OpenAI models to build the state-of-the-art conversational QA models. Second, we find that using human-annotated data achieved significant improvements on QuAC and DoQA datasets. This can be attributed to the fact that the human-annotated data have higher quality on unanswerable cases which exists in QuAC and DoQA datasets. Eventually, it leads to the overall improvements on these two datasets. Detail results and analyses on unanswerable cases can be found in §[6.5](#S6.SS5 "6.5 Evaluation of Unanswerable Case ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

Ours WinTieGPT-4 WinAverage13.81%69.09%17.10%Doc2Dial14.29%68.00%17.71%QuAC11.67%73.33%15.00%QReCC11.11%77.22%11.67%CoQA7.78%80.00%12.22%DoQA22.78%57.78%19.44%ConvFinQA16.67%67.78%15.55%SQA11.11%61.67%27.22%TopiOCQA19.31%60.69%20.00%HybridDial7.78%78.33%13.89%INSCIT15.56%66.11%18.33%

*Table 3: Human evaluation (A/B testing) comparing our ChatQA-70B to GPT-4 over 10 datasets. In terms of average scores, our model and GPT-4 are tie most of the time (69.09%), and GPT-4 achieves slightly higher win rate ($\sim$3.3%) than ours.*

#### 6.1.6 Human Evaluation

Despite F1 scores being the most commonly used metrics for evaluating the quality of QA models, there are often multiple ways to answer questions, which makes the automatic metrics less than perfect. Therefore, we use human evaluations to further compare our ChatQA-70B with GPT-4. In this human evaluation, we ask annotators to verify the facts in ChatQA-70B and GPT-4’s outputs and determine which model provides a more accurate response to the question666More human evaluation setup can be found in the Appendix[D](#A4 "Appendix D Human Evaluation ‣ ChatQA: Building GPT-4 Level Conversational QA Models")..
The human evaluation results over the 10 datasets are shown in Table[3](#S6.T3 "Table 3 ‣ 6.1.5 Human Annotated Data vs. GPT-3.5-Turbo Synthetic Data ‣ 6.1 Main Results ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"). We first find that our ChatQA-70B and GPT-4 are tie most of the time (69.09%), and GPT-4 achieves slightly higher win rate ($\sim$3.3%) than ours. This further confirms our model has powerful capability to produce correct answers. Second, we find that our model achieves slightly better win rate than GPT-4 in ConvFinQA, which indicates the strong arithmetic calculation ability of our model. Third, we find that GPT-4 achieves significantly better win rate in SQA task, which suggests that there is still a gap between our model and GPT-4 on tabular reasoning task.

ModelsAvg-textAvg-tableAvg-retAvg-nonretChatQA-13B45.8062.6840.0161.72ChatQA-70B48.8866.4242.3365.96GPT-3.5-turbo (4k)46.0760.4040.4160.33GPT-4 (8k)46.9670.1041.5866.22

*Table 4: Fine-grained studies on average scores of different dataset types. Avg-text covers datasets where the documents only have text, including Doc2Dial, QuAC, QReCC, CoQA, DoQA, TopiOCQA, and INSCIT.
Avg-table covers datasets with table in the documents, including ConvFinQA, SQA, and HybridDial.
Avg-ret covers datasets with long documents requiring retrieval, including Doc2Dial, QuAC, QReCC, TopiOCQA, and INSCIT.
Avg-nonret covers datasets with short documents which do not require retrieval, including CoQA, DoQA, ConvFinQA, SQA, and HybridDial.*

### 6.2 Fine-grained Analyses

In Table[4](#S6.T4 "Table 4 ‣ 6.1.6 Human Evaluation ‣ 6.1 Main Results ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we further compared our models and OpenAI models across different dataset types in our conversational QA benchmarks. In the comparison between ChatQA-70B and GPT-4, ChatQA-70B achieves better results in text-only documents (avg-text), demonstrating its superior text understanding ability. Meanwhile, GPT-4 shows better QA capability in tabular data given the comparisons in avg-table.
Regarding the datasets that require or do not require retrieval, ChatQA-70B and GPT-4 are comparable (gaps are within an average score of 1).

In the comparison between ChatQA-13B and GPT-3.5-turbo, ChatQA-13B demonstrates better tabular QA capability, given the scores in Avg-table. ChatQA-13B also shows better scores in datasets where documents do not need retrieval, while being on par with GPT-3.5-turbo in text-only documents and documents that need retrieval.

ModelsAvg-CQAAvg-retAvg-nonretChatQA-70B54.1442.3365.96- w/ “top-5” chunks54.0442.9165.16

*Table 5: Ablation study on using “top-5” retrieved chunk as the context for the stage-2 instruction tuning. We report average scores on all 10 conversational QA datasets (Avg-CQA), five retrieval datasets (Avg-ret) and five non-retrieval datasets (Avg-nonret).*

ModelsAvg-BothAvg-QuACQuAC (no*)QuAC (yes*)Avg-DoQADoQA (no*)DoQA (yes*)Avg-CQAChatQA-70B- 1k unanswerable†76.8880.8975.1086.6772.8864.4981.2654.16- 1.5k unanswerable†77.2580.7677.6683.8573.7468.8178.6754.14- 2k unanswerable†77.1080.8277.5984.0573.3867.9578.8053.86- 2.5k unanswerable†75.8778.8173.7683.8572.9366.5479.3153.78- SyntheticConvQA♢69.8472.9255.3890.4266.7745.0988.4554.08GPT-3.5-turbo (4k)73.2778.3461.9194.7668.2151.9984.4350.37GPT-4 (8k)80.7387.4283.4591.3874.0574.2873.8253.90

*Table 6: Accuracies on answerable and unanswerable samples across QuAC and DoQA datasets. Avg-Both is the averaged score between QuAC and DoQA, and Avg-CQA is the average score across the 10 conversational QA datasets. ♢ denotes that the HumanAnnotatedConvQA is replaced with the SyntheticConvQA. *“no” denotes the unanswerable samples, while “yes” denotes the answerable samples. †We conduct ablation studies in terms of the number of unanswerable samples in HumanAnnotatedConvQA. We use 1.5k unanswerable samples for our final ChatQA-70B, as it produces both high-quality generation and less hallucination.*

### 6.3 Top-*k* Chunks for Stage-2 Instruction Tuning

For all the datasets used in stage-2 tuning, the context is provided as continuous paragraphs or documents containing the answer. In contrast, the model needs to handle the top-*k* retrieved chunks during inference for long documents.
To address this train/test mismatch, we investigate whether replacing some continuous paragraphs with retrieved top-*k* chunks will enhance the model’s robustness.

We use NarrativeQA for this study, since each question has a corresponding long document. Initially, we use a summary of the long document as the context, which contains the answer. To incorporate discontinuous contexts, we first cut the long document into 300-word chunks. Then, we use the Dragon retriever to retrieve the top-4 chunks related to the question as additional context. Finally, we consider the retrieved four chunks along with the summary of the long document as the “top-5” chunks. 777Note that, we do not directly use top-5 retrieved chunks for training, because they may not contain the answer. In such cases, fine-tuning the model to generate answer could encourage hallucination. We use this reconstructed NarrativeQA to replace the original one for the stage-2 instruction tuning.

In Table[5](#S6.T5 "Table 5 ‣ 6.2 Fine-grained Analyses ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we observe that using “top-5” chunks as the context for training leads to improvements on the datasets that require retrieval. But it degrades the performance on non-retrieval datasets. Overall, these two models perform comparable.
It is because incorporating “top-5” retrieved chunks in stage-2 tuning aligns with the inference stage where retrieval is needed, which improves the Avg-ret score. However, mixing continuous and discontinuous documents could make the stage-2 tuning less stable, leading to sub-optimal results on non-retrieval datasets. We believe more future work can be conducted regarding the balance of incorporating continuous context and top-*k* retrieved chunks in stage-2 tuning.

ModelsAvg.D2DQuACQReCCTopiOINSCITChatQA-70B42.3139.1938.3348.7351.3033.98- # of ctx: top-341.9137.2038.3548.9452.7832.27- # of ctx: top-1040.7137.0636.9547.6149.4032.53- ctx reverse ordering42.4839.0838.8549.6351.1633.69- ctx swing ordering42.3039.3538.0949.0950.9833.99- ctx random ordering42.0139.3238.2848.7950.1333.51- Dragon Retrieval40.5037.9238.4447.8850.3927.87

*Table 7: Ablation studies on input context across datasets that require retrieval. All models use SyntheticConvQA. D2D denotes Doc2Dial, and TopiO denotes TopiOCQA. We study the number of contexts used in inputs (# of ctx), context ordering (reverse, swing, random), and the use of retrieved context from the original Dragon. In comparison, ChatQA-70B (default setting) uses “Dragon + Fine-tune” to retrieve the top-5 contexts, and arranges them sequentially from the first to the fifth context in top-5.*

### 6.4 Ablation Studies for Inference Stage

In Table[7](#S6.T7 "Table 7 ‣ 6.3 Top-k Chunks for Stage-2 Instruction Tuning ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we show ablation studies on how the number of retrieved context/chunks, context ordering, and different retrievers affect the conversational QA results.

First, we find that using more contexts as inputs do not always improve the results. Utilizing top-5 contexts as input yields better results compared to using either top-3 or top-10 contexts. Intuitively, more contexts have higher probability to contain correct answers (better recall scores). As a result, using top-5 contexts achieves better results than using top-3. However, as the number of contexts further increases, the model may suffer from “lost in the middle” phenomenon*(Liu et al., [2023](#bib.bib41 ""))* and the difficulty of extracting answers from the provided context could also increase, which leads to inferior results by using top-10 contexts.

Second, we study how using different orderings of top-5 contexts affects the results. We compare sequential ordering (from $1$st context to $5$th context) to reverse ordering (from $5$th to $1$st context), swing ordering (given the “lost in the middle” phenomenon, we arrange the most relevant context to appear at the beginning and the end of the input context. Hence, the ordering becomes {$1$st, $3$rd, $5$th, $4$th, $2$nd}), and random ordering (random shuffle the top-5 contexts). We find using sequential ordering is comparable to using reverse and swing orderings, and random shuffling is slightly worse. Results indicate that our model excels in extracting the correct answer from lengthy contexts, regardless of the answer’s location. This is because, during the ChatQA fine-tuning, the answer’s location is randomly distributed within the context.

Third, we observe that when we replace “Dragon + Fine-tune” with the original non-finetuned Dragon retriever, the average score drops by 1.81 (from 42.31 to 40.50). In addition, the score drops significantly in INSCIT dataset (from 33.98 to 27.87) due to the large performance gap between the two retrievers (as shown in Table[1](#S3.T1 "Table 1 ‣ 3.2.3 Training Blends ‣ 3.2 Stage-2: Context-Enhanced Instruction Tuning ‣ 3 ChatQA ‣ ChatQA: Building GPT-4 Level Conversational QA Models")). Basically, when the quality of retrieval improves, it directly enhances the performance of question answering.

ContextConversationAssistant ResponsesThe process for Board hearings has changed. If you’re appealing a VA decisionunder the new process, keep reading below to learn how to request and preparefor a hearing. You’ll follow the new process if you’re : Appealing a VA decisionyou received on or after February 19 , 2019. Participating in the Rapid AppealsModernization Program Opting into the new process using VA Form 10182 Ifyou re using VA Form 9 to appeal a VA decision you received before February19, 2019, go to the Board of Veterans’ Appeals BVA Hearing page … The Boardof Veterans Appeals schedules hearings in docket order the order in which theyreceive requests …User: What can I learn at my Board hearing?Assistant: Do you want to learn exactly what to expect?…User: Is it necessary for me to consider requesting a Board hearing?Assistant: You can request a video conference hearing if you wantto provide testimony to a judge about your appeal.User: when will I follow the new process for a Board hearing?Assistant:ChatQA-70B: If you’re appealing a VA decision you received on orafter February 19, 2019, you’ll follow the new process.GPT-4: You’ll follow the new process for a Board hearing if you’reappealing a VA decision you received on or after February 19, 2019,participating in the Rapid Appeals Modernization Program, or optinginto the new process using VA Form 10182.Ground Truth: You will follow the new process if appealing a VAdecision if you received it on or after February 19, 2019.title: How did Arya survive this confrontation unscathed?There was no time or intention to turn her into White Walker. Let us look at thescene from S04E04. A white walker brings a baby holding in his lap and puts iton an ice table (or whatever it is) for the arrival of the Night King. … When Aryadrops the dagger, Night King looks surprised. This shows he doesn’t know whatwould happen next. There was no time to think and turn her into a white walkerby touch. There was no planning in this scene like S04E04. …User: How did Arya survive this confrontation unscathed?Assistant: Arya was quick enough to react, drop the dagger anddo the rest.User: How old is the child that they lay on the altar of ice?Assistant:ChatQA-70B: The child appeared to be a baby.GPT-4: The text does not provide information on the exact age of thechild that was laid on the ice altar.Ground Truth: a babyMassifType of nature reservePreserved areaChornohoraCarpathian Biosphere2476.8 haUholka / Wide MeadowCarpathian Biosphere11860 haSvydovetsCarpathian Biosphere3030.5 haMaramorosCarpathian Biosphere2243.6 haRozokPresov Preserved67.1 haJasmundJasmund National Park492.5 ha…User: which of the forests is a carpathian biosphere reserve?Assistant: Chornohora, Uholka / Wide Meadow, Svydovets,Maramoros, Kuziy / TrybushanyUser: which of those covers more than 3100 ha?Assistant:ChatQA-70B: Uholka / Wide MeadowGPT-4: Chornohora, Uholka / Wide Meadow, Svydovets, MaramorosGround Truth: Uholka / Wide Meadowtitle: Summary of enemy aircraft damaged/destroyed (John B. England)DateLocationNumberTypeStatus24 April 1944Munich, Germany3Me-110Destroyed27 May 1944Strasbourg, France1Me-109Damaged18 August 1944Paris, France0.5Me-109Destroyed13 September 1944Nordhausen, Germany2.5Me-109Destroyed6 October 1944Berlin, Germany1Me-109Damaged6 October 1944Berlin, Germany2Me-109Destroyed27 November 1944Magdeburg, Germany4FW-190Destroyed14 January 1945Berlin, Germany1Me-109Destroyed…User: what were all of the dates in which john b. england destroyed ordamaged enemy planes?Assistant: 24 April 1944, 27 May 1944, 18 August 1944, 13 September1944, 6 October 1944, 27 November 1944, 14 January 1945 …User: on which of those dates did john b. england destroy or damagemore than one enemy plane?Assistant:ChatQA-70B: 6 October 1944, 13 September 1944, 27 November 1944(missing 24 April 1944)GPT-4: 24 April 1944, 13 September 1944, 6 October 1944,27 November 1944Ground Truth: 24 April 1944, 13 September 1944, 6 October 1944,27 November 1944

*Table 8: Case studies on ChatQA-70B and GPT-4. We use blue color to highlight the relevant context for the last user question. And we use green and red color to highlight the correct and incorrect parts in the assistant responses.*

### 6.5 Evaluation of Unanswerable Case

#### 6.5.1 Evaluation Setup

In this section, we study another aspect of the model’s capability, which is to discern if a question can be answered within the provided context. Generating an answer in unanswerable case will lead to hallucination.
To allow this evaluation, we require the model to indicate it when no answer can be found in the given context.

We use QuAC and DoQA datasets which have such unanswerable cases to evaluate such capability. Specifically, for unanswerable case, we consider the model indicating that the question cannot be answered as correct888We collect a set of heuristic matching patterns from all generated samples to determine if the model suggests that the question is unanswerable. More details can be found in the Appendix[E](#A5 "Appendix E Unanswerable Case Evaluation ‣ ChatQA: Building GPT-4 Level Conversational QA Models")., and as for answerable cases, we consider the model not indicating the question is unanswerable as correct (i.e., the model giving an answer). Note that for answerable cases, we only select the samples where correct context is retrieved. Our model checkpoints (ChatQA-70B w/ 1.5k unanswerable and ChatQA-70B w/ SyntheticConvQA) are same ones reported in Table[2](#S5.T2 "Table 2 ‣ 5.2.1 Long Document Datasets ‣ 5.2 Evaluation Benchmarks ‣ 5 Experimental Setup ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

In the end, we calculate the average accuracy score of unanswerable and answerable cases as the final metric.
We consider this average accuracy as a reliable metric since it is in the same spirit of F1 metric which measures the harmonic mean of precision and recall scores.

#### 6.5.2 Results

In Table[6](#S6.T6 "Table 6 ‣ 6.2 Fine-grained Analyses ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we compare our models with OpenAI models across QuAC and DoQA datasets. First, we find that compared to using SyntheticConvQA, using HumanAnnotatedConvQA significantly increase the average accuracies on both QuAC and DoQA. It is because the unanswerable annotations are in higher quality for human-annotated data, which significantly improve the accuracy for unanswerable cases. Second, OpenAI models show powerful capability in this task, especially for GPT-4. Compared to them, our best model achieved a significantly better average accuracy than GPT-3.5-turbo, while we still has a slight gap compared to GPT-4 (around 3.5%).
Third, we find that models achieving higher accuracy on unanswerable samples tends to get lower accuracy on answerable samples, and vice versa. We speculate that when a model tends to be “aggressive” and offer somewhat relevant answers to those unanswerable questions, it will boost the accuracy for answerable cases, but reduces accuracy for unanswerable ones. Conversely, when a model is more “conservative” and strictly checks if the question can be answered, it will result in the opposite effects.

We conduct ablation studies in terms of the number of unanswerable samples in HumanAnnotatedConvQA. We find that using a small amount of unanswerable samples (e.g., 1.5k) is able to achieve remarkable results on the unanswerable evaluation, as well as the overall conversational QA score (i.e., Avg-CQA). Interestingly, incorporating more unanswerable samples does not necessarily lead to higher accuracy scores in the unanswerable evaluation.

### 6.6 Case Study

In Table[8](#S6.T8 "Table 8 ‣ 6.4 Ablation Studies for Inference Stage ‣ 6 Results ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we show four examples of ChatQA-70B and GPT-4 outputs. More examples can be found in Appendix[F](#A6 "Appendix F Case Study ‣ ChatQA: Building GPT-4 Level Conversational QA Models").
The first example (about Board hearing) is a simple information seeking problem, and both ChatQA-70B and GPT-4 answer correctly. In the second example (about Arya), the model needs to find implicit information (highlighted in blue) for giving the answer. GPT-4 tends to be safe on providing an answer, and it replies that the context does not provide exact information about the age which is also correct.

Both the third and fourth examples require the model to have good tabular understanding and reasoning ability. In the third example (about Massif), ChatQA-70B gives correct answer by comparing the size of preserved area against 3100 ha, while GPT-4 fails to do so.
In the fourth example (about John B. England), ChatQA-70B correctly lists three dates but misses one date, while GPT-4 correctly answers the question.

7 Conclusion
------------

In this paper, we build a family of ChatQA models, varying in model sizes from 7B to 70B.
Comprehensive evaluations on 10 conversational QA datasets show that our best ChatQA-70B model can remarkably outperform GPT-3.5-turbo and perform on par with GPT-4 without using any synthetic data from ChatGPT models.
In addition, we demonstrate that fine-tuning a single-turn query retriever using our curated conversational QA data performs comparably to the state-of-the-art LLM-based query rewriting model, without incurring extra computational time and potential API costs associated with rewriting.
Furthermore, we show that incorporating a small amount of “unanswerable” samples can significantly enhance our model’s capability to handle scenarios where answers are unavailable. The unanswerable case evaluation highlights that our best model ChatQA-70B only has a slight gap compared to GPT-4.

References
----------

* Adlakha et al. (2022)Adlakha, V., Dhuliawala, S., Suleman, K., de Vries, H., and Reddy, S.Topiocqa: Open-domain conversational question answering with topic switching.*TACL*, 2022.
* Aliannejadi et al. (2021)Aliannejadi, M., Kiseleva, J., Chuklin, A., Dalton, J., and Burtsev, M.Building and evaluating open-domain dialogue corpora with clarifying questions.In *EMNLP*, 2021.
* Anantha et al. (2021)Anantha, R., Vakulenko, S., Tu, Z., Longpre, S., Pulman, S., and Chappidi, S.Open-domain question answering goes conversational via question rewriting.In *NAACL*, 2021.
* Anthropic (2023a)Anthropic.Introducing 100k context windows, 2023a.
* Anthropic (2023b)Anthropic.Introducing Claude, 2023b.
* Brabant et al. (2022)Brabant, Q., Lecorvé, G., and Barahona, L. M. R.Coqar: Question rewriting on coqa.In *LREC*, 2022.
* Campos et al. (2020)Campos, J. A., Otegi, A., Soroa, A., Deriu, J. M., Cieliebak, M., and Agirre, E.Doqa-accessing domain-specific faqs via conversational qa.In *ACL*, 2020.
* Chen et al. (2022a)Chen, Z., Li, S., Smiley, C., Ma, Z., Shah, S., and Wang, W. Y.Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering.In *EMNLP*, 2022a.
* Chen et al. (2022b)Chen, Z., Zhao, J., Fang, A., Fetahu, B., Rokhlenko, O., and Malmasi, S.Reinforced question rewriting for conversational question answering.In *EMNLP*, 2022b.
* Choi et al. (2018)Choi, E., He, H., Iyyer, M., Yatskar, M., Yih, W.-t., Choi, Y., Liang, P., and Zettlemoyer, L.Quac: Question answering in context.In *EMNLP*, 2018.
* Chu et al. (2020)Chu, Z., Chen, M., Chen, J., Wang, M., Gimpel, K., Faruqui, M., and Si, X.How to ask better questions? a large-scale multi-domain dataset for rewriting ill-formed questions.In *AAAI*, 2020.
* Chung et al. (2022)Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Castro-Ros, A., Pellat, M., Robinson, K., Valter, D., Narang, S., Mishra, G., Yu, A., Zhao, V., Huang, Y., Dai, A., Yu, H., Petrov, S., Chi, E. H., Dean, J., Devlin, J., Roberts, A., Zhou, D., Le, Q. V., and Wei, J.Scaling instruction-finetuned language models.*arXiv preprint arXiv: 2210.11416*, 2022.
* Conover et al. (2023a)Conover, M., Hayes, M., Mathur, A., Meng, X., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., et al.Free dolly: Introducing the world’s first truly open instruction-tuned llm, 2023a.
* Conover et al. (2023b)Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R.Free Dolly: Introducing the world’s first truly open instruction-tuned llm, 2023b.
* Dai et al. (2022)Dai, Z., Chaganty, A. T., Zhao, V., Amini, A., Green, M., Rashid, Q., and Guu, K.Dialog inpainting: Turning documents to dialogs.In *ICML*, 2022.
* Dasigi et al. (2019)Dasigi, P., Liu, N. F., Marasović, A., Smith, N. A., and Gardner, M.Quoref: A reading comprehension dataset with questions requiring coreferential reasoning.In *EMNLP*, 2019.
* Del Tredici et al. (2021)Del Tredici, M., Barlacchi, G., Shen, X., Cheng, W., and de Gispert, A.Question rewriting for open-domain conversational qa: Best practices and limitations.In *CIKM*, 2021.
* Deng et al. (2022)Deng, Y., Lei, W., Zhang, W., Lam, W., and Chua, T.-S.Pacific: Towards proactive conversational question answering over tabular and textual data in finance.In *EMNLP*, 2022.
* Du et al. (2022)Du, Z., Qian, Y., Liu, X., Ding, M., Qiu, J., Yang, Z., and Tang, J.Glm: General language model pretraining with autoregressive blank infilling.In *ACL*, 2022.
* Dua et al. (2019)Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M.Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs.In *NAACL*, 2019.
* Elgohary et al. (2019)Elgohary, A., Peskov, D., and Boyd-Graber, J.Can you unpack that? learning to rewrite questions-in-context.In *EMNLP*, 2019.
* Fan et al. (2019)Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J., and Auli, M.Eli5: Long form question answering.In *ACL*, 2019.
* Feng et al. (2020)Feng, S., Wan, H., Gunasekara, C., Patel, S., Joshi, S., and Lastras, L.doc2dial: A goal-oriented document-grounded dialogue dataset.In *EMNLP*, 2020.
* Galimzhanova et al. (2023)Galimzhanova, E., Muntean, C. I., Nardini, F. M., Perego, R., and Rocchietti, G.Rewriting conversational utterances with instructed large language models.In *IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology*, 2023.
* Gao et al. (2022)Gao, C., Zhang, W., and Lam, W.Unigdd: A unified generative framework for goal-oriented document-grounded dialogue.In *ACL*, 2022.
* Gao et al. (2023)Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., and Wang, H.Retrieval-augmented generation for large language models: A survey.*arXiv preprint arXiv:2312.10997*, 2023.
* Google (2023)Google.Introducing bard, 2023.
* Guo et al. (2021)Guo, M., Zhang, M., Reddy, S., and Alikhani, M.Abg-coqa: Clarifying ambiguity in conversational question answering.In *AKBC*, 2021.
* Honovich et al. (2022)Honovich, O., Scialom, T., Levy, O., and Schick, T.Unnatural instructions: Tuning language models with (almost) no human labor.*arXiv preprint arXiv:2212.09689*, 2022.
* Ishii et al. (2022)Ishii, E., Xu, Y., Cahyawijaya, S., and Wilie, B.Can question rewriting help conversational question answering?In *Proceedings of the Third Workshop on Insights from Negative Results in NLP*, 2022.
* Iyer et al. (2022)Iyer, S., Lin, X. V., Pasunuru, R., Mihaylov, T., Simig, D., Yu, P., Shuster, K., Wang, T., Liu, Q., Koura, P. S., et al.Opt-iml: Scaling language model instruction meta learning through the lens of generalization.*arXiv preprint arXiv:2212.12017*, 2022.
* Izacard \& Grave (2021)Izacard, G. and Grave, É.Leveraging passage retrieval with generative models for open domain question answering.In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics*, 2021.
* Izacard et al. (2022)Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., and Grave, E.Unsupervised dense information retrieval with contrastive learning.*Transactions on Machine Learning Research*, 2022.
* Kim et al. (2022)Kim, H., Hessel, J., Jiang, L., Lu, X., Yu, Y., Zhou, P., Bras, R. L., Alikhani, M., Kim, G., Sap, M., et al.Soda: Million-scale dialogue distillation with social commonsense contextualization.*arXiv preprint arXiv:2212.10465*, 2022.
* Kočiskỳ et al. (2018)Kočiskỳ, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., and Grefenstette, E.The narrativeqa reading comprehension challenge.*TACL*, 2018.
* Köpf et al. (2023)Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., et al.Openassistant conversations–democratizing large language model alignment.*arXiv preprint arXiv:2304.07327*, 2023.
* Köpf et al. (2023)Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., ES, S., Suri, S., Glushkov, D., Dantuluri, A., Maguire, A., Schuhmann, C., Nguyen, H., and Mattick, A.Openassistant conversations - democratizing large language model alignment.*arXiv preprint arXiv: 2304.07327*, 2023.
* Lin et al. (2019)Lin, K., Tafjord, O., Clark, P., and Gardner, M.Reasoning over paragraph effects in situations.In *Proceedings of the 2nd Workshop on Machine Reading for Question Answering*, 2019.
* Lin et al. (2023a)Lin, S.-C., Asai, A., Li, M., Oguz, B., Lin, J., Mehdad, Y., Yih, W.-t., and Chen, X.How to train your dragon: Diverse augmentation towards generalizable dense retrieval.*arXiv preprint arXiv:2302.07452*, 2023a.
* Lin et al. (2023b)Lin, X. V., Chen, X., Chen, M., Shi, W., Lomeli, M., James, R., Rodriguez, P., Kahn, J., Szilvasy, G., Lewis, M., et al.Ra-dit: Retrieval-augmented dual instruction tuning.*arXiv preprint arXiv:2310.01352*, 2023b.
* Liu et al. (2023)Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P.Lost in the middle: How language models use long contexts.*arXiv preprint arXiv:2307.03172*, 2023.
* Longpre et al. (2023)Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., et al.The flan collection: Designing data and methods for effective instruction tuning.*arXiv preprint arXiv:2301.13688*, 2023.
* Mele et al. (2021)Mele, I., Muntean, C. I., Nardini, F. M., Perego, R., Tonellotto, N., and Frieder, O.Adaptive utterance rewriting for conversational search.*Information Processing \& Management*, 2021.
* Mishra et al. (2022)Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H.Cross-task generalization via natural language crowdsourcing instructions.In *ACL*, 2022.
* Mo et al. (2023)Mo, F., Mao, K., Zhu, Y., Wu, Y., Huang, K., and Nie, J.-Y.Convgqr: Generative query reformulation for conversational search.*arXiv preprint arXiv:2305.15645*, 2023.
* Muennighoff et al. (2022)Muennighoff, N., Wang, T., Sutawika, L., Roberts, A., Biderman, S., Scao, T. L., Bari, M. S., Shen, S., Yong, Z.-X., Schoelkopf, H., et al.Crosslingual generalization through multitask finetuning.*arXiv preprint arXiv:2211.01786*, 2022.
* Nakamura et al. (2022)Nakamura, K., Levy, S., Tuan, Y.-L., Chen, W., and Wang, W. Y.Hybridialogue: An information-seeking dialogue dataset grounded on tabular and textual data.In *Findings of ACL*, 2022.
* Nguyen et al. (2016)Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., and Deng, L.Ms marco: A human generated machine reading comprehension dataset.*choice*, 2016.
* OpenAI (2022)OpenAI.Introducing ChatGPT, 2022.
* OpenAI (2023)OpenAI.GPT-4, 2023.
* Ouyang et al. (2022)Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.Training language models to follow instructions with human feedback.*NeurIPS*, 2022.
* Pasupat \& Liang (2015)Pasupat, P. and Liang, P.Compositional semantic parsing on semi-structured tables.In *ACL*, 2015.
* Qu et al. (2020)Qu, C., Yang, L., Chen, C., Qiu, M., Croft, W. B., and Iyyer, M.Open-retrieval conversational question answering.In *SIGIR*, 2020.
* Rajpurkar et al. (2016)Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P.Squad: 100,000+ questions for machine comprehension of text.In *EMNLP*, 2016.
* Rajpurkar et al. (2018)Rajpurkar, P., Jia, R., and Liang, P.Know what you don’t know: Unanswerable questions for squad.In *ACL*, 2018.
* Raposo et al. (2022)Raposo, G., Ribeiro, R., Martins, B., and Coheur, L.Question rewriting? assessing its importance for conversational question answering.In *ECIR*, 2022.
* Reddy et al. (2019)Reddy, S., Chen, D., and Manning, C. D.Coqa: A conversational question answering challenge.*TACL*, 2019.
* Saeidi et al. (2018)Saeidi, M., Bartolo, M., Lewis, P., Singh, S., Rocktäschel, T., Sheldon, M., Bouchard, G., and Riedel, S.Interpretation of natural language rules in conversational machine reading.In *EMNLP*, 2018.
* Sanh et al. (2022)Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al.Multitask prompted training enables zero-shot task generalization.In *ICLR*, 2022.
* Touvron et al. (2023)Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.Llama 2: Open foundation and fine-tuned chat models.*arXiv preprint arXiv:2307.09288*, 2023.
* Trischler et al. (2017)Trischler, A., Wang, T., Yuan, X., Harris, J., Sordoni, A., Bachman, P., and Suleman, K.Newsqa: A machine comprehension dataset.In *Proceedings of the 2nd Workshop on Representation Learning for NLP*, 2017.
* Vakulenko et al. (2021a)Vakulenko, S., Longpre, S., Tu, Z., and Anantha, R.Question rewriting for conversational question answering.In *WSDM*, 2021a.
* Vakulenko et al. (2021b)Vakulenko, S., Voskarides, N., Tu, Z., and Longpre, S.A comparison of question rewriting methods for conversational passage retrieval.In *ECIR*, 2021b.
* Wang et al. (2023a)Wang, B., Ping, W., McAfee, L., Xu, P., Li, B., Shoeybi, M., and Catanzaro, B.Instructretro: Instruction tuning post retrieval-augmented pretraining.*arXiv preprint arXiv:2310.07713*, 2023a.
* Wang et al. (2022a)Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F.Text embeddings by weakly-supervised contrastive pre-training.*arXiv preprint arXiv:2212.03533*, 2022a.
* Wang et al. (2022b)Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H.Self-instruct: Aligning language model with self generated instructions.*arXiv preprint arXiv:2212.10560*, 2022b.
* Wang et al. (2022c)Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Naik, A., Ashok, A., Dhanasekaran, A. S., Arunkumar, A., Stap, D., et al.Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks.In *EMNLP*, 2022c.
* Wang et al. (2023b)Wang, Y., Ivison, H., Dasigi, P., Hessel, J., Khot, T., Chandu, K. R., Wadden, D., MacMillan, K., Smith, N. A., Beltagy, I., et al.How far can camels go? exploring the state of instruction tuning on open resources.*arXiv preprint arXiv:2306.04751*, 2023b.
* Wei et al. (2022a)Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V.Finetuned language models are zero-shot learners.In *ICLR*, 2022a.
* Wei et al. (2022b)Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al.Chain-of-thought prompting elicits reasoning in large language models.*NeurIPS*, 2022b.
* Wu et al. (2022)Wu, Z., Luan, Y., Rashkin, H., Reitter, D., Hajishirzi, H., Ostendorf, M., and Tomar, G. S.Conqrr: Conversational query rewriting for retrieval with reinforcement learning.In *EMNLP*, 2022.
* Wu et al. (2023)Wu, Z., Parish, R., Cheng, H., Min, S., Ammanabrolu, P., Ostendorf, M., and Hajishirzi, H.Inscit: Information-seeking conversations with mixed-initiative interactions.*TACL*, 2023.
* Xu et al. (2023a)Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., and Jiang, D.Wizardlm: Empowering large language models to follow complex instructions.*arXiv preprint arXiv:2304.12244*, 2023a.
* Xu et al. (2023b)Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B.Retrieval meets long context large language models.*arXiv preprint arXiv:2310.03025*, 2023b.
* Ye et al. (2023)Ye, F., Fang, M., Li, S., and Yilmaz, E.Enhancing conversational search: Large language model-aided informative query rewriting.In *EMNLP*, pp. 5985–6006, 2023.
* Yu et al. (2020)Yu, S., Liu, J., Yang, J., Xiong, C., Bennett, P., Gao, J., and Liu, Z.Few-shot generative conversational query rewriting.In *SIGIR*, 2020.
* Zhang et al. (2023a)Zhang, H., Diao, S., Lin, Y., Fung, Y. R., Lian, Q., Wang, X., Chen, Y., Ji, H., and Zhang, T.R-tuning: Teaching large language models to refuse unknown questions.*arXiv preprint arXiv:2311.09677*, 2023a.
* Zhang et al. (2023b)Zhang, S., Dong, L., Li, X., Zhang, S., Sun, X., Wang, S., Li, J., Hu, R., Zhang, T., Wu, F., et al.Instruction tuning for large language models: A survey.*arXiv preprint arXiv:2308.10792*, 2023b.
* Zhou et al. (2023)Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al.Lima: Less is more for alignment.*arXiv preprint arXiv:2305.11206*, 2023.
* Zhu et al. (2021)Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., and Chua, T.-S.Tat-qa: A question answering benchmark on a hybrid of tabular and textual content in finance.In *ACL*, 2021.

Appendix A ChatQA Instruction Tuning
------------------------------------

### A.1 Stage-1: Supervised Fine-tuning

The format template of LLM inputs in stage-1 is as follows:

```

System: This is a chat between a user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user’s questions
based on the context. The assistant should also indicate when the answer cannot be
found in the context.

User: {Question 1}

Assistant: {Answer 1}

...

User: {Latest Question}

Assistant:

```

We use the ${\texttt{Latest Answer}}$ from Assistant as the supervision for model outputs.

### A.2 Stage-2: Context-Enhanced Instruction Tuning

Based on the stage-1 format template, the LLM inputs in stage-2 adds {Context for Latest Question} and {Instruction} from User, as depicted below:

```

System: This is a chat between a user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user’s questions
based on the context. The assistant should also indicate when the answer cannot be
found in the context.

{Context for Latest Question}

User: {Instruction} + {Question 1}

Assistant: {Answer 1}

...

User: {Latest Question}

Assistant:

```

Same as Stage-1, we use the ${\texttt{Latest Answer}}$ from Assistant as the supervision for model outputs.

As for the ${\texttt{Instruction}}$, we use different instructions for different datasets based on the answer types. Details can be found below:

“Please give a full and complete answer for the question.” This is for datasets with long answers. We use it for the HumanAnnotatedConvQA or SyntheticConvQA.

“Answer the following question with a short span. The answer needs to be just in a few words.” This is for datasets with short answers. We use it for SQuAD1.1, SQuAD2.0, NarrativeQA, DROP, ROPES, NewsQA, and Quoref.

“Answer the following question with a number from context or the math arithmetic using +, -, *, or /.” This is for datasets using arithmetic calculation or extracting numbers from the context. We use it for the TAT-QA dataset where questions need the arithmetic calculation or extracting numbers from the context.

“Answer the following question with a short span, or a full and complete answer.” This is for datasets with both short and long answers. The model needs to tell whether to generate short or long answers based on the questions. We use it for the TAT-QA dataset where questions do not need the arithmetic calculation, since we observe the lengths of the corresponding answers can be both short and long.

For the stage-1 SFT data used in stage-2 instruction tuning, we keep the same format template as in stage-1. In other words, the {Context for Latest Question} and {Instruction} are empty.

### A.3 Prompts for Synthetic Data Generation

We use a two-step prompting on GPT-3.5-turbo for obtaining a high-quality multi-turn QA conversation. In step-1, we prompt GPT-3.5-turbo to generate a multi-turn QA conversation based on a given document. However, we find that the generated conversation lacks pronouns or common nouns to refer back to words or phrases in dialogue history. Hence, in step-2, we prompt GPT-3.5-turbo to revise the generated conversation by adding pronouns or common nouns. For both stages, we give GPT-3.5-turbo few-shot examples to ensure a high-quality generation.

The in-context few-shot prompt in step-1 is as follows (aims to generate multi-turn QA conversation):

```

You are a helpful, pattern-following assistant.

role-user:
PersonA has access to information in the following passage:
###There are certain important aspects of college preparation. The student preparing
for college should define their goals and interests. The student should also plan
financially and academically. We have a checklist that helps students get ready
financially and academically.###
Generate a conversation of around four turns between PersonA and PersonU where
PersonU needs information from PersonA that is in the passage. Use only information
found in the passage.
PersonU starts the conversation and only asks questions. Most of PersonU’s questions
are follow-up or continuations of previous questions.
Conversation:

role-assistant:
PersonU: I need to know how to prepare for college.
PersonA: You should first define your goals and interest and be sure to know the
costs and how to plan financially and academically for college.
PersonU: Is there something I can use as a guide to get ready for it?
PersonA: you can use our checklist to get ready financially and academically.
PersonU: Any other tip?
PersonA: Be sure to check out the reasons to attend a College or Career schools.
PersonU: Ok, that sounds good. Can you give me more tips?
PersonA: Another thing to take into consideration is to explore your career options.

role-user:
PersonA has access to information in the following passage:
###There are several disability compensation benefits. One of the benefits is the
annual clothing allowance. This allowance covers the cost of purchasing new clothes
up to a limit of 200 dollars annually.###
Generate a conversation of around two turns between PersonA and PersonU where
PersonU needs information from PersonA that is in the passage. Use only information
found in the passage.
PersonU starts the conversation and only asks questions. Most of PersonU’s questions
are follow-up or continuations of previous questions.
Conversation:

role-assistant:
PersonU: I heard about a disability benefit for clothing. What is it about?
PersonA: This is a disability compensation benefit known as an annual clothing
allowance.
PersonU: I am quite interested about such benefits.
PersonA: Would you like to find out if you are eligible?

role-user:
PersonA has access to information in the following passage:
###Vascepa (icosapent ethyl) is a prescription fish oil medication approved for
treating adults with extremely high triglycerides, defined as a triglyceride level
of 500 mg/dL or higher. While using this medicine, you may need frequent blood tests
at your doctor’s office. Vascepa is only part of a treatment program that may also
include diet, exercise, and weight control. Follow your doctor’s instructions very
closely. You should not use Vascepa if you are allergic to icosapent. To make sure
Vascepa is safe for you, tell your doctor if you have: 1 liver disease; 2 diabetes;
3 a thyroid disorder; 4 problems with your pancreas; 5 a bleeding or blood-clotting
disorder; 6 if you are allergic to fish or shellfish; or. 7 if you drink large
amounts of alcohol, Vascepa (icosapent) works in the liver and bloodstream to reduce
very low-density triglycerides.###
Generate a conversation of around three turns between PersonA and PersonU where
PersonU needs information from PersonA that is in the passage. Use only information
found in the passage.
PersonU starts the conversation and only asks questions. Most of PersonU’s questions
are follow-up or continuations of previous questions.
Conversation:

role-assistant:
PersonU: I heard about a medication called Vascepa. What is it used for?
PersonA: Vascepa is a prescription fish oil medication used for treating adults with
extremely high triglyceride levels.
PersonU: Are there any side effects or precautions I should be aware of?
PersonA: While using Vascepa, you may need frequent blood tests, and it’s important
to follow your doctor’s instructions closely. It’s important to talk to your doctor
about any allergies or existing conditions you have.
PersonU: Is it effective in lowering cholesterol levels?
PersonA: Vascepa is used together with a low-fat diet and other treatments to lower
total cholesterol in people with severe high cholesterol.

role-user:
PersonA has access to information in the following passage:

{Document}

Generate a conversation of around three turns between PersonA and PersonU where
PersonU needs information from PersonA that is in the passage. Use only information
found in the passage.
PersonU starts the conversation and only asks questions. Most of PersonU’s questions
are follow-up or continuations of previous questions.
Conversation:

role-assistant:

```

The in-context few-shot prompt in step-2 is as follows (aims to revise the multi-turn QA conversation by adding pronouns and common nouns):

```

You are a helpful, pattern-following assistant.

role-user:
Given the following conversation between PersonU and PersonA:
###PersonU: I’ve heard about adobo sauce in Mexican cuisine. Can you tell me more
about it?
PersonA: Adobo sauce is a dark red sauce made from ground chiles, herbs, and
vinegar. It’s used as a marinade and cooking sauce in Mexican and Southwestern
cuisines.
PersonU: What are the main ingredients in adobo sauce?
PersonA: The main ingredients are typically chile peppers, garlic, and vinegar.
However, there can be variations in the recipe.
PersonU: How can I make adobo sauce at home?
PersonA: One way to make adobo sauce at home is by using smoked jalapeno peppers
called chipotles. You can stew the chipotles in a sauce with tomatoes, garlic,
vinegar, salt, and spices. It’s also possible to find canned adobo sauce in the
Hispanic foods section of many grocery stores.
PersonU: Are there any other uses for adobo sauce?
PersonA: Adobo sauce can be used as a condiment or as a marinade for various dishes.
It enhances the flavor of meats and can be used as a cooking and serving sauce.
Additionally, adobo is also used in a stew-like presentation of meat prepared in
this sauce.
PersonU: Is adobo sauce easy to preserve?
PersonA: Yes, you can preserve adobo sauce by freezing or canning it when you’re
done making it. This allows you to store it for up to a year and enjoy its flavors
whenever you want.###
Modify PersonU’s questions to use pronouns or common nouns to refer back to words
or phrases in PersonU’s previous questions or PersonA’s previous responses.

role-assistant:
PersonU: I’ve heard about adobo sauce in Mexican cuisine. Can you provide more
information about it?
PersonA: Adobo sauce is a dark red sauce made from ground chiles, herbs, and vinegar.
It’s used as a marinade and cooking sauce in Mexican and Southwestern cuisines.
PersonU: What are the main ingredients in the sauce?
PersonA: The main ingredients are typically chile peppers, garlic, and vinegar.
However, there can be variations in the recipe.
PersonU: How can I make the sauce at home?
PersonA: One way to make it at home is by using smoked jalapeno peppers called
chipotles. You can stew them in a sauce with tomatoes, garlic, vinegar, salt, and
spices. It’s also possible to find canned adobo sauce in the Hispanic foods section
of many grocery stores.
PersonU: Are there any other uses for it?
PersonA: It can be used as a condiment or as a marinade for various dishes. It
enhances the flavor of meats and can be used as a cooking and serving sauce.
Additionally, it is also used in a stew-like presentation of meat prepared in
this sauce.
PersonU: Is it easy to preserve?
PersonA: Yes, you can preserve it by freezing or canning it when you’re done
making it. This allows you to store it for up to a year and enjoy its flavors
whenever you want.

role-user:
Given the following conversation between PersonU and PersonA:
###PersonU: I’ve heard about a test called urinalysis. What is it used for?
PersonA: Urinalysis is a test that evaluates a sample of your urine. It is used to
detect and assess a wide range of disorders, such as urinary tract infection,
kidney disease, and diabetes. It examines the appearance, concentration, and content
of urine.
PersonU: Can it help diagnose bladder cancer?
PersonA: Yes, urinalysis can be used as a screening and diagnostic tool for
bladder cancer. One way to test for bladder cancer is to check for blood in the
urine, which is called hematuria. This can be done during a urinalysis by examining
the urine cells under a special microscope.
PersonU: What other conditions or diseases can be detected through urinalysis?
PersonA: Urinalysis is commonly used to diagnose urinary tract or kidney infections,
evaluate causes of kidney failure, screen for the progression of chronic conditions
like diabetes mellitus and high blood pressure, and it can be used in combination
with other tests to diagnose certain diseases. It is a versatile test that helps
detect abnormalities in the urinary tract.
PersonU: Where can I get a urinalysis done?
PersonA: Urinalysis can be performed in various healthcare settings, including
doctors’ offices, urgent care facilities, laboratories, and hospitals. You can
provide a urine sample in a specimen cup, and the sample can be analyzed on-site
or sent to a laboratory for testing.
PersonU: How much urine is typically needed for a urinalysis?
PersonA: Usually, only small amounts of urine, ranging from 30 to 60 ml, are required
for urinalysis testing. It’s a non-invasive and relatively simple test that provides
valuable information about your health.###
Modify PersonU’s questions to use pronouns or common nouns to refer back to words
or phrases in PersonU’s previous questions or PersonA’s previous responses.

role-assistant:
PersonU: I’ve heard about a test called urinalysis. What is it used for?
PersonA: Urinalysis is a test that evaluates a sample of your urine. It’s used to
detect and assess a wide range of disorders, such as urinary tract infection,
kidney disease, and diabetes. It examines the appearance, concentration, and
content of urine.
PersonU: Can it help diagnose bladder cancer?
PersonA: Yes, urinalysis can be used as a screening and diagnostic tool for bladder
cancer. One way to test for bladder cancer is to check for blood in the urine, which
is called hematuria. This can be done during a urinalysis by examining the urine
cells under a special microscope.
PersonU: What other conditions or diseases can be detected through this test?
PersonA: Urinalysis is commonly used to diagnose urinary tract or kidney infections,
evaluate causes of kidney failure, screen for the progression of chronic conditions
like diabetes mellitus and high blood pressure, and it can be used in combination
with other tests to diagnose certain diseases. It is a versatile test that helps
detect abnormalities in the urinary tract.
PersonU: Where can I go to get this test done?
PersonA: Urinalysis can be performed in various healthcare settings, including
doctors’ offices, urgent care facilities, laboratories, and hospitals. You can
provide a urine sample in a specimen cup, and the sample can be analyzed on-site
or sent to a laboratory for testing.
PersonU: How much urine is typically needed for the test?
PersonA: Usually, only small amounts of urine, ranging from 30 to 60 ml, are
required for urinalysis testing. It’s a non-invasive and relatively simple test
that provides valuable information about your health.

role-user:
Given the following conversation between PersonU and PersonA:

{multi-turn QA conversation}

Modify PersonU’s questions to use pronouns or common nouns to refer back to words
or phrases in PersonU’s previous questions or PersonA’s previous responses.

role-assistant:

```

Appendix B More Details and Results for Retrieval in Conversational QA
----------------------------------------------------------------------

### B.1 Query Rewriting Prompts for GPT-3.5-turbo

The in-context few-shot query rewriting prompt we use for GPT-3.5-turbo is as follows:

```

You are a helpful, pattern-following assistant.

role-user:
Given the following conversation between PersonU and PersonA:
PersonU: Hello, I would like to know what to do if I do not agree with any decision.
PersonA: disagree with our decision about your monthly income adjustment amounts?
PersonU: no. Where can I find my SHIP contact information?
PersonA: You can find your local SHIP contact information in the back of your
Medicare & You 2020 Handbook online.
PersonU: and how do they calculate the adjustments?
Instead of having this entire conversation, how can PersonU get what he or she is
looking for using a single question? Respond with that question.

role-assistant:
How is the calculation for adjustments made by SHIP determined?

role-user:
Given the following conversation between PersonU and PersonA:
PersonU: I need to know how to prepare for college.
PersonA: You should first define your goals and interest and be sure to know the
costs and how to plan financially and academically for college.
PersonU: Is there something I can use as a guide to get ready for it?
Instead of having this entire conversation, how can PersonU get what he or she is
looking for using a single question? Respond with that question.

role-assistant:
What resources or guides can I use to help me prepare for college?

role-user:
Given the following conversation between PersonU and PersonA:

{Dialogue History + Latest Question}

Instead of having this entire conversation, how can PersonU get what he or she is
looking for using a single question? Respond with that question.

role-assistant:

```

### B.2 More Results for Retrieval in Conversational QA

ModelsAverageDoc2DialQuACQReCCTopiOCQAINSCITtop-1top-5top-1top-5top-1top-5top-1top-5top-5*top-20*top-5*top-20*Dragon (w/ dialog history)46.2973.0943.3375.6156.882.8646.1781.9657.6878.8027.4946.22Dragon + Rewrite (w/ dialog history)47.5774.1244.5476.9857.2383.0446.4582.6060.9481.7428.6946.22Dragon + Rewrite (w/ single query only)54.4680.1347.6080.6047.1077.1551.7385.7873.0788.1952.7968.92Dragon + Fine-tune (w/ dialog history)52.7280.6748.9483.0152.6481.9550.7387.1767.8686.2843.4364.94Dragon + Fine-tune + Rewrite (w/ dialog hisotry)53.1780.8449.3084.6455.0483.2351.2387.9960.5081.0349.8067.33

*Table 9: Comprehensive multi-turn retrieval results across five datasets.*

In Table[9](#A2.T9 "Table 9 ‣ B.2 More Results for Retrieval in Conversational QA ‣ Appendix B More Details and Results for Retrieval in Conversational QA ‣ ChatQA: Building GPT-4 Level Conversational QA Models"), we show comprehensive comparisons between fine-tuning and rewriting methods.

Interestingly, we find that compared to only using the rewritten query as input (Dragon + Rewrite (w/ single query only)), giving additional dialog history (Dragon + Rewrite (w/ dialog history)) makes the average scores significantly drop. This is because Dragon is originally pretrained on single-turn queries, it will naturally have better generalization ability when a single-turn rewritten query instead of a multi-turn conversation is provided. And the rewritten query already contains sufficient information from the dialog history.

In addition, we observe that “Dragon + Fine-tune” performs on par with “Dragon + Fine-tune + Rewrite”. In other words, for the multi-turn fine-tuning method, replacing original query with rewritten query as inputs yields comparable results. This is because rewritten query will not provide much additional information for the model since the dialog history has already been provided, and it might even cause negative effects (e.g., results on the TopiOCQA dataset) since it makes the whole conversation input not natural. This once again demonstrates how effective the fine-tuning method is in equipping the model with the capability to comprehend the multi-turn context.

Appendix C Conversational QA Benchmarks
---------------------------------------

### C.1 Data Statistics

##### Doc2Dial

We use the test set of Doc2Dial for the evaluation. It consists of 719 dialogs with 3939 user-agent turns.

##### QuAC

We use the validation set of QuAC for the evaluation since its test set cannot be directly obtained. Its validation set consists of 1000 dialogs with 7354 user-agent turns. Among these 7354 user-agent turns, there are 1486 (around 20.2%) unanswerable questions. For the combined evaluation of answerable and unanswerable questions, we set the ground truth response for the unanswerable question as “Sorry. I cannot find the answer based on the context.”, same as the one we set in our stage-2 tuning. For the fair comparison, we replace the unanswerable responses in the baseline models (i.e., Llama2-SFT/Chat, GPT-3.5-turbo, GPT-4) with the same sentence (Details can be found in Appendix[E](#A5 "Appendix E Unanswerable Case Evaluation ‣ ChatQA: Building GPT-4 Level Conversational QA Models")).

##### QReCC

We use the test set of QReCC for the evaluation. Its test set includes some dialogue samples source from the QuAC dataset. To avoid overlap with other benchmark datasets, we remove those QuAC-sourced samples, resulting in 2805 user-agent turns.

##### TopiOCQA

We use the validation set of TopiOCQA since its test set is not available yet. Its validation set consists of 205 dialogs with 2514 user-agent turns. Each question has a total of four human-written answers. It has unanswerable cases. However, we do not include this dataset in unanswerable case evaluation because we only find 46 questions with no answer being found, which is not sufficient enough for the unanswerable case evaluation. Given that this dataset use the whole Wikipedia as the retrieval corpus, we leverage its topic information for multi-turn retrieval since we focused more on conversational QA over documents which typically narrow down to a specific topic or domain.

##### INSCIT

We use the validation set of TopiOCQA since its test set is not available yet. Its validation set consists of 86 dialogues with 502 user-agent turns. Each question has an average of 1.9 human-written answers. It has unanswerable response for a few questions. However, we do not include this dataset in unanswerable case evaluation because every question always has at least one answerable response, which means every question is answerable. INSCIT also use the whole Wikipedia as the retrieval corpus. Hence, we also leverage its topic information for multi-turn retrieval for the same reason mentioned in TopiOCQA.

##### CoQA

We use the validation set of CoQA since its test set cannot be directly obtained. Its validation set consists of 500 dialogues with 7983 user-agent turns. Each question has a total of four human-written answers. It has unanswerable cases. However, we do not include this dataset in unanswerable case evaluation for the same reason as in the TopiOCQA dataset. We only find 13 questions with no answer being found, which is not sufficient enough for the unanswerable case evaluation.

##### DoQA

We use the test set of DoQA for the evaluation. Its test set consists of 1200 dialogues with 5394 user-agent turns across Cooking, Travel and Movie domains. Among 5394 user-agent turns, there are 1479 (around 27.4%) unanswerable questions. We use the same strategy as the one mentioned in QuAC dataset to incorporate unanswerable samples into the evaluation.

##### ConvFinQA

We use the validation set of ConvFinQA since its test set cannot be directly obtained. Its validation set consists of 421 dialogues with 1490 user-agent turns.

##### SQA

We use the test set of SQA for the evaluation. Its test set consists of 1025 dialogues with 3100 user-agent turns.

##### HybridDial

We use the test set of HybridDial for the evaluation. Its test set consists of 243 dialogues with 1111 user-agent turns.

### C.2 Details of Incorporating Table as Context

We use the markdown format to process the table in the document (i.e., using “$\mid$” to separate each cell in the table).
However, as the number of rows in a table increases, it is harder for the model to link the values of cells in the last few rows to their corresponding headers in the table. To address this issue, we append the headers to the corresponding cell values in each row, so as to make it easier for the model to capture the connections between the cells and their respective headers. For example, a original markdown table

```

| Boat | Crew | Nation | Date | Meet | Location |
| M1x | Mahe | FR | 2009 | | Poznan |
| M2- | Hamish | US | 2012 | Olympics | Lucerne |
| M2+ | Igor | DE | 1994 | | Indianapolis |

```

will be converted to

```

| M1x (Boat) | Crew: Mahe | Nation: FR | Date: 2009 | Meet: | Location: Poznan |
| M2- (Boat) | Crew: Hamish | Nation: US | Date: 2012 | Meet: Olympics | Location: Paris |
| M2+ (Boat) | Crew: Igor | Nation: DE | Date: 1994 | Meet: | Location: Indianapolis |

```

We apply this table processing strategy to all tabular datasets.

### C.3 Prompts for the Benchmarks

We use the prompt format mentioned in §[A.2](#A1.SS2 "A.2 Stage-2: Context-Enhanced Instruction Tuning ‣ Appendix A ChatQA Instruction Tuning ‣ ChatQA: Building GPT-4 Level Conversational QA Models") for benchmarks in ChatQA, Llama2-Chat, GPT-3.5-turbo and GPT-4. We keep the {Context for Latest Question} the same for all models, while we adjust the {Instruction} for the baselines (i.e., Llama2-Chat, GPT-3.5-turbo and GPT-4) to ensure they do not get sub-optimal results.
We use different instructions for test benchmarks with different answer types (e.g., long answer, short answer, arithmetic calculation). The {Instruction} for all models on the benchmarks are as follows:

#### C.3.1 ChatQA

We keep the {Instruction} consistent between the data blends in stage-2 and test benchmarks based on different answer types.

We use “Please give a full and complete answer for the question.” for DoQA, INSCIT, HybridDial, Doc2Dial, QuAC, and QReCC, since these datasets generally have long answers for the questions.

We use “Answer the following question with a short span, or a full and complete answer.” for SQA and TopiOCQA, since these datasets have both short and long answers based on the questions.

We use “Answer the following question with a short span. The answer needs to be just in a few words.” for CoQA, since it generally has short answers for the questions.

We use “Answer the following question with a number from context or the math arithmetic using +, -, *, or /.” for ConvFinQA, since this dataset requires the model to either extract numbers from the context or do arithmetic calculation. We will calculate the number based on the arithmetic formula the model generates and compare it with the gold answer.

#### C.3.2 Llama2-Chat

Note that we tried original Llama2-Chat prompt template999[https://gpus.llm-utils.org/llama-2-prompt-template/](https://gpus.llm-utils.org/llama-2-prompt-template/ ""), which gives slightly worse results compared to using the one in §[A.2](#A1.SS2 "A.2 Stage-2: Context-Enhanced Instruction Tuning ‣ Appendix A ChatQA Instruction Tuning ‣ ChatQA: Building GPT-4 Level Conversational QA Models"). We have tried several {Instruction} for Llama2-Chat. We find the ones below works the best.

We use “Please give an answer in just one sentence.” for DoQA, INSCIT, HybridDial, Doc2Dial, QuAC, and QReCC, since these datasets generally have long answers but within one sentence. We notice that it is important to give Llama2-Chat specific instruction like “one sentence” instead of “full and complete answer” to prevent the model from generating very long answers.

We use “Answer the following question with a short span, or one sentence.” for TopiOCQA, since this dataset has both short and long answers based on the questions, and the long answers are generally within one sentence.

We use “Answer the following questions with one or a list of entities.” for SQA, since the answer for this dataset always consists of one or a list of entities from the context.

We use “Answer the following question with a short span. The answer needs to be just in a few words.” for CoQA, since it generally has short answers for the questions.

We use “Answer the following question with just a number from context or just the math arithmetic using +, -, *, or /.” for ConvFinQA, since this dataset requires the model to either extract numbers from the context or do arithmetic calculation. We extract the arithmetic formula generated by the model and use a calculator to get its final result.

#### C.3.3 GPT-3.5-turbo \& GPT-4

We have tried several {Instruction} for GPT-3.5-turbo and GPT-4, we find the ones below works the best (applied for both GPT-3.5-turbo and GPT-4).

We use “Please give an answer in just one sentence.” for DoQA, INSCIT, HybridDial, Doc2Dial, QuAC, and QReCC, since these datasets generally have long answers but within one sentence. Similar to Llama2-Chat, we also find OpenAI models tend to generate quite long answers given the instruction of “full and complete answer”. Hence, we make the instruction more specific (i.e., “one sentence”) to prevent the model from generating very long answers.

We use “Answer the following questions in JUST a few words or one sentence.” for TopiOCQA, since this dataset has both short and long answers based on the questions, and the long answers are generally within one sentence.

We use “Answer the following questions with one or a list of entities. Do not give a detailed explanation. Answer needs to be as short as possible.” for SQA. We find that OpenAI models often give detailed explanation for SQA dataset unless specifically instructed not to.

We use “Answer the following question with a short span. The answer needs to be just in a few words.” for CoQA, since it generally has short answers for the questions.

We use “Answer the following questions with just a number from context or just the math arithmetic using +, -, *, or /.” for ConvFinQA, since this dataset requires the model to either extract numbers from the context or do arithmetic calculation. We extract the arithmetic formula generated by the model and use a calculator to get its final result.

Appendix D Human Evaluation
---------------------------

We conduct human evaluation across the ten test benchmark datasets. We randomly select 60 samples for each dataset, and each sample is labelled by three annotators, which results in a total of 1800 annotations.

We ask the annotators to verify the facts in models’ outputs and determine which model provides a more accurate response to the question. We use Amazon Mechanical Turk platform to conduct the human evaluation. The details of our human evaluation instructions and the interface shown to annotators are illustrated in Figure[3](#A4.F3 "Figure 3 ‣ Appendix D Human Evaluation ‣ ChatQA: Building GPT-4 Level Conversational QA Models").

<img src='x3.png' alt='Refer to caption' title='' width='461' height='303' />

*Figure 3: Human evaluation instructions and the interface for annotators.*

Appendix E Unanswerable Case Evaluation
---------------------------------------

For unanswerable case, we consider the model indicating that the question cannot be answered as correct. For answerable cases, we consider the model not indicating the question is unanswerable as correct (i.e., the model giving an answer).
We find that ChatQA always generates “Sorry. I cannot find the answer based on the context” when it finds the question is unanswerable, since we use this sentence as a response for unanswerable questions in the stage-2 tuning. As for Llama2-Chat, GPT-3.5-turbo, and GPT-4, their outputs for unanswerable questions usually follow some specific patterns. Below, we list the heuristic matching patterns derived from all generated samples in DoQA and QuAC datasets to determine if the model suggests that the question is unanswerable. When we find the generated output contains the pattern listed below, we interpret it as an indication that the question is unanswerable, and we will replace its answer with “Sorry. I cannot find the answer based on the context.” for the F1 evaluation.

```

i’m not sure, cannot find, does not provide, cannot provide, cannot answer,
cannot be found, cannot be determined, don’t have information, do not have
information, couldn’t find, no information in the context, does not mention,
not explicitly mentioned, i don’t have any, i do not have any, does not
specify, doesn’t provide, not able to, unable to, doesn’t specify, there is
no information, there is no mention, not mentioned, i don’t have enough
information, there is no specific information, there is no specific mention,
no information found, I don’t have that information

```

ContextConversationAssistant ResponsesPosRiderTime/Retired1Loris Capirossi38:04.7302Valentino Rossi+0.1803Jeremy McWilliams+0.5344Tohru Ukawa+0.5375Shinya Nakano+0.7426Ralf Waldmann+7.0197Franco Battaini+20.8898Stefano Perugini+20.8919Jason Vincent+21.31010Anthony West+26.81611Alex Hofmann+26.933…23Arno Visscher+1:40.63524Henk Van De Lagemaat+1 LapRetMaurice BolwerkRetirementRetAndre RomeinRetirementRetRoberto RolfoRetirementRetMarcellino LucchiRetirementUser: what racers are shown?Assistant: Loris Capirossi, Valentino Rossi,Jeremy McWilliams, Tohru Ukawa, … ,Henk Van De Lagemaat, Maurice Bolwerk,Roberto Rolfo, Marcellino Lucchi.User: of those, what times were recorded?Assistant: 38:04.730, +0.180, +0.534,+0.537, … , +1:40.635, +1 Lap, Retirement.User: of those times, which were notactual times?Assistant: +1 Lap, RetirementUser: which of was an actual finish?Assistant:ChatQA-13B: Loris CapirossiChatQA-70B: +1 LapGPT-3.5-turbo: 38:04.730GPT-4: 38:04.730Ground Truth: +1 Lap2014 compared to 2013 mst 2019s net salesdecreased $ 305 million, or 3% in 2014 ascompared to 2013 … space systems 2019operating results included the following(in millions):net salesbacklog at year-end2015$ 9105$ 174002014$ 9202$ 203002013$ 9288$ 214002015 compared to 2014 space systems 2019net sales in 2015 decreased $ 97 million, or1%, compared to 2014.
these decreases werepartially offset by higher net sales ofapproximately $ 235 million for businessesacquired in 2014; and approximately $ 75million for the orion program due toincreased volume.User: what was the average backlog in 2015?Assistant: 17400User: what was it in 2014?Assistant: 20300User: what is the sum of those years?Assistant: 17400 + 20300User: what was the average backlog in 2013?Assistant: 21400User: what is the total sum for all 3 years?Assistant:ChatQA-13B: 17400 + 20300 + 21400ChatQA-70B: 17400 + 20300 + 21400GPT-3.5-turbo: 9288 + 9202 + 9105GPT-4: 17400 + 20300 + 21400Ground Truth: 17400 + 20300 + 21400The XVI World Rhythmic GymnasticsChampionships were held in Brussels,Belgium, on November 20-22, 1992.RankGymnastCountryPoint1Oxana KostinaRussia9.7752Maria PetrovaBulgaria9.7003Diana PopovaBulgaria9.6254Carmen AcedoSpain9.6255Irina DeleanuRomania9.5506Ekaterina S.Ukraine9.5257Viktoriya YaniUkraine9.4758Maria SansaridouGreece9.425User: What is the XVI World RhythmicGymnastics Championships?Assistant: The XVI World RhythmicGymnastics Championships were held inBrussels, Belgium, on November20-22, 1992User: How many Gymnastics participatedthis championship?Assistant:ChatQA-13B: There are 8 Gymnasticsparticipated in the championship.ChatQA-70B: There are 8 gymnasts*participated in the championship.GPT-3.5-turbo: There were a total of 8gymnasts* who participated inthis championship.GPT-4: The context does not providethe total number of gymnasts whoparticipated in the XVI WorldRhythmic Gymnastics Championships.Ground Truth: There are 8 gymnastsparticipated this championship.*Both models correct the User’s typo of“Gymnastics” to “gymnasts”

ContextConversationAssistant ResponsesThe 3000 metres steeplechase or 3000-metersteeplechase is the most common distance for thesteeplechase in track and field. It is an obstaclerace over the distance of the 3000 metres, whichderives its name from the horse racing steeplechase.YearAthleteLocation1968Jouko Kuha (FIN)Stockholm1969Vladimiras Dudinas (URS)Kiev1970Kerry O’Brien (AUS)Berlin1971Kerry O’Brien (AUS)Adelaide1972Anders Gärderud (SWE)Helsinki1973Ben Jipcho (KEN)Helsinki1974Anders Gärderud (SWE)Helsinki1975Anders Gärderud (SWE)Stockholm1976Anders Gärderud (SWE)Montréal1977Michael Karst (FRG)Stockholm…User: Hello, can you tell me about the 3000metres steeplechase?Assistant: The 3000 metres steeplechase is themost common distance for the steeplechase intrack and field that goes over the distance ofthe 3000 metres, which derives its name fromthe horse racing steeplechase.User: Can you tell me the men that are part of it?Assistant: Some of the men are Henry Rono andMichael Karst.User: Which locations did Kerry O’Brien ran for?Assistant:ChatQA-13B: He ran in Berlin, Adelaide, and Stockholm.ChatQA-70B: Kerry O’Brien ran in Adelaide and Berlin.GPT-3.5-turbo: Kerry O’Brien ran in Berlin, Adelaide,and Champaign.GPT-4: Kerry O’Brien ran in Berlin and Adelaide.Ground Truth: He ran for both Berlin and Adelaide.title: Non-EU family member traveling to Schengencountries. source: You can travel visa-free only if youhave a “residence” card of a family member of a unioncitizen” issued under article 10 of the free movementdirective (or if you are a citizen of a country whosenationals don’t require visas in the Schengen area). Aresidence card issued by Romania to the spouse of aRomanian citizen is not necessarily an article 10 card.If you do not have an article 10 card, EU countriesother than Romania must issue you a visa free of chargeand on the basis of an expedited procedure. The groundsfor refusal of that visa are strictly limited. Theseprovisions are also dependent on your traveling with orjoining your spouse. In case you are not, you’ll need aregular Schengen visa. As Relaxed pointed out in acomment, it appears that your concern stems in partfrom the fact that Romania is not yet a full participantin the Schengen agreement. These rules apply to allEU countries, not just Schengen countries, becausethey are based on the EU right of freedom of movement.User: Can I travel from Romania to the Schengenarea visa-free?Assistant: You can travel visa-free only if youhave a “residence card of a family member of aunion citizen” issued under article 10 of the freemovement directiveUser: I have a temporary residence card also, canwe travel together visa-free?Assistant: If you do not have an article 10 card, EUcountries other than Romania must issue you a visafree of charge and on the basis of an expeditedprocedure.User: How do I get a article 10 card?Assistant:ChatQA-13B: The grounds for refusal of that visa arestrictly limited. These provisions are also dependenton your traveling with or joining your spouse. Incase you are not, you’ll need a regular Schengen visa.ChatQA-70B: Sorry. I cannot find the answer based onthe context.GPT-3.5-turbo: To obtain an article 10 card, you wouldneed to meet the requirements set out in the freemovement directive and apply for it through theappropriate channels in Romania.GPT-4: The text does not provide information on howto obtain an article 10 card.*Ground Truth: Sorry. I cannot find the answer basedon the context.*In the evaluation, we will detect this unanswerablepattern and replace it with “Sorry. I cannot find theanswer based on the context” to match with theground truth.title: Bengal fox. “Vulpes bengalensis” is a relativelysmall fox with an elongated muzzle, long, pointedears, and a bushy tail about 50 to 60% of the lengthof the head and body. Its dorsal pelage is very variable,but mostly grayish and paler ventrally; its legs tendto be brownish or rufous. It is more daintily built than“Vulpes vulpes”. The tail is bushy with a prominentblack tip which distinguishes it from ”V. vulpes”. Backof ears are dark brown with black margin. Its rhinariumis naked and the lips are black, with small black hairpatches on upper part of nuzzle (shaft) in front of eyes.title: Red fox. The red fox is considered a morespecialised form of ”Vulpes” than the Afghan, corsacand Bengal foxes in the direction of size and adaptationto carnivory; the skull displays far fewer neotenoustraits than in other species …title: Bengal fox. The ears have the same colour as thenape or maybe darker, but not having a dark patch as in“V. vulpes”. Extensive variation in coat colour existsacross populations and seasonally within populations,but generally varies from grey to pale brown …User: what is red fox?Assistant:
It is the largest of the true foxes andone of the most widely distributed members ofthe order Carnivora.User: where did it originate?Assistant: The species is Eurasian in origin.User: how did it evolve?Assistant:
It is considered a more specialised formof “Vulpes” than the Afghan, corsac and Bengalfoxes in the direction of size and adaptationto carnivory.User: what is the appearance of this third typeof fox?Assistant:ChatQA-13B:The ears have the same colour as thenape or maybe darker, but not having a dark patchas in “V. vulpes”. Extensive variation in coat colourexists across populations and seasonally withinpopulations, but generally varies from grey to palebrown.ChatQA-70B:“Vulpes bengalensis” is a relativelysmall fox with an elongated muzzle, long, pointedears, and a bushy tail about 50 to 60% of the lengthof the head and body.GPT-3.5-turbo: I’m sorry, but I couldn’t findany information about a third type of fox in thegiven context.GPT-4: The context does not provide information ona third type of fox.Ground Truth: It is a relatively small fox with anelongated muzzle, long, pointed ears, and a bushy tailabout 50 to 60% of the length of the head and body.

*Table 10: Case study for ChatQA-13B, ChatQA-70B, GPT-3.5-turbo, and GPT-4. We use blue color to highlight the relevant context for the last user question. And we use green and red color to highlight the correct and incorrect parts in the assistant responses.*

Appendix F Case Study
---------------------

More examples on ChatQA-13B, ChatQA-70B, GPT-3.5-turbo, and GPT-4 can be found in Table[10](#A5.T10 "Table 10 ‣ Appendix E Unanswerable Case Evaluation ‣ ChatQA: Building GPT-4 Level Conversational QA Models"). We use blue color to highlight the relevant context for the last user question. And we use green and red color to highlight the correct and incorrect parts in the assistant responses.

Appendix G Human Annotated Conversational QA Data
-------------------------------------------------

### G.1 Statistics of User-Agent Turns

<img src='x4.png' alt='Refer to caption' title='' width='368' height='193' />

*Figure 4: Statistics regarding the number of user-agent turns for each dialogue*

Figure[4](#A7.F4 "Figure 4 ‣ G.1 Statistics of User-Agent Turns ‣ Appendix G Human Annotated Conversational QA Data ‣ ChatQA: Building GPT-4 Level Conversational QA Models") illustrates the statistics regarding the number of user-agent turns per dialogue over the 7k human annotated conversational QA data.

### G.2 Guidelines for Conversational QA Data Collection

In this section, we demonstrate the guidelines we provide for the conversational QA data collection. We request the annotators to act as both user and agent to construct the conversational QA sample based on the given document, which makes the annotation more efficient.
The guidelines consists of three parts: 1) What does conversational QA samples look like; 2) What kinds of conversational QA samples we need; and 3) What we need to annotate.

#### G.2.1 What does conversational QA samples look like

One conversational QA sample is based on a document that we provide. The user and agent behaviors are as follows:

* •

    User behavior: 1) Ask questions to agent based on the given document; 2) Answer questions from agent when agent wants to clarify something.

* •

    Agent behavior: 1) Answer questions from user based on the document; 2) Ask questions to the user when the user’s question is not clear, or too general/broad.

#### G.2.2 What kinds of multi-turn QA samples we need

We list the requirements for user’s questions and agent’s responses below.

##### User’s Questions

* •

    User’s questions can refer to his/her previous (or multiple turns before) question.

* •

    User’s questions can also refer to the previous (or multiple turns before) answer from the agent.

* •

    Try to use pronouns or common nouns to replace the entities mentioned before.

* •

    Try to make the user’s questions diverse. For the same type of questions, try to use different ways to express it in different dialogue annotations.

* •

    Other user question types we need

    + –
            Given the answer from agent, ask for extra information (e.g., What else …; any other …; is there more …).

        + –
            Switch the topic and starting a new thread in conversation.

        + –
            Ask two questions at the same time.

##### Agent’s Response

* •

    Try to make the agent’s answer within 1-2 sentences. If the answer has to be long, try to make it concise.

* •

    Try not to directly copy the whole relevant context from the document as the answer. Instead, try to construct the answer by paraphrasing the chosen context.

* •

    Try to come up with a small proportion of cases where agent asks questions to user to clarify something. Specifically, when the user’s question is too broad or not clear, the agent needs to narrow down the scope by asking clarification questions to see which specific aspects that the user cares more about.

#### G.2.3 What we need to annotate

We list what we need to annotate for each conversation below.

* •

    For each document, we need to annotate the user’s questions and corresponding agent’s responses. The average number of user-agent turns per conversation needs to be around five.

* •

    For each user’s question, we need to annotate all the relevant context within the document.
