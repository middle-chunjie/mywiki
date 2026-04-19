# PROMPTAGATOR: FEW-SHOT DENSE RETRIEVAL FROM 8 EXAMPLES

Zhuyun Dai\*, Vincent Y. Zhao\*, Ji Ma\*, Yi Luan\*, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall and Ming-Wei Chang

Google Research

{zhuyundai, vzhao, maji, luanyi, mingweichang}@google.com

*equal contributions † corresponding authors

# ABSTRACT

Much recent research on information retrieval has focused on how to transfer from one task (typically with abundant supervised data) to various other tasks where supervision is limited, with the implicit assumption that it is possible to generalize from one task to all the rest. However, this overlooks the fact that there are many diverse and unique retrieval tasks, each targeting different search intents, queries, and search domains. In this paper, we suggest to work on Few-shot Dense Retrieval, a setting where each task comes with a short description and a few examples. To amplify the power of a few examples, we propose Prompt-base Query Generation for Retriever (PROMPTAGATOR), which leverages large language models (LLM) as a few-shot query generator, and creates task-specific retrievers based on the generated data. Powered by LLM's generalization ability, PROMPTAGATOR makes it possible to create task-specific end-to-end retrievers solely based on a few examples without using Natural Questions (Kwiatkowski et al., 2019) or MS MARCO (Nguyen et al., 2016) to train dual encoders. Surprisingly, LLM prompting with no more than 8 examples allows dual encoders to outperform heavily engineered models trained on MS MARCO like ColBERT v2 (Santhanam et al., 2022) by more than  $1.2\mathrm{nDCG}$  on average on 11 retrieval sets. Further training standard-size re-rankers using the same generated data yields another 5.0 point nDCG improvement. Our studies determine that query generation can be far more effective than previously observed, especially when a small amount of task-specific knowledge is given.

# 1 INTRODUCTION

Recently, major progress has been made on neural retrieval models such as dual encoders, which can retrieve knowledge from a large collection of documents containing millions to billions of passages (Yih et al., 2011; Lee et al., 2019; Karpukhin et al., 2020). However, Thakur et al. (2021) recently proposed the BEIR heterogeneous retrieval benchmark, and showed that it is still difficult for neural retrievers to perform well on a wide variety of retrieval tasks that lack dedicated training data. Thus, previous approaches focus on transferring knowledge from question answering (QA) datasets such as MS MARCO (Nguyen et al., 2016). To best transfer from QA datasets, expressive retrievers are developed that allow fine-grained token-level interaction such as ColBERT (Khattab & Zaharia, 2020; Santhanam et al., 2022) and SPLADE (Formal et al., 2021) but with higher inference cost. Data augmentation via synthetic question generation has previously been explored (Ma et al., 2021; Shakeri et al., 2020), but these question generators are typically only trained on popular QA datasets.

We argue that it is hard to expect models based on one or two QA datasets to perform well across different retrieval tasks. First, different retrieval tasks have very different search intents; in other words, different definitions of "relevance". For example, as illustrated in Figure 1(a), both Dbpedia-Entity (Hasibi et al., 2017) and FEVER (Thorne et al., 2018) are tasks to retrieve documents from Wikipedia. Dbpedia-Entity is a task to retrieve entities that are mentioned in the query, while FEVER is a task to find evidence that either supports or refutes a given statement. Which document is relevant to the query can be very different from one task to another task even if they share the same

Figure 1: Few-shot retrieval with PROMPTAGATOR. Left (a): Retrieval tasks from BEIR differ in query distribution, retrieval corpus, and search intents. Middle (b): Most prior work uses supervised setting (2) which trains model on a large QA retrieval datasets and transfer to other retrieval tasks. Right (c): Few-shot PROMPTAGATOR performance. Average nDCG@10 on 11 datasets from BEIR from our PROMPTAGATOR models and previously MS MARCO-supervised models (SPLADE v2).



domain. Moreover, different tasks have distinct distributions of queries even when their search intents are similar. For example, in the BEIR benchmark, queries in HotpotQA (Yang et al., 2018) are long compositional questions, while queries in FiQA (Maia et al., 2018) are short financial questions.

In this paper, we advocate to work on the setting of Few-shot Retrieval for diverse retrieval (§2), where each task comes with a short description and a few annotated examples to clearly illustrate the search intents. Given that only a few examples are available, we propose Prompt-base Query Generation for Retriever (PROMPTAGATOR) (§3) which aims to resolve the data scarcity issue while retaining the efficiency of a small dual encoder, by harnessing the power of large language models (LLM) such as FLAN (Wei et al., 2022a). PROMPTAGATOR combines prompting with LLMs as a query generator without fine-tuning (§3.1), and can generate good queries with minimal supervision - shown in Figure 1(b), it solely relies on a few supervised examples from the target task without using annotated query-document pairs from Natural Questions (Kwiatkowski et al., 2019) or MS MARCO (Nguyen et al., 2016) to train the retriever directly. The key insight of PROMPTAGATOR is to amplify the power of few-shot examples by creating task-specific prompting, which in turn enables generating a large set of synthetic queries for training retrievers suited for the task. To ensure the generated data quality, we develop a filtering technique that ensures round-trip consistency using generated data only (§3.2). Our filter is tailored to retrieval, which removes ambiguous, generic, and low-quality questions, and significantly improves retrieval performance.

While PROMPTAGATOR is not the first application of LLM for retrieval, prior attempts of using LLMs often come with higher serving cost. Neelakantan et al. (2022) proposes to use the GPT-3 (Brown et al., 2020) embeddings in dual encoder models. However, the embedding size is  $12\mathrm{k}$  and hence makes the search index footprint and inference cost high. Sachan et al. (2022) and Bonifacio et al. (2022) have applied prompting and LLMs for reranking, while leaving the retriever untouched. With PROMPTAGATOR, we show that LLMs can be used to generate efficient end-to-end retriever with high accuracy. The contributions of the paper are as follows:

- We analyze the previously overlooked differences across retrieval tasks in their search intents and query distributions, and propose a Few-Shot Retrieval setting for the BEIR dataset. Our prompt and fewshot examples will be released to facilitate future research.  
- We propose PROMPTAGATOR, a simple recipe for few-shot retrieval by prompting with a LLM to generate synthetic task-specific training data. For the first time, end-to-end retrievers solely based on a few supervised examples can be strong and efficient to serve with PROMPTAGATOR.  
- Our experimental results show that, surprisingly, PROMPTAGATOR with two-to-eight examples produced significantly better retrievers compared to recent models trained on MS MARCO or NQ that have over 500K human annotated examples (Figure 1(c)). PROMPTAGATOR outperforms ColBERT v2 and SPLADE v2 on 11 retrieval tasks we tested, while reranking boosts results by another 5 points on standard retrieval evaluation metric.

# 2 FEW-SHOT RETRIEVAL TASK

In this section, we first introduce the definition of a retrieval task and the differences among different retrieval tasks. We then propose a new Few-Shot Retrieval setting for the BEIR benchmark.

# 2.1 RETRIEVAL TASK

Given a large corpus, a retrieval model is responsible to find documents that are most relevant to a provided query  $q$  according to a pre-defined relevancy. Formally, we define a retrieval task as:

$$
T = \{\mathcal {D}, \mathcal {Q}, \mathcal {I} \},
$$

where  $\mathcal{D} = \{d_1, d_2, \dots, d_n\}$  is a large corpus of documents for retrieval,  $\mathcal{Q}$  is a query distribution, and  $\mathcal{I}$  is the underlying search intent for the task. Depending on the task,  $\mathcal{D}$  can be any document collection, such as web or Wikipedia.  $Q$  also varies across tasks, e.g., short keyword search queries, questions, arguments, etc. If  $\mathcal{I}(q, d) = 1$ , it means search intent of  $q$  has been satisfied by the document  $d$ . For example, in QA tasks such as Natural Questions (NQ) the search intent is to find passages that provide the answer to the question, meaning  $\mathcal{I}_{\mathrm{NQ}}(q, d) = 1$  if  $d$  answers  $q$ . Importantly, for the same pair of  $(q, d)$ , their relevance can be completely different under different search intents. For example, some argument retrieval tasks look for supporting arguments, while other tasks need to retrieve counter arguments.

In this work, we target the scenario where a target retrieval corpus  $\mathcal{D}_{\mathcal{T}}$  is available, but the amount of annotated query-document pairs for the new task is limited. Most prior of research efforts were put into adapting retrievers to new corpus  $\mathcal{D}_{\mathcal{T}}$ , but the divergence in queries  $\mathcal{Q}_{\mathcal{T}}$  and intents  $\mathcal{I}_{\mathcal{T}}$  remains under-explored. Next, we explore how search intent can be expressed with a short description and very few number of examples.

# 2.2 FEW-SHOT BEIR SETTING

In this paper, we argue that it is important to let retrievers be aware of task-specific query distribution and search intent, as opposed to merely focusing on the domain adaptation of  $\mathcal{D}$ . Prior belief is that it is expensive to collect enough in-distribution queries and relevance labels to train a neural retriever, but intuitively, a person can understand a retrieval task by reading a short instruction and going over a few examples. In this work, we ask if a few (8 or fewer) examples are sufficient for the machines to learn a task-specific retriever. To facilitate our study and future research of few-shot retrieval, we define a new few-shot retrieval evaluation setting built upon the BEIR heterogeneous retrieval benchmark (Thakur et al., 2021).

BEIR has 18 information retrieval datasets across 9 domains, including Bio-Medical, Finance, News, Twitter, Wikipedia, StackExchange, Quora, Scientific, and Misc. These datasets also cover a diverse range of search intents: QA retrieval (question-to-document), duplicate question discovery (question-to-question), fact checking (claim-to-document), etc. Following Santhanam et al. (2022) and Formal et al. (2021), we narrow our focus to the publicly-available datasets in BEIR. The original BEIR evaluation used a zero-shot set up, where no queries or relevant query-document pairs from the evaluation datasets can be used to train the retrievers.

We extend BEIR to the few-shot setting by randomly taking a few (2 to 8) in-domain relevant query-document examples as the task-specific supervision. The examples are sampled from the development set when it is available. For the BEIR tasks which only have a test set, we use samples from the test data as few-shot examples. To make the evaluation fair, when evaluating few-shot retriever models, these test-set examples should be treated as 'failed to retrieve' even if the model successfully retrieves them. The prompt and few-shot examples will be released to the public.

# 3 PROMPTAGATOR

To approach the goal of creating retrievers from few-shot examples, we propose Prompt-base Query Generation for Retriever (PROMPTAGATOR). The key idea of PROMPTAGATOR is to transform the few examples into many more examples by prompting a LLM, instead of using them to train a retriever directly.

PROMPTAGATOR consists of three components: prompt-based query generation, consistency filtering, and retriever training. During prompt-based query generation, a task-specific prompt will be combined with a large language model to produce queries for the target task using  $\mathcal{D}_T$ . Then a filtering step cleans the generated data based on round-trip consistency; surprisingly, we found a retriever trained only on the synthetic data can be used to filter the synthetic data. Finally, a retriever (in this paper, dual encoders) and a cross attention reranker will be trained based on the generated data. Figure 5 in Appendix shows the overall procedure.

# 3.1 PROMPT-BASE QUERY GENERATION

PROMPTAGATOR constructs instruction prompt that consists task-specific query passage descriptions and  $k$  annotated query-document examples from the target dataset. Specifically, let  $\{(q_i,d_i)\}^k$  be  $k$  relevant query-document pairs from the target task  $T$ , where  $q_{i}\sim \mathcal{Q}_{T}$ ,  $d_{i}\in \mathcal{D}_{T}$ , and  $\mathcal{I}_T(q_i,d_i) = 1$ . Following FLAN (Wei et al., 2022a), we use instruction prompts with the following form:

$$
\left(e _ {d o c} \left(d _ {i}\right), e _ {q u e r y} \left(q _ {1}\right), \dots , e _ {d o c} \left(d _ {k}\right), e _ {q u e r y} \left(q _ {k}\right), e _ {d o c} (d)\right)
$$

where  $e_{doc}(d)$  and  $e_{query}(q)$  are task-specific document, query descriptions respectively, and  $d$  is a new document. Take ArguAna for example, we set  $e_{doc}(d) = \text{"Argument:} \{d\}$  and  $e_{query} = \text{"Counter Argument:} \{q\}$  to inform the LLM to generate counter arguments. The LLM will be expected to generate  $e_{query}(\hat{q})$ . If the LLM does not generate query description correctly, we consider it a generation failure and drop the output; otherwise we accept  $q$  and form a synthetic relevant example  $(q, d)$ .

Running the prompt on all documents from  $\mathcal{D}_T$ , we can create a large set of synthetic  $(q, d)$  examples, amplifying the information from few examples into a large synthetic dataset whose query distribution is similar to true task distribution  $\mathcal{Q}_T$  and query-document pairs convey the true search intent  $\mathcal{I}_T$ .

We use FLAN (Wei et al., 2022a) as the LLM for query generation in this work. FLAN is trained on a collection of tasks described via instructions and was shown to have good zero/few-shot performance on unseen tasks. We use the 137B FLAN checkpoint provided by the authors. During prompt engineering, we use at most 8 examples, and reduce the number if they exceed the input length limit of FLAN. we also manually truncate individual query and document in the examples if they are too long. We randomly sample up to 1 million documents from each corpus and generate 8 questions per document using sampling decoding with temperature 0.7. The set of templates can be found in Table 4 in the Appendix.

# 3.2 CONSISTENCY FILTERING USING ONLY GENERATED DATA

The filtering step improves the quality of generated queries by ensuring the round-trip consistency (Alberti et al., 2019): a query should be answered by the passage from which the query was generated. In our retrieval case, the query should retrieve its source passage. Consistency filtering (Alberti et al., 2019; Lewis et al., 2021) has been shown crucial for synthetic question generation on QA tasks. However, these techniques typically rely on an external question-answering model as the filter, trained on existing supervised QA data. Since we want to address different search intents, using a single external filtering model does not work for us.

Surprisingly, we find out that consistency filtering based on the generated data alone can work well over the different search intents observed in BEIR. We first use the generated query and document pairs to train an initial retriever. Given a synthetic query-document pair  $(q,d)$ , we use the initial retriever to predict the most relevant passages for  $q$ . We keep  $q$  only when  $d$  occurs among the Top- $K$  passages returned by the retriever. This may seem unintuitive because the filtering model (the initial retriever) is trained on the same noisy synthetic data that it will filter. We show this filter substantially reduces the number of synthetic queries and significantly improves retrieval performance.

# 3.3 FEW-SHOT PROMPTAGATOR RETRIEVER

Our synthetically generated data allows training task-specific neutral retrievers for tasks where supervised in-domain fine-tuning is challenging due to data scarcity. In this work, we use the standard dual-encoder retrieval architecture and we propose a simple pretrain/fine-tune recipe.

<table><tr><td></td><td>Retrieval Supervision</td><td>Cross-Attn Distillation</td><td>Retriever</td><td>Token-level Retrieval</td><td>Serving Model Size</td><td>#Reranking Doc.</td><td>QGen Model</td></tr><tr><td>Contriever</td><td>NA</td><td></td><td>self</td><td></td><td>110M</td><td>0</td><td></td></tr><tr><td>GTR-XXL</td><td>MS MARCO(500K)</td><td></td><td>self</td><td></td><td>6B</td><td>0</td><td></td></tr><tr><td>Splade v2</td><td>MS MARCO(500K)</td><td>✓</td><td>self</td><td>✓</td><td>110M</td><td>0</td><td></td></tr><tr><td>ColBERT v2</td><td>MS MARCO(500K)</td><td>✓</td><td>self</td><td>✓</td><td>110M</td><td>0</td><td></td></tr><tr><td>GenQ</td><td>MS MARCO(500K)</td><td>✓</td><td>self</td><td></td><td>110M</td><td>0</td><td>T5 (MS MARCO)</td></tr><tr><td>GPL</td><td>MS MARCO(500K)</td><td>✓</td><td>self</td><td></td><td>110M</td><td>0</td><td>T5 (MS MARCO)</td></tr><tr><td>MonoT5</td><td>MS MARCO(500K)</td><td></td><td>BM25</td><td>✓</td><td>3B</td><td>1000</td><td></td></tr><tr><td>InPars</td><td>Few (3)</td><td></td><td>BM25</td><td>✓</td><td>3B</td><td>1000</td><td>GPT-3</td></tr><tr><td>UPR</td><td>NA</td><td></td><td>Contriever</td><td></td><td>110M+3B</td><td>1000</td><td>T0*</td></tr><tr><td>PROMPTAGATOR</td><td>Few (0-8)</td><td></td><td>self</td><td></td><td>110M</td><td>0</td><td>FLAN</td></tr><tr><td>PROMPTAGATOR++</td><td>Few (0-8)</td><td></td><td>PROMPTAGATOR</td><td></td><td>110M+110M</td><td>200</td><td>FLAN</td></tr></table>

Table 1: Comparison of settings, resources and model size for different frameworks. Our models are just a 110M-size dual encoder PROMPTAGATOR and a 110M-size reranker PROMPTAGATOR++, as good quality generated data allows simple models/pipeline to achieve strong performance. See text for more details for UPR's QGen model<sup>1</sup>.

Following prior work (Ni et al., 2021), we initialize the dual encoder using the Transformer encoder from a T5 (Raffel et al., 2020) checkpoint. We then pretrain our retriever on C4 with the independent cropping task from Contriever (Izacard et al., 2022a), where we treat two random crops from the same document as positive retrieval pairs and train with a cross-entropy loss over in-batch random negatives. Next, we fine-tune the dual-encoder on the query-document pairs generated from our prompt-base QGen, again with cross-entropy loss over in-batch random negatives. After training for a set number of epochs, we apply round-trip filtering on our synthetic data as described in ( $\S 3.2$ ) using this initial dual encoder, and continue to fine-tune the dual encoder on the filtered data.

We also propose PROMPTAGATOR++, a reranker trained on the same synthetic data generated from our prompt-base QGen, which refines the retrieved candidates using a slower but more accurate cross-attention model. We train the reranker using a cross-entropy loss with 31 sampled negatives from top 200 passages retrieved by the PROMPTAGATOR retriever, which approximates the inference time distribution (reranking top 200 from the retriever).

# 3.4 ZERO-SHOT PROMPTAGATOR RETRIEVER

The prompt-based query generation can also run in a zero-shot manner, where we universally apply the following prompt irrespective of the target task:  $f' \{d\}$  Read the passage and generate a query. Here  $d$  denotes the document text. We train retrievers and rerankers on the zero-shot prompt generated data, leading to zero-shot PROMPTAGATOR and zero-shot PROMPTAGATOR++.

# 3.5 DISCUSSION

Table 1 compares the PROMPTAGATOR recipe to some recently proposed approaches. Our dual encoder does not rely on hard negative mining or distillation; it uses a standard dual encoder model without adding the token-level matching inductive biases that ColBERT and SPLADE have. Our reranker also uses a 110M model instead of larger models. We aim to use this simplified recipe to highlight the power of few-shot data, as we will shown in (§4.3). Comparing PROMPTAGATOR to these approaches, the ability to use a prompt and few-shot examples with a LLM makes PROMPTAGATOR be able to generate efficient models with high accuracy. While other LLM approaches such as InPars (Bonifacio et al., 2022) and UPR (Sachan et al., 2022) have focused on reranking, PROMPTAGATOR focuses on retrieval.

# 4 EXPERIMENTS

We report quantitative evaluation of PROMPTAGATOR by measuring its retrieval performance on the BEIR benchmark. We then dive deeper into the results through ablation studies and qualitative analysis.

# 4.1 IMPLEMENTATION

The original FLAN training set overlapped with 2 datasets in the BEIR benchmark:  $\mathrm{NQ}^2$  and Quora<sup>3</sup>. Most of existing systems use all of the supervised data from MS MARCO in their system. Therefore we exclude MS MARCO, NQ and Quora from our main evaluations. We report nDCG@10, the standard retrieval evaluation metric on BEIR.

For PROMPTAGATOR's prompt-based query generation, we sample questions from the LLM with a temperature of 0.7. For round-trip filtering, we use MS MARCO as validation set and tune  $K$ . We find setting  $K$  to 1 leads to the best results and thus use 1 for all BEIR datasets, i.e. we keep a  $(q,d)$  pair only when  $d$  is ranked in the top 1 place by the initial dual encoder.

We implement PROMPTAGATOR's dual encoders following GTR (Ni et al., 2021); in particular, we use a shared Transformer encoder initialized from T5, take the mean pooling of the top encoder layer, and project it to a fixed 768-dimensional embedding. To ensure efficiency, we use the T5-base version 1.1 encoder architecture consisting of 110M parameters. For PROMPTAGATOR++ reranking models, we use the standard Transformer cross attention encoder, also initialized with a 110M T5-base encoder checkpoint. At inference time, we rerank the top 200 candidates retrieved from the PROMPTAGATOR dual encoder retriever.

We mostly follow the hyper-parameters used in the Ni et al. (2021). The default batch size in this recipe is 6k; however, some of the corpora in BEIR contain only a few thousand documents, making multiple relevant documents appear in the same batch which interacts negatively with the in-batch softmax loss. We found it important to use appropriate batch sizes and training steps for those small datasets. We split the datasets into three groups based on corpus size: small datasets (<50k), middle datasets (50k-500k), large datasets (>500k). For dual encoder training, we use 128 batch size for small datasets and 6k for others. We finetune for 5k steps for large datasets and 1k for others. For ranking models, we use batch size of 64 for all datasets and finetune large datasets for 20k steps, 5k for others.

# 4.2 MAIN RESULTS

Table 2 shows the experimental results. We first notice that zero-shot PROMPTAGATOR already serves as a strong baseline, comparing favorably to other retrieval baselines trained on  $\mathcal{O}(100\mathrm{K})$  examples from MS MARCO. Nonetheless, few-shot PROMPTAGATOR markedly improves upon zero-shot PROMPTAGATOR, increasing the averaged nDCG@10 by over 2 points, which highlights the impact of adapting the LLM to the target task. Few-shot PROMPTAGATOR, being relatively simple in training steps and model architecture, outperforms strong baselines such as GenQ (Thakur et al., 2021) and GPL (Wang et al., 2022) which also use query generation to augment training data, as well as ColBERT v2 (Santhanam et al., 2022) and SPLADE v2 (Formal et al., 2021) which rely on token level interaction architectures and distillation recipes.

Our reranker PROMPTAGATOR++ further boosts performance with another 5 points gain on nDCG@10. It significantly outperforms UPR (Sachan et al., 2022) whose reranker uses T0 (Sanh et al., 2021), an instruction tuned LLM similar to FLAN. It also outperforms monoT5-3B (Nogueira et al., 2020), which achieved previous state-of-the-art reranking performance on BEIR in a recent study by Rosa et al. (2022). Note most of these reranker approach uses a 3B model for its better generalization ability than smaller models, while PROMPTAGATOR++ uses a standard 110M rereanker.

Comparing few-shot PROMPTAGATOR to baselines, the biggest improvement is on Touche-2020 (touché), followed by ArguAna (arg). Touche-2020's goal is to retrieve documents for a controversial topic, e.g., "should felons who have completed their sentence be allowed to vote?". ArguAna's goal is to find the counter-arguments that oppose the input argument, and the input arguments are often several-sentence long. Both tasks are extremely different from traditional QA retrieval data that other

<table><tr><td></td><td>arg</td><td>touché</td><td>covid</td><td>nfc</td><td>hotpot</td><td>dbp</td><td>climate</td><td>fever</td><td>scifact</td><td>scidocs</td><td>fiqa</td><td>AVG.</td></tr><tr><td colspan="13">Retriever</td></tr><tr><td colspan="13">Unsupervised</td></tr><tr><td>BM25</td><td>31.5</td><td>36.7</td><td>65.6</td><td>32.5</td><td>60.3</td><td>31.3</td><td>21.3</td><td>75.3</td><td>66.5</td><td>15.8</td><td>23.6</td><td>41.8</td></tr><tr><td>Contriever</td><td>37.9</td><td>19.3</td><td>27.4</td><td>31.7</td><td>48.1</td><td>29.2</td><td>15.5</td><td>68.2</td><td>64.9</td><td>14.9</td><td>24.5</td><td>34.7</td></tr><tr><td colspan="13">Supervised [MS MARCO]</td></tr><tr><td>GTR-XXL</td><td>54.0</td><td>25.6</td><td>50.1</td><td>34.2</td><td>59.9</td><td>40.8</td><td>26.7</td><td>74.0</td><td>66.2</td><td>16.1</td><td>46.7</td><td>44.9</td></tr><tr><td>SPLADE v2</td><td>47.9</td><td>27.2</td><td>71.0</td><td>33.4</td><td>68.4</td><td>43.5</td><td>23.5</td><td>78.6</td><td>69.3</td><td>15.8</td><td>33.6</td><td>46.6</td></tr><tr><td>ColBERT v2</td><td>46.3</td><td>26.3</td><td>73.8</td><td>33.8</td><td>66.7</td><td>44.6</td><td>17.6</td><td>78.5</td><td>69.3</td><td>15.4</td><td>35.6</td><td>46.2</td></tr><tr><td>GenQ</td><td>49.3</td><td>18.2</td><td>61.9</td><td>31.9</td><td>53.4</td><td>32.8</td><td>17.5</td><td>66.9</td><td>64.4</td><td>14.3</td><td>30.8</td><td>40.1</td></tr><tr><td>GPL</td><td>55.7</td><td>25.5</td><td>70.0</td><td>34.5</td><td>58.2</td><td>38.4</td><td>23.5</td><td>75.9</td><td>67.4</td><td>16.9</td><td>34.4</td><td>45.5</td></tr><tr><td colspan="13">PROMPTAGATOR (110M)</td></tr><tr><td>Zero-shot</td><td>53.8</td><td>26.6</td><td>72.7</td><td>33.4</td><td>60.4</td><td>36.4</td><td>21.4</td><td>76.2</td><td>62.3</td><td>16.3</td><td>40.4</td><td>45.5</td></tr><tr><td>Few-shot</td><td>59.4</td><td>34.5</td><td>75.6</td><td>33.4</td><td>61.4</td><td>38.0</td><td>16.8 (24.0*)</td><td>77.0</td><td>65.0</td><td>18.4</td><td>46.2</td><td>47.8</td></tr><tr><td colspan="13">Retriever + Reranker</td></tr><tr><td colspan="13">Unsupervised</td></tr><tr><td>UPR (3B)</td><td>50.3</td><td>21.3</td><td>60.4</td><td>33.3</td><td>72.2</td><td>33.8</td><td>9.5</td><td>57.3</td><td>69.6</td><td>17.3</td><td>45.0</td><td>42.7</td></tr><tr><td>InPars (3B)</td><td>-</td><td>-</td><td>78.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="13">Supervised [MS MARCO]</td></tr><tr><td>monoT5 (220M)</td><td>13.2</td><td>27.7</td><td>77.8</td><td>35.7</td><td>69.5</td><td>41.9</td><td>24.5</td><td>80.2</td><td>73.6</td><td>16.5</td><td>41.4</td><td>45.6</td></tr><tr><td>monoT5 (3B)</td><td>28.8</td><td>20.0</td><td>79.5</td><td>38.4</td><td>75.9</td><td>47.8</td><td>28.0</td><td>85.0</td><td>77.7</td><td>19.7</td><td>51.4</td><td>51.1</td></tr><tr><td colspan="13">PROMPTAGATOR++ (110M + 110M)</td></tr><tr><td>Zero-shot</td><td>52.1</td><td>27.8</td><td>76.0</td><td>36.0</td><td>71.2</td><td>41.3</td><td>22.6</td><td>83.8</td><td>73.2</td><td>19.1</td><td>45.9</td><td>49.9</td></tr><tr><td>Few-shot</td><td>63.0</td><td>38.1</td><td>76.2</td><td>37.0</td><td>73.6</td><td>43.4</td><td>20.3 (24.1*)</td><td>86.6</td><td>73.1</td><td>20.1</td><td>49.4</td><td>52.8</td></tr></table>

Table 2: Main Results. nDCG@10 on BEIR. Retriever Comparisons (Upper Half): Among the various kind of retrievers, both zero-shot and few-shot PROMPTAGATOR produce strong results. Note that ColBERT v2 and SPLADE v2 allows token-level interactions, but PROMPTAGATOR DE models do not. Retriever+Reranker Comparisons (Lower Half): In the scenario where speed is not a concern, reranker is often used. We train PROMPTAGATOR++ use the same generated data and get significant improvement. See text for more details for Climate-Fever. $^4$

models use, which are dominated by factoid questions. On the other hand, few-shot PROMPTAGATOR can successfully adapt to this task with a few examples.

# 4.3 ABLATION STUDY

Next, we study our results in greater detail and analyze factors contributing to performance.

Impact of consistency filtering. In Figure 2(a), we show quality difference between few-shot PROMPTAGATOR with and without round-trip filtering. We can see that filtering improves performance on 8 out of 11 datasets, and leads to 2.5 points improvement on average. These results demonstrate the effectiveness of our filtering strategy. There are nonetheless datasets where filtering hurts model quality such as NFCCorpus and SciFact. Note these are the smallest datasets in terms of generated queries. We conjecture further tuning dual encoder models on the filtered data results in overfitting.

Manually examining the query document pairs removed by the filter, we find the majority cases are either the query being too general which matches many documents, or the query contains hallucination irrelevant to the document. There are also cases where high quality query document pairs were incorrectly removed since the initial dual encoder model ranks other documents higher. We suspect designing query-specific K values would help retain such query document pairs and further improve model performance. We leave this to future exploration.

Can generated queries replace human annotated queries? In Figure 2(b), we compare the dual encoders trained on examples generated from the 8-shot PROMPTAGATOR vs the dual encoders trained on supervised data. Note that we did not add other components to make the comparison simple. We choose MS MARCO as there are enough labeled data for this task and neither FLAN nor our models are trained on MS MARCO examples. The results showed that the eight examples plus LLM can replace a significant portion of the supervised examples.

How does PROMPTAGATOR compare to other query generation approaches? Figure 2(c) compares zero-shot PROMPTAGATOR to two query generation approaches: GenQ (Thakur et al., 2021) uses a MS MARCO trained T5 query generation model, and NQ-QGen is our in-house T5 QGen model

Figure 2: Left (a): Consistency filter. Delta in nDCG@10 between few-shot PROMPTAGATOR with and without round-trip filter. Middle (b): Comparing the effect of the generated data versus the number of supervised data on MS MARCO. The LLM can amplify the power of the few examples, making 8 examples to catch up with 50k labeled examples, when simple dual encoders are used. Right (c): Ablation on query generation model. GenQ is a prior query generation system from Thakur et al. (2021), while NQ-QGen is our in-house T5 query generation model trained on NQ. Other than the generated data, NQ-QGen and PROMPTAGATOR uses the same hyper parameters.



fine-tuned on NQ. The figure shows the advantages of zero-shot PROMPTAGATOR, outperforming both fine-tuned QGen models by large margins. NQ-QGen uses the same filtering, dual-encoder training, batch sizes and training steps as PROMPTAGATOR, providing an apple-to/apple comparison of the query generators. These results indicate that the main contributing factor to PROMPTAGATOR is the prompted LLM query generation, not the specific training recipe or hype-parameters.

Does Few-shot always improve over Zero-shot? As shown in Table 2, few-shot PROMPTAGATOR almost always outperforms zero-shot PROMPTAGATOR. The only exception is Climate-FEVER (climate). After examining this dataset, we realized that in the original Climate-FEVER dataset, a query-document pair can be annotated as either "supports", "refutes", or "not enough info". BEIR treats these three annotations all as relevant; however, a "not enough info" document may not be related to the query. Using such pairs in the few-shot prompts can hurt query generation. Therefore, we tried switching to FEVER's few-shot prompt, as the two datasets share same corpus and similar search intents. With the better annotated examples, few-shot PROMPTAGATOR indeed surpass zero-shot. This result provides some evidence that low quality few-shot examples negatively affect PROMPTAGATOR.

<table><tr><td></td><td>arg</td><td>touché</td><td>covid</td><td>nfc</td><td>hotpot</td><td>dbp</td><td>climate</td><td>fever</td><td>scifact</td><td>scidocs</td><td>fiqa</td><td>AVG.</td></tr><tr><td>FLAN original</td><td>59.4</td><td>34.5</td><td>75.6</td><td>33.4</td><td>61.4</td><td>38.0</td><td>(24.0*)</td><td>77.0</td><td>65.0</td><td>18.4</td><td>46.2</td><td>48.5</td></tr><tr><td>FLAN w/o NQ and Quora</td><td>58.8</td><td>33.3</td><td>70.2</td><td>33.7</td><td>61.7</td><td>34.4</td><td>(23.5*)</td><td>76.2</td><td>63.8</td><td>18.3</td><td>43.0</td><td>47.0</td></tr></table>

Table 3: Impact of different FLAN version. We use Fever model for Climte Fever for this study. See (§4.3) for more details.

Impact of FLAN versions In the main experiments we have used the FLAN model described in Wei et al. (2022a). This model was trained on a collection of datasets including question-answer datasets; specifically, it includes Natural Questions (NQ) and Quora. FLAN was not trained on query-document pairs from NQ or Quora; however, in order to determine whether the inclusion of this data biased the results on the final retrieval evaluation, we designed an additional ablation experiment. Following the recipe from Wei et al. (2022a) used to train the original FLAN models, we trained an additional LLM excluding both the NQ and Quora datasets. Table 3 shows the results for PROMPTAGATOR trained with and without NQ and Quora. While the accuracy drops slightly, the overall performance still outperform prior retrievers.

(a) Gold queries

(b) Few-shot

(c) NQ-QGen

(d) Prompts (4 examples)  
Figure 3: Top first word distribution on queries generated from different models in the ArguAna dataset. Left (a)(b)(c): Compare gold queries (a) and generated queries (b)(c). Queries generated by few-shot models has closer distribution to the gold queries, while the NQ-QGen queries are mostly questions. Right (d): The few shot FLAN can generate diverse queries even though there are only 4 examples in the prompt. Statistics of more datasets are available in the Appendix (Figure 4).

# 4.4 QUALITATIVE ANALYSIS

In order to understand the advantages of few-shot PROMPTAGATOR, we analyze the distribution of the first words of the queries generated by different query generation methods for the ArguAna task in Figure 3. Note that the distribution of few-shot PROMPTAGATOR (Fig. 3b) is much closer to the real distribution (Fig. 3a) while the NQ-QGen (Fig. 3c) mostly generated questions even when query of the tasks are arguments. Examples are showcased in Table 5 in the Appendix.

# 5 RELATED WORK

Neural retrieval models The success of pre-trained large language models (Devlin et al., 2019; Raffel et al., 2020; Brown et al., 2020) has fostered a lush growth in the field of neural retrieval models. Neural retrieval models can be grouped into two categories, namely representation based models and interaction based models.

Representation based models (Palangi et al., 2016; Gillick et al., 2018; Karpukhin et al., 2020) encode a query and passage independently into a common dense space, and scores their relevance based on vector dot-product or cosine similarity. Recent research on representation based models has primarily focused on the following aspects: developing better pre-training tasks (Lee et al., 2019; Chang et al., 2020; Khattab & Zaharia, 2020; Izacard et al., 2022a; Oguz et al., 2022) or pre-training architectures (Gao & Callan, 2021; 2022), improving expressiveness using multi-vector representations (Luan et al., 2021), improving negative contrast (Qu et al., 2021; Xiong et al., 2021; Lu et al., 2021), and improving generalization across different domains (Thakur et al., 2021; Ren et al., 2022). Different techniques have been explored to improve the generalization, such as using query generation for data augmentation (Ma et al., 2021; Wang et al., 2022), using contrastive learning for better pre-training (Izacard et al., 2022a), using knowledge distillation (Chen et al., 2021; Wang et al., 2022) and scaling the model size (Ni et al., 2021).

Although encoding the query and document into a single vector enables fast retrieval via approximate nearest neighbor search (Johnson et al., 2021; Wu et al., 2019), it also constrains the representational power of these models thus leading to sub-optimal predictions. Interaction based models on the other hand explicitly model the interaction between query and document terms (Guo et al., 2016; Hui et al., 2017; Xiong et al., 2017; Dai et al., 2018; McDonald et al., 2018; Nogueira & Cho, 2019), and therefore make more informed decisions. These models are typically more expensive, and thus are used for reranking or rescoring. Distilling interaction based models into representation based models has been shown effective in closing the gap between the two (Hofstätter et al., 2020; Ren et al., 2021a; Lin et al., 2021; Ren et al., 2021b; Reddi et al., 2021; Zhang et al., 2022). Another attempt to combine the best of both worlds is by postponing the interaction until the last layer of the model (Gao et al., 2021a; Khattab & Zaharia, 2020), blurring the boundary between representation and interaction models.

Few-shot Learning The development of pre-trained large language models also popularize the few-shot learning paradigm, which utilizes a few examples as context for model inputs (Brown et al., 2020; Wei et al., 2022b). Two approaches are commonly used. One approach is to provide the LLM an instruction of the task in natural language with a few examples and do not update any parameter of LLM (Brown et al., 2020; Bonifacio et al., 2022). The other approach provides the LLM the

instruction, a few examples and also performs model fine-tuning (Schick & Schütze, 2021a;b;c; Gao et al., 2021b; Logan IV et al., 2022; Izacard et al., 2022b). Our work adopts the first approach. Usually 10-100 examples are used. For example, 32 examples are used in the few-shot setting in GPT3. In the context of retrieval, Bonifacio et al. (2022) provides GPT3 three question-document pairs and uses it as the question generator for training interaction based models.

Prompt-based Query Generation The idea of using prompted LLMs for query generation has previously been proposed for improving retrieval reranking. In UPR (Sachan et al., 2022), they proposed to use prompted LLMs to rerank the passages directly. InPars (Bonifacio et al., 2022) is probably the most closely related work to ours, where they proposed to use few-shot prompting with GPT-3 to generate synthetic data for training a T5-based reranker applied to a BM25 retriever. In this paper, we propose few-shot prompted LLMs and show that generated data can produce efficient and strong end-to-end retrievers. Moreover, we show the quality of generated data can be improved by task-specific prompts and consistency filtering.

Retrievers with late interactions While dual encoder models are very efficient at retrieval due to the MIPS algorithms, their expressivity is limited due to the fact that their score is just a dot-product between the query vector and the document vector. ColBERT (Santhanam et al., 2022; Khattab & Zaharia, 2020) and SPLADE (Formal et al., 2021) are the models to increase the interactions between the query and document by allowing token-level interactions. Because these models are not just dot product between queries and documents, MIPS algorithms can not be used directly. Hence, these models usually have much higher serving cost compared to dual encoders.

# 6 CONCLUSION AND DISCUSSIONS

In this paper, we have presented PROMPTAGATOR, a novel approach to few-shot retrieval. We showed that it is possible to create task-specific, end-to-end retrievers with only a few annotated examples. The few-shot examples, amplified by prompt-based LLM query generation, simplifies the complexity of training neural retrievers for a new tasks and leads to promising retrieval performance gains. It hopefully inspires future research to further push the limit of few-shot retrieval, towards generalizable retrieval systems that can seamlessly and efficiently adapt to many tasks.

While we demonstrate that query generation can be very effective, many questions remain for the roles of question generation and large language models. One of the key issue that needs thorough investigation is on the generated data efficiency. We have not yet explored exactly how many query-document pairs are needed for each task, or how to use these generated examples more efficiently. Another issue that is worthwhile understanding is the sensitivity of final retriever's performance with respect to the prompt. Finally, we would like to draw a connection from PROMPTAGATOR to distillation, as the final dual encoders definitely benefit a lot from the large language model. Analyzing the headroom and understanding how we can better transfer knowledge from LLMs to retrievers would be a critical topic for the future.

# 7 COMPUTE USAGE AND ENVIRONMENTAL IMPACT

We used the 137B large language model FLAN for query generation. FLAN is based on the same pretrained model as LaMDA (Thoppilan et al., 2022). LaMDA was pre-trained on a large corpus consisting of 1.56T words, costing 451 MWh energy and 25.2 tCO2e carbon footprint. In PROMPTAGATOR, we generated 29.23M queries * 2 prompts = 58.46M queries, for a total of 610M words. As mentioned in (\$6), PROMPTAGATOR can be viewed as distilling LLM to standard-sized dual encoders via prompt-based query generation. While the distillation process is computationally expensive, it significantly reduces cost for inference.

# ACKNOWLEDGEMENTS

We thank Kenton Lee, Tom Kwiatkowski, and Daniel Gillick for technical discussion and providing feedback on our manuscript. We thank Alex Salcianu for developing a bulk inference pipeline for large language models.

# REFERENCES

Chris Alberti, Daniel Andor, Emily Pitler, Jacob Devlin, and Michael Collins. Synthetic QA corpora generation with roundtrip consistency. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 6168-6173, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1620. URL https://aclanthology.org/P19-1620.  
Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, and Rodrigo Nogueira. Inpars: Unsupervised dataset generation for information retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '22, pp. 2387-2392, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450387323. doi: 10.1145/3477495.3531863. URL https://doi.org/10.1145/3477495.3531863.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877-1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bcbd4967418bfb8ac142f64a-Paper.pdf.  
Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, and Sanjiv Kumar. Pre-training tasks for embedding-based large-scale retrieval. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rkg-mA4FDr.  
Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Anchit Gupta, Patrick Lewis, Stan Peshterliev, Yashar Mehdad, Sonal Gupta, and Wen-tau Yih. Salient phrase aware dense retrieval: Can a dense retriever imitate a sparse one? CoRR, 2021. URL https://arxiv.org/abs/2110.06918.  
Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. Convolutional neural networks for soft-matching n-grams in ad-hoc search. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM '18, pp. 126-134, New York, NY, USA, 2018. Association for Computing Machinery. URL https://doi.org/10.1145/3159652.3159659.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. URL https://aclanthology.org/N19-1423.  
Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE v2: Sparse lexical and expansion model for information retrieval. CoRR, abs/2109.10086, 2021. URL https://arxiv.org/abs/2109.10086.  
Luyu Gao and Jamie Callan. Condenser: a pre-training architecture for dense retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 981-993, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.75. URL https://aclanthology.org/2021.emnlp-main.75.  
Luyu Gao and Jamie Callan. Unsupervised corpus aware language model pre-training for dense passage retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2843-2853, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.203. URL https://aclanthology.org/2022.acl-long.203.

Luyu Gao, Zhuyun Dai, and Jamie Callan. COIL: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3030-3042, Online, June 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021.nacl-main.241. URL https://aclanthology.org/2021.nacl-main.241.  
Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3816-3830, Online, August 2021b. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.295. URL https://aclanthology.org/2021.acl-long.295.  
Daniel Gillick, Alessandro Presta, and Gaurav Singh Tomar. End-to-end retrieval in continuous space. CoRR, abs/1811.08008, 2018. URL https://arxiv.org/abs/1811.08008.  
Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, CIKM '16, pp. 55-64, New York, NY, USA, 2016. URL https://doi.org/10.1145/2983323.2983769.  
Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, Krisztian Balog, Svein Erik Bratsberg, Alexander Kotov, and Jamie Callan. Dbpedia-entity v2: A test collection for entity search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '17, pp. 1265-1268, New York, NY, USA, 2017. ISBN 9781450350228. URL https://doi.org/10.1145/3077136.3080751.  
Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. Improving efficient neural ranking models with cross-architecture knowledge distillation. ArXiv, abs/2010.02666, 2020. URL https://arxiv.org/abs/2010.02666.  
Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. PACRR: A position-aware neural IR model for relevance matching. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 1049-1058, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1110. URL https://www.aclweb.org/anthology/D17-1110.  
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022a. URL https://openreview.net/forum?id=jKN1pXi7b0.  
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning with retrieval augmented language models, 2022b. URL https://arxiv.org/abs/2208.03299.  
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547, 2021. doi: 10.1109/TBDATA.2019.2921572. URL https://doi.org/10.1109/TBDATA.2019.2921572.  
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769-6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550.  
Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (eds.), Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pp. 39-48. ACM, 2020. doi: 10.1145/3397271.3401075. URL https://doi.org/10.1145/3397271.3401075.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, March 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026.  
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 6086–6096, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1612. URL https://aclanthology.org/P19-1612.  
Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Kuttler, Aleksandra Piktus, Pontus Stenetorp, and Sebastian Riedel. PAQ: 65 million probably-asked questions and what you can do with them. Transactions of the Association for Computational Linguistics, 9:1098-1115, 2021. doi: 10.1162/tacl_a_00415. URL https://aclanthology.org/2021.tacl-1.65.  
Sheng-Chieh Lin, Zheng-Hong Yang, and Jimmy Lin. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021), pp. 163-173, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.repl4nlp-1.17. URL https://aclanthology.org/2021.repl4nlp-1.17.  
Robert Logan IV, Ivana Balazevic, Eric Wallace, Fabio Petroni, Sameer Singh, and Sebastian Riedel. Cutting down on prompts and parameters: Simple few-shot learning with language models. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2824-2835, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.222. URL https://aclanthology.org/2022-findings-acl.222.  
Jing Lu, Gustavo Hernandez Abrego, Ji Ma, Jianmo Ni, and Yinfei Yang. Multi-stage training with improved negative contrast for neural passage retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6091-6103, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.492. URL https://aclanthology.org/2021.emnlp-main.492.  
Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics, 9:329-345, 04 2021. ISSN 2307-387X. doi: 10.1162/tacl_a_00369. URL https://doi.org/10.1162/tacl_a_00369.  
Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan McDonald. Zero-shot neural passage retrieval via domain-targeted synthetic question generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 1075-1088, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.92. URL https://aclanthology.org/2021.eacl-main.92.  
Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. Ww'18 open challenge: financial opinion mining and question answering. In Companion proceedings of the web conference 2018, pp. 1941-1942, 2018. URL https://doi.org/10.1145/3184558.3192301.  
Ryan McDonald, George Brokos, and Ion Androutsopoulos. Deep relevance ranking using enhanced document-query interactions. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1849-1860, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1211. URL https://www.aclweb.org/anthology/D18-1211.  
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas A. Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David P. Schnurr, Felipe Petroski Such, Kenny Sai-Kin Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang,

Peter Welinder, and Lilian Weng. Text and code embeddings by contrastive pre-training. ArXiv, abs/2201.10005, 2022. URL https://arxiv.org/abs/2201.10005.  
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco: A human generated machine reading comprehension dataset. In CoCo@NIPS, 2016. URL http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf.  
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Y. Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. Large dual encoders are generalizable retrievers. CoRR, abs/2112.07899, 2021. URL https://arxiv.org/abs/2112.07899.  
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert. arXiv, 2019. URL https://arxiv.org/abs/1901.04085.  
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document ranking with a pretrained sequence-to-sequence model. In Trevor Cohn, Yulan He, and Yang Liu (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pp. 708-718. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020-findings-emnlp.63. URL https://doi.org/10.18653/v1/2020-findings-emnlp.63.  
Barlas Oguz, Kushal Lakhotia, Anchit Gupta, Patrick Lewis, Vladimir Karpukhin, Aleksandra Piktus, Xilun Chen, Sebastian Riedel, Scott Yih, Sonal Gupta, and Yashar Mehdad. Domain-matched pre-training tasks for dense retrieval. In Findings of the Association for Computational Linguistics: NAACL 2022, pp. 1524-1534, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022-findings-naacl.114. URL https://aclanthology.org/2022-findings-naacl.114.  
Hamid Palangi, Li Deng, Yelong Shen, Jianfeng Gao, Xiaodong He, Jianshu Chen, Xinying Song, and Rabab Ward. Deep sentence embedding using long short-term memory networks: Analysis and application to information retrieval. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(4):694-707, 2016. URL https://doi.org/10.1109/TASLP.2016.2520371.  
Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 5835-5847, June 2021. doi: 10.18653/v1/2021.naacl-main.466. URL https://aclanthology.org/2021.naacl-main.466.  
Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21/140:1-67, 2020. URL http://jmlr.org/papers/v21/20-074.html.  
Sashank J. Reddi, Rama Kumar Pasumarthi, Aditya Krishna Menon, Ankit Singh Rawat, Felix X. Yu, Seungyeon Kim, Andreas Veit, and Sanjiv Kumar. Rankdistil: Knowledge distillation for ranking. In AISTATS, pp. 2368-2376, 2021. URL http://proceedings.mlr.press/v130/reddi21a.html.  
Ruiyang Ren, Shangwen Lv, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. PAIR: Leveraging passage-centric similarity relation for improving dense passage retrieval. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 2173-2183, Online, August 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021-findings-acl.191. URL https://aclanthology.org/2021-findings-acl.191.  
Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. RocketQAv2: A joint training method for dense passage retrieval and passage re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 2825-2835, Online and Punta Cana, Dominican Republic, November 2021b.

Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.224. URL https://aclanthology.org/2021.emnlp-main.224.  
Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qifei Wu, Yuchen Ding, Hua Wu, Haifeng Wang, and Ji-Rong Wen. A thorough examination on zero-shot dense retrieval, 2022. URL https://arxiv.org/abs/2204.12755.  
Guilherme Moraes Rosa, Luiz Bonifacio, Vitor Jeronymo, Hugo Abonizio, Marzieh Fadaee, Roberto Lotufo, and Rodrigo Nogueira. No parameter left behind: How distillation and model size affect zero-shot retrieval. arXiv preprint arXiv:2206.02873, 2022.  
Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. Improving passage retrieval with zero-shot question generation. arXiv, 2022. URL https://arxiv.org/abs/2204.07496.  
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.  
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3715-3734, July 2022. URL https://aclanthology.org/2022.naacl-main.272.  
Timo Schick and Hinrich Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 255–269, Online, April 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.20. URL https://aclanthology.org/2021.eacl-main.20.  
Timo Schick and Hinrich Schütze. It's not just size that matters: Small language models are also few-shot learners. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2339-2352, Online, June 2021b. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.185. URL https://aclanthology.org/2021.naacl-main.185.  
Timo Schick and Hinrich Schütze. Few-shot text generation with natural language instructions. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 390-402, Online and Punta Cana, Dominican Republic, November 2021c. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.32. URL https://aclanthology.org/2021.emnlp-main.32.  
Siamak Shakeri, Cicero Nogueira dos Santos, Henghui Zhu, Patrick Ng, Feng Nan, Zhiguo Wang, Ramesh Nallapati, and Bing Xiang. End-to-end synthetic data generation for domain adaptation of question answering systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5445-5460. Association for Computational Linguistics, 2020. URL https://aclanthology.org/2020.emnlp-main.439.  
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=wCu6T5xFjeJ.  
Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Kathleen S. Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew

Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed H. Chi, and Quoc Le. Lamda: Language models for dialog applications. CoRR, abs/2201.08239, 2022. URL https://arxiv.org/abs/2201.08239.  
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERIFICATION. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 809-819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1074. URL https://aclanthology.org/N18-1074.  
Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2345-2360, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.168. URL https://aclanthology.org/2022.naacl-main.168.  
Jason Wei, Maarten Paul Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew Mingbo Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=gEZrGcozdqR.  
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022b. URL https://openreview.net/forum?id=yzkSU5zdwd.  
Xiang Wu, Ruiqi Guo, David Simcha, Dave Dopson, and Sanjiv Kumar. Efficient inner product approximation in hybrid spaces. arXiv, 2019. URL https://arxiv.org/abs/1903.08690.  
Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '17, pp. 55-64, New York, NY, USA, 2017. URL https://doi.org/10.1145/3077136.3080809.  
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=zeFrfgyZln.  
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369-2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/D18-1259.  
Wen-tau Yih, Kristina Toutanova, John C. Platt, and Christopher Meek. Learning discriminative projections for text similarity measures. In Proceedings of the Fifteenth Conference on Computational Natural Language Learning, pp. 247-256, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL https://aclanthology.org/W11-0329.  
Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, and Weizhu Chen. Adversarial retriever-ranker for dense text retrieval. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=MR7XubKUFB.

# A ANALYSIS ON PROMPTS

Table 4 shows the list of prompt templates on different BEIR datasets. In order to further analysis the difference between zero-shot and few-shot prompts, we compare the few-shot and zero-shot generated queries given the same paragraph, randomly sampled from three datasets in Table 5. We observe that in general, the few-shot generated queries are closer to the original queries, while zero-shot queries are mostly questions. For example, in the ArguAna dataset, the few-shot queries are in general longer and more claim-like. In contrary, the zero-shot queries are most short question-like queries. Interestingly, for the HotpotQA dataset, even though both few-shot and zero-shot queries are generating questions-like queries, few-shot queries sometimes generate multi-hop questions, while zero-shot mostly generates single-hop questions. We further conduct first word distribution across different generation models for all datasets in Figure 4.

<table><tr><td>Dataset</td><td>Prompt</td></tr><tr><td>ArguAna</td><td>0 Argument: passage X 1 Counter argument: query X</td></tr><tr><td>FiQA</td><td>0 passage X 1 query X</td></tr><tr><td>HotpotQA</td><td>0 Evidence: passage X 1 Vexed question: query X</td></tr><tr><td>DBPedia-Entity</td><td>0 entity: passage X 1 query: query X</td></tr><tr><td>NFCorpus</td><td>0 Article: passage X 1 Query: query X</td></tr><tr><td>Touché-2020</td><td>0 passage X 1 Debate: query X</td></tr><tr><td>TREC-Covid</td><td>0 passage X 1 Question: query X</td></tr><tr><td>SciFact</td><td>0 passage X 1 Finding: query X</td></tr><tr><td>SCIDOCS</td><td>0 passage X 1 The passage is about query X</td></tr><tr><td>FEVER</td><td>0 passage X 1 Is it true that query X</td></tr></table>

# B DETAILED IMPLEMENTATION

Figure 5 shows the overall process of PROMPTAGATOR++, the details of which are in Section 3.

# C QUERY GENERATION STATISTICS

In Table 6, we analyze the length of the generated questions by different query generation systems. Note that NQ-QGen always generates short queries due to the query generation models being fine-tuned on the NQ dataset, and all of the generated questions have similar length to those questions of NQ. Interestingly, zero-shot PROMPTAGATOR already obtains more variance in terms of length compared to NQ-QGen. Finally, few-shot PROMPTAGATOR offers significantly more variance in terms of the length of generated queries.

Table 4: Prompt template for each dataset.  

<table><tr><td></td><td>Few-shot</td><td>Zero-shot</td><td>NQ QGen</td></tr><tr><td>ArguAna</td><td>98.2</td><td>26.0</td><td>9.7</td></tr><tr><td>Touché-2020</td><td>7.8</td><td>13.4</td><td>9.8</td></tr><tr><td>TREC-Covid</td><td>10.8</td><td>11.4</td><td>10.2</td></tr><tr><td>NFCorpus</td><td>8.3</td><td>11.5</td><td>10.3</td></tr><tr><td>HotpotQA</td><td>11.2</td><td>12.2</td><td>8.8</td></tr><tr><td>DBPedia-Entity</td><td>8.2</td><td>13.8</td><td>8.8</td></tr><tr><td>Fever</td><td>12.1</td><td>10.7</td><td>8.8</td></tr><tr><td>Climate-Fever</td><td>12.9</td><td>10.7</td><td>8.8</td></tr><tr><td>SciFact</td><td>12.6</td><td>12.4</td><td>10.0</td></tr><tr><td>SCIDOCS</td><td>7.4</td><td>15.7</td><td>10.7</td></tr><tr><td>FiQA-2018</td><td>12.5</td><td>10.1</td><td>9.5</td></tr><tr><td>AVG.</td><td>17.8</td><td>13.5</td><td>9.6</td></tr></table>

# D AUTHOR CONTRIBUTIONS

Table 6: Average query length.

Zhuyun Dai: Propose few-shot retrieval idea. Perform early experiments to validate the approach. Retrieval pipeline infrastructure. PROMPTAGATOR experiment.

Vincent Zhao: Propose few-shot retrieval idea. Main developer for the retrieval pipeline infrastructure. FLAN data generation. PROMPTAGATOR experiment.

Ji Ma: Advise research directions. Retrieval infrastructure design and implementation. Propose round trip filtering. NQ QGEN experiments. PROMPTAGATOR experiment.

<table><tr><td>Paragraph</td><td>Few-shot</td><td>Zero-shot</td><td>Analysis</td></tr><tr><td>ArguAna: Banning skin whiten-ing on such a basis also requires the acceptance of the racial overtones. Some form of tan is popular in many societies of people of European ethnic origins – that is not a racial matter, it is more based on economic social perceptions (that of holidays to warmer climates). Ascribing a racial element to every thing to do with skin tone is at best a lazy analysis. Irrespective of issues of race and perceptions of ethnic origins, and its intersec- tion with beauty standards, some people will be given advantages in life due to their appearance. Ban- ning a certain form of cosmetic, even if it can have some racial and ethnic undertones, won&#x27;t change that.</td><td>Query 1: Banning skin whitening on such a bas-sis also requires the accept- tance of the racial overtones. However, the racial over- tones are simply a side ef- fect of the product being used incorrectly in order to attain a lighter skin tone than they are naturally born with. The product itself is not racist, but rather the peo- ple who use it. For example, a person of Asian de- scent may use a skin whiten- ing cream to attain a lighter skin tone so that they may fit into a more white-dominant society. The cream itself is not racist, but rather the fact that the person has used it to attain a lighter skin tone so that they may fit into a more white-dominant society.</td><td>Query 1: What did they try to ban? 
Query 2: Is skin whitening racist? 
Query 3: Banning skin whiten- ing on such a basis also re- quires the acceptance of the racial overtones. Some form of tan is popular in many so- cieties of people of European ethnic origins – that is not a racial matter, it is more based on economic social perceptions (that of holidays to warmer clim- mates). 
Query 4: What are some of the racial overtones in skin- whitening? 
Query 5: Are tans a sign of af- fluence?</td><td>In ArguAna dataset, the few-shot exam- ples are more state- ment like and are longer than zero-shot.</td></tr><tr><td>Touché-2020: Psychologically,diet-wise and academically. Housewives can be both educated and uneducated but the fact that they spend more time with their children means that they put in a greater effort and since they are not preoccupied with their job(read: cellphone); they pay the close attention needed to keep their children out of trouble. Think of housewives as inspectors in your home...</td><td>Query 1: Housewives are better parents 
Query 2: Should house- wives be considered better parents?</td><td>Query 1: ,diet-wise and aca- demically. 
Query 2: People want to be relaxed all the time(At home AND at work) that is why all the research on work-life bal- ance. 
Query 3:has shown that home- makers are more relaxed, less stressed, healthier, happier - the list goes on.</td><td>The few-shot gener- ates argument-like queries that are more controversial, while zero-shot generate random statements that sometimes even have grammar problems.</td></tr><tr><td>HotpotQA: The Yung See San Fong House, or Young&#x27;s Home in the Heart of the Hills; is a house in Los Gatos in the U.S. state of California. It is a combination of Asian decorative motifs and pagoda roofs together with West- ern massing and layout. It was completed in 1917 for Ruth Com- fort Mitchell Young, a writer, and her husband, Sanborn Young, a gentleman farmer, conservationist and later California State Senator.</td><td>Query 1: Which author was the husband of Ruth Com- fort Mitchell Young? 
Query 2: Which is known as the Yount-Lee House 
Query 3: Who is the archi- tect of the Yung See San Fong House? 
Query 4: The Yung See San Fong House, or Young&#x27;s Home in the Heart of the Hills, is a house in which U.S. state?</td><td>Query 1: Yung See San Fong House was completed in 1917 for Ruth Comfort Mitchell Young, a writer, and her hus- band, Sanborn Young. 
Query 2: How old is the Yung See San Fong House? 
Query 3: Yung See San Fong House was completed in 1917 for Ruth Comfort Mitchell Young, a writer, and her hus- band, Sanborn Young.</td><td>Few-shot examples sometimes will create multihop questions indicated in blue, which rarely hap- pens in zero-shot examples.</td></tr></table>

Table 5: Few-shot and zero-shot generated queries randomly sampled from ArguAna, FiQA and HotpotQA dataset.

Yi Luan: Conduct all PROMPTAGATOR++ experiments. Analyzing PROMPTAGATOR generated data. Early distillation experiments.

Jianmo Ni: Main developer for transformer dual encoder and reranking modeling development. PROMPTAGATOR experiment.

Jing Lu: Early distillation experiments. Reranking modeling development.

Anton Bakalov: Reranking evaluation code support. Discussion.

Kelvin Guu: Early few-shot retrieval idea. Advise research directions.

Keith B. Hall: Advise research directions. Mentor researchers. Prompt design and analysis.

Ming-Wei Chang: Project initiator. Team organization. Advise research directions. Mentor researchers. Prompt design and analysis.

Gold queries

Few-shot examples

SciFact  
NQ-QGen

Prompts (4 examples)

Gold queries

HotpotQA  
Few-shot examples

NQ-QGen

Prompts (6 examples)

Gold queries

FiQA  
Few-shot examples

NQ-QGen

Prompts (6 examples)

Gold queries

Fever  
Few-shot examples

NQ-QGen

Prompts (3 examples)

Gold queries

Climate-Fever  
Few-shot examples

NQ-QGen

Prompts (3 examples)

Gold queries

TREC-Covid  
Few-shot examples

NQ-QGen

Prompts (3 examples)

Gold queries

Few-shot examples

Touche-2020  
NQ-QGen

Prompts (3 examples)


SCIDOCS




NFCORPUS



Figure 4: Top first word distribution on queries generated from different models in all other BEIR datasets.

DBPedia-Entity



Figure 5: PROMPTAGATOR++ Training pipeline.

# Footnotes:

Page 4: <sup>1</sup>UPR uses T0 query generation for reranking, instead of for synthetic data augmentation that other QGen approaches do. 
Page 5: ${}^{2}$  FLAN is only trained on question-to-answer tasks and never observes the question-passage supervision needed for retrieval training. Additionally, FLAN has not been fine-tuned on query generation tasks on QA datasets. <sup>3</sup>We study the impact of NQ and Quora on FLAN query generation in (§4.3) ${}^{4}$  Climate-FEVER's relevant  $\left( {q,d}\right)$  pairs in BEIR are not well-defined (§4.3),so we also tried running query- generation with FEVER's few-shot prompt on Climate-FEVER. We report the results with FEVER prompt in ( ), but they are not used for computing the average. 
