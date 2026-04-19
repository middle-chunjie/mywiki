*[inlinelist,1]label\=(),

Evaluating Retrieval Quality in Retrieval-Augmented Generation
===============================================================

Alireza SalemiUniversity of Massachusetts AmherstAmherstMAUnited States[asalemi@cs.umass.edu](mailto:asalemi@cs.umass.edu)andHamed ZamaniUniversity of Massachusetts AmherstAmherstMAUnited States[zamani@cs.umass.edu](mailto:zamani@cs.umass.edu)

(2024)

###### Abstract.

Evaluating retrieval-augmented generation (RAG) presents challenges, particularly for retrieval models within these systems. Traditional end-to-end evaluation methods are computationally expensive. Furthermore, evaluation of the retrieval model’s performance based on query-document relevance labels shows a small correlation with the RAG system’s downstream performance. We propose a novel evaluation approach, eRAG, where each document in the retrieval list is individually utilized by the large language model within the RAG system. The output generated for each document is then evaluated based on the downstream task ground truth labels. In this manner, the downstream performance for each document serves as its relevance label. We employ various downstream task metrics to obtain document-level annotations and aggregate them using set-based or ranking metrics. Extensive experiments on a wide range of datasets demonstrate that eRAG achieves a higher correlation with downstream RAG performance compared to baseline methods, with improvements in Kendall’s $\tau$ correlation ranging from 0.168 to 0.494. Additionally, eRAG offers significant computational advantages, improving runtime and consuming up to 50 times less GPU memory than end-to-end evaluation.

Evaluation; Retrieval Quality; Retrieval-Augmented Generation

††journalyear: 2024††copyright: rightsretained††conference: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 14–18, 2024; Washington, DC, USA.††booktitle: Proceedings of the 47th Int’l ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA††ccs: Computing methodologies Natural language generation††ccs: Information systems Evaluation of retrieval results

1. Introduction
----------------

Retrieval-augmented generation (RAG) has emerged as a prominent approach in natural language processing, combining the strengths of retrieval and generation models *(reml)*, with use cases in decreasing hallucination *(agrawal2023knowledge; shuster-etal-2021-retrieval-augmentation)*, knowledge-grounding *(rag; fid; srag)*, and personalization *(salemi2023lamp; salemi2024optimization)*. Evaluating RAG systems is important as it ensures the effectiveness of integrating retrieval-based methods with generative models *(ares; ragas)*. Traditionally, RAG evaluation has primarily relied on end-to-end assessment, which entails comparing the generated output with one or more ground truth references *(kilt)*. While this is crucial, it presents several limitations, especially, for evaluating retrieval models in RAG systems.

First, end-to-end evaluation lacks transparency regarding which retrieved document contributed to the generated output, hindering interpretability of the system’s behavior. Secondly, it is resource-intensive, consuming significant time and computational power, particularly when dealing with a large set of retrieval results consumed by the LLM. To process long input sequences resulting from the utilization of all retrieved documents by the LLM, GPUs with substantial memory capacities are essential for end-to-end evaluation. Moreover, many ranking systems rely on interleaving (i.e., replacing one or more documents in the result list) for evaluation and optimization, which further complicates the evaluation, as slight variations in retrieval results necessitate re-computation of the RAG pipeline. Finally, optimizing ranking models often requires document-level feedback, such as user clicks *(10.1145/3539618.3591639; 10.1145/1498759.1498818)*. However, end-to-end evaluation only provides list-level feedback for the retrieval results. That said, this paper studies retrieval evaluation in RAG.

Human annotations can be a potential solution for evaluating retrieval models in RAG, however, accurate annotations are often challenging and costly to obtain. More recently, with the emergence of large language models (LLMs) and their advanced capabilities in reasoning and text comprehension, they have been utilized to annotate documents for retrieval evaluation *(ares; ragas)*. Nevertheless, these approaches predominantly evaluate the retriever in RAG systems based on human preferences, whereas the primary objective of the retrieval model in RAG is to serve the LLM that leverages the retrieved results *(reml)*. That said, our extensive investigation on a diverse set of RAG systems for open-domain question answering, fact verification, and dialogue systems reveals that employing human annotations, such as the provenance labels in the KILT benchmark *(kilt)*, for evaluating the retrieval models within a RAG system exhibits only a minor correlation with the downstream RAG performance. This indicates a lack of meaningful relationship between the evaluated metrics and the downstream performance of RAG.

In this paper, we propose eRAG, a new approach for evaluating retrievers in RAG systems, where we apply the LLM in RAG system on each document in the retrieval result list individually and use the LLM’s output to provide document-level annotations. These annotations can be obtained using any arbitrary downstream task metric, such as accuracy, exact match, or ROUGE *(lin-2004-rouge)*. We can then apply a set-based or ranking metric as an aggregation function to obtain a single evaluation score for each retrieval result list.

We evaluate our proposed approach on question answering, fact-checking, and dialogue generation from the knowledge-intensive language tasks (KILT) benchmark *(kilt)*. Our results demonstrate that our proposed approach achieves the highest correlation with the downstream performance of the RAG system in comparison with the baselines. Specifically, we observe an absolute improvement in Kendall’s tau correlation ranging between 0.168 and 0.494 across the evaluated datasets. Furthermore, we investigate the impact of different retrieval augmentation methods, the quantity of retrieved documents, and the LLM size on correlation. Finally, we demonstrate that our approach offers significant computational advantages, consuming up to 50 times less memory compared to end-to-end evaluation. To facilitate research in this domain, we make eRAG’s implementation publicly available at: <https://github.com/alirezasalemi7/eRAG>.

2. Evaluating Retrievers in RAG
--------------------------------

Generally, two predominant methods are used for obtaining relevance labels for retrieval evaluation. The first approach involves human judgment to assess the relevance of a query to documents within a corpus. The main issue with this approach is that human annotation can be costly and is often impractical for evaluating all documents in a corpus *(scott-etal-2012-corpus)*. Moreover, human annotation relies on human preferences to judge the relevance of documents to a query. However, a document deemed relevant based on human preferences may not be useful for an LLM in fulfilling its task.

The second approach utilizes the downstream ground truth output associated with the query to provide weak relevance labels. In this method, a retrieved document containing the downstream ground truth is considered relevant *(dpr; izacard2021distilling; 10.1145/3578337.3605137; 10.1145/3539618.3591629)*. This method also presents its own challenges. This approach is impractical, particularly in scenarios where the task involves long-text generation or text classification, as downstream task labels might not exist within documents. Also, one document can be useful for an LLM in fulfilling its task without containing the ground truth labels.

Even though we are not aware any work that use LLMs for evaluating retrieval models in RAG, LLMs can be leveraged to label documents based on their relevance to a query. Inspired by *thomas2023large*, the LLM functions as a binary classifier, indicating whether a document is relevant to the query or not. The mentioned challenges persist even with the judgment of LLMs, especially if the LLM responsible for labeling differs from the LLM in the RAG pipeline. Besides, employing LLMs as judges in this scenario can pose challenges due to the computational cost of running them on a large set of retrieved documents and memory constraints.

To mitigate these problems, we propose eRAG, a novel approach that involves utilizing the LLM in RAG system itself as the arbiter for generating labels to evaluate the retrieval model.

### Using Downstream Large Language Model in RAG as Document Annotator

Consider a retrieval model $\mathcal{R}$ that produces a ranked list $\mathbf{R}_{k}$ with $k$ documents for the LLM $\mathcal{M}$ tasked with performing a specific task, utilizing a downstream evaluation function $\mathcal{E}_{\mathcal{M}}$. The LLM $\mathcal{M}$ takes a ranked list of documents as its input along with the query $q$, and generates an output represented as $\bar{y}\=\mathcal{M}(q,\mathbf{R}_{k})$.
For the documents in $\mathbf{R}_{k}$, we feed each document individually to the LLM $\mathcal{M}$ with the query and evaluate the generated answer to create the label for each document, expressed as:

| (1) |  | ${\mathcal{G}_{q}}[d]\=\mathcal{E}_{\mathcal{M}}(\mathcal{M}(q,{d}),y)\quad:% \quad\forall d\in\mathbf{R}_{k}$ |  |
| --- | --- | --- | --- |

where $y$ is the expected downstream output for the query. We can employ the created $\mathcal{G}_{q}$ to utilize any ranking metric to evaluate $\mathcal{R}$.

Note that the runtime cost of a vanilla transformer *(NIPS2017_3f5ee243)* scales quadratically with its input length. Consequently, for end-to-end evaluation, the cost of running a transformer on a ranked list with $k$ documents, with an average length of $d$, to generate an output with length $l$ is $O(lk^{2}d^{2})$. Conversely, in our approach, as each document is individually fed to the LLM for k times, the cost is $O(lkd^{2})$, proving to be more efficient than end-to-end evaluation.

### Retrieval Evaluation Metrics

For a ranked list $\mathbf{R}_{k}$, comprising $k$ retrieved documents generated by a retrieval model $\mathcal{R}$, an evaluation metric $\mathcal{E}_{\mathcal{R}}$ assigns a score ${\mathcal{E}_{\mathcal{R}}(\mathbf{R}_{k},\mathcal{G}_{q})}\in[0,1]$, by comparing the ranked list with the relevance scores $\mathcal{G}_{q}$, which is a function that maps each document to a scalar relevance score for the document with respect to the query $q$ (i.e., $\mathcal{G}_{q}(d)\=s_{d}$). Various definitions exist for the evaluation metric $\mathcal{E}_{\mathcal{R}}$; in this paper, we examine Precision (P), Recall (R), Mean Average Precision (MAP), Mean Reciprocal Rank (MRR) *(mrr)*, Normalized Discounted Cumulative Gain (NDCG) *(ndcg)*, and Hit Rate. Note that when dealing with non-binary relevance labels, precision considers the average value of relevance labels, while Hit Ratio considers the maximum value among them.

3. Experiments
---------------

*Table 1. The correlation between each evaluation approach and the downstream performance of the LLM. T5-small with FiD with 50 retrieved documents is used. We do not report correlation for the Answers method for FEVER and WOW datasets because the answers to queries do not exist in the document since FEVER is a classification dataset and WoW is long-text generation. For the WoW dataset, we only report correlation on Precision and Hit Ratio because other metrics do not support non-integer relevance labels. Tau is Kendall’s tau and rho is Spearman’s rho.*
