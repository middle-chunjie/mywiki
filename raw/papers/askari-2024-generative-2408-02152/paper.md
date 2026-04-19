11institutetext: Leiden University, Leiden, The Netherlands  
11email: {a.askari, z.ren, s.verberne}@liacs.leidenuniv.nl22institutetext: The University of Edinburgh, Edinburgh, United Kingdom  
22email: wai33institutetext: University of Amsterdam, Amsterdam, The Netherlands  
33email: {m.aliannejadi, e.kanoulas}@uva.nl

Generative Retrieval with Few-shot Indexing
============================================

Arian Askari†Chuan Meng†Mohammad AliannejadiZhaochun RenEvangelos KanoulasSuzan Verberne

###### Abstract

Existing generative retrieval (GR (generative retrieval)) methods rely on training-based indexing, which fine-tunes a model to memorise associations between queries and the document identifiers of relevant documents.
Training-based indexing suffers from high training costs, under-utilisation of pre-trained knowledge in large language models, and limited adaptability to dynamic document corpora.
To address the issues, we propose a few-shot indexing-based GR (generative retrieval) framework (Few-Shot GR).
It has a few-shot indexing process without any training, where we prompt an LLM (large language model) to generate docids (document identifier) for all documents in a corpus, ultimately creating a docid (document identifier) bank for the entire corpus.
During retrieval, we feed a query to the same LLM (large language model) and constrain it to generate a docid (document identifier) within the docid (document identifier) bank created during indexing, and then map the generated docid (document identifier) back to its corresponding document.
Moreover, we devise few-shot indexing with one-to-many mapping to further enhance Few-Shot GR.
Experiments show that Few-Shot GR achieves superior performance to state-of-the-art GR (generative retrieval) methods requiring heavy training.

††footnotetext: †These two authors contributed equally.

1 Introduction
--------------

\Acf

GR*[cheng2025descriptive, cai2025exploring, zeng2024planning, zeng2024scalable, kuo2024survey, li2024survey, li2024matching]* is a new paradigm in information retrieval (IR (information retrieval)).
Unlike traditional IR that decouples indexing and retrieval, GR (generative retrieval) unifies both processes into a single model*[tay2022transformer]*.
Studies in GR (generative retrieval) typically regard indexing and retrieval as training and inference processes, respectively.
The indexing (training) process typically trains a seq2seq model*[raffel2020exploring]* to map queries to the docids (document identifier) corresponding to relevant documents, using extensive training data of query–docid (document identifier) pairs*[zhuang2022bridging]*.
In the retrieval (inference) process, the trained model takes a query text as input and directly generates potentially relevant docids (document identifier).

Limitations.
Existing studies typically rely on training-based indexing to memorise the associations between a query and its docid (document identifier).
The nature of training-based indexing has two main limitations: (i) The approach has a high training overhead*[li2024survey]*.
Existing studies typically use an LLM (large language model) *[lee2023glen, li2024corpuslm]* as the backbone and then fine-tune it with a new learning objective: mapping query text to docids (document identifier).
Fine-tuning an LLM (large language model) with a new objective demands large-scale query–docid (document identifier) pairs, considerable time, and numerous GPUs.(ii) The approach does not make effective use of LLMs (large language model)’ pre-trained knowledge.
Because there is a gap between the learning objectives of LLMs (large language model) pre-training (text generation) and GR (generative retrieval) fine-tuning (query–docid (document identifier) mapping), fine-tuning an LLM (large language model) with GR (generative retrieval)’s objective may cause the LLM (large language model) to forget its pre-trained knowledge*[li2024survey]*.
Little research has explored mainly using LLMs (large language model)’ pre-trained knowledge for GR (generative retrieval) indexing, without heavy training*[li2024survey]*.

A new perspective on GR (generative retrieval).
To address the limitation, we propose a few-shot indexing-based GR (generative retrieval) framework (Few-Shot GR).
Unlike previous GR (generative retrieval) approaches based on training-based indexing, Few-Shot GR has a few-shot indexing process, where we index a document corpus without requiring any training.
Specifically, in the few-shot indexing process, Few-Shot GR prompts an LLM (large language model) in a few-shot way to generate a free-text docid (document identifier) for each document in a corpus.
This process ultimately produces a docid (document identifier) bank for all documents in an entire corpus.
During the retrieval process (inference), the same LLM (large language model) used in few-shot indexing takes a query as input and uses constrained beam search*[de2020autoregressive]* to ensure the generated docid (document identifier) matches a valid docid (document identifier) created during few-shot indexing.

However, the implementation of Few-Shot GR brings one new challenge:
We found that generating only one docid (document identifier) per document during few-shot indexing results in limited retrieval quality.
This occurs because a document can be relevant to multiple diverse queries; during retrieval, when the LLM (large language model) is fed with different queries that share the same relevant document, it is hard for the LLM (large language model) to always point to one docid (document identifier).
We therefore further improve Few-Shot GR to address the challenge.
Unlike most GR (generative retrieval) studies that generate a single docid (document identifier) per document, we devise few-shot indexing with one-to-many mapping, which enhances few-shot indexing by, for each document, generating multiple docids (document identifier).
This approach allows a relevant document to be mapped back by multiple various docids (document identifier) that are generated in response to different queries during retrieval.

Experiments.
We equip Few-Shot GR with LLMs (large language model) for few-shot indexing and retrieval.
Experiments on Natural Questions (NQ)*[kwiatkowski2019natural]* and MS MARCO show that Few-Shot GR outperforms or performs comparably to state-of-the-art GR (generative retrieval) methods*[lee2023glen, sun2024learning]*.
Moreover, our analyses reveal that two critical factors contribute to the success of Few-Shot GR:
conducting one-to-many mapping during few-shot indexing, and selecting an effective LLM (large language model).
Finally, we demonstrate that few-shot indexing is significantly more efficient than training-based indexing.

Our main contributions are as follows:

* •

    We propose Few-Shot GR, a novel GR (generative retrieval) framework, which conducts GR (generative retrieval) indexing solely with prompting an LLM (large language model) without requiring any training.

* •

    We devise few-shot indexing with one-to-many mapping to further enhance Few-Shot GR’s performance.

* •

    Experiments show that Few-Shot GR achieves superior performance to state-of-the-art GR (generative retrieval) methods that require heavy training.


*Figure 1: Prompt used for indexing and retrieval. The three queries in the demonstration examples are sampled from NQ’s training set*[kwiatkowski2019natural]*, while their corresponding docids (document identifier) are annotated by the authors.*

2 Methodology
-------------

Few-shot indexing with one-to-many mapping.
Let $C\={d_{1},\cdots,$$d_{i},\cdots,d_{|C|}}$ be a corpus with $|C|$ documents; this step aims to use an LLM (large language model) to generate $n$ distinct free-text docids (document identifier) ${id_{1},\cdots,id_{j},\cdots,id_{n}}$ for each document $d$ in the corpus $C$.
Ultimately, we create a docid (document identifier) bank $B$ that contains docids (document identifier) for all documents ($n$ docids (document identifier) for each document) in $C$.

Following the GR (generative retrieval) literature*[zhuang2022bridging, pradeep2023does]*, which shows that replacing documents with their corresponding pseudo queries during indexing results in better retrieval quality, we use only pseudo queries for indexing.
Specifically, we first generate $n$ pseudo queries ${\hat{q}_{1},\cdots,\hat{q}_{j},\cdots,\hat{q}_{n}}$ for a document $d_{i}$ and only feed the generated pseudo queries to the LLM (large language model) to generate $n$ corresponding docids (document identifier) ${id_{1},\cdots,id_{j},\cdots,id_{n}}$, formally:

|  | $\begin{split}\hat{q}_{j}\=\&\mathrm{QG}(d_{i}),\\ id_{j}\=\&\mathrm{LLM}(\hat{q}_{j}),\end{split}$ |  | (1) |
| --- | --- | --- | --- |

where $QG$ is a pseudo query generator, $i\=1,\cdots,|C|$ and $j\=1,\cdots,n$.
As depicted in Figure[1], we prompt the LLM (large language model) in a few-shot manner.

After few-shot indexing, we deduplicate docids (document identifier) in the docid (document identifier) bank $B$.
The devised one-to-many mapping technique during few-shot indexing effectively captures diverse relevance signals, addressing limitations faced by prior methods relying on single identifier generation per document.

Retrieval with constrained beam search.
Given a user query $q$ and the docid (document identifier) bank $B$ created in the previous stage, this step aims to use the same prompt (see Figure[1]) and the LLM (large language model) (see Equation[1]) from the indexing phase to generate a docid (document identifier) $id$, formally:

|  | $id\=\mathrm{LLM}(q),$ |  | (2) |
| --- | --- | --- | --- |

Where we use constrained beam search*[de2020autoregressive]* to the LLM (large language model)’s decoding, ensuring the generated docid (document identifier) $id$ matches a valid docid (document identifier) in the docid (document identifier) bank $B$.
Finally, we map the matched valid docid (document identifier) back to its corresponding document.
Note that the docid (document identifier) bank $B$ undergoes de-duplication, ensuring that each docid (document identifier) uniquely corresponds to a single document.

3 Experimental setup
--------------------

Datasets.
We evaluate on NQ320K*[lee2023glen, sun2024learning, tay2022transformer]* and MS300K*[wang2023novo, mekonnen2025lightweight]*; both have widely been used for GR (generative retrieval) evaluation.
NQ320K is a version of Natural Questions (NQ)*[kwiatkowski2019natural]*; NQ320K consists of 320k relevant query–document pairs, 100k documents, and 7,830 test queries.
MS300K is a version of MS MARCO; MS300K contains 300k query–document pairs, 320k documents, and 5,187 test queries.

Baselines.
We use non-GR (generative retrieval) and GR (generative retrieval) baselines.
Following *[lee2023glen]*, we use the following non-GR (generative retrieval) baselines: BM25*[robertson2009probabilistic]*, DPR*[karpukhin2020dense]*, SentenceT5*[ni2022sentence]*, and GTR-base*[ni2022large]*.
We use the following GR (generative retrieval) baselines (training-based indexing): (i) SEAL*[bevilacqua2022autoregressive]* learns to generate n-grams-based docids (document identifier) and applies FM-index*[ferragina2000opportunistic]*. (ii) DSI*[tay2022transformer]* learns to generate numeric identifiers.(iii) DSI-QG*[zhuang2022bridging]* augments DSI training by using pseudo queries; we replicate DSI-QG using the pseudo query generator provided by the original paper.(iv) DSI-QG (InPars) uses the pseudo query generator from InPars*[bonifacio2022inpars]*.(v) TOME*[ren2023tome]* learns to generate document URLs.(vi) GLEN*[lee2023glen]* learns dynamic lexical docids (document identifier).(vii) GenRET*[sun2024learning]* learns to assign numeric docids (document identifier) based on an auto-encoding scheme.(viii) NOVO*[wang2023novo]* learns interpretable docids (document identifier).

*Table 1: Retrieval quality of Few-Shot GR and baselines on NQ320K and MS300K.
DSI-QG (InPars) and Few-Shot GR use the query generator from InPars*[bonifacio2022inpars]* to generate pseudo queries.
Methods marked † are our reimplementations; all other results are from the corresponding papers*[wang2023novo, sun2024learning, ren2023tome, wang2023novo]*.
The best value in each column is marked in bold, and the second best is underlined.*

| Method | NQ320K | | | MS300K | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Recall@1 | Recall@10 | MRR@100 | Recall@1 | Recall@10 | MRR@10 |
| BM25 | 29.7 | 60.3 | 40.2 | 39.1 | 69.1 | 48.6 |
| DocT5Query | 38.0 | 69.3 | 48.9 | 46.7 | 76.5 | 56.2 |
| ANCE | 50.2 | 78.5 | 60.2 | 45.6 | 75.7 | 55.6 |
| SentenceT5 | 53.6 | 83.0 | 64.1 | 41.8 | 75.4 | 52.8 |
| GTR-base | 56.0 | 84.4 | 66.2 | – | – | – |
| SEAL | 59.9 | 81.2 | 67.7 | 25.9 | 68.6 | 40.2 |
| DSI | 55.2 | 67.4 | 59.6 | 32.4 | 69.9 | 44.3 |
| NCI | 66.4 | 85.7 | 73.6 | 30.1 | 64.3 | 41.7 |
| DSI-QG† | 63.1 | 80.7 | 69.5 | 41.0 | 71.2 | 50.7 |
| DSI-QG (InPars)† | 63.9 | 82.0 | 71.4 | 41.3 | 71.5 | 50.0 |
| TOME | 66.6 | – | – | – | – | – |
| GLEN | 69.1 | 86.0 | 75.4 | – | – | – |
| GenRET | 68.1 | 88.8 | 75.9 | 47.9 | 79.8 | 58.1 |
| NOVO | 69.3 | 89.7 | 76.7 | 49.1 | 80.8 | 59.2 |
| Few-Shot GR | 70.1 | 87.6 | 77.4 | 49.6 | 81.2 | 59.1 |

Evaluation metrics.
In line with recent GR (generative retrieval) work*[wang2023novo, lee2023glen, sun2024learning]*, we report Recall@1,10 on both datasets, plus MRR@100 (NQ320K) and MRR@10 (MS300K).

Implementation details.
We equip Few-Shot GR with llama-3-8B-Instruct for indexing and retrieval.We generate 10 docids (document identifier) per document during few-shot indexing.
We set the maximum and minimum lengths for docid (document identifier) generation to 15 and 3 tokens, respectively.
We employ the query generator from InPars*[bonifacio2022inpars]* for generating pseudo queries in Equation[1].
We conduct parameter tuning on the training set of NQ320K or MS300K.

4 Result and analysis
---------------------

Comparison with baselines.
Table[1] shows the retrieval quality of Few-Shot GR and all baselines on NQ320K and MS300K.
The leading observation is that Few-Shot GR outperforms all baselines across all metrics, except GenRET/NOVO on Recall@10 (NQ320K)/MRR@10 (MS300K).
This shows that our proposed few-shot indexing is highly effective versus training-based indexing.
Notably, while GenRET/NOVO is slightly better on those metrics, it requires large training corpora and heavy model-specific training, which may not be feasible in low-resource settings.
In contrast, Few-Shot GR achieves strong results using only a small set of examples, making it more practical.

The impact of # docids (document identifier) generated per document.
Figure[2] shows Few-Shot GR’s performance w.r.t. # generated docids (document identifier) per document during few-shot indexing on NQ320K; we equip Few-Shot GR with llama-3-8B-Instruct or Zephyr-7B-$\beta$*[tunstall2023zephyr]*.
We found that Few-Shot GR’s performance improves as it generates more docids (document identifier) per document during indexing, reaching saturation when generating 10 docids (document identifier).
E.g., with Llama-3, increasing the number of generated docids (document identifier) from 1 to 10 yields a 27.2% improvement in Recall@10.
It suggests that our devised “one-to-many mapping” is key to the success of few-shot indexing.
The trend is similar on MS300K; we report only NQ320K hereafter due to space.

<img src='x1.png' alt='Refer to caption' title='' width='540' height='335' />

*Figure 2: Few-Shot GR’s retrieval quality w.r.t. # generated docids (document identifier) per document in few-shot indexing on NQ320K.*

*Table 2: Retrieval quality of Few-Shot GR with different LLMs (large language model) on NQ320K.*

| Method | Recall@1 | Recall@10 | MRR@100 |
| --- | --- | --- | --- |
| T5-base | 52.4 | 66.4 | 55.8 |
| Zephyr-7B-$\beta$ | 69.9 | 87.2 | 77.8 |
| llama-3-8B-Instruct | 70.1 | 87.6 | 77.4 |

The impact of LLMs (large language model) choices.
Table[2] shows Few-Shot GR’s performance using different LLMs (large language model) on NQ320K; here we compare T5-base, Zephyr-7B-$\beta$, and llama-3-8B-Instruct.
We found that Llama-3-8B-Instruct performs the best across most metrics, followed by Zephyr-7B-$\beta$.
However, both markedly outperform T5-base in terms of performance.
It suggests that selecting an effective LLM (large language model) is another critical factor contributing to the success of Few-Shot GR.

*Table 3: Efficiency of indexing and retrieval for Few-Shot GR and training-based GR (generative retrieval) baselines on NQ320K.
Few-Shot GR uses llama-3-8B-Instruct and generates 10 docids (document identifier) per document during few-shot indexing.*

| Method | Indexing (hr) | Retrieval (ms) |
| --- | --- | --- |
| DSI-QG | 240 | 72 |
| GenRET | $\approx$16,800 | 72 |
| Few-Shot GR | 37 | 98 |

Efficiency of indexing and retrieval.
Table[3] presents the indexing time and retrieval latency for Few-Shot GR compared to two training-based GR (generative retrieval) methods, DSI-QG*[zhuang2022bridging]* and GenRET*[sun2024learning]*.
The time cost of indexing is measured in hours (hr) on the training set of NQ320K, while the retrieval query latency is measured in milliseconds (ms) on the test set of NQ320K.
We perform all measurements on a single A100 GPU (80GB) with a batch size of 16, except for the indexing (training) of GenRET.
We inquired with the authors of GenRET*[sun2024learning]* about GenRET’s indexing (training) time, and they indicated it took 7 days on 100 A100 GPUs.
This implies it may take approximately 16,800 hours on a single A100 GPU.
We found that Few-Shot GR is significantly more efficient in indexing than existing GR (generative retrieval) methods.
Also, Few-Shot GR achieves similar retrieval query latency compared to existing GR (generative retrieval) methods.

5 Conclusions \& Future Work
----------------------------

We have proposed a new, efficient, and effective GR (generative retrieval) paradigm, Few-Shot GR, featuring a few-shot indexing process that solely relies on prompting an LLM (large language model) to record associations between queries and their docids (document identifier), eliminating the need for any training steps.
We have designed few-shot indexing with one-to-many mapping to further enhance Few-Shot GR’s indexing.
Experimental results show that GR (generative retrieval) achieves superior performance to training-intensive state-of-the-art GR (generative retrieval) methods.

Suitability for dynamic corpora.
Training-based indexing struggles with dynamic corpora, as training on new documents often causes forgetting of old ones*[li2024matching]*.
Although several studies attempt to mitigate this issue*[mehta2022dsi++, kishore2023incdsi, chen2023continual, guo2024corpusbrain++]*, it remains inherent to training-based methods.
Few-Shot GR alleviates this challenge by enabling easy addition or removal of docids (document identifier) in the few-shot indexing docid (document identifier) bank, thus avoiding catastrophic forgetting.
Future work can further explore this direction.

The datasets used in this paper, NQ320K and MS300K, contain corpora of 100K and 320K documents, respectively.
So it is worthwhile to test whether Few-Shot GR’s effectiveness would generalise to a million-document corpus.
Also, it is worth testing Few-Shot GR on other datasets (e.g., BEIR*[thakur2021beir]* and conversational search domains*[mo2025conversational, meng2026conversational1, meng2025bridging, meng2023query]*).
Finally, exploring automatic retrieval quality prediction for generative retrieval methods is another promising direction*[10.1145/3769733.3769743, meng2025query]*.
