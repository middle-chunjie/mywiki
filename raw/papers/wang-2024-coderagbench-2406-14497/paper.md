# CODERAG-BENCH: Can Retrieval Augment Code Generation?

Zora Zhiruo Wang\* Akari Asai\* Xinyan Velocity Yu Frank F. Xu Yiqing Xie Graham Neubig Daniel Fried Carnegie Mellon University University of Washington University of Southern California https://code-rag-bench.github.io/

# Abstract

While language models (LMs) excel at generating code, many programs are difficult to generate using only parametric knowledge. Despite the success of retrieval-augmented generation (RAG) in text-centric tasks, its potential for code generation remains under-explored. This work introduces CODERAG-BENCH, a holistic retrieval-augmented code generation benchmark covering tasks like basic programming, open-domain, and repository-level problems and provide reproducible evaluations on both retrieval and end-to-end code generation performance. We further create a diverse, open data-tore for code retrieval, aggregating sources such as competition solutions, tutorials, library documentation, StackOverflow posts, and GitHub repositories. Based on CODERAG-BENCH, we conduct large-scale evaluations of 10 retrievers and 10 LMs and systematically analyze when retrieval can benefit code generation models and identify remaining challenges. We find that while retrieving high-quality contexts improves code generation, retrievers often struggle to fetch useful contexts, and generators face limitations in using those contexts effectively. We hope CODERAG-BENCH encourages further development in code-oriented RAG methods.

# 1 Introduction

Generating code from natural language has rapidly advanced with language models (LMs; Chen et al. 2021; Li et al. 2022, 2023; Roziere et al. 2023). However, most models follow an NL (Natural Language)-to-code approach without integrating external context, which is crucial in complex scenarios like using unfamiliar libraries (Zhou et al., 2023; Jimenez et al., 2024). Relying solely on parametric knowledge also limits adaptation to new data distributions at test time, such as evolving public libraries or private code bases not seen during training (Zhang et al., 2023; Jimenez et al., 2024).

Retrieval-augmented generation (RAG; Lewis et al. 2020; Guu et al. 2020) addresses this by retrieving relevant documents at inference time, reducing reliance on model parameters (Asai et al., 2024) and improving accuracy across tasks (Izacard et al., 2022b). Despite success in text-based tasks, its application to diverse coding problems and retrieval sources remains under-explored (Zhou et al., 2023; Su et al., 2024).

We present CODERAG-BENCH, a holistic benchmark designed to advance research in retrieval-augmented code generation (RACG; §2). CODERAG-BENCH (as in Figure 1) covers six programming tasks across four categories: basic programming, open-domain coding, repository-level, and code retrieval tasks. For each task, we manually annotate canonical documents as references for evaluating RACG systems. We also compile a diverse corpus of documents from five sources: programming solutions, online tutorials, Python library documentation, StackOverflow posts, and GitHub files. In total, CODERAG-BENCH has 9k coding tasks and 25 million retrieval documents, providing a robust foundation for reproducible and reliable evaluations in retrieval and RACG.

We conduct holistic evaluations in retrieval, generation, and RACG (§3). Code generation models significantly benefit from access to canonical documents (i.e., from the canonical retrieval corpus) in various scenarios. For example, GPT-4o achieves a  $27.4\%$  gain on SWE-Bench and a  $6.9\%$  gain on the harder ODEX subset when canonical documents are provided. In RACG settings, where models retrieve top relevant documents, some even surpass their performance when using gold documents, highlighting the strong potential of retrieval-augmented approaches for enhancing code generation. However, current retrieval models face challenges in selecting useful documents, particularly for open-domain and repository-level tasks. Additionally, generation models with limited context

Reproducible retrieval and end-to-end evaluation

5 Document Sources for Retrieval  
Figure 1: Overview of CODERAG-BENCH.


windows exhibit smaller improvements, suggesting considerable room for future advancements.

Beyond canonical retrieval, we also explore RACG with open retrieval, i.e., retrieving documents from various sources with different chunking strategies (§4). We find that models can benefit from functionally relevant snippets from certain sources, and chunking documents to 200–800 tokens often gives the best results. For instance, by retrieving from StackOverflow or online tutorials, both StarCoder and GPT4o can significantly improve, while on repository-level tasks, the gains are rather limited. Overall, we hope CODERAG-BENCH can serve as a testbed for future work exploring, analyzing, and improving RACG systems.

# 2 The CODERAG-BENCH

For CODERAG-BENCH (Figure 1), the curation is driven by three factors: (i) Diverse tasks: Code generation spans multiple levels (line, function, repository) across closed and open domains. (ii) Rigorous evaluation: We offer high-quality ground-truth annotations for retrieval and execution-based evaluation to measure functional correctness. (iii) Unified interface: Our codebase provides a consistent interface for retrieval, augmented generation, and evaluation, unlike current datasets with varied pipelines.

In this section, we introduce the creation process of CODERAG-BENCH: programming problem integration (§2.1), retrieval source collection (§2.2), canonical document annotation (§2.3), and the evaluation pipeline (§2.4). Examples with canonical documents are available in §A.

# 2.1 Programming Problems

We categorize existing Python-based coding datasets into four types: code retrieval, basic programming, open-domain problems, and repository-level problems. To ensure the diversity of datasets, we choose and unify multiple frequently adopted datasets for each category, as listed in Table 1.

Basic programming problems This category includes interview-style problems that mostly require Python built-in operations and pose algorithmic challenges. We select the two most widely used datasets: HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021), which ask the model to complete a function from an NL problem description. However, due to limited public knowledge about model training data, it is unclear whether models suffer from data contamination on HumanEval and MBPP (Jain et al., 2024). Hence, we also include LiveCodeBench (Jain et al., 2024) with problems collected from coding websites after the training cutoff of LMs that we consider, to decrease the risk of contamination.

Open-domain problems Open-domain coding problems require Python libraries beyond the standard libraries used in basic programming problems. We adopt the DS-1000 (Lai et al., 2023) and ODEX (Wang et al., 2023b) datasets that cover data-science and general open-domain coding problems. DS-1000 collects data science problems with programs using seven common data-related libraries such as pandas and numpy. ODEX cov

<table><tr><td>Type</td><td>Dataset</td><td># Examples</td><td># Corpus</td><td>Ground-Truth Docs</td><td>Evaluation</td></tr><tr><td rowspan="3">Basic programming</td><td>HumanEval</td><td>164</td><td>164</td><td>program solutions</td><td>execution</td></tr><tr><td>MBPP</td><td>500</td><td>500</td><td>program solutions</td><td>execution</td></tr><tr><td>LiveCodeBench</td><td>400</td><td>-</td><td>-</td><td>execution</td></tr><tr><td rowspan="2">Open-domain</td><td>DS-1000</td><td>1000</td><td>34,003</td><td>docs</td><td>execution</td></tr><tr><td>ODEX</td><td>945</td><td>34,003</td><td>docs, stackoverflow</td><td>execution</td></tr><tr><td rowspan="2">Repository-level</td><td>RepoEval (function)</td><td>373</td><td>237</td><td>github repository</td><td>execution</td></tr><tr><td>SWE-bench-Lite</td><td>300</td><td>40,868</td><td>github repository</td><td>execution</td></tr><tr><td>Code retrieval</td><td>CodeSearchNet-Py</td><td>22,177</td><td>22177</td><td>CSN functions</td><td>ndcg@10</td></tr></table>

ers problems using a broader range of 79 libraries, such as web requests with requests and database operations with sqlchemy.

Repository-level coding problems Beyond function-level, some problems require editing files in the context of an entire GitHub repository. We thus adopt RepoEval (Zhang et al., 2023) and SWE-bench (Jimenez et al., 2024) for repository-level code generation and issue-solving tasks. We integrate all three splits of RepoEval but only report its function split, as it is the only split supporting execution-based evaluation. Notably, our code-base is the first to enable reproducible execution evaluation on RepoEval. SWE-bench focuses on resolving GitHub issues by asking models to edit multiple files that pass the required test cases. We use SWE-bench-Lite, a 300-problem subset whose results can be reproduced, with a packaged Docker container (Wang et al., 2024).

Code retrieval problems In addition to retrieval for augmenting generations, we adopt the Python split of CodeSearchNet (CSN) as a code retrieval task. CSN searches for the correct implementation of an NL query from a pool of functions collected from GitHub repositories. Instead of monitoring how generation changes with various retrieval results, CSN can directly measure retrieval quality.

# 2.2 Retrieval Sources

We collect retrieval documents from five commonly used resources for program developers, listed in Table 2. CODERAG-BENCH supports two retrieval setups: canonical retrieval—retrieves documents from only the canonical datastore (§2.3), and open retrieval—retrieves documents from any datastore.

Programming solutions We create one document from each basic programming problems that have canonical solutions (i.e., HumanEval and

MBPP), following VoyageAI (2024), by concatenating its NL problem and program solution.

Online tutorials We collect tutorials from multiple websites including GeeksforGeeks, W3Schools, tutorialspoint, and Towards Data Science, $^{4}$  via the raw HTML pages obtained from ClueWeb22 (Overwijk et al., 2022), a large-scale crawled web corpus. Each page contains code snippets and their text explanations, covering topics from basic programming techniques to advanced library usage.

Library documentation We collect the official documentation provided by devdocs.io for all Python libraries following (Zhou et al., 2023). These could be especially useful for open-domain and repository-level problems that use some library functions to realize complex setups.

StackOverflow posts StackOverflow (SO) is among the most frequently visited sites for developers. We collect all SO posts from the RedPajama-1T (Computer, 2023) stackexchange split. We treat each post as a retrievable document, that has a question, code responses, and textual explanations.

GitHub repository We collect high-quality repositories from GitHub, using the git thub split of RedPajama-1T (Computer, 2023), as developers often refer to popular repositories when writing their programs. Following this practical paradigm, we enable LMs to retrieve files from other repositories as contexts to write the current program.

Table 1: Overview of the datasets in CodeRAG-Bench. CSN stands for CodeSearchNet.  

<table><tr><td>Resource</td><td>Corpus size</td><td>Avg. length</td></tr><tr><td>Programming solutions</td><td>1.1k</td><td>194.6</td></tr><tr><td>Online tutorials</td><td>79.4k</td><td>1502.5</td></tr><tr><td>Library documentation</td><td>34k</td><td>953.4</td></tr><tr><td>StackOverflow posts</td><td>23.5M</td><td>689.2</td></tr><tr><td>Github files</td><td>1.7M</td><td>5135.4</td></tr></table>

Table 2: Five sources to form our retrieval dataset.

# 2.3 Canonical Document Annotation

To ensure reliable retrieval evaluation and estimate the upper bound of a RACG system with an ideal retriever, it's essential that all examples include canonical documents—the documents containing the necessary context to solve the programming problem. As most existing datasets lack these canonical documents, we annotate them from the corresponding retrieval pool, as shown in Table 1.

Basic programming problems The canonical document for examples in HumanEval and MBPP is the documents we created in §2.2 in the programming solutions pool. Since LiveCodeBench does not provide solutions to its problems, we do not annotate canonical documents for it.

Open-domain problems Since open-domain problems require libraries, we annotate the canonical library documentation for DS-1000 and ODEX examples. We first automatically parse out the library functions used in each program, and find their corresponding documentation entries. Then, we manually verify the functions and remove incorrect ones. This yields an average of 1.4 and 1.2 entries for DS-1000 and ODEX.

Repository-level problems We adopt canonical code from the original dataset as our canonical documents: 20-line code snippets of the missing functions in RepoEval, and the ground-truth edited files in SWE-bench. We obtain these from the completed local repositories from the original datasets.

# 2.4 Evaluation Metrics

For retrieval, we evaluate NDCG, Precision and Recall (Thakur et al., 2021) and use NDCG@10 percentage as our primary metric, following prior work (Izacard et al., 2022a). For code generation, we adopt the pass@k metric (Chen et al., 2021) to measure the execution correctness of programs. We evaluate the final RAG performance both in canonical and open retrieval setups.

# 3 Canonical RACG

We evaluate 10 top retrieval and 10 generation models on CODERAG-BENCH with canonical data sources. We report results of document retrieval (§3.2), direct NL-to-code generation (§3.3), and end-to-end RACG with retrieved context (§3.4).

# 3.1 Experimental Setup

Retrieval baselines We adopt 10 top-performing retrievers from three categories: sparse, dense, and proprietary APIs. For sparse retrievers,

we use BM25 (Robertson and Zaragoza, 2009), known for its robustness in domain adaptation (Thakur et al., 2021). Dense retrievers include BGE-base/large (Xiao et al., 2023), GIST-base/large (Solatorio, 2024), and SFR-Embedding-Mistral (Meng et al., 2024), all top-ranked on the MTEB leaderboard (Muennighoff et al., 2022). We also include open code embedding models, Codesage-small (Zhang et al., 2024) and Jina-v2-code Günther et al., 2023, which are specifically trained for code retrieval. Proprietary APIs include voyage-code-2 (VoyageAI, 2024), optimized for code retrieval, and openai-text-embedding-small-03, selected for its cost-effectiveness. Finally, we apply reranking with BGE-eranker-base(Xiao et al., 2023) on top-100 openai results before generation.

Generation baselines We adopt both code-specific LMs and strong general text-oriented LMs. For code-specific LMs, we use StarCoder2 (Lozhkov et al., 2024), CodeGemma (Team, 2024), CodeLlama (Roziere et al., 2023), and DeepSeekCoder (Guo et al., 2024) in various sizes. For general text LMs, we include three top-performing models: Llama3 (Meta, 2024), Command-R (CohereAI, 2024) specially optimized for RAG, and proprietary GPT models gpt-3.5-turbo-0125 and gpt-4o. We use the instruct version of all generation models if available, since they often perform better than the base versions.

Experimental setup For retrieval, we implement BM25 retrievers using pyserini (Lin et al., 2021) with parameter  $k_{1} = 1.2$  and  $b = 0.75$ , and use sentence-transformers (Reimers and Gurevych, 2019) for all dense models with open checkpoints. We prepend the top-5 retrieved documents to the original problems (we study the number of documents in §E), and do not include other contexts such as few-shot examples. For code generation, we use temperature  $t = 0.2$ ,  $top\_p = 0.95$  and sample one response for all generations, following prior work (Li et al., 2023). Specifically on SWEbench-Lite, we adopt the  $n = 21$  way sampling and majority-vote reranking strategy proposed by Agentless (Xia et al., 2024).

# 3.2 Retrieval Results

Table 3 shows retrieval results on six tasks.

<table><tr><td rowspan="2">Method</td><td colspan="3">Problem Solutions</td><td colspan="2">Library Docs</td><td colspan="2">In-Repository Files</td><td rowspan="2">Avg. All</td></tr><tr><td>HumanEval</td><td>MBPP</td><td>CSN</td><td>DS-1000</td><td>ODEX</td><td>RepoEval</td><td>SWE-bench-Lite</td></tr><tr><td>BM25</td><td>100.0</td><td>98.6</td><td>89.1</td><td>5.2</td><td>6.7</td><td>93.2</td><td>43.0</td><td>57.7</td></tr><tr><td>GIST-base (768)</td><td>98.0</td><td>98.0</td><td>89.9</td><td>12.0</td><td>12.1</td><td>81.2</td><td>46.8</td><td>58.0</td></tr><tr><td>GIST-large (1024)</td><td>100.0</td><td>98.9</td><td>89.6</td><td>13.6</td><td>28.0</td><td>82.9</td><td>47.8</td><td>61.7</td></tr><tr><td>BGE-base (768)</td><td>99.7</td><td>98.0</td><td>90.0</td><td>10.8</td><td>22.0</td><td>77.5</td><td>44.9</td><td>58.8</td></tr><tr><td>BGE-large (1024)</td><td>98.0</td><td>99.0</td><td>90.6</td><td>8.9</td><td>11.5</td><td>80.4</td><td>40.1</td><td>56.3</td></tr><tr><td>SFR-Mistral (4096)</td><td>100.0</td><td>99.0</td><td>-</td><td>19.3</td><td>37.1</td><td>83.8</td><td>62.7</td><td>67.0</td></tr><tr><td>Codesage-small (768)</td><td>100.0</td><td>96.3</td><td>90.7</td><td>8.9</td><td>14.3</td><td>94.1</td><td>47.1</td><td>60.1</td></tr><tr><td>Jina-v2-code (768)</td><td>100.0</td><td>97.7</td><td>-</td><td>26.2</td><td>19.9</td><td>90.5</td><td>58.3</td><td>65.4</td></tr><tr><td>OpenAI-03 (1536)</td><td>100.0</td><td>98.9</td><td>-</td><td>18.2</td><td>16.5</td><td>93.0</td><td>43.3</td><td>61.7</td></tr><tr><td>Voyage-code (1536)</td><td>100.0</td><td>99.0</td><td>-</td><td>33.1</td><td>26.6</td><td>94.3</td><td>29.1</td><td>63.7</td></tr></table>

Comparison of lexical and neural retrievers BM25 has been widely used as a primary retrieval model in recent RACG work (Zhou et al., 2023; Jimenez et al., 2024), yet comprehensive comparisons against diverse retrieval systems are often under-explored. While prior studies indicate that neural retrieval systems often underperform BM25 baselines in out-of-domain scenarios (Thakur et al., 2021), our analysis of CODERAG-BENCH reveals that dense embedding models frequently surpass BM25. We hypothesize that this is because many competitive retrieval models are trained on diverse tasks across various domains, including code data (Asai et al., 2023; Su et al., 2023), enhancing their robustness in code retrieval setups.

Do code retrieval models perform better? At similar parameter scales, models specifically trained for code retrieval tasks typically show superior performance. Notably, Jina-v2-code outperforms GIST-base and BGE-base by 7.4 and 6.6 average NDCG@10, respectively, while Voyage-code significantly outperforms OpenAI-03.

Do larger retrieval models perform better? Among dense retrieval models, increasing model size often leads to better retrieval performance, similar to the trends observed in LMs (Brown et al., 2020). In particular, GIST-large  $(340M)$  constantly outperforms GIST-base  $(110M)$ , and SFR-Mistral  $(7B)$  achieves the best among all open sparse and dense models on all tasks, surpassing proprietary embedding models on several tasks.

Efficiency While larger retrieval models often outperform smaller ones, they often introduce significant costs. We analyze efficiency, focusing on (i) encoding latency: latency to encode documents offline, and (ii) search latency: latency to encode queries/documents and calculate their similarities,

Table 3: Retrieval performance (NDCG@10) on code generation datasets. LiveCodeBench is excluded due to lack of ground-truth solutions. RepoEval is at the function level with 2k context tokens. Embedding dimension sizes are listed next to method names. Bold indicates best performance, underline indicates second-best. Highlighted models are specifically trained for code domains. Avg. reflects overall scores, excluding CodeSearchNet.  

<table><tr><td>Method</td><td>Encoding</td><td>Search</td><td>Model</td><td>Index</td></tr><tr><td>BM25</td><td>0.15ms</td><td>0.02ms</td><td>-</td><td>141MB</td></tr><tr><td>GIST-base</td><td>3.7ms</td><td>9.7ms</td><td>440MB</td><td>307MB</td></tr><tr><td>GIST-large</td><td>13ms</td><td>18ms</td><td>1300MB</td><td>409MB</td></tr><tr><td>SFR-Mistral</td><td>316ms</td><td>113ms</td><td>14220MB</td><td>1638 MB</td></tr><tr><td>Voyage-code</td><td>22ms</td><td>40ms</td><td>-</td><td>1172MB</td></tr><tr><td>OpenAI-03</td><td>31ms</td><td>47ms</td><td>-</td><td>1172MB</td></tr></table>

Table 4: Efficiency analysis for document retrieval.

(iii) model storage requirements, and (iv) index storage requirements. We conduct efficiency analysis on sampled CodeSearchNet Python data.<sup>7</sup> See experimental details in §B. As shown in Table 4, BM25 indexing and searching takes only seconds to finish. Compared to base-size GIST-base, the SFR-Mistral model is more powerful in retrieval, yet requires over  $5 \times$  larger index storage, and adds nearly  $100 \times$  latency to encode documents, suggesting that the efficiency aspect should also be carefully studied for RAG pipelines.

# 3.3 Generation with Canonical Documents

We first evaluate possible lower- and upper-bounds on RACG results by testing generation (i) without any retrieval, and (ii) with ground-truth documents. We report both results in Table 5. Compared to the base generation without contexts, incorporating canonical contexts improves in most setups, and substantially so on basic programming problems.

On open-domain tasks, most code-specific LMs increase up to 5.2 points, signifying that most models can benefit from indirectly helpful documents. In contrast, GPTs show no gains with retrieval. We hypothesize that this is because both datasets mostly test on common Python libraries, which

<table><tr><td rowspan="3">Method</td><td colspan="5">Basic Programming</td><td colspan="5">Open-Domain</td><td colspan="4">Repo-Level</td><td></td></tr><tr><td colspan="2">HumanEval</td><td colspan="2">MBPP</td><td rowspan="2">LCB w/o</td><td colspan="2">DS-1000</td><td colspan="2">ODEX</td><td colspan="2">ODEX-hard</td><td colspan="2">RepoEval</td><td>SWE-bench</td><td></td></tr><tr><td>w/o</td><td>gold</td><td>w/o</td><td>gold</td><td>w/o</td><td>gold</td><td>w/o</td><td>gold</td><td>w/o</td><td>gold</td><td>w/o</td><td>gold</td><td>w/o</td><td></td></tr><tr><td>StarCoder2-7B</td><td>31.7</td><td>94.5</td><td>10.4</td><td>34.8</td><td>1.5</td><td>29.2</td><td>30.0</td><td>14.6</td><td>17.5</td><td>10.3</td><td>17.2</td><td>26.5</td><td>42.0</td><td>0.0</td><td>0.7</td></tr><tr><td>CodeGemma-7B</td><td>49.4</td><td>77.4</td><td>48.0</td><td>52.2</td><td>21.5</td><td>20.1</td><td>19.8</td><td>18.9</td><td>18.2</td><td>13.8</td><td>13.8</td><td>24.7</td><td>32.2</td><td>0.0</td><td>0.3</td></tr><tr><td>CodeLlama-7B</td><td>34.8</td><td>87.2</td><td>23.8</td><td>42.8</td><td>13.5</td><td>21.8</td><td>26.1</td><td>35.8</td><td>41.0</td><td>27.6</td><td>31.0</td><td>24.1</td><td>38.3</td><td>0.0</td><td>0.0</td></tr><tr><td>CodeLlama-34B</td><td>42.7</td><td>84.8</td><td>51.2</td><td>88.0</td><td>5.8</td><td>34.7</td><td>37.0</td><td>34.9</td><td>38.0</td><td>17.2</td><td>27.6</td><td>29.8</td><td>42.6</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeekCoder-7B</td><td>70.1</td><td>87.8</td><td>60.8</td><td>63.6</td><td>30.5</td><td>41.4</td><td>43.2</td><td>39.2</td><td>41.7</td><td>17.2</td><td>24.1</td><td>28.2</td><td>43.7</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeekCoder-33B</td><td>78.0</td><td>95.7</td><td>61.0</td><td>92.2</td><td>33.8</td><td>40.2</td><td>40.1</td><td>28.0</td><td>28.9</td><td>24.1</td><td>31.0</td><td>32.4</td><td>45.3</td><td>0.3</td><td>0.7</td></tr><tr><td>Llama3-8B</td><td>57.9</td><td>65.2</td><td>35.6</td><td>52.8</td><td>2.8</td><td>28.9</td><td>31.1</td><td>37.4</td><td>33.7</td><td>13.8</td><td>17.2</td><td>26.0</td><td>43.2</td><td>0.0</td><td>0.3</td></tr><tr><td>Command-R</td><td>43.3</td><td>51.2</td><td>37.2</td><td>37.8</td><td>10.0</td><td>25.8</td><td>28.5</td><td>35.5</td><td>36.0</td><td>20.7</td><td>20.7</td><td>23.9</td><td>37.0</td><td>0.0</td><td>0.3</td></tr><tr><td>GPT-3.5-turbo</td><td>72.6</td><td>91.5</td><td>70.8</td><td>72.6</td><td>35.3</td><td>43.7</td><td>42.9</td><td>41.7</td><td>40.3</td><td>17.2</td><td>24.1</td><td>23.9</td><td>39.1</td><td>0.7</td><td>6.3</td></tr><tr><td>GPT-4o</td><td>75.6</td><td>92.6</td><td>79.4</td><td>81.4</td><td>43.8</td><td>52.7</td><td>51.2</td><td>44.6</td><td>44.2</td><td>20.7</td><td>27.6</td><td>32.4</td><td>46.1</td><td>2.3</td><td>30.7</td></tr></table>

Table 5: Code generation pass@1 without additional contexts  $(w / o)$ , and with ground-truth documents (gold). We only report  $w / o$  for LCB because LCB does not have ground-truth documents. We highlight results showing gold  $>$ $w / o$  with green (darker green when having  $10+$  increases), and with red if gold  $< w / o$ .

powerful models may have already memorized, similar to their memorization of factual knowledge (Mallen et al., 2023; Kandpal et al., 2023), thereby reducing the need for retrieval. To verify this hypothesis, we build an ODEX subset of examples with the 20 least used libraries, i.e., ODEX-hard. As shown in Table 5, adding documents retrieved with most methods improves the results by  $20.3 - 40.1\%$ , showing the effectiveness of RACG on challenging coding tasks using unfamiliar libraries.

Repository-level challenges All models show gains of 7.5-17.2 points with canonical snippets in RepoEval, but SWE-bench Lite proves much more challenging — only GPT-3.5-turbo and GPT-4o achieve non-trivial results, consistent with previous findings (Yang et al., 2024). Notably, GPT-4o shows a  $27.4\%$  increase when using gold documents on SWE-bench, indicating that retrieval significantly enhances performance when paired with strong core generation capabilities, even in highly challenging coding tasks.

# 3.4 Retrieval-Augmented Code Generation

We now experiment with top-performing retrieval and generation models in the full RACG setting, which requires both retrieve documents and generating conditioned on the documents. We select the best retrieval models from each type: BM25, GIST-large, Voyage, and OpenAI embeddings. For generation, we select (i) StarCoder2-7B: a weaker model that benefits the most from contexts; (ii) DeepSeekCoder-7B: one of the strongest open code LMs; and (iii) GPT-3.5-turbo: one of the top proprietary models. For each dataset, we retrieve the most relevant contexts from its canonical source marked in Table 1, and retrieve programming solutions for LiveCodeBench. Table 6 shows the results. Note that we exclude canonical docs (answers) from the retrieval corpora for basic programming tasks.

Overall, the best retrieval models vary depending on the task and underlying LMs. In some cases, top-performing retrieval models do not lead to the best RACG outcomes, highlighting the need to evaluate RACG systems holistically across varied tasks.

Basic programming problems Most retrieved contexts help StarCoder2 generations. On MBPP, RACG even outperforms canonical setup by 15.6-17.8. However, RACG does not improve DeepSeekCoder generations, which we observe is due to over-complicated and ungrammatically repetitive generations when with additional contexts. In comparison, GPT-3.5-turbo can effectively improve with added contexts, showing its better ability to leverage augmented contexts.

Open-domain problems The weaker StarCoder2 benefits from retrieved library documentation across all datasets, while DeepSeekCoder and GPT-3.5 show gains mainly on ODEX-hard problems. This aligns with findings from the canonical document setup, indicating that RACG is particularly effective for less popular libraries. Interestingly, despite relatively low NDCG@10 scores, the best-performing RACG combinations match their canonical results on ODEX-hard.

Repository-level problems All models show improvements with retrieved code snippets on RepoEval, with RACG using strong retrievers like openai-embeddings performing on par with—or even surpassing—the canonical setup, likely due to the additional context provided to the models. On SWE-Bench, the best-performing combination, Retrieval-then-Rerank and GPT4o, yields a 21-point improvement over the no-retrieval baseline. However, there remains a 9-point gap compared to the gold setup, indicating room for improvement on the retrieval side, as reflected in the limited code retrieval performance shown in Table 3.

<table><tr><td rowspan="2">Method</td><td colspan="3">Basic Programming</td><td colspan="3">Open-Domain</td><td colspan="2">Repo-Level</td></tr><tr><td>HumanEval</td><td>MBPP</td><td>LCB</td><td>DS-1000</td><td>ODEX</td><td>ODEX-hard</td><td>RepoEval</td><td>SWE-bench</td></tr><tr><td colspan="9">w/ StarCoder2-7B</td></tr><tr><td>None</td><td>31.7</td><td>2.4</td><td>1.5</td><td>29.2</td><td>14.6</td><td>10.3</td><td>26.5</td><td>0.0</td></tr><tr><td>BM25</td><td>43.9</td><td>51.8</td><td>1.0</td><td>36.7</td><td>14.1</td><td>13.8</td><td>36.7</td><td>0.0</td></tr><tr><td>GIST-large</td><td>38.7</td><td>50.4</td><td>0.5</td><td>35.9</td><td>17.3</td><td>13.8</td><td>40.8</td><td>0.3</td></tr><tr><td>Voyage, code</td><td>39.0</td><td>52.6</td><td>0.3</td><td>36.0</td><td>15.3</td><td>10.3</td><td>45.8</td><td>0.3</td></tr><tr><td>OpenAI, small</td><td>39.0</td><td>52.6</td><td>1.5</td><td>35.5</td><td>15.9</td><td>17.2</td><td>51.2</td><td>0.0</td></tr><tr><td>OpenAI, rerank</td><td>34.8</td><td>53.4</td><td>0.5</td><td>33.4</td><td>14.1</td><td>17.2</td><td>53.9</td><td>0.3</td></tr><tr><td>Gold</td><td>94.5</td><td>34.8</td><td>-</td><td>30.0</td><td>17.5</td><td>17.2</td><td>42.0</td><td>0.7</td></tr><tr><td colspan="9">w/ DeepseekCoder-7B-instruct</td></tr><tr><td>None</td><td>70.1</td><td>60.8</td><td>30.5</td><td>41.4</td><td>39.2</td><td>17.2</td><td>28.2</td><td>0.7</td></tr><tr><td>BM25</td><td>68.9</td><td>60.0</td><td>31.8</td><td>36.6</td><td>37.8</td><td>20.7</td><td>37.3</td><td>0.0</td></tr><tr><td>GIST-large</td><td>66.3</td><td>56.6</td><td>33.8</td><td>35.9</td><td>34.9</td><td>20.7</td><td>44.5</td><td>0.3</td></tr><tr><td>Voyage, code</td><td>66.5</td><td>56.4</td><td>31.8</td><td>35.9</td><td>39.4</td><td>17.2</td><td>46.6</td><td>0.3</td></tr><tr><td>OpenAI, small</td><td>68.9</td><td>58.6</td><td>32.0</td><td>35.5</td><td>37.1</td><td>20.7</td><td>55.2</td><td>0.3</td></tr><tr><td>OpenAI, rerank</td><td>53.0</td><td>60.6</td><td>31.5</td><td>36.5</td><td>37.1</td><td>24.1</td><td>55.5</td><td>0.3</td></tr><tr><td>Gold</td><td>87.8</td><td>63.6</td><td>-</td><td>43.2</td><td>41.7</td><td>24.1</td><td>48.1</td><td>0.0</td></tr><tr><td colspan="8">w/ GPT-3.5-turbo</td><td>GPT-4o</td></tr><tr><td>None</td><td>72.6</td><td>70.8</td><td>35.3</td><td>43.7</td><td>41.7</td><td>17.2</td><td>23.9</td><td>2.3</td></tr><tr><td>BM25</td><td>73.2</td><td>72.4</td><td>35.5</td><td>36.9</td><td>41.0</td><td>24.1</td><td>30.8</td><td>6.7</td></tr><tr><td>GIST-large</td><td>73.2</td><td>68.2</td><td>34.8</td><td>36.7</td><td>36.2</td><td>13.8</td><td>38.3</td><td>19.3</td></tr><tr><td>Voyage, code</td><td>75.0</td><td>66.8</td><td>34.5</td><td>37.4</td><td>41.0</td><td>20.7</td><td>43.2</td><td>15.7</td></tr><tr><td>OpenAI, small</td><td>73.8</td><td>68.4</td><td>35.8</td><td>36.9</td><td>40.3</td><td>17.2</td><td>48.0</td><td>21.0</td></tr><tr><td>OpenAI, rerank</td><td>64.0</td><td>72.6</td><td>33.5</td><td>37.4</td><td>40.5</td><td>17.2</td><td>49.6</td><td>21.7</td></tr><tr><td>Gold</td><td>91.5</td><td>72.6</td><td>-</td><td>42.9</td><td>40.3</td><td>24.1</td><td>39.1</td><td>30.7</td></tr></table>

Table 6: Performance of retrieval-augmented code generation, with top retrieval and generation models. We bold-type the best RACG results. We test gpt-4o on SWE-bench to show non-trivial results than gpt-3.5-turbo. Note that we exclude code canonical answer from the retrieval corpora for basic programming tasks.  

<table><tr><td rowspan="2">Model</td><td rowspan="2">Retriever</td><td colspan="7">HumanEval</td><td colspan="7">ODEX</td></tr><tr><td>w/o</td><td>Prog</td><td>Tut</td><td>Docs</td><td>SO</td><td>GitHub</td><td>All</td><td>w/o</td><td>Prog</td><td>Tut</td><td>Docs</td><td>SO</td><td>GitHub</td><td>All</td></tr><tr><td rowspan="3">StarCoder</td><td>BM25</td><td></td><td>97.6</td><td>27.4</td><td>29.3</td><td>32.9</td><td>30.5</td><td>97.6</td><td></td><td>18.2</td><td>13.4</td><td>14.1</td><td>11.6</td><td>15.9</td><td>16.2</td></tr><tr><td>GIST</td><td>31.7</td><td>67.1</td><td>34.8</td><td>26.7</td><td>32.3</td><td>32.9</td><td>69.1</td><td>14.6</td><td>14.6</td><td>15.7</td><td>17.3</td><td>11.4</td><td>15.5</td><td>17.1</td></tr><tr><td>OpenAI</td><td></td><td>97.6</td><td>29.3</td><td>24.4</td><td>36.0</td><td>31.1</td><td>97.6</td><td></td><td>18.7</td><td>14.1</td><td>15.9</td><td>10.9</td><td>16.9</td><td>15.3</td></tr><tr><td>GPT-4o</td><td>OpenAI</td><td>75.6</td><td>94.5</td><td>90.2</td><td>90.9</td><td>91.5</td><td>84.8</td><td>95.1</td><td>44.6</td><td>49.2</td><td>44.2</td><td>47.6</td><td>40.3</td><td>39.4</td><td>39.6</td></tr></table>

Table 7: Comparing five retrieval sources on HumanEval and ODEX, using StarCoder2 (top) and GPT-4o (bottom).

# 4 RACG with Open Retrieval

Besides retrieving documents from the canonical source, we explore RACG with open retrieval from all sources (§2.2) on three category-representative datasets: HumanEval, ODEX, and RepoEval. We also study mixed retrieval with documents from all sources, where we aggregate the top-1 documents from all five sources as augmented contexts.<sup>8</sup> We use the three top retrievers along with StarCoder2 and OpenAI retrieval with GPT-4o generation, to study open RACG with weak and strong LMs.

General programming: HumanEval Among all sources, SO posts can improve the results by 1.8-4.3, regardless of the choice of retrievers. Tutorialis can improve results by 2.1 only with the GIST retriever.From manual examinations of the results, many retrieved posts and tutorials are about the same programming problem as the HumanEval ex

ample, with code and detailed textual explanations, hence could hint or disclose the answer. Other retrieval sources do not often contain relevant content thus do not bring improvements. Surprisingly, generation with mixed documents performs as well as using the gold documents, suggesting that the model can discern and integrate the most useful content from a mixture of texts.

Open-domain: ODEX Programming solutions are the most helpful source by bringing 3.8-4.3 gains, even surpassing gains of canonical documentation. Notably, both GPT-4o and StarCoder using OpenAI retrieval from programming solutions, outperform their variants retrieving from documentation by 3.2 and 1.6 points. Although the retrieved content is only sometimes functionally relevant to the ODEX examples, they can exemplify the correct usage of libraries such as regex in solutions and requests in GitHub files, thus guiding the generation to be more functionally correct. Similar to HumanEval, GIST-large is particularly good

at retrieving tutorials, while BM25 and OpenAI embeddings find higher-quality program solutions, indicating their respective domain advantages.

Repository-level: RepoEval Open sources are less useful than code snippets in the local repository. Understanding local code contexts is crucial and irreplaceable than external resources. When using both local and open-source contexts  $(L + O)$ , models surpass the no-retrieval baseline, yet are still only comparable with Local, suggesting more efforts and insights to benefit from both sources.

<table><tr><td>Method</td><td>w/o</td><td>Local</td><td>Prog</td><td>Tut</td><td>Docs</td><td>SO</td><td>GitHub</td><td>Open</td><td>L+O</td></tr><tr><td colspan="10">StarCoder2-7B</td></tr><tr><td>BM25</td><td></td><td>36.7</td><td>23.6</td><td>25.2</td><td>23.9</td><td>23.6</td><td>25.5</td><td>23.6</td><td>31.4</td></tr><tr><td>GIST</td><td>26.5</td><td>40.8</td><td>24.1</td><td>23.3</td><td>21.7</td><td>24.7</td><td>24.4</td><td>24.1</td><td>41.8</td></tr><tr><td>OpenAI</td><td></td><td>51.2</td><td>23.9</td><td>24.1</td><td>24.1</td><td>23.1</td><td>22.8</td><td>24.9</td><td>50.9</td></tr><tr><td colspan="10">GPT-4o</td></tr><tr><td>OpenAI</td><td>32.4</td><td>62.2</td><td>35.4</td><td>28.7</td><td>27.8</td><td>29.0</td><td>28.2</td><td>30.3</td><td>54.2</td></tr></table>

Exploring optimal chunking strategies Adding many documents may exceed model context limits hence impairing RACG, we thus explore various chunking strategies to better integrate retrieval. Compared to the no-chunking baseline, we study (i) post-retrieval chunking that takes the first N-tokens of each document, (ii) post-retrieval with reranking using BGE-eranker-base ( $\S 3.1$ ) to find the most relevant N-token chunk from each document, and (iii) pre-retrieval chunking that chunks documents beforehand and retrieves N-token pieces directly.

Figure 2: Performance with different chunking sizes.

We compare (i) using the first N-tokens for N from 200 to 1500 (Figure 2). Most sources are best represented by the first 800 tokens except for SO posts. However, we find (ii) reranking within this optimal range of 200-800 tokens greatly degrades the results, showing limited utility of current rerankers. Lastly, (iii) pre-retrieval achieves the highest scores on almost all document sources (Table 9).

Table 8: RACG with open retrieval on RepoEval.  

<table><tr><td>Method</td><td>Tutorials</td><td>Docs</td><td>SO</td><td>GitHub</td></tr><tr><td>Full text</td><td>6.7</td><td>17.7</td><td>28.0</td><td>3.7</td></tr><tr><td>First chunk</td><td>27.4</td><td>29.3</td><td>30.5</td><td>30.5</td></tr><tr><td>w/ reranking</td><td>9.1</td><td>9.1</td><td>14.0</td><td>13.4</td></tr><tr><td>Pre-retrieval</td><td>31.1</td><td>32.9</td><td>33.5</td><td>29.3</td></tr></table>

Table 9: Comparing chunking strategies on HumanEval.

# 5 Related Work

Code generation Neural code generation has been a crucial task (Lu et al., 2021), and increasingly strong code LMs have been created (Roziere et al., 2023; Li et al., 2023; Guo et al., 2024; Team, 2024) to solve various tasks (Chen et al., 2021; Lai et al., 2023; Jimenez et al., 2024). However, most LMs generate code solely based on NL queries and model parametric knowledge, without using external programming sources (e.g., tutorials). To fill in this gap and allow systematic studies of RACG, we integrate various datasets and retrieval sources to build CODERAG-BENCH.

Retrieval augmented generation (RAG) RAG has been widely used in knowledge-intensive tasks (Lewis et al., 2020; Guu et al., 2020), however, mostly on text-centric tasks using general domain corpora such as Wikipedia (Asai et al., 2024). Some works used programming context retrieved from repositories (Ding et al., 2023; Yang et al., 2024) or documentations (Zhou et al., 2023), yet none of them considered RACG across varied coding tasks and knowledge sources. In text-centric tasks, unified benchmarks such as BEIR (Thakur et al., 2021) and KILT (Petroni et al., 2020) aggregate retrieval and generation tasks and facilitate its progress (Muennighoff et al., 2022). To similarly enable systematic studies of RACG across coding tasks and retrieval sources, we curate a unified benchmark and release its RACG codebase.

# 6 Conclusion

In this work, we propose CODERAG-BENCH, a benchmark for retrieval-augmented code generation with various coding tasks and retrieval sources. With our experiments with top-performing retrieval and generation models, we show that retrieving external documents can greatly benefit code generation. However, current retrieval models struggle to find useful documents, and generation models have limited context capacity and RAG abilities, both leading to suboptimal RACG results. We hope CODERAG-BENCH can serve as a solid testbed to advance future endeavors in this direction.

# Limitations

We propose a new paradigm, retrieval-augmented code generation, equipped with a comprehensive benchmark CODERAG-BENCH. However, as an initial exploration in this field, our work could be extended in task and language diversity, as well as model and methodological improvements.

We aggregate various existing code generation tasks, but many interesting scenarios such as code debugging remain under-explored. Meanwhile, we focus on coding tasks using Python programming language, but extrapolating to other languages may bring additional challenges.

Meanwhile, for benchmarking purposes, we mostly experimented with vanilla retrieval, reranking, and generation methods, but better backbone models and advanced methods for each RACG component are yet fully explored. Our results may not represent all model behaviors, and we encourage future works to build methods that break certain limitations we observe in current systems.

# Acknowledgment

We thank Shuyan Zhou and Xinran Zhao for the helpful discussions in the early stage of this project; Saujas Vaduguru, Jing Yu Koh, Alex Xie, and Andy Liu for providing valuable feedback for the draft.

# References

Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2023. Task-aware retrieval with instructions. In *Findings of the Association for Computational Linguistics: ACL* 2023, pages 3650-3675, Toronto, Canada. Association for Computational Linguistics.  
Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and Wen-tau Yih. 2024. Reliable, adaptable, and attributable language models with retrieval. arXiv preprint arXiv:2403.03187.  
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens

Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. ArXiv, abs/2005.14165.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.  
CohereAI. 2024. Command r.  
Together Computer. 2023. Redpajama: An open source recipe to reproduce llama training dataset.  
Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. 2023. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.  
Michael Gunther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrupas, Saba Sturua, Bo Wang, et al. 2023. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. arXiv preprint arXiv:2310.19923.  
Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. 2024. Deepseek-coder: When the large language model meets programming-the rise of code intelligence. arXiv preprint arXiv:2401.14196.  
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929-3938. PMLR.  
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022a. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research.  
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane A. Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022b. Few-shot learning with retrieval augmented language models. *ArXiv*, abs/2208.03299.  
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations.  
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023. Large language models struggle to learn long-tail knowledge. In International Conference on Machine Learning, pages 15696-15707. PMLR.  
Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2023. Ds-1000: a natural and reliable benchmark for data science code generation. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org.  
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459-9474. Curran Associates, Inc.  
Raymond Li, Loubna Ben allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia LI, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Joel Lamy-Poirier, Joao Monteiro, Nicolas Gontier, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Ben Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason T Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Urvashi Bhattacharyya, Wenhao Yu, Sasha Luccioni, Paulo Villegas, Fedor Zhdanov, Tony Lee, Nadav Timor, Jennifer Ding, Claire S Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Carolyn Jane Anderson, Brendan DolanGavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Munoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro Von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you! Transactions on Machine Learning Research. Reproducibility Certification.  
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Remi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen, Po Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. 2022. Competition-level code generation with alphanumeric. Science, 378(6624):1092-1097.

Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Zheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021).  
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems.  
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.  
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. CoRR, abs/2102.04664.  
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802-9822, Toronto, Canada. Association for Computational Linguistics.  
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Sfremedding-mistral:enhance text retrieval with transfer learning.  
Meta. 2024. Introducing meta llama 3: The most capable openly available llm to date.  
Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2022. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316.  
Arnold Overwijk, Chenyan Xiong, Xiao Liu, Cameron VandenBerg, and Jamie Callan. 2022. Clueweb22: 10 billion web documents with visual and semantic information. arXiv preprint arXiv:2211.15848.  
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. 2020. Kilt: a benchmark for knowledge intensive language tasks. arXiv preprint arXiv:2009.02252.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.  
Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3:333-389.  
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.  
Aivin V. Solatorio. 2024. Gistembed: Guided in-sample selection of training negatives for text embedding fine-tuning.  
Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao Shi, Che Liu, Qian Liu, and Tao Yu. 2024. Arks: Active retrieval in knowledge soup for code generation. arXiv preprint arXiv:2402.12317.  
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2023. One embedder, any task: Instruction-finetuned text embeddings. In Findings of the Association for Computational Linguistics: ACL 2023, pages 1102-1121, Toronto, Canada. Association for Computational Linguistics.  
CodeGemma Team. 2024. Codegemma: Open code models based on gemma.  
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).  
VoyageAI. 2024. voyage-code-2: Elevate your code retrieval.  
Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. 2024. Opendevin: An open platform for ai software developers as generalist agents. arXiv preprint arXiv:2407.16741.  
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. 2023a. Learning to filter context for retrieval-augmented generation. arXiv preprint arXiv:2311.08377.  
Zhiruo Wang, Shuyan Zhou, Daniel Fried, and Graham Neubig. 2023b. Execution-based evaluation for open-domain code generation. In *Findings of the Association for Computational Linguistics: EMNLP* 2023. Association for Computational Linguistics.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2024. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489.  
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. arXiv.  
John Yang, Carlos E Jimenez, Alexander Wettig, Killian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793.  
Dejiao Zhang, Wasi Ahmad, Ming Tan, Hantian Ding, Ramesh Nallapati, Dan Roth, Xiaofei Ma, and Bing Xiang. 2024. Code representation learning at scale. arXiv preprint arXiv:2402.01935.  
Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code completion through iterative retrieval and generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.  
Shuyan Zhou, Uri Alon, Frank F. Xu, Zhengbao Jiang, and Graham Neubig. 2023. Docprompting: Generating code by retrieving the docs. In *The Eleventh International Conference on Learning Representations*.

# A Example Illustrations

# A.1 Example with Canonical Documents

To present our canonical document annotation (§2.3) more concretely, we illustrate examples with their annotated canonical documents. Figure 3 shows the general-programming examples, with one HumanEval and one MBPP example, respectively. Figure 4 shows two open-domain coding examples with canonical library documentation from DS-1000 and ODEX, respectively.

# A.2 RACG with Helpful and Distracting Documents

Beyond the numerical numbers reported in experiment sections, here we provide some concrete examples that: (i) benefit from RACG when relevant documents are retrieved, and (ii) distracted by irrelevant documents retrieved hence results in degraded performance.

# B Additional Details about Retrieval Efficiency

For open access models, we use the same single A100 GPU with 80 GRAM, with a batch size of 64 for GIST base and large, and 8 for SFR-Mistral. For proprietary models, we estimate their efficiency using a batch size of 64. We then average the time for each batch for each query and document. For Voyage-code, we apply a "dynamic-batching" technique that makes sure the total tokens in the batch won't exceed the token limit. For both open and proprietary models, we define the search efficiency as the time it takes to embed individual query and the time to calculate similarities. Note that the time for both can be optimized by tokenizing all documents and all queries, then taking the dot product. The actual runtime for API models varies for each organization with different rate limits and the batch size. For this experiment, we set the maximum context length to match the maximum length of the original models. This notably increases the encoding latency of SFR Mixtral, which has a longer maximum context window size than smaller embedding models.

# C Result Reproduction

In Table 5 in §3, we are able to reproduce most results reported in the original papers, but with minor variances. Here we explain the differences in implementation and (potential) reasons that lead to these small performance variances.

Our approach To keep a fair comparison among all models, we use the same prompt for each dataset when evaluating all models. Meanwhile, we use zero-shot prompts without any additional instructions, i.e., only input the original problem description of the example, to prevent unknown effects on the model performance when using different instructions and/or in-context examples.

According to this setup, we next describe the differences in prompts used by the original works and how they may affect the results.

StarCoder2 The StarCoder2 technical report (Lozhkov et al., 2024) reported results on the HumanEval, MBPP, and DS-1000 datasets. On HumanEval, our reproduced results (31.7) is slightly lower than their number (35.4), possibly because the original paper additionally input the test cases as additional information in the prompt, whereas in our basic NL-to-code setup, no test cases are provided. This additional information may cause their results to be higher.

On MBPP dataset, they adopt a subset of MBPP, i.e., 399 out of 427 examples that have additional test cases populated by Liu et al. (2023). In contrast, we evaluate on the entire dataset, which is likely to cause the variance in results.

On DS-1000, the original paper samples 40 generations and report the pass@1 rate, while we only generate one program with greedy decoding. This difference in decoding strategy may cause slight variance in the results.

CodeGemma The CodeGemma technical report (Team, 2024) reported results on HumanEval and MBPP datasets, but does not provide any details about the instructions, few-shot examples, or other parts of the prompt that they use. We were able to roughly reproduce their reported results, but with 3-5 points less in pass@1.

CodeLlama The CodeLlama technical report (Roziere et al., 2023) reports results on HumanEval and MBPP datasets. We were able to perfectly reproduce their results on the HumanEval dataset under the zero-shot setting. However, for MBPP experiments, they use 3-shot prompting, which could potentially explain that our zero-shot results are 4 points lower in pass@1.

DeepSeekCoder The DeepSeekCoder technical report (Guo et al., 2024) reports results on HumanEval and MBPP for the 7B-instruct-v1.5 and the 33B-instruct models, the report additionally report DS1000 results for the 33B-instruct model. We could reproduce the original results on HumanEval

# Canonical Document

def truncate_number(number: float) -> float:
    "" Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1).

Return the decimal part of the number.  
>>> truncate_number(3.5)  
0.5  
return number % 1.0

# Problem

def truncate_number(number: float) -> float:
    "" Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1).

Return the decimal part of the number.  
>>> truncate_number(3.5)  
0.5

# Canonical Document

Write a python function to remove first and last occurrence of a given character from the string.

def remove_Occ(s,ch):  
    for i in range(len(s)):  
        if (s[i] == ch):  
            s = s[0:i] + s[i+1:]  
            break  
    for i in range(len(s) - 1,-1,-1):  
        if (s[i] == ch):  
            s = s[0:i] + s[i+1:]  
            break  
    return s

# Problem

Write a python function to remove first and last occurrence of a given character from the string.

Figure 3: HumanEval (left) and MBPP (right) examples with annotated canonical solutions.

and DS-1000, but got slightly worse results on MBPP because they used few-shot prompting, which should outperform our zero-shot method.

Llama3 Since there is no technical report available yet, the official blog post  $^{10}$  report results on HumanEval, without any descriptions on prompting construction or the inference process. Our reproduced results are about 4 points lower than their original results.

# D Analysis on Open-Domain Coding Problems

In §3.4, providing the documentation of required libraries brings limited benefits, especially with strong proprietary models such as GPT and Gemini. While we hypothesize that these strong models are sufficiently familiar with the required libraries and in turn barely benefit from additional information about them, in this section, we quantitatively investigate this issue and verify its validity.

Concretely, we construct a subset of ODEX containing only examples with less common libraries. We use the real-world distribution of all libraries involved in ODEX and select examples that use the top 20 least common libraries (e.g., sqlite3, ftplib, flask). We then evaluate model performance on this subset and compare the results with and without documentation in model contexts.

With varied retrieval models Aligning with §3.4, we examine the RACG results using documentation retrieved by different retrieval models. As shown in Table 10, augmenting documentation retrieved with most methods improves the results by  $20.3 - 40.1\%$ . Compared to the entire ODEX set where most queries require common libraries, this hard-library split more clearly demonstrates the effectiveness of augmenting library documentation. This result verifies our hypothesis that strong GPT models are familiar with most common libraries, and can only benefit from additional library information when harder libraries are required.

<table><tr><td>Model</td><td>none</td><td>BM25</td><td>GIST</td><td>Voyage</td><td>OpenAI</td><td>Gold</td></tr><tr><td>GPT-3.5-turbo</td><td>17.2</td><td>24.1</td><td>13.8</td><td>20.7</td><td>17.2</td><td>24.1</td></tr><tr><td>GPT-4</td><td>20.7</td><td>24.1</td><td>17.2</td><td>27.6</td><td>24.1</td><td>27.6</td></tr></table>

Table 10: RACG results on the subset of ODEX examples using the least common libraries.

# E How Many Documents to Augment?

Different models have varied context length limits and context utilization abilities. Therefore, we study how model performance varies when providing different numbers of documents in the context. We experiment with one representative dataset for each task category: HumanEval since it is the most commonly used dataset, ODEX for its broad domain coverage, and RepoEval for its solvable dif

# Canonical Document

pandas.reference api.pandas.dataframe.groupby pandas.DataFrame.groupby DataFrame.groupby(by==None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=True, no_default=no_default, observed=False, dropna=True)[source]

中

pandas.reference api.pandas.dataframe.squeeze pandas.DataFrame.squeeze

DataFrame}squeeze(axis=None)[source]

Squeeze 1 dimensional axis objects into scalars. Series or

DataFrames with a single element are squeezed to a scalar.

# Problem

What is best way to achieve this? closest I got was with the zip function but haven't managed to make it work for more then one level (two columns).

A:

<code>

import pandas as pd

df = pd.DataFrame({'name': ['A', 'A', 'B', 'C', 'B', 'A'], 'v1': ['A1', 'A2', 'B1',

'C1', 'B2', 'A2'], 'v2': ['A11', 'A12', 'B12', 'C11', 'B21', 'A21'], 'v3': [1, 2, 3, 4, 5, 6])

</code>

result  $=$  ... # put solution in this variable

BEGIN SOLUTION

<code>

# Canonical Document

python.library(socket#socket(socket.send

socket.send(bytes[ flags])

Send data to the socket. The socket must be connected to a remote socket. The optional flags argument has the same meaning as for recv() above. Returns the number of bytes sent. Applications are responsible for checking that all data has been sent; if only some of the data was transmitted, the application needs to attempt delivery of the remaining data. For further information on this topic, consult the Socket Programming HOWTO. Changed in version 3.5: If the system call is interrupted and the signal handler does not raise an exception, the method now retries the system call instead of raising an InterruptedError exception (see PEP 475 for the rationale).

# Problem

# sending http headers to `client'

Figure 4: DS-1000 (left) and ODEX (right) examples with annotated canonical library documentation.

ficulty. We compare RACG performance when providing top-1, 2, 5, and 10 documents.

As shown by Figure 6, adding five documents yields the best results in most settings, except for StarCoder2 on RepoEval which best uses 8 documents. Despite the drastic variance in length limits of StarCoder2  $(16k)$  and DeepseekCoder  $(4k)$ , the sweet spot is consistently 5 documents. While adding a few documents brings helpful contexts, adding more low-ranked documents may introduce noise and deteriorate generation due to the imperfections of retrieval systems (Wang et al., 2023a).

# F Does RACG Help Stronger Models?

We have shown that RACG with open retrieval improves a relatively weaker model, StarCoder2 (§4). To see if this improvement of RACG with open retrieval generalizes to stronger models, we experiment with a series of top-performing proprietary models: GPT-4o, Claude-3-haiku and sonnet, and Gemini-1.5-flash and pro.

Basic programming: HumanEval RACG can consistently improve the performance of GPT-4 and Claude-3-sonnet when leveraging all sources of documents. However, for weaker models such as Claude-3-haiku and Gemini-1.5-flash, RACG only helps when aggregating multiple sources yet

falls short when grounding on one source (even the canonical solution source). Interestingly, the stronger Claude-3-sonnet performs worse than the weaker Claude-3-haiku, but can benefit from all retrieval sources and outperform haiku with documents from the canonical programming source, suggesting its potentially better RAG ability. While the stronger Claude effectively benefits from extra contexts, the stronger Gemini-1.5-pro behaves similarly to its weaker counterpart and cannot do RACG effectively with non-canonical sources.

Open domain: ODEX All models experience limited improvements by leveraging library documentation to complex the ODEX task, with the only exception that GPT-4o improves 4.6 points by incorporating programming solutions into the context.

As results degrade in most cases, we conduct a manual analysis to examine when most models fail. We find that most models tend to copy functions in the context, sometimes even overwriting the function being queried, thus failing all the test cases specific to the queried function. Further, possibly affected by the plethora of programs in context, models tend to generate over-complicated programs which, however, do not often pass the test cases.

Question:

Split string 'Words, words, words.' using a regex ' $\backslash \mathsf{W} + \}$

Solution

re.split("\\W+)", 'Words, words, words.')

Context (helpful)

Write a function to put spaces between words starting with capital letters in a given string by using regex. import re   
def capital_wordspaces(str1): return re.sub(r"\w([A-Z]",r"\1\2",str1)

Model Prediction

import re   
def f_2195340(): 'split string Words, words, words: using a regex   
 $\mathrm{W + }^{\mathrm{m m e n}}$  text  $=$  Words, words, words. return re.split(r' $\mathrm{W + }$ '),text)

Question

Find intersection data between series `s1` and series `s2`

Solution

pd.Series(list(set(s1).intersection(set(s2))))

Context (distracting)

skimage segmentation.join Segmentations(s1, s2) [source]

Return the join of the two input segmentations. The join J of S1 and S2 is defined as the segmentation in which two voxels are in the same segment if and only if they are in the same segment in both S1 and S2. Parameters class Intersection(expr1, expr2, **extra) set_extremes(e1, e2)[source]

Model Prediction

fromskimage segmentation import join segmentations   
deff_18079563(s1,s2): "Find intersection data between series's1' and series s2. return join segmentations(s1,s2)

Figure 6: Comparing RACG performance with various numbers of documents.

Figure 5: RACG helps with relevant contexts (left) and hurts with distracting contexts (right).


In general, most models can be easily distracted or disturbed by additional contexts (Wang et al., 2023a), and fail to conduct the designated code generation task, indicating much room for improvement for RACG.

Repository level: RepoEval While GPT-4o can solve the RepoEval task with a reasonable success rate, all Claude models are challenged by the task and achieve less than  $10\%$  pass@1 for most scenarios. We find Claude models mostly respond with explanations of the incomplete input code, instead of the to-be-completed code even with proper instructions, possibly caused by some properties of the unknown training data. Gemini-1.5-flash also barely solves the task and often generates textual explanations; however its stronger pro variant gets about 10–25 point improvements, demonstrating its stronger repository-level code completion abilities.

<table><tr><td>Method</td><td>Baseline</td><td>Program</td><td>Tutorial</td><td>Docs</td><td>SO</td><td>GitHub</td><td>All</td></tr><tr><td>GPT-4o</td><td>75.6</td><td>94.5</td><td>90.2</td><td>90.9</td><td>91.5</td><td>84.8</td><td>95.1</td></tr><tr><td>Claude-3-haiku</td><td>74.4</td><td>77.4</td><td>77.4</td><td>71.3</td><td>67.7</td><td>73.2</td><td>82.9</td></tr><tr><td>Claude-3-sonnet</td><td>65.9</td><td>78.7</td><td>66.5</td><td>68.9</td><td>70.7</td><td>73.8</td><td>80.5</td></tr><tr><td>Gemini-1.5-flash</td><td>72.0</td><td>91.5</td><td>75.0</td><td>70.1</td><td>68.9</td><td>68.9</td><td>95.1</td></tr><tr><td>Gemini-1.5-pro</td><td>82.9</td><td>95.7</td><td>79.9</td><td>77.4</td><td>79.9</td><td>80.5</td><td>86.6</td></tr></table>

Table 11: RACG on HumanEval with strong code LMs.  

<table><tr><td>Method</td><td>Baseline</td><td>Program</td><td>Tutorial</td><td>Docs</td><td>SO</td><td>GitHub</td><td>All</td></tr><tr><td>GPT-4o</td><td>44.6</td><td>49.2</td><td>44.2</td><td>47.6</td><td>40.3</td><td>39.4</td><td>39.6</td></tr><tr><td>Claude-3-haiku</td><td>48.5</td><td>42.6</td><td>39.2</td><td>44.6</td><td>33.7</td><td>40.5</td><td>35.1</td></tr><tr><td>Claude-3-sonnet</td><td>41.0</td><td>37.6</td><td>35.3</td><td>38.0</td><td>34.2</td><td>42.4</td><td>38.0</td></tr><tr><td>Gemini-1.5-flash</td><td>50.6</td><td>48.3</td><td>46.7</td><td>46.2</td><td>41.9</td><td>44.9</td><td>43.1</td></tr><tr><td>Gemini-1.5-pro</td><td>57.2</td><td>54.4</td><td>45.6</td><td>51.0</td><td>46.5</td><td>39.6</td><td>46.0</td></tr></table>

Table 12: RACG on ODEX with strong code LMs.  

<table><tr><td>Method</td><td>Baseline</td><td>Local</td><td>Program</td><td>Tutorial</td><td>Docs</td><td>SO</td><td>GitHub</td><td>All</td><td>L+E</td></tr><tr><td>GPT-4o</td><td>32.4</td><td>62.2</td><td>35.4</td><td>28.7</td><td>27.8</td><td>29.0</td><td>28.2</td><td>30.3</td><td>54.2</td></tr><tr><td>Claude-3-haiku</td><td>9.1</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.2</td><td>0.2</td><td>0.5</td></tr><tr><td>Claude-3-sonnet</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Gemini-1.5-flash</td><td>1.3</td><td>16.9</td><td>4.0</td><td>2.1</td><td>3.2</td><td>2.1</td><td>3.2</td><td>2.7</td><td>11.8</td></tr><tr><td>Gemini-1.5-pro</td><td>10.5</td><td>39.1</td><td>15.1</td><td>13.4</td><td>15.8</td><td>15.3</td><td>11.8</td><td>12.3</td><td>33.0</td></tr></table>

Table 13: RACG on RepoEval with strong code LMs.

# Footnotes:

Page 0: *Equal contribution. 
Page 1: <sup>1</sup>In this work we focus on Python-related tasks because it is the most widely-used programming language for benchmarking code generation. We leave extensions to other programming languages for future work. 
Page 2: 2Two other splits (API and line) are evaluated by lexical measures that have been shown as ineffective in signifying functional correctness (Chen et al., 2021; Wang et al., 2023b). <sup>3</sup>https://www.swebench.com/lite.htm1 4https://geeksforgeeks.org; https://www. w3schools.com/; https://www.tutorialspoint.com/; https://towardsdatascience.com 
Page 3: <sup>5</sup>https://sbert.net/ <sup>6</sup>We found that without these approaches, performance even with state-of-the-art GPT4o remains around 1-2%. 
Page 4: Due to the costs, we randomly sample 10k queries and 100k from CodeSearchNet Python split. For API models, we use a batch size of 64 for encoding. 
Page 6: <sup>8</sup>We use the first 500 tokens of each document for all experiments in this section, which we show to be optimal in ablation studies (§4), and satisfies all model context limits. 
Page 7: <sup>9</sup>We do not chunk programming solutions since they are typically short (average  $< 200$  tokens as in Table 2). 
Page 12: 10 https://ai.meta.com/blog/meta-llama-3/ 
