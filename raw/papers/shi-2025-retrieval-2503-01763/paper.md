\scalerel*[Uncaptioned image] Retrieval Models Aren’t Tool-Savvy:  Benchmarking Tool Retrieval for Large Language Models
===========================================================================================================================

Zhengliang Shi1Yuhan Wang1Lingyong Yan2  
Pengjie Ren1Shuaiqiang Wang2Dawei Yin2Zhaochun Ren$3$  
1Shandong University, Qingdao, China2Baidu Inc., Beijing, China  
3Leiden University, Leiden, The Netherlands  
\faGithub [Tool-Retrieval-Benchmark](https://github.com/mangopy/tool-retrieval-benchmark "")  
shizhl@mail.sdu.edu.cn z.ren@liacs.leidenuniv.nlCorresponding author.

###### Abstract

Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks.
Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step.
However, the performance of IR models in tool retrieval tasks remains underexplored and unclear.
Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios.
In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets.
We benchmark six types of models on ToolRet.
Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet.
This low retrieval quality degrades the task pass rate of tool-use LLMs.
As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models.111Resource is available on <img src='extracted/6480629/figure/hf.png' alt='[Uncaptioned image]' title='' width='12' height='12' /> [Huggingface](https://huggingface.co/spaces/mangopy/ToolRet-demo "") and \faGithub [Website](https://mangopy.github.io/tool-retrieval-benchmark/ "").

\scalerel

*<img src='x2.png' alt='[Uncaptioned image]' title='' width='355' height='355' />Retrieval Models Aren’t Tool-Savvy:  
Benchmarking Tool Retrieval for Large Language Models

  
Zhengliang Shi1Yuhan Wang1Lingyong Yan2Pengjie Ren1Shuaiqiang Wang2Dawei Yin2Zhaochun Ren$3$††thanks: Corresponding author.1Shandong University, Qingdao, China2Baidu Inc., Beijing, China3Leiden University, Leiden, The Netherlands\faGithub [Tool-Retrieval-Benchmark](https://github.com/mangopy/tool-retrieval-benchmark "")shizhl@mail.sdu.edu.cn z.ren@liacs.leidenuniv.nl

1 Introduction
--------------

Large language models (LLMs) have demonstrated remarkable progress across various natural language processing (NLP) tasks, such as text summarization*chang2023survey*.
However, they suffer from inherent inabilities to interact with the physical world and access vast, up-to-date knowledge*qin2024tool*.
To alleviate these drawbacks, tool learning is proposed to equip LLMs with external tools, augmenting them as agents to manipulate tools for practical task-solving*qu2025tool; wang2024tools*.

<img src='x3.png' alt='Refer to caption' title='' width='789' height='384' />

*Figure 1: Correlation between the tool retrieval performance (e.g., Recall@10) of IR models and the end-to-end task pass rate of tool-use agents.*

In practical applications, retrieving useful tools from toolsets for LLM agents typically serves as the initial step*wang2024toolgen; xu2024enhancing; song2023restgpt*.
This step becomes particularly critical in real-world scenarios where the candidate tools are usually large-scale and many of them are similar in functionality*qu2024colt*.
However, most existing work*guo2024stabletoolbench; qian2023toolink* simplifies this retrieval process by manually pre-selecting a small set of 10-20 relevant tools for each evaluation task.
For example, the ToolACE*toolace* and ToolBench*qin2023toolllm* annotate about 10 tools per task.
While recent information retrieval (IR) techniques such as semantic matching*qu2024colt; xu2024enhancing*, can assist with tool retrieval, they are often trained on ad-hoc tool-use datasets, lacking comprehensive evaluation on diverse scenarios, especially for unseen tasks.
To further explore the importance of tool retrieval, we conduct a pilot experiment on ToolBench*qin2023toolllm*.
As shown in Figure[1], we observe that (i) the agent’s performance substantially drops when replacing the officially annotated toolset with the retrieved tools; and
(ii) even strong retrievers like colbertv2*colbertv2*, struggle to retrieve target tools effectively.
These findings highlight the necessity to (i) systematically evaluate IR models on diverse tool retrieval tasks; and (ii) analyze the impact of retrieval on the end-to-end task pass rate.

In this work, we introduce ToolRet, the first large-scale tool retrieval benchmark comprising 7.6k diverse retrieval tasks and a corpus of 43k tools, which comprehensively evaluates IR models across diverse retrieval scenarios.
Specifically, we collect query-tool datasets from the following sources: (i) Tool-use agent benchmarks from published research papers in AI conferences, such as ACL and NeurIPS; (ii) Related conference resources such as AppBench in EMNLP and ToolLens in CIKM; and (iii) Other publicly available datasets from the open-source community, e.g., HuggingFace.
The collected data is carefully curated to cover a wide range of practical tool requirements, comprising diverse types of tool documentation, domains, and varying query lengths.
Then, we standardize the format of all the collected tasks, aligning them with retrieval tasks similar to the format in MTEB, where each retrieval task contains a query and target tools (e.g., labels).
To support the instructional retrieval*weller2024followir* setting of our benchmark, we also introduce a target-aware strategy to supplement each query with an instruction using the powerful LLMs (i.e., gpt-4o).

We systematically evaluate five types of IR models such as embedding models and LLM re-ranking, under various experimental settings.
Our results reveal that even the best model (i.e., NV-embedd-v1) that demonstrates strong performance in conventional IR benchmarks, achieves an nDCG@10 of only 33.83 in our benchmark.
This highlights the challenges of the tool retrieval tasks.
We identify two key factors contributing to this performance gap:
(i) Lower term overlap between queries and target tools in tool retrieval tasks, which demands higher representation abilities for IR models to accurately match query intent with the correct tools; and
(ii) Task shift from conventional information-seeking tasks (e.g., document retrieval) to tool retrieval, leading to suboptimal performance of IR models that are not explicitly optimized.

To enhance the retrieval performance and enable IR models to augment tool-use agents, we further propose the ToolRet-train, a large-scale training dataset containing more than 200k retrieval tasks.
We extend our data collection process from ToolRet to include the training set of three mainstream tool-use datasets, including ToolACE*toolace*, APIGen*liu2024apigen* and ToolBench*qin2023toolllm*.
To enable the training, we pair each retrieval task with 10 negative tools retrieved by the NV-embed-v1.
Finally, each training example contains the query, an generated instruction, the target tools, and the negative tools.
Results show that the IR models trained over ToolRet-train, exhibit significant improvements in the retrieval process, leading to a higher end-to-end task pass rate when integrated with tool-use LLMs.

Our contributions are summarized as follows:
(i) We introduce ToolRet, the first evaluation benchmark for tool retrieval tasks.
(ii) We evaluate the tool retrieval performance of various IR models and analyze the impact of retrieval on the end-to-end task pass rate of tool-use LLMs; and
(iii) We contribute to a large-scale training dataset that enhances the performance of IR models, improving their ability to augment tool-use LLMs effectively.

2 Related work
--------------

Tool learning with foundation models. Tool learning aims to equip LLMs with tools, such as web API*song2023restgpt* and python packages*codeact*, expanding their utility*qin2023toolllm*.
Existing work teaching LLMs to use tools can be broadly classified into tuning-free*chameleon* and tuning-based methods*gao2024confucius*.
The former prepends the description of candidate tools in the LLMs’ context, prompting them to select and invoke tools*huang2023metatool*.
The latter enables LLMs to learn the usage of each tool through training on synthetic data*toolace; gao2024confucius*.
However, both two paradigms struggle when facing the large-scale toolset in practice*qu2024towards; liu2024apigen*.
First, real-world toolsets are typically massive, making it less possible to incorporate all tools within the limited context of LLMs.
For example, the RapidAPI platform contains more than 52k tools while the PyPI222<https://pypi.org/> hosts over 600k frequently updated packages.
Second, since tools are frequently updated, it is cost-intensive to re-train the LLMs to memorize all tools*quexplore*.
Although recent studies address this challenge using semantic retrievers*qin2023toolllm; wang2024toolgen*, these solutions are typically ad-hoc and lack systematic evaluation across diverse tool retrieval scenarios.
To fill this gap, we present the first comprehensive tool retrieval benchmark with systematic analysis.

Information retrieval benchmark. Conventional information retrieval (IR) benchmarks are typically designed for information-seeking tasks, such as Nature Question*kwiatkowski-etal-2019-natural* for question answering and MS-MARCO*bajaj2016ms* for passage re-ranking.
Recent work also explores the IR technique in various downstream tasks, such as table retrieval*chen2024tablerag; zhang2020web* and scientific retrieval*ajith2024litsearch*, which substantially augments the downstream models.
However, tool retrieval, a crucial step for tool-use agents, remains underexplored.
Compared with traditional IR tasks, retrieving useful tools is more challenging since solving a task typically requires the combination of multiple tools*qu2024towards*.
Most existing benchmarks simplify this retrieval process by manually annotating a small set of tools that fit the LLMs’ context, which is far from reality with a large toolset.
In this work, we evaluate IR models on diverse tool retrieval tasks and contribute over 200k training data to facilitate future research.

3 Benchmark construction
------------------------

### 3.1 Data collection

To build a comprehensive benchmark for tool retrieval evaluation, we collect data from the following well-known sources:
(i) Tool-use LLM benchmarks: A wide range of benchmarks published in leading AI conferences such as ACL and NeurIPS;
(ii) Conference Resources: Datasets from resource tracks in IR and NLP conferences (e.g., CIKM and EMNLP); and
(iii) Other high-quality dataset: We identify related datasets released on open-source platforms like HuggingFace and their technique reports can be found in public submissions like arXiv. We include them to enrich ToolRet.

Given the rapid development of benchmarks from these sources, we collect datasets released between the August 2023 to December 2024 in this version.333Our team will maintain and update the benchmark. We download these data from official channels based on their usage requirements and totally collect more than 30 datasets.
Since the data sources are diverse and their original formats vary substantially, we perform necessary data cleaning operations such as deduplication and text normalization to ensure consistency and quality.

We observe that most of the collected datasets are originally designed to evaluate the tool-use capability of LLMs, where the LLM is required to correctly call a sequence of target tools given an input query.
To facilitate retrieval evaluation in ToolRet, we align the format of all collected tasks with the well-established IR benchmark like BEIR and MTEB.
Specifically, each task consists of a query as input and target tools as label (a.k.a, ground truth), where a tool is identified by a unique identifier and paired with detailed documentation to describe its functionality.
Endpoints of the collected datasets and concrete examples of our formatted dataset are provided in AppendixLABEL:sec:app:benchmark.

### 3.2 Data sampling

After collecting the datasets, we observe data size imbalances across different datasets.
Besides, some datasets are extremely large with substantial redundant content, making comprehensive model evaluation both inefficient and unnecessary.
Therefore, we streamline them through effective data sampling while maintaining its evaluation integrity.

Task sampling. For each collected dataset, we encode the tasks using the embedding model, i.e., NV-embedd-v1, and apply the K-means clustering algorithm on the text embeddings.
We set the number of clusters to the size of the corresponding toolset and randomly sample one task from each cluster.
If the toolset size exceeds the number of queries, we retain all queries.
For example, the original ToolEyes*ye2024tooleyes* dataset contains 500 queries and 95 tools; Thus, we set the cluster number as min(500, 95) \= 95 for clustering.

Toolset sampling. To eliminate redundancy, we manually review the documentation of each raw dataset to identify and merge identical toolsets.
For example, since the COLT*qu2024colt* toolset overlaps with the Toolbench*qin2023toolllm* , we merge their intersecting tools.
Ultimately, we merge all toolsets from the 34 datasets to form the final corpus, resulting in a total of 43k tools.
Each tool is assigned a unique identifier.

After sampling, we obtain 7.6k retrieval tasks and a corpus of 43k tools.

### 3.3 Instruction construction

Instructional information retrieval*Sun2024MAIR; weller2024followir* is an active research area, where an additional instruction is paired with the input query to guide IR models in retrieving target information.
This instruction-following capability is especially critical in tool retrieval, as IR models are often used to augment LLM agents and receive additional context from the agents beyond the input query.
To support this instructional IR scenario, we construct the instructions as part of ToolRet.

Considering manually writing instructions is cost-intensive and challenging to scale, we introduce a target-aware strategy using powerful LLMs to automate this process.
Specifically, we first invite three human experts with strong NLP and IR backgrounds to manually craft 100 seed instructions.
In line with the well-defined format from*asai2022task*, our instruction outlines the relevance criteria by bridging the query intent and the functionality of the target tools.
For example, for the transcribing the audio to text task, the instruction is presented as “retrieve tools that process audio inputs to produce accurate textual transcriptions aligned with the user requirements”.
Next, we employ a powerful LLM, i.e., GPT-4o, as an automatic instruction generator and guide it to generate instruction for each task through in-context learning.
To enhance the diversity, we randomly sample in-context examples from the pool of both the generated and handcrafted instructions.
A detailed pseudo algorithm is provided in AppendixLABEL:sec:app:benchmark.

After the above three processes, we obtain ToolRet, which consists of 7.6k tasks, each paired with an instruction, and a corpus of 43k diverse tools, providing a comprehensive testbed and supporting various evaluation settings.

4 Benchmark statistic
---------------------

Statistic# size of retrieval task7,615 - # of web API retrieval task4,916 - # of code function retrieval task950 - # of customized app retrieval task1,749# size of tool43,215 - # of web API36,978 - # of code function3,794 - # of customized app2,443avg. query / instruction length (tokens)46.87 / 43.43avg. tool documentation length (token)174.56

*Table 1: Basic statistics of our benchmark ToolRet.*
