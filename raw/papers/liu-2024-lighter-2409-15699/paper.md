\@ACM@balancefalse\@ACM@balancefalse

Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation
==========================================================================================

Chenyuan WuUniversity of Science and Technology of China [wu.chenyuan@ustc.edu.cn](mailto:wu.chenyuan@ustc.edu.cn),Ninglu ShaoGaoling School of AI, Renmin University of China [shao.ninglu@ruc.edu.cn](mailto:shao.ninglu@ruc.edu.cn),Zheng LiuBeijing Academy of Artificial Intelligence [zhengliu1026@gmail.com](mailto:zhengliu1026@gmail.com),Shitao XiaoBeijing Academy of Artificial Intelligence [stxiao@baai.ac.cn](mailto:stxiao@baai.ac.cn),Chaozhuo LiBeijing University of Posts and
Telecommunications [cli@bupt.edu.cn](mailto:cli@bupt.edu.cn)andDefu LianUniversity of Science and Technology of China [lian.defu@ustc.edu.cn](mailto:lian.defu@ustc.edu.cn)

###### Abstract.

The existing Retrieval-Augmented Generation (RAG) systems face significant challenges in terms of cost and effectiveness. On one hand, they need to encode the lengthy retrieved contexts before responding to the input tasks, which imposes substantial computational overhead. On the other hand, directly using generic Large Language Models (LLMs) often leads to sub-optimal answers, while task-specific fine-tuning may compromise the LLMs’ general capabilities.
To address these challenges, we introduce a novel approach called FlexRAG (Flexible Context Adaptation for RAG). In this approach, the
retrieved contexts are compressed into compact embeddings before being encoded by the LLMs. Simultaneously, these compressed embeddings are optimized to enhance downstream RAG performance.
A key feature of FlexRAG is its flexibility, which enables effective support for diverse compression ratios and selective preservation of important contexts.
Thanks to these technical designs, FlexRAG achieves superior generation quality while significantly reducing running costs. Comprehensive experiments on various question-answering datasets validate our approach as a cost-effective and flexible solution for RAG systems.

Retrieval Augmented Generation, Large Language Models, Question Answering, Context Compression and Optimization

1. Introduction
----------------

Large language models (LLMs) are growing as a general foundation of artificial intelligence. However, the existing LLMs are still limited by incomplete and outdated knowledge due to their static nature, and this limitation is particularly pronounced when dealing with knowledge-intensive tasks *(Sun et al., [2024]; Liu et al., [2023]; Ji et al., [2023])*. To mitigate this limitation, people resort to retrieval-augmented generation (RAG). With proper information retrieved from external databases, the generation process can be conducted on top of knowledge-grounded contexts. Consequently, it substantially contributes to LLMs’ generation quality in terms of truthfulness and credibility *(Gao et al., [2023])*. In recent years, influential prototyping systems, e.g., WebGPT, SearchGPT *(OpenAI, [2024])*, Perplexity *(Perplexity, [2024])*, and developing frameworks, such as Llama-Index, Lainchain, are continuously proposed by the community, facilitating both application and research in this area.

### 1.1. The Challenges

Despite the widespread popularity, existing RAG systems still face significant challenges, particularly in terms of running costs and effectiveness. Firstly, RAG systems often require processing lengthy contexts to handle knowledge-intensive tasks. For instance, solving multi-hop QA tasks may involve working with a series of correlated documents *(Trivedi et al., [2022]; Ho et al., [2020]; Yang et al., [2018])*, while general language modeling tasks may call for iterative retrieval of diverse knowledge sources *(Jiang et al., [2023c]; Asai et al., [2023])*. In such situations, it will take substantial computation costs in order to encode the lengthy contexts for LLMs. Secondly, the RAG systems can be limited by their answer quality if generic LLMs are directly utilized. This limitation is especially evident for many public models of moderate scale, which often struggle to effectively utilize the retrieved knowledge, particularly in complex and noisy contexts *(Gao et al., [2023])*. While task-specific fine-tuning can improve the answer quality *(Zhang et al., [2024])*, it may come at the cost of reduced instruction-following capabilities and diminished performance on other general tasks *(Luo et al., [2023])*.

<img src='x1.png' alt='Refer to caption' title='' width='822' height='207' />

*Figure 1. Comparison of related techniques. 1) Context compression: token embeddings are compressed into compact summary vectors. 2) Context filtering: important token embeddings are filtered from the input prompt. 3) Prompt tuning: soft-prompt is learned to improve the downstream task. 4) FlexRAG: unifying all functions in one framework, with compressive embeddings (summary vectors) down-sampled by importance (filtering) and learned to optimize the RAG performance (prompt tuning).*

### 1.2. Our Approach

To address the above challenges, we propose a novel approach in this paper, called FlexRAG (shown in Figure [1]. 4). It transforms the retrieved contexts into compact and more usable forms, which substantially improves the cost-effectiveness of RAG systems.

First of all, FlexRAG helps to reduce the running cost of RAG. Essentially, FlexRAG pre-encodes external documents into compressive embeddings during the offline stage and performs down-sampling of the compressive embeddings when corresponding documents are retrieved for specific RAG tasks. Since the down-sampled token embeddings are significantly shorter compared to directly tokenized documents, this approach substantially reduces the computation cost for RAG systems. A key characteristic of FlexRAG is its flexibility. Previous methods typically perform static compression of input context based on predefined compression ratios *(Chevalier et al., [2023]; Mu et al., [2024]; Ge et al., [2023])*. In contrast, our approach supports arbitrary compression ratios specified by the user, and enables selective compression of the contexts based on their importance in specific scenarios. This means that the critical parts of the context are preserved as much as possible, while the less important parts are assigned with large compression ratios. Consequently, useful information within the context can be better presented for downstream RAG tasks.

Secondly, FlexRAG optimizes the performance of RAG in a compatible way. In our work, a two-stage training workflow is designed for FlexRAG. In the first stage, we employ task-generic pre-training using a generic corpus, like RedPajama *(Computer, [2023])*, which establishes the preliminary alignment between the compression module and downstream LLM. In the second stage, we perform task-specific fine-tuning using various instruction-tuning datasets, which optimizes the answer quality for downstream RAG tasks. Throughout the entire training process, the compression module remains learnable while the LLM parameters are kept fixed. As no modification is made to the LLM’s original parameters, we can optimize the performance in RAG without compromising the performance in other general tasks.

We perform comprehensive empirical studies using a variety of question-answering datasets. In our experiments, FlexRAG exhibits three key advantages. 1) Superior cost-effectiveness, where substantial improvements can be achieved over generic LLMs and other context compression methods with significantly reduced costs. 2) Flexibility of usage, as it effectively supports various compression ratios and compression methods. 3) General usability, as the competitive performance can be well-preserved across various datasets and working conditions. These results validate FlexRAG as an effective and economical component for RAG systems.

To summarize, the following technical contributions are highlighted for our paper:

* •

    We propose a novel method, FlexRAG, for compressive and optimized adaptation of the retrieved contexts for RAG.

* •

    FlexRAG realizes flexible compression of the retrieved contexts by various ratios, and enables selective preservation of useful information leveraging estimated importance.

* •

    We design a two-stage training workflow for FlexRAG. By making sufficient utilization of available data, it effectively enhances the downstream performance of RAG.

* •

    We perform comprehensive experiments with a variety QA datasets, whose result verifies the cost-effectiveness, flexibility, and general usability of FlexRAG.

2. Related Works
-----------------

In this section, the related works are discussed from three perspectives: 1) retrieval-augmented generation, 2) context compression, 3) fine-tuning for RAG optimization.

### 2.1. Retrieval-augmented Generation

Retrieval-Augmented Generation has emerged as a crucial paradigm for language models*(Lewis et al., [2020])*, particularly with the rise of LLMs. A typical RAG system consists of two components: retrieval tools that access external databases and a language model that generates knowledge-grounded content based on the retrieval results. By introducing relevant knowledge, RAG significantly enhances the truthfulness and credibility of LLM-generated outputs, making it a valuable approach for mitigating hallucinations *(Ji et al., [2023]; Liu et al., [2023])*. Additionally, RAG offloads internal knowledge to external memory, contributing to improved cost-effectiveness of LLMs *(Borgeaud et al., [2022]; Izacard et al., [2023])*.

In recent years, RAG has become a significant research focus in both academia and industry. Researchers have continuously proposed advanced architectures beyond the basic direct prompting approach, such as fusion-in-decoder *(Izacard and Grave, [2020])* and internal knowledge injection *(Borgeaud et al., [2022])*, which facilitate the effective use of the retrieved knowledge. Meanwhile, the training of retriever and generator has been improved from simple independent training to more advanced forms of joint training *(Izacard et al., [2023]; Shi et al., [2023]; Zhang et al., [2023])*, offering users the flexibility to select the most suitable method for their specific applications. Moreover, designing appropriate mechanisms for RAG, such as determining when and where to apply retrieval augmentation and which information to retrieve, is crucial. Significant progress has been made in this field, with approaches like uncertainty-based methods (e.g., FLARE *(Jiang et al., [2023c])*), self-prompting methods (e.g., ToolFormer *(Schick et al., [2024])*, Self-RAG *(Asai et al., [2023])*), and reflection-based methods (e.g., ReAct *(Yao et al., [2022])*, Reflexion *(Shinn et al., [2024])*) proposed for more effective control of RAG in complex real-world scenarios. Additionally, RAG’s application has been actively explored beyond traditional question-answering *(Lewis et al., [2020])*, such as long-context modeling *(Xu et al., [2023a])*, in-context learning *(Zhang et al., [2023]; Wang et al., [2023])*, code generation *(Wang et al., [2024])*, and multi-modal processing *(Yasunaga et al., [2022])*, etc.

### 2.2. Context Compression

Running costs pose a significant challenge for RAG systems, primarily due to the need to encode lengthy retrieved contexts. To address this, an important strategy involves compressing the retrieved contexts before they are processed by downstream LLMs. In line with this approach, various compression techniques have been developed in recent years.
One notable work is Gist*(Mu et al., [2024])*, which implicitly compresses long contexts using a small number of Gist tokens. Similarly, ICAE*(Ge et al., [2023])* fine-tunes LLMs as specialized context compressors through LoRA, while AutoCompressor*(Chevalier et al., [2023])* integrates compression learning with the autoregressive language modeling process.
In addition to these implicit methods, another line of research focuses on explicit filtering of contexts, where less important tokens are removed directly. A representative study in this area is LLMLingua *(Jiang et al., [2023a], [b])*, which uses coarse-to-fine approaches to compress contexts based on given budgets. Besides, RECOMP *(Xu et al., [2023b])* presents both extractive and abstractive compressor, which selects useful sentences and generates summaries from long-contexts, respectively. While explicit methods are generally agnostic to downstream LLMs and therefore more practical, they may suffer from higher compression loss due to the over-removal of input tokens.

### 2.3. RAG Fine-tuning

While RAG systems can be constructed using off-the-shelf retrievers and LLMs, this native approach often results in sub-optimal performance. Issues may include insufficient utilization of retrieved knowledge, vulnerability to retrieval noise, and misalignment with the required answer format or human preferences *(Gao et al., [2023])*. Consequently, continual fine-tuning is often necessary to enhance RAG performance. In the simplest scenarios, retrievers and LLMs can be independently fine-tuned using their respective training data *(Zhang et al., [2024])*. However, more advanced approaches involve joint training of retrievers and LLMs. In these cases, retrievers are optimized to select contexts that are more conducive to the LLM’s processing, while LLMs are trained to adapt to the specific contexts provided by the retrievers *(Izacard et al., [2023]; Shi et al., [2023])*.
Nevertheless, fine-tuning RAG systems isn’t without its drawbacks. It can lead to a reduction in the general capacity of LLMs, a phenomenon known as catastrophic forgetting. To mitigate this issue, parameter-efficient fine-tuning (PEFT) is often employed, where only specialized learnable adapters are fine-tuned, leaving the LLM’s original parameters intact *(Ding et al., [2023])*. Among various PEFT methods, prompt-tuning is particularly effective in minimizing the impact on LLMs *(Lester et al., [2021]; Liu et al., [2021])*. In this approach, adaptation modules, i.e. soft prompts, are introduced as external components. Following this spirit, our encoder is designed, which produces both compressive and RAG-optimized embeddings for downstream LLMs.

<img src='x2.png' alt='Refer to caption' title='' width='788' height='307' />

*Figure 2. Architecture of FlexRAG. It transforms the retrieved contexts $\mathbf{X}_{retr}$ into compressive embeddings $\mathbf{E}_{retr}$ using the compressive encoder. With estimated importance, it down-samples $\mathbf{E}_{retr}$ into $\mathbf{E}^{\prime}_{retr}$ as the compressed context for RAG.*

3. Methodology
---------------

In this section, we delve into the technical aspects of FlexRAG. We begin by formulating the problem of context adaptation for RAG. Following this, we introduce the architecture of FlexRAG, focusing on its two basic components: the compressive context adapter and the selective compression mechanism. Finally, we introduce the optimization of FlexRAG using various types of data.

### 3.1. Problem Formulation

Retrieval-Augmented Generation (RAG) is a specialized working paradigm for Large Language Models (LLMs), which is designed to facilitate knowledge-intensive tasks, such as question-answering and knowledge-grounded dialogue systems. It presents the task’s prompt (denoted as $\mathrm{X}_{task}$) together with the retrieved context (denoted as $\mathrm{X}_{retr}$) as the input for LLM, based on which the ground-truth answer (denoted as $\mathrm{X}_{ans}$) is predicted. The retrieved context is expected to include necessary knowledge to the task, such that the ground-truth answer can be better predicted. In spite of widespread popularity, the directly application of LLMs for RAG is constrained by suboptimal performance and high computational costs. To address these challenges, we propose Flexible Context Adaptation for RAG with the following objectives: 1) the retrieved context is compressed into a concise form: $|\phi(\mathrm{X}_{retr})|<|\mathrm{X}_{retr}|$ ($\phi(\cdot)$ stands for the compressor); 2) the compressed context helps to deliver optimized performance for RAG. These objectives can be generalized and formulated as the following optimization problem:

| (1) |  | $\begin{split}\&\max\limits_{\phi}~{}P_{LLM}(\mathrm{X}_{ans}|\phi(\mathrm{X}_{% retr}),\mathrm{X}_{task})\\ \&~{}~{}s.t.~{}|\phi(\mathrm{X}_{retr})|\=k,~{}~{}\text{where }k<|X_{retr}|\end{split}$ |  |
| --- | --- | --- | --- |

In other words, we aim to realize the optimal compression, where the compressed context of the predefined size $k$ can maximize the generation likelihood of the ground-truth answer. Additionally, the compressor is expected to achieve flexible compression, allowing the retrieved context to be compressed to any length within the original context length $|X_{retr}|$.

### 3.2. Architecture

The workflow of FlexRAG consists of the following steps (Figure [2]). First, the retrieved context is tokenized and jointly encoded as a sequence of embeddings, denoted as $\mathbf{E}_{retr}$, using a specialized context encoder $\psi(\cdot)$: $\mathbf{E}_{retr}\leftarrow\psi(X_{retr})$. Next, the well-encoded embeddings $\mathbf{E}_{retr}$ are down-scaled by a sampling function $\gamma(\cdot)$:

| (2) |  | $\mathbf{E}^{\prime}_{retr}\leftarrow\gamma(\mathbf{E}_{retr},k),~{}~{}~{}~{}% \text{where}~{}~{}~{}~{}|\mathbf{E}^{\prime}_{retr}|\=k,$ |  |
| --- | --- | --- | --- |

In this place, $k$ indicates the predefined size of compressed context. The down-scaled embeddings $\mathbf{E}^{\prime}_{retr}$ serve as compact yet informative representations of the retrieved context, which are passed to the downstream LLM for retrieval-augmented generation.

The above workflow consists of two basic modules. The first is the compressive encoder, which implements the encoding function $\psi(\cdot)$ to transform the retrieved context into informative and flexible-to-sample embeddings. The second is the importance estimator, which assesses the importance of each part of the context. Based on the estimation results, selective compression is conducted through down-sampling, i.e. $\gamma(\cdot)$, allowing for the preservation of the most critical information with higher emphasis.

### 3.3. Compressive Encoder

As described, the compressive encoder transforms the tokenized retrieved contexts $\mathbf{X}_{retr}$ into compressive embeddings $\mathbf{E}_{retr}$, which can be flexibly down-sampled to provide a high-quality compression of the original input. As a result, the realization of compressive encoder is guided by the following considerations.

First, to serve as compressive representations of input, each element of $\mathbf{E}_{retr}$ needs to fully capture the information for its nearby context. This requires a highly expressive encoding backbone to generate rich-semantic embeddings. To achieve this goal, we exploy LLMs as the foundation of our compressive encoder.

Second, it needs to ensure a seamless connection between the compressive embeddings and the downstream LLM. To facilitate this objective, the compressive embeddings must resemble the input token embeddings of the downstream LLM. With this consideration, we employ the same backbone as the downstream LLM. Besides, we choose to leverage the first-$n$ layers instead of the entire LLM, considering that the intermediate embeddings from the mid-layers are more similar to the LLM’s token embeddings.

Third, the length of retrieved context is likely to exceed the window size of compressive encoder, making it inevitable to chunk the input and encode each segment individually. However, it has to avoid over-chunking so as to maintain the coherence of input. Therefore, the chunking size is expanded to the maximum extent in each specific scenario.

Therefore, the compressive encoding can be formulated as the following workflow:

| (3) |  | $\psi(\mathbf{X}_{retr})\rightarrow\bigl{[}\mathrm{LLM}^{e}_{:\mathrm{n}}(% \mathbf{X}_{retr}^{1}),~{}...~{},\mathrm{LLM}^{e}_{:\mathrm{n}}(\mathbf{X}_{% retr}^{m})\bigr{]}\rightarrow\mathbf{E}_{retr}$ |  |
| --- | --- | --- | --- |

In this place, $\mathrm{LLM}^{e}$ is the LLM backbone employed for compressive encoding, while “$:\mathrm{n}$” indicates that the first-n layers are utilized.

### 3.4. Selective Compression Mechanism

Once the retrieved contexts are encoded as compressive embeddings, selective compression is applied, which produces the compressed context for RAG through down-sampling. It emphasizes the useful information to RAG tasks, where the related contexts are assigned with a high sampling ratio. In contrast, it neglects the less useful information, whose related contexts are assigned with a small sampling ratio. With this processing, the useful information can be better preserved from compression.

Selective compression calls for accurate estimation of context importance. In our work, we propose two alternative approaches to achieve this goal, providing users with the flexibility to choose the most suitable option for their specific applications.

#### 3.4.1. Token-level estimation

The first alternative estimates context importance on the token basis, which is a popular principle adopted by many studies *(Jiang et al., [2023a]; Pan et al., [2024])*. Given the input prompt of RAG task $X_{task}$, the importance tokens within the retrieved contexts $X_{retr}$ are generally favored by the LLM, leading to relatively higher generation likelihood compared to other less useful tokens. Based on this principle, we introduce the following relationship as an approximate indicator of token importance:

| (4) |  | $\text{for}~{}~{}x_{i}\in X_{retr}:~{}~{}w_{i}\leftarrow P_{LLM}(x_{i}|{X}_{% task},X_{retr}[:x_{i}])$ |  |
| --- | --- | --- | --- |

where $X_{retr}[:x_{i}]$ represents the prefix of $x_{i}$. In other words, $w_{i}>w_{j}$ if $x_{i}$ is more important than $x_{j}$.
Despite simplicity, the above indicator can basically identify useful contexts as demonstrated by related works *(Jiang et al., [2023a]; Pan et al., [2024])*. However, it might suffer from incoherence and broken semantic as discrete tokens are sampled from context.

#### 3.4.2. Sentence-level estimation

The second alternative estimates the importance for each sentence within the retrieved contexts. To this end, we employ an ad-hoc model to estimate the sentence’s relevance to the task’s prompt, where the relevance score is used as the importance. Although the original retriever which produce the retrieved contexts is a desirable option, it is not always always in practice. Therefore, we leverage a general purpose embedder, such as E5 *(Wang et al., [2022])* and BGE *(Xiao et al., [2023])*, as the relevance oracle, denoted as $\mathcal{M}$. Therefore, the importance is computed as:

| (5) |  | $\text{for}~{}~{}sent_{i}\in X_{retr}:~{}~{}w_{i}\leftarrow\mathcal{M}(X_{task}% ,sent_{i}),$ |  |
| --- | --- | --- | --- |

where $sent_{i}$ stands for the $i$-th sentence in the retrieved contexts. Compared to the token-level method, the sentence-level approach is able to better maintain semantic coherence, as the retrieved contexts are down-sampled on a sentence basis.

#### 3.4.3. Compression ratio allocation

Although the estimated importance is positively correlated with the usefulness of context, it serves more as an indicator of relative relationships rather than a direct basis for sampling ratios. To address this, we propose a stepped scheme for allocating the sampling ratios.
This method partitions the retrieved contexts into groups, where higher-priority groups receive a greater sampling ratio. The allocation process is straightforward which involves three simple steps.
First, we rank different parts of the contexts based on their estimated importance (by tokens with the token-level estimation, or by sentences with the sentence-level estimation). Next, we introduce $k$ groups: $g_{1}$, $g_{2}$, … , $g_{k}$, with increasing priorities. We also define sampling ratio for these $k$ groups: $w_{1}$, $w_{2}$, … , $w_{k}$, in an ascending order. Finally, we make linear allocation of the contexts to the $k$ groups to ensure that the following relationship holds:

| (6) |  | $w_{1}*n_{1}+w_{2}*n_{2}+...w_{k}*n_{k}\=\alpha*n$ |  |
| --- | --- | --- | --- |

Here, $n_{i}$ and $n$ represent the length for the $g_{i}$ and $X_{retr}$, respectively, while $\alpha$ denotes the required compression ratio. In the simplest case where a binary partition is made (i.e., forming a low-priority group $g_{1}$ and a high-priority group $g_{2}$ are formed), the context allocation can be directly calculated once the sampling ratios are determined.

### 3.5. Training Workflow

The training process takes place to optimize the compressive encoder, enabling it to generate high-quality compressed contexts that enhance RAG performance. Given the abundance of unlabeled data (e.g., general corpora like Pile, RedPajama) and the limited availability of labeled data (e.g., for question answering), we design a two-stage training workflow to fully optimize the model based on the accessible data resource. First, we perform auto-regressive pre-training on the unlabeled data, where language modeling is conducted based on the compressed contexts. In this stage, the following training objective is maximized:

| (7) |  | $\max\sum\nolimits_{x_{i}\in X_{pre}}P_{LLM}(x_{i}|\phi(X_{pre}[:x_{i}])).$ |  |
| --- | --- | --- | --- |

Here, $X_{pre}$ stands for a sample of unlabeled data for pre-training, $x_{i}$ is the $i$-th token, and $\phi(X_{pre}[:x_{i}])$ is the compression of $x_{i}$’s prefix. With the first stage of training, the connection is established between the compressive encoder and the downstream LLM. Next, we move on to perform fine-tuning based on label QA datasets. For each training sample, the ground-truth answer $X_{ans}$ is predicted based on the question $X_{q}$ and the compressed retrieved contexts $\phi(X_{retr})$ (i.e. relevant docs to the question). As a result, we can formulate the following objective function:

| (8) |  | $\max\sum\nolimits_{x_{i}\in X_{ans}}P_{LLM}(x_{i}|\phi(X_{retr}),X_{q},X_{ans}% [:x_{i}]).$ |  |
| --- | --- | --- | --- |

Thanks to the second stage of training, the compressive encoder can be further enhanced to optimize the RAG performance.

The training process is further enhanced in two ways. First, we introduce two-stream processing during pre-training. This involves encoding all chunks of each input at the very beginning (encoding stream) and then performing auto-regressive decoding based on the pre-encoded contexts (decoding stream). This approach allows for parallelized auto-regressive language modeling, making it more sample-efficient than traditional recurrent method *(Chevalier et al., [2023])*. Second, we randomly sample the compressive embeddings using a dynamic rate during training. In other words, selective compression is only made during inference. This approach lets the entire output of the encoder to be trained as compressive embeddings; meanwhile, it also enables the model to flexibly accommodate various compression ratios.

4. Experiments
---------------

Our experiments are dedicated to the following research questions. RQ. 1 Can FlexRAG bring forth cost-effective compression results for general RAG tasks. RQ. 2 Can FlexRAG flexibly support diverse compression ratios and compression methods. RQ. 3 Extended analysis of various aspects, including FlexRAG’s robustness to different working conditions and the impact of each technical factor.

### 4.1. Settings

#### 4.1.1. Datasets

The experiments focus on evaluating RAG performance, where two types of tasks are used: Long-sequence Multi-doc QA (LMQA) and conventional Open-Domain QA (ODQA). For LMQA, the retrieved contexts consist of multiple long documents, whose entire length is usually longer than the LLM’s window size. We include the following datasets for LMQA: HoptpotQA, 2WikiMQA, Musique, where we use the curated version offered by LongBench *(Bai et al., [2023])*. The retrieved contexts have been well-presented in these datasets, thus no additional retriever is needed. For ODQA, the retrieved contexts consist of short passages from Wikipedia corpus, whose entire length is usually within the context window of LLM. We include the following datasets for ODQA: Natural Questions (NQ), PopQA, and TriviaQA, where we use the curated version offered by KILT*(Petroni et al., [2020])*. The retrieved contexts are not presented in these datasets, therefore, we employ various retrievers to undertake this role in our experiment. Following the requirements from LongBench and KILT, we use F1 and Exact Match (EM) as the metrics for LMQA and ODQA, respectively.

#### 4.1.2. Baselines

We make comparison with various types of popular baselines in our experiment. First, we introduce two methods which employ LLaMA-2-7B (chat) for question answering: 1) Llama (retrieval), which directly makes use of the retrieved contexts without compression, and Llama (w/o retrieval) which answers the question without using retrieved contexts. Second, we include two types of context compression methods. One is the context compression methods, which contains ICAE*(Ge et al., [2023])* and AutoCompressor *(Chevalier et al., [2023])*. Both methods generate summary vectors as the compressed inputs for RAG. The other one is the context filtering methods, including LLMLingua *(Jiang et al., [2023a])*, LongLLMLingua *(Jiang et al., [2023b])*, and TF-IDF (as implemented by Gist*(Mu et al., [2024])*). These methods filtering important tokens from the retrieved contexts for RAG tasks.

|  |  | LMQA | | | | ODQA | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | CP. Ratio | HotpotQA | 2WikiMQA | Musique | Average | NQ | PopQA | TriviaQA | Average |
| Llama (w/o retrieval) | – | 19.07 | 27.78 | 5.65 | 17.50 | 14.17 | 18.01 | 53.67 | 28.62 |
| Llama (w. retrieval) | – | 27.20 | 32.21 | 7.61 | 22.34 | 25.13 | 31.04 | 55.59 | 37.25 |
| ICAE (Ge et al., [2023]) | 8 $\times$ | 19.56 | 25.07 | 5.73 | 16.79 | 10.53 | 4.44 | 8.21 | 7.73 |
| AutoCompressor (Chevalier et al., [2023]) | 8 $\times$ | 13.80 | 17.31 | 7.05 | 12.72 | 13.20 | 17.78 | 49.55 | 26.84 |
| TF-IDF (Mu et al., [2024]) | 8 $\times$ | 19.20 | 24.77 | 6.28 | 16.75 | 11.49 | 14.05 | 46.73 | 24.09 |
| LLMLingua (Jiang et al., [2023a]) | 8 $\times$ | 21.07 | 26.40 | 5.46 | 17.64 | 10.22 | 11.92 | 35.75 | 19.30 |
| LongLLMLingua (Jiang et al., [2023b]) | 8 $\times$ | 21.55 | 24.77 | 7.15 | 17.82 | 18.15 | 22.74 | 52.77 | 31.22 |
| LLMLingua-2 (Pan et al., [2024]) | 8 $\times$ | 29.51 | 26.37 | 11.89 | 22.59 | 14.03 | 16.67 | 44.19 | 24.96 |
| FlexRAG w/o SC. | 8 $\times$ | 33.81 | 38.76 | 12.56 | 28.38 | 31.44 | 24.55 | 65.07 | 40.35 |
| FlexRAG w. SC. | 8 $\times$ | 36.30 | 39.15 | 14.33 | 29.93 | 33.45 | 35.96 | 66.71 | 45.37 |

*Table 1. Cost-effectiveness Analysis on LMQA and ODQA (FlexRAG w. SC. stands for FlexRAG with selective compression).*

| Model | CP. Ratio | EM | CUDA Time (s) | TFLOPs |
| --- | --- | --- | --- | --- |
| Llama (w.r.) | 1 $\times$ | 37.25 | 7.78 | 14.17 |
| FlexRAG | 2 $\times$ | 47.23 | 4.97 | 10.48 |
| | 4 $\times$ | 47.25 | 3.13 | 7.03 |
| 8 $\times$ | 45.37 | 2.48 | 5.39 |
| 16 $\times$ | 38.93 | 2.20 | 4.59 |

*Table 2. Efficiency analysis using CUDA Time and TFLOPs. Compression ratios are varied from 2$\times$ to 16$\times$.
Experiments are performed on ODQA with EM as the quality metric.*

#### 4.1.3. Implementations

We initialize FlexRAG with the first 8 layers of LLaMA-2-7B (chat), and
we leverage LLaMA-2-7B (chat) *(Touvron et al., [2023])* as our downstream LLM. This ensures a fair comparison with the baselines and maintains the economical running of the experiments. The pre-traininig is performed with 90K sampled instances from Redpajama *(Computer, [2023])* and 10K training instances from LongAlpaca *(Chen et al., [2023])*, while the fine-tuning is performed with 10K sampled instances from a blend of HotpotQA*(Yang et al., [2018])* and Natural Questions dataset *(Kwiatkowski et al., [2019])*. During training, the compression ratios are randomly sampled from 1, 2, 4, and 8. The training takes place on a Nvidia 8×A800 GPU machine. By default, we use BGE-EN-large *(Xiao et al., [2023])* as the retriever, where the top-5 documents are returned during the testing stage. Meanwhile, FlexRAG’s compression is made by sentence-level importance estimation and selection.

### 4.2. Cost-Effectiveness Analysis

#### 4.2.1. Primary results

We first evaluate the primary question answering performance under the default setting, where a uniform compression ratio (CP. Ratio) of 8$\times$ is applied. For LMQA, all retrieved contexts are confined within 32K tokens, allowing them to be fully utilized by the downstream LLM after compression. In contrast, Llama (w. retrieval) must truncate the retrieved contexts to fit within the 4K window of Llama-2, as implemented by Longbench *(Bai et al., [2023])*. We compare two variants of our method: FlexRAG w/o SC, which disables selective compression and uniformly down-samples the retrieved contexts at an interval of 8 tokens, and FlexRAG w. SC, the default method using selective compression. The following observations can be drawn from the experiment results in Table [1].

First, FlexRAG demonstrates superior performance in the experiment. Even without using selective compression, FlexRAG w/o SC already outperforms all baselines with notable advantages. With the enhancement from selective compression, FlexRAG’s performance is further improved, which leads to the optimal question answering quality across all datasets. The above result preliminarily validates the effectiveness FlexRAG, indicating that the context compression and RAG optimization are realized simultaneously.

Second, FlexRAG’s effectiveness is more pronounced on ODQA tasks, whose retrieved contexts are much shorter than those from LMQA, e.g., it notably improves upon the best of the compression baselines (LongLLMLingual) from 18.15 to 33.45 on PopQA. This is because when retrieved contexts are concise, they are more likely to suffer from information loss once compressed. It is a more challenging task to handle the compression on ODQA, and the increased challenge expands the gap between FlexRAG and baselines.

Third, Llama (w. retrieval) falls behind many compression baselines in LMQA, whereas it outperforms all of them in ODQA. As mentioned, the compression baselines can fully leverage the retrieved contexts with a uniform compression ratio 8$\times$; while Llama (w. retrieval) has to truncate the retrieved contexts, which incurs information. However, the retrieved context is concise on ODQA, where no truncation is needed for Llama (w. retrieval); by comparison, the compression baselines will not leverage any extra information, but suffer from the information loss caused by compression.

|  |  | LMQA | | | | ODQA | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | CP. Ratio | HotpotQA | 2WikiMQA | Musique | Average | NQ | PopQA | TriviaQA | Average |
| Llama (w/o retrieval) | – | 19.07 | 27.78 | 5.65 | 17.50 | 14.17 | 18.01 | 53.67 | 28.62 |
| Llama (w. retrieval) | – | 27.20 | 32.21 | 7.61 | 22.34 | 25.13 | 31.04 | 55.59 | 37.25 |
| FlexRAG w/o SC. | 1 $\times$ | 30.83 | 35.01 | 12.86 | 26.23 | 37.01 | 33.92 | 66.93 | 45.95 |
| | 2 $\times$ | 32.06 | 35.72 | 12.02 | 26.60 | 36.62 | 31.03 | 67.74 | 45.13 |
| 4 $\times$ | 31.23 | 34.19 | 11.83 | 25.75 | 35.28 | 28.09 | 67.10 | 43.49 |
| 8 $\times$ | 33.81 | 38.76 | 12.56 | 28.38 | 31.44 | 24.55 | 65.07 | 40.35 |
| FlexRAG w. SC. | 2 $\times$ | 34.31 | 37.41 | 13.28 | 28.33 | 37.19 | 36.10 | 68.41 | 47.23 |
| | 4 $\times$ | 35.06 | 36.40 | 14.93 | 28.80 | 36.02 | 37.15 | 68.59 | 47.25 |
| 8 $\times$ | 36.30 | 39.15 | 14.33 | 29.93 | 33.45 | 35.96 | 66.71 | 45.37 |

*Table 3. Flexibility analysis on LMQA and ODQA using different compression ratios (1$\times$, 2$\times$, 4$\times$, 8$\times$).*

| Method | Estimator | HP. Prop. | Compression Ratio | | | LMQA | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | HP. | LP. | Overall | HotpotQA | 2WikiMQA | Musique | Average |
| Token w/o SC | Uniform | N.A. | N.A. | N.A. | 8 $\times$ | 33.81 | 38.76 | 12.56 | 28.38 |
| Token w. SC | Likelihood | 1: 16 | 1 $\times$ | $\sim$ 16 $\times$ | 8 $\times$ | 33.18 | 37.80 | 11.82 | 27.60 |
| | | | 2 $\times$ | $\sim$ 11 $\times$ | 8 $\times$ | 35.17 | 38.60 | 11.84 | 28.54 |
| Sentence w. SC | Embedding | 1 $\times$ | $\sim$ 16 $\times$ | 8 $\times$ | 36.30 | 39.15 | 14.33 | 29.93 |
| | | 2 $\times$ | $\sim$ 11 $\times$ | 8 $\times$ | 35.70 | 39.96 | 12.52 | 29.39 |

*Table 4. Flexibility Analysis on LMQA using different importance estimators and compression ratios (HP/LP: high-priority / low-priority contexts. In our experiment, the top 1/16 (1:16) of the retrieved contexts are allocated with the high priority).*

#### 4.2.2. Efficiency

We further explore the efficiency of FlexRAG using Torch Profiler*(Torch, [2024])*.
The evaluation takes place on one NVIDIA A800 GPU, with a batch size of 8 with BF16 precision. FlexRAG is compared with Llama (w. retrieval), where both methods employ the identical input and output lengths. The experiment is conducted with the ODQA datasets, where EM is the quality metric. The compression ratio is varied from 2$\times$ to 8$\times$ for FlexRAG.

As shown in Table[2], FlexRAG is faster than Llama (w.r.) while improving the RAG’s performance simultaneously. With the increasing of compression ratio, FlexRAG becomes even faster, leading to 3.54$\times$ reduction in CUDA time and 3.09$\times$ reduction in TFLOPs at a 16$\times$ compression ratio. Moreover, FlexRAG substantially improves upon Llama (w.r.) in terms of RAG quality, achieving a 10% improvement in ODQA datasets at a 4$\times$ compression ratio.

### 4.3. Flexibility Analysis

#### 4.3.1. Flexibility in compression ratio

We make evaluation of four compression ratios: $1\times$ (a special case without down-sampling), $2\times$, $4\times$, $8\times$. Both FlexRAG w. SC and FlexRAG w/o SC are included in our experiment, allowing us to evaluate the impact where selective compression is enabled or disabled. The experiments are performed on both LMQA and ODQA datasets. For LMQA, FlexRAG’s compressed contexts may still exceed Llama-2’s window size in some cases (unless $8\times$); as a result, truncation will be used when it has to. We utilize Llama (w/o retrieval) and Llama (w. retrieval) as the baselines for comparison. The experiment results are shown in Table [3], where the following observations can be made.

First, FlexRAG consistently outperforms the baselines across all compression ratios. Even without selective compression, FlexRAG w/o SC already surpasses all baselines. When selective compression is enabled, FlexRAG w. SC further extends its empirical advantage. Such an observation indicates that the diverse compression ratios are effectively supported by FlexRAG across different use cases. Besides, it’s worth noting that when the compression ratio is $1\times$, FlexRAG w/o SC receives the same truncated contexts as Llama (with retrieval). Therefore, any improvement it shows over Llama (with retrieval) provides a direct measure of FlexRAG’s enhancement on RAG performance.

Second, FlexRAG’s performance is consistently improved on LMQA datasets when the compression ratio grows. The optimal result is achieved when the highest compression ratio $8\times$ is used. In contrast, FlexRAG exhibits an opposite tendency on ODQA datasets, where the lowest compression ratio $1\times$ presents the optimal result. As introduced, the retrieved contexts on LMQA exceed the window size of Llama-2; therefore, higher compression ratios help to bring in more information. However, the retrieved contexts on ODQA is concise; as a result, it will not introduce extra information with higher compression ratios, but only increase the information loss for the well-presented contexts.

| Retriever | BM25 | LLM-Embedder | BGE-base | BGE-large | E5-large | Average |
| --- | --- | --- | --- | --- | --- | --- |
| Llama (w. retrieval) | 31.20 | 39.22 | 35.88 | 37.25 | 39.82 | 36.67 |
| FlexRAG | 44.58 | 46.75 | 45.80 | 45.95 | 48.47 | 46.31 |

*Table 5. Average performance on ODQA datasets with
various retrievers.*

#### 4.3.2. Flexibility in compression method

We make exploration of three alternative compression methods in our experiment. 1) Token w/o SC: this method is identical to FlexRAG w/o SC, operating at the token level with uniform down-sampling at an interval of 8 tokens. 2) Token w. SC: this method performs selective compression through token-level down-sampling, utilizing the likelihood-based importance estimator described in Eq. [4]. 3) Sentence w. SC: this method performs selective compression via sentence-level down-sampling, employing the embedding-based importance estimator as outlined in Eq. [5]. We make further variation for the compression ratio: with an uniform overall compression ratio $8\times$ and high-priority proportion 1:16, we use two alternative sets of compression ratios for the high-priority (HP.) and low-priority (LP.) group. One is (HP: $1\times$, LP: $16\times$), also the default setting in our experiment; the other one is (HP: $2\times$, LP: $11\times$). The following observations can be made from the experiment results in Table [4].

First, FlexRAG consistently maintains competitive performance across various scenarios. Notably, Sentence w. SC delivers the best results in our experiments. In contrast, Token w. SC shows sub-optimal performance, only outperforming Token w/o SC in specific cases. As previously discussed, token-level down-sampling can lead to incoherence, which diminishes the effectiveness of selective compression, especially under higher compression ratios.

Second, FlexRAG exhibits varied performance given different allocation of compression ratios between the high-priority (HP) and low-priority (LP) groups. HotpotQA and Musique prefer the default allocation (HP: $1\times$, LP: $16\times$), while 2WikiMQA benefits more from a different setup (HP: $2\times$, LP: $11\times$). With a constrained overall compression ratio, assigning lower compression ratios to the HP group helps to preserve crucial information. However, this approach can also lead to the omission of less salient but still important content. Thus, finding the optimal allocation of compression ratios is about striking a delicate balance between these competing factors.

<img src='x3.png' alt='Refer to caption' title='' width='830' height='351' />

*Figure 3. Performance on NQ and TriviaQA with varied #Doc.*

### 4.4. Extended Analysis

#### 4.4.1. Robustness to working conditions

We first analyze the impact of using different retrievers. Beyond BGE-large, which was applied for both training and testing, we explore several alternative retrievers for testing, including BM25 *(Robertson et al., [2009])*, LLM-Embedder *(Zhang et al., [2023])*, BGE-base *(Xiao et al., [2023])*, and E5-large *(Wang et al., [2022])*. The results are presented in Table[5], leading to the following key observations. First, FlexRAG consistently outperforms llama (w. retrieval) across all retrievers tested. Notably, this includes not only BGE-large, but also other retrievers differ from the one employed during training, demonstrating the strong generalizability of FlexRAG. Second, FlexRAG already achieves a superior performance even with a relatively weaker retriever, e.g., BM25. Additionally, its performance improves further when paired with stronger retrievers, like LLM-Embedder and E5-large.

We further investigate the effect of the number of retrieved documents. In our experiment, we vary the number of retrieved contexts from the top 1 to the top 10 documents returned by the retriever. The results, shown in Figure[3], lead to the following observations. First, our method (FlexRAG) consistently outperforms the baseline (Llama w. retrieval), and it demonstrates a greater stability, indicating that FlexRAG effectively handles variations in the number of retrieved documents. Second, FlexRAG’s performance improves as the number of retrieved documents increases from 1 to 5, knowing that more useful information can be continually introduced. However, beyond this threshold, no further benefits are observed. According to previous studies*(Yoran et al., [2023])*, this plateau can be attributed to increased noise from irrelevant documents. Notably, FlexRAG experiences a much smaller performance decline compared to the baseline, suggesting it is more robust to noise.

| Factor | Setting | LMQA | ODQA |
| --- | --- | --- | --- |
| Training stage | w/o pre-training | 22.32 | 39.91 |
| | w/o rag fine-tuning | 24.90 | 39.18 |
| default setting* | 29.93 | 45.37 |
| Encoder Arch. | first 4 layer | 25.07 | 39.85 |
| | first 12 layer | 26.28 | 40.46 |
| first 8 layer* | 29.93 | 45.37 |

*Table 6. Ablation studies of FlexRAG on LMQA and ODQA datasets. Default settings are marked with “*”.*

#### 4.4.2. Ablation studies

We first examine the significance of the two-stage training paradigm. In addition to the default method where both training stages are applied, we assess the impact of using only one of the stages: either w/o pre-training (i.e., RAG fine-tuning only) or w/o fine-tuning (i.e., pre-training only). The results, presented in Table [6], indicate that pre-training with unlabeled data (w/o fine-tuning) significantly boosts FlexRAG’s performance, as this stage alone already delivers competitive results. The subsequent RAG fine-tuning further enhances the performance, with the default two-stage method achieving the best results in the experiment.

Next, we evaluate the architecture of the compressive encoder, with three alternatives tested: the first 4 layers of Llama-2, the first 8 layers of Llama-2 (default), and the first 12 layers of Llama-2. As shown in Table [6], the best performance is achieved when the encoder uses the first 8 layers of Llama-2 as its backbone. In comparison, the smaller encoder (first 4 layers) is constrained by its limited expressiveness, while the larger encoder (first 12 layers) introduces greater disparity with the input layer of the downstream LLM. Therefore, selecting the appropriate encoder architecture requires balancing these considerations.

5. Conclusion And Future Work
------------------------------

In this paper, we have presented FlexRAG which brings forth compressed and optimized contexts for RAG tasks. FlexRAG’s architecture allows for flexible production of compressed contexts across various compression ratios, while also supporting selective compression to preserve critical information. By leveraging a two-stage training workflow, FlexRAG effectively utilizes diverse training data, resulting in significant performance optimization. Our experiments across multiple QA datasets demonstrate FlexRAG’s cost-effectiveness, flexibility, and generalizability in different working conditions. Building on the progress of this preliminary work, future research will explore broader applications with more extensive LLM backbones and RAG tasks beyond question answering.

References
----------

* (1)
* Asai et al. (2023)Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.Self-rag: Learning to retrieve, generate, and critique through self-reflection.*arXiv preprint arXiv:2310.11511* (2023).
* Bai et al. (2023)Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023.LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.*arXiv preprint arXiv:2308.14508* (2023).
* Borgeaud et al. (2022)Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.Improving language models by retrieving from trillions of tokens. In *International conference on machine learning*. PMLR, 2206–2240.
* Chen et al. (2023)Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023.Longlora: Efficient fine-tuning of long-context large language models.*arXiv preprint arXiv:2309.12307* (2023).
* Chevalier et al. (2023)Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023.Adapting language models to compress contexts.*arXiv preprint arXiv:2305.14788* (2023).
* Computer (2023)Together Computer. 2023.*RedPajama: an Open Dataset for Training Large Language Models*.[https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data "")
* Ding et al. (2023)Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2023.Parameter-efficient fine-tuning of large-scale pre-trained language models.*Nature Machine Intelligence* 5, 3 (2023), 220–235.
* Gao et al. (2023)Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023.Retrieval-augmented generation for large language models: A survey.*arXiv preprint arXiv:2312.10997* (2023).
* Ge et al. (2023)Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2023.In-context autoencoder for context compression in a large language model.*arXiv preprint arXiv:2307.06945* (2023).
* Ho et al. (2020)Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps.*arXiv preprint arXiv:2011.01060* (2020).
* Izacard and Grave (2020)Gautier Izacard and Edouard Grave. 2020.Leveraging passage retrieval with generative models for open domain question answering.*arXiv preprint arXiv:2007.01282* (2020).
* Izacard et al. (2023)Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023.Atlas: Few-shot learning with retrieval augmented language models.*Journal of Machine Learning Research* 24, 251 (2023), 1–43.
* Ji et al. (2023)Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.Survey of hallucination in natural language generation.*Comput. Surveys* 55, 12 (2023), 1–38.
* Jiang et al. (2023a)Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a.LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. 13358–13376.
* Jiang et al. (2023b)Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b.Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression.*arXiv preprint arXiv:2310.06839* (2023).
* Jiang et al. (2023c)Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023c.Active retrieval augmented generation.*arXiv preprint arXiv:2305.06983* (2023).
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019.Natural questions: a benchmark for question answering research.*Transactions of the Association for Computational Linguistics* 7 (2019), 453–466.
* Lester et al. (2021)Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.The power of scale for parameter-efficient prompt tuning.*arXiv preprint arXiv:2104.08691* (2021).
* Lewis et al. (2020)Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020.Retrieval-augmented generation for knowledge-intensive nlp tasks.*Advances in Neural Information Processing Systems* 33 (2020), 9459–9474.
* Liu et al. (2021)Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2021.P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks.*arXiv preprint arXiv:2110.07602* (2021).
* Liu et al. (2023)Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2023.Trustworthy LLMs: A survey and guideline for evaluating large language models’ alignment.*arXiv preprint arXiv:2308.05374* (2023).
* Luo et al. (2023)Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023.An empirical study of catastrophic forgetting in large language models during continual fine-tuning.*arXiv preprint arXiv:2308.08747* (2023).
* Mu et al. (2024)Jesse Mu, Xiang Li, and Noah Goodman. 2024.Learning to compress prompts with gist tokens.*Advances in Neural Information Processing Systems* 36 (2024).
* OpenAI (2024)OpenAI. 2024.SearchGPT Prototype.[https://openai.com/index/searchgpt-prototype/](https://openai.com/index/searchgpt-prototype/ "")
* Pan et al. (2024)Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, et al. 2024.Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression.*arXiv preprint arXiv:2403.12968* (2024).
* Perplexity (2024)Perplexity. 2024.Perplexity.<https://www.perplexity.ai/>
* Petroni et al. (2020)Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. 2020.KILT: a benchmark for knowledge intensive language tasks.*arXiv preprint arXiv:2009.02252* (2020).
* Robertson et al. (2009)Stephen Robertson, Hugo Zaragoza, et al. 2009.The probabilistic relevance framework: BM25 and beyond.*Foundations and Trends® in Information Retrieval* 3, 4 (2009), 333–389.
* Schick et al. (2024)Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2024.Toolformer: Language models can teach themselves to use tools.*Advances in Neural Information Processing Systems* 36 (2024).
* Shi et al. (2023)Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023.Replug: Retrieval-augmented black-box language models.*arXiv preprint arXiv:2301.12652* (2023).
* Shinn et al. (2024)Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2024.Reflexion: Language agents with verbal reinforcement learning.*Advances in Neural Information Processing Systems* 36 (2024).
* Sun et al. (2024)Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. 2024.Trustllm: Trustworthiness in large language models.*arXiv preprint arXiv:2401.05561* (2024).
* Torch (2024)Torch. 2024.Torch Profiler.<https://pytorch.org/docs/stable/profiler.html>
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.Llama 2: Open foundation and fine-tuned chat models.*arXiv preprint arXiv:2307.09288* (2023).
* Trivedi et al. (2022)Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022.MuSiQue: Multihop Questions via Single-hop Question Composition.*Transactions of the Association for Computational Linguistics* 10 (2022), 539–554.
* Wang et al. (2022)Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022.Text Embeddings by Weakly-Supervised Contrastive Pre-training.*arXiv preprint arXiv:2212.03533* (2022).
* Wang et al. (2023)Liang Wang, Nan Yang, and Furu Wei. 2023.Learning to retrieve in-context examples for large language models.*arXiv preprint arXiv:2307.07164* (2023).
* Wang et al. (2024)Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. 2024.CodeRAG-Bench: Can Retrieval Augment Code Generation?*arXiv preprint arXiv:2406.14497* (2024).
* Xiao et al. (2023)Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023.C-Pack: Packaged Resources To Advance General Chinese Embedding.arXiv:2309.07597 [cs.CL]
* Xu et al. (2023b)Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023b.Recomp: Improving retrieval-augmented lms with compression and selective augmentation.*arXiv preprint arXiv:2310.04408* (2023).
* Xu et al. (2023a)Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2023a.Retrieval meets long context large language models.*arXiv preprint arXiv:2310.03025* (2023).
* Yang et al. (2018)Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018.HotpotQA: A dataset for diverse, explainable multi-hop question answering.*arXiv preprint arXiv:1809.09600* (2018).
* Yao et al. (2022)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022.React: Synergizing reasoning and acting in language models.*arXiv preprint arXiv:2210.03629* (2022).
* Yasunaga et al. (2022)Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2022.Retrieval-augmented multimodal language modeling.*arXiv preprint arXiv:2211.12561* (2022).
* Yoran et al. (2023)Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023.Making retrieval-augmented language models robust to irrelevant context.*arXiv preprint arXiv:2310.01558* (2023).
* Zhang et al. (2023)Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023.Retrieve anything to augment large language models.*arXiv preprint arXiv:2310.07554* (2023).
* Zhang et al. (2024)Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. 2024.Raft: Adapting language model to domain specific rag.*arXiv preprint arXiv:2403.10131* (2024).
