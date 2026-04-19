Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy
=================================================================================================

Zhihong Shao1, Yeyun Gong2, yelong shen3, Minlie Huang1, Nan Duan2, Weizhu Chen3  
1 The CoAI Group, DCST, Institute for Artificial Intelligence,  
1 State Key Lab of Intelligent Technology and Systems,  
1 Beijing National Research Center for Information Science and Technology,  
1 Tsinghua University, Beijing 100084, China  
2 Microsoft Research Asia 3 Microsoft Azure AI  
szh19@mails.tsinghua.edu.cn aihuang@tsinghua.edu.cn*Corresponding author: Minlie Huang.

###### Abstract

Retrieval-augmented generation has raise extensive attention as it is promising to address the limitations of large language models including outdated knowledge and hallucinations.
However, retrievers struggle to capture relevance, especially for queries with complex information needs.
Recent work has proposed to improve relevance modeling by having large language models actively involved in retrieval, i.e., to guide retrieval with generation.
In this paper, we show that strong performance can be achieved by a method we call Iter-RetGen, which synergizes retrieval and generation in an iterative manner:
a model’s response to a task input shows what might be needed to finish the task, and thus can serve as an informative context for retrieving more relevant knowledge which in turn helps generate a better response in another iteration.
Compared with recent work which interleaves retrieval with generation when completing a single output, Iter-RetGen processes all retrieved knowledge as a whole and largely preserves the flexibility in generation without structural constraints.
We evaluate Iter-RetGen on multi-hop question answering, fact verification, and commonsense reasoning, and show that it can flexibly leverage parametric knowledge and non-parametric knowledge, and is superior to or competitive with state-of-the-art retrieval-augmented baselines while causing fewer overheads of retrieval and generation.
We can further improve performance via generation-augmented retrieval adaptation.

1 Introduction
--------------

Generative Large Language Models (LLMs) have powered numerous applications, with well-perceived utility.
Despite being powerful, LLMs lack knowledge that is under-represented in their training data, and are prone to hallucinations, especially in open-domain settings *OpenAI ([2023](#bib.bib21 ""))*.
Retrieval-augmented LLMs, therefore, have raised widespread attention as LLM outputs can be potentially grounded on external knowledge.

Previous retrieval-augmented LMs *Izacard et al. ([2022b](#bib.bib13 "")); Shi et al. ([2023](#bib.bib27 ""))* typically adopted one-time retrieval, i.e., to retrieve knowledge using only the task input (e.g., a user question for open-domain question answering).
One-time retrieval should suffice to fulfill the information needs if they are clearly stated in the original input, which is applicable to factoid question answering *Kwiatkowski et al. ([2019](#bib.bib17 ""))* and single-hop fact verification *Thorne et al. ([2018](#bib.bib28 ""))*, but not to tasks with complex information needs, e.g., multi-hop reasoning *Yang et al. ([2018](#bib.bib33 ""))* and long-form question answering *Fan et al. ([2019](#bib.bib5 ""))*.

To fulfill complex information needs, recent work proposes to gather required knowledge multiple times throughout the generation process, using partial generation *Trivedi et al. ([2022a](#bib.bib30 "")); Press et al. ([2022](#bib.bib23 ""))*) or forward-looking sentence(s) *Jiang et al. ([2023](#bib.bib14 ""))* as search queries.
However, such structured workflows of interleaving retrieval with generation have the following limitations:
(1) as intermediate generation is conditioned on knowledge retrieved before, with no awareness of knowledge retrieved afterwards, they fail to process all retrieved knowledge as a whole during the generation process;
(2) they require multi-round retrieval to gather a comprehensive set of knowledge, and may frequently change the prompts by updating newly retrieved knowledge, thus increasing the overheads of both retrieval and generation.

In this paper, we find it simple but effective to enhance retrieval-augmented LLMs through iterative retrieval-generation synergy (Iter-RetGen, Fig [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy")).
Iter-RetGen iterates retrieval-augmented generation and generation-augmented retrieval:
Retrieval-augmented generation outputs a response to a task input based on all retrieved knowledge (initially using the task input as the query).
This output shows what might be needed to fulfill the task, and thus can serve as an informative context to retrieve more relevant knowledge, i.e., generation-augmented retrieval.
The newly retrieved knowledge can benefit another iteration of retrieval-augmented generation.
We can also leverage model generations to adapt retrieval, by distilling knowledge from a re-ranker with access to model generations to a dense retriever with access to task inputs only, which may be beneficial in scenarios where user inputs can be easily collected, but relevant knowledge or desirable outputs are not annotated.

We evaluate our method on three tasks, including multi-hop question answering, fact verification, and commonsense reasoning.
Our method prompts an LLM to produce a chain of reasoning steps followed by the final answer under a few-shot setting.
For in-context demonstrations, we focus on problem-solving and follow *Wei et al. ([2022](#bib.bib32 ""))* to annotate chains of thoughts, without explicitly considering how generation-augmented retrieval might be affected, which makes it conceptually simple and easy to implement.
Our method achieves up to 8.6% absolute gains over previous state-of-the-art retrieval-augmented methods on four out of six datasets while being competitive on the remaining two.
According to our experiments, generation generally benefits from more iterations, with two iterations giving the most performance gains.
One may customize the performance-cost tradeoffs by choosing an appropriate number of iterations.
We can further improve performance and also reduce iterations via the aforementioned generation-augmented retrieval adaptation.

We summarize our findings as follows:

* •

    Automatic metrics such as exact match can significantly underestimate the performance of LLMs in question answering tasks. Moreover, improvements in exact match do not always reflect improvements in generations. Evaluation using LLMs may be more reliable.

* •

    Iter-RetGen is superior to or competitive with state-of-the-art retrieval-augmented methods, while being simpler and causing fewer overheads of retrieval and generation. With generation-augmented retrieval adaptation, we can further improve performance and also reduce overheads (by reducing iterations).

* •

    It is desirable for an LLM to leverage both parametric knowledge and non-parametric knowledge effectively. Iter-RetGen consistently outperforms Self-Ask on question answering tasks, regardless of whether in-context non-parametric knowledge mentions the answers or not.

2 Related Work
--------------

In recent months, there has been a surge in LLM-powered applications, such as ChatGPT, Bing Chat, and CoPilot *Chen et al. ([2021](#bib.bib4 ""))*.
While showing an unprecedented level of performance, LLMs are subject to the following limitations:
(1) Due to a high demand for compute and data, it remains an open research question to continually update LLMs both efficiently and effectively *Scialom et al. ([2022](#bib.bib24 ""))*;
(2) LLMs also tend to hallucinate *OpenAI ([2023](#bib.bib21 ""))*, i.e., generating plausible but non-factual texts.
To alleviate these issues, there is a growing trend of augmenting LLMs with tools *Mialon et al. ([2023](#bib.bib19 "")); Gou et al. ([2023](#bib.bib9 ""))*, e.g., a code interpreter *Gao et al. ([2022b](#bib.bib7 "")); Shao et al. ([2023](#bib.bib25 ""))* or a search engine *Nakano et al. ([2021](#bib.bib20 ""))*, in an attempt to offload sub-tasks to more qualified experts, or to enrich the input context for LLMs by providing more relevant information.

<img src='x1.png' alt='Refer to caption' title='' width='433' height='157' />

*Figure 1:  Iter-RetGen iterates retrieval and generation.
In each iteration, Iter-RetGen leverages the model output from the previous iteration as a specific context to help retrieve more relevant knowledge, which may help improve model generation (e.g., correcting the height of Hesse Hogan in this figure).
We only show two iterations in this figure for brevity.
Solid arrows connect queries to the retrieved knowledge, and dashed arrows denote retrieval-augmented generation.*

Retrieval augmentation is a mainstream direction to connect LLMs to the external world.
Previous retrieval-augmented LMs *Izacard and Grave ([2021](#bib.bib12 "")); Shao and Huang ([2022](#bib.bib26 ""))* typically receive retrieved knowledge in a passive way:
knowledge is retrieved based on the task inputs without LMs’ intervention.
As it is difficult for a retriever to capture relevance, especially in the zero-shot setting, recent work shows a shift towards having LLMs actively involved in retrieval to improve relevance modeling, e.g., to provide a specific context for retrieval with model generations (e.g., generated search queries *Nakano et al. ([2021](#bib.bib20 "")); Press et al. ([2022](#bib.bib23 "")); Yao et al. ([2022](#bib.bib34 ""))*, partial generation *Trivedi et al. ([2022a](#bib.bib30 ""))*, or forward-looking sentences *Jiang et al. ([2023](#bib.bib14 ""))*). *Khattab et al. ([2022](#bib.bib16 ""))* proposed a DSP programming framework that supports various retrieval-augmented methods.

Recent work interleaves retrieval with generation when completing a single output.
Such a structured workflow may reduce the flexibility in generation *Yao et al. ([2022](#bib.bib34 ""))*.
Iter-RetGen avoids interrupting generation with retrieval, but iterates retrieval and generation, i.e., to leverage the complete generation from the previous iteration to retrieve more relevant information which helps improve generation in the next iteration.
Iter-RetGen also has the advantage of processing all retrieved knowledge as a whole during the generation process, and is conceptually simpler and easier-to-implement, while being empirically strong in multi-hop question answering, fact verification, and commonsense reasoning.

A closely related work called Gar *Mao et al. ([2021](#bib.bib18 ""))* augments queries with generated background information.
HyDE *Gao et al. ([2022a](#bib.bib6 ""))* also shares a similar spirit, but focuses on zero-shot information retrieval, and proposes to first prompt an LLM to produce “hypothetical” paragraphs that cover the information needed to answer a given question, and then use the generated paragraphs to retrieve the real ones.
RepoCoder *Zhang et al. ([2023](#bib.bib36 ""))* focuses on repository-level code completion, and proposes a 2-iteration retrieval-generation paradigm where the second iteration leverages the intermediate code completion for retrieval.
By contrast, we propose to synergize retrieval and generation with Iter-RetGen on various natural language tasks, and explore how we can further adapt retrieval with model generations.

3 Iterative Retrieval-Generation Synergy
-----------------------------------------

### 3.1 Overview

Given a question $q$ and a retrieval corpus $\mathcal{D}\={d}$ where $d$ is a paragraph, Iter-RetGen repeats retrieval-generation for $T$ iterations;
in iteration $t$, we
(1) leverage the generation $y_{t-1}$ from the previous iteration, concatenated with $q$, to retrieve top-$k$ paragraphs,
and then (2) prompt an LLM $\mathcal{M}$ to produce an output $y_{t}$, with both the retrieved paragraphs (denoted as $\mathcal{D}_{y_{t-1}||q}$) and $q$ integrated into the prompt.
Therefore, each iteration can be formulated as follows:

|  | $y_{t}\=\mathcal{M}(y_{t}|\text{{prompt}}(\mathcal{D}_{y_{t-1}||q},q)),\ \ \forall 1\leq t\leq T$ |  | (1) |
| --- | --- | --- | --- |

The last output $y_{T}$ will be produced as the final response.

### 3.2 Generation-Augmented Retrieval

There are many natural language tasks with complex information needs.
For example, in open-domain multi-hop question answering, specific information needs may manifest themselves only after correctly answering some prerequisite sub-questions.
In other words, there may exist semantic gaps between the original question $q$ and its supporting knowledge, which can not be effectively addressed by a retriever with a representation bottleneck.
In the first iteration, we can retrieve knowledge with only the question $q$.
In later iterations, the LLM output from the previous iteration, though having no guarantee of correctness, shows what might be needed to answer the question, and thus can be leveraged to bridge the semantic gaps;
with improved retrieval, an LLM can potentially produce a better output.

### 3.3 Retrieval-Augmented Generation

In each iteration, we generate an output using Chain-of-Thought prompting except that we also prepend retrieved knowledge to the question $q$.
Though there may exist more advanced prompting variants, e.g., incorporating previous generations into the prompt to enable direct refinements, we leave the explorations for future work, and focus on investigating the synergy between retrieval and generation in a straightforward manner.

### 3.4 Generation-Augmented Retrieval Adaptation

Model generations not only provide specific contexts for retrieval, but can also be leveraged to optimize the retriever, so that information needs in a question can be better captured by the retriever.

##### Dense Retriever

We adopted dense retrieval in our experiments.
Given a dense retriever parametrized by $\mathbf{\theta}\={\mathbf{\theta}_{q},\mathbf{\theta}_{d}}$ where $\mathbf{\theta}_{q}$ and $\mathbf{\theta}_{d}$ denote parameters of the query encoder and the paragraph encoder, respectively, the similarity score between a query and a paragraph is calculated as the inner product of their encoded vectors:

|  | $s_{\mathbf{\theta}}(q,d)\=\langle\mathbf{E}(q;\mathbf{\theta}_{q}),\mathbf{E}(d;\mathbf{\theta}_{d})\rangle$ |  | (2) |
| --- | --- | --- | --- |

##### Re-ranker

A re-ranker, parametrized by $\mathbf{\phi}$, outputs the probability of a paragraph being relevant to a query; we denote the probability as $s_{\mathbf{\phi}}(q,d)$.

##### Distillation

A re-ranker is typically better at capturing relevance between a query and a paragraph than a retriever.
Therefore, we distill knowledge from a re-ranker to a retriever.
To help the retriever better address the semantic gaps between a question and its supporting knowledge, we allow access to $y_{1}$ for the re-ranker (where $y_{1}$ is the LLM output from the first iteration).
We optimize only the query encoder of the retriever using the following training objective:

|  | $\begin{split}\mathbf{\theta}_{q}^{*}\&\=arg\min_{\mathbf{\theta}_{q}}\ \text{KL}(P_{\mathbf{\phi}}(\cdot|y_{1},q),P_{\mathbf{\theta}}(\cdot|q))\\ P_{\mathbf{\phi}}(d|y_{1},q)\&\=\frac{\exp(s_{\mathbf{\phi}}(y_{1}||q,d)/\tau)}{\sum_{d^{\prime}\in\mathcal{D}_{y_{1}||q}}\exp(s_{\mathbf{\phi}}(y_{1}||q,d^{\prime})/\tau)}\\ P_{\mathbf{\theta}}(d|q)\&\=\frac{\exp(s_{\mathbf{\theta}}(q,d)/\tau)}{\sum_{d^{\prime}\in\mathcal{D}_{y_{1}||q}}\exp(s_{\mathbf{\theta}}(q,d^{\prime})/\tau)}\\ \end{split}$ |  | (3) |
| --- | --- | --- | --- |

where $\text{KL}(\cdot,\cdot)$ denotes the KL divergence between two probabilistic distributions.

4 Experiments
-------------

### 4.1 Datasets

We experimented on six datasets of three reasoning tasks:
(1) Multi-hop question answering, including HotPotQA *Yang et al. ([2018](#bib.bib33 ""))*, 2WikiMultiHopQA *Ho et al. ([2020](#bib.bib10 ""))*, MuSiQue *Trivedi et al. ([2022b](#bib.bib31 ""))*, and Bamboogle *Press et al. ([2022](#bib.bib23 ""))*.
On MuSiQue, we followed *Press et al. ([2022](#bib.bib23 ""))* to use only 2-hop questions;
(2) Fact Verification, including Feverous *Aly et al. ([2021](#bib.bib1 ""))*;
(3) Commonsense reasoning, including StrategyQA *Geva et al. ([2021](#bib.bib8 ""))*.
Examples are presented in Table [1](#S4.T1 "Table 1 ‣ 4.1 Datasets ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy").

We used the October 2017 *Yang et al. ([2018](#bib.bib33 ""))* and the December 2018 *Karpukhin et al. ([2020](#bib.bib15 ""))* Wikipedia dump as the retrieval corpus for HotPotQA and 2WikiMultiHopQA, respectively, and used the December 2021 Wikipedia dump *Izacard et al. ([2022b](#bib.bib13 ""))* for the other datasets.

DatasetsExampleHotPotQAWhat is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?2WikiMultiHopQAWhich film came out first, Blind Shaft or The Mask Of Fu Manchu?MuSiQueIn which year did the publisher of In Cold Blood form?BamboogleWhen did the first prime minister of the Russian Empire come into office?FeverousIs it true that Based on the same platform as the Chevrolet Sail, the Baojun 310 was launched on 2017 Beijing Auto Show where the price ranges from 36.800 yuan to 60.800 yuan?StrategyQAIs it common to see frost during some college commencements?

*Table 1:  Example questions from six datasets.*

### 4.2 Evaluation Settings

We conducted evaluations on all 125 questions from Bamboogle, the first 500 questions from the train set of StrategyQA, and the first 500 questions from the development sets of the other datasets.
All methods are evaluated under the 3-shot setting, sharing the same questions in demonstrations.

Evaluation metrics are exact match (EM) and F1 for multi-hop question answering datasets, and accuracy for both fact verification and commonsense reasoning datasets.
For more robust evaluation, we also evaluate the correctness of model outputs using text-davinci-003, the resulting metric denoted as Acc†.
The prompt used for evaluation is as follows, where {question}, {model output}, and {answer} are placeholders.


### 4.3 Baselines

Direct Prompting *Brown et al. ([2020](#bib.bib3 ""))* prompts an LLM to directly generate the final answer without an explanation.
When augmenting Direct prompting with retrieval, we used the question to retrieve knowledge which will be placed before the question in the prompt.

CoT Prompting *Wei et al. ([2022](#bib.bib32 ""))* prompts an LLM to generate natural language reasoning steps followed by the final answer.

ReAct *Yao et al. ([2022](#bib.bib34 ""))* interleaves reasoning, action, and observation steps, until reaching the action of finalizing an answer.
An action can be either generating a query to search for information or finalizing an answer.
An observation is the concatenation of retrieved paragraphs.

Self-Ask *Press et al. ([2022](#bib.bib23 ""))* interleaves (i) follow-up question generation, (ii) retrieval using the follow-up, and (iii) answering the follow-up conditioned on the retrieved knowledge, until no more follow-up questions are generated and the LLM gives an answer to the original question.
We followed *Yoran et al. ([2023](#bib.bib35 ""))* to prepend newly retrieved paragraphs to the original question.
On our evaluated tasks, Self-Ask is conceptually similar to ReAct, with the main difference being that Self-Ask accumulates retrieved knowledge before the original question in the prompt, while ReAct places retrieved knowledge right after its query.
Self-Ask and IRCoT *Trivedi et al. ([2022a](#bib.bib30 ""))* also share the spirit of synergizing reasoning and retrieval.

DSP *Khattab et al. ([2022](#bib.bib16 ""))* comprises a multi-hop retrieval stage and an answer prediction stage.
For each hop within the retrieval stage, the model is prompted to generate search queries and to summarize retrieve knowledge for subsequent use.
In the prediction stage, DSP generates the answer using CoT based on the summarized knowledge and retrieved documents.

### 4.4 Implementation Details

We used text-davinci-003 version of InstructGPT *Ouyang et al. ([2022](#bib.bib22 ""))* as the backend LLM.
We also present experiments using the open-source Llama-2 models *Touvron et al. ([2023](#bib.bib29 ""))* in Appendix [A](#A1 "Appendix A Experiments Using Llama-2 ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy").
All experiments used greedy decoding.
Contriever-MSMARCO *Izacard et al. ([2022a](#bib.bib11 ""))* was used for retrieval.
We retrieved top-5 paragraphs for each query.
We allowed at most 5 interactions with retrieval for ReAct and Self-Ask.
We adapted the implementation of DSP 111https://github.com/stanfordnlp/dspy/issues/85 to use the same generation model and retrieval systems as the other methods.

Note that the first iteration of Iter-RetGen is CoT prompting with retrieval augmentation.
Therefore, Iter-RetGen and CoT prompting share the same annotated in-context demonstrations.
All prompts are presented in the Appendix.

### 4.5 Main Results

MethodHotPotQA2WikiMultiHopQAMuSiQueBamboogleFeverousStrategyQAEMF1Acc†EMF1Acc†EMF1Acc†EMF1Acc†AccAcc†AccAcc†Without RetrievalDirect21.936.844.821.329.233.97.018.715.811.224.428.060.160.166.566.7CoT30.044.150.030.039.644.019.430.928.643.251.160.059.859.871.071.0With RetrievalDirect31.644.753.327.335.443.613.928.226.517.631.843.269.869.865.665.6ReAct24.944.761.128.038.545.923.437.037.921.831.040.366.466.466.966.9Self-Ask36.855.264.837.348.855.927.641.542.931.541.254.870.770.770.270.2DSP43.855.060.8-------------Iter-RetGen 139.253.965.533.745.255.424.238.638.136.847.757.667.067.072.072.0Iter-RetGen 244.158.671.234.947.058.126.441.141.038.448.759.268.868.873.073.0Iter-RetGen 345.259.971.434.847.858.325.741.440.837.647.059.269.069.072.372.3Iter-RetGen 445.861.173.436.047.458.526.741.840.838.449.660.071.571.573.873.8Iter-RetGen 545.260.372.835.547.558.825.740.739.639.249.760.870.370.373.273.2Iter-RetGen 645.961.073.335.548.159.425.940.539.840.050.059.270.970.972.472.4Iter-RetGen 745.160.472.935.547.458.426.142.041.040.050.760.870.570.574.174.1

*Table 2:  Evaluation results on multi-hop question answering, fact verification, and commonsense reasoning datasets. Acc† is the accuracy of model outputs evaluated with text-davinci-003.
For Iter-RetGen, we evaluated LLM outputs in different iterations (up to 7 iterations).
Underlined metric values are higher than those of Self-Ask.*

MethodHotPotQA2WikiMultiHopQAMuSiQueBamboogleFeverousStrategyQA# API# Doc# API# Doc# API# Doc# API# Doc# API# Doc# API# DocReAct2.914.33.015.02.914.42.814.12.110.62.814.2Self-Ask3.216.03.215.93.014.83.014.92.311.33.015.1

*Table 3:  Average numbers of API calls to text-davinci-003 and retrieved paragraphs for ReAct and Self-Ask.
Note that Iter-RetGen ($T\=2$) achieves significantly higher or competitive Acc† with fewer API calls (i.e., 2) and fewer retrieved paragraphs (5 per iteration, 10 in total).*

As shown by Table [2](#S4.T2 "Table 2 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Iter-RetGen ($T\geq 2$) achieve significantly higher Acc† than retrieval-augmented baselines on HotPotQA, 2WikiMultiHopQA, Bamboogle, and StrategyQA, while being competitive with the best method (i.e., Self-Ask) on MuSiQue and Feverous.

When increasing the number of iterations for Iter-RetGen, performance generally improves, with the second iteration giving the greatest boost.
It is worth noting that, as shown by Table [3](#S4.T3 "Table 3 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Iter-RetGen ($T\=2$) is superior to or competitive with ReAct and Self-Ask using fewer API calls to the LLM (i.e., 2) and fewer retrieved paragraphs (i.e., 5 per iteration, 10 in total).
Iter-RetGen is also conceptually simple, which is to iterate retrieval-augmented CoT, without complex processing.

We also compared Iter-RetGen with DSP which also generates the answer using CoT based on retrieved knowledge but differs in information collection and processing.
In each iteration, Iter-RetGen retrieves knowledge based on (1) the question and (2) the previous model output which shows what may be needed to answer the question.
With the number of iterations increasing, we tend to obtain a more comprehensive and relevant set of knowledge.
Besides, unlike DSP, we do not summarize the retrieved documents for answer generation, and thus will not introduce summarization errors.
As shown in Table [2](#S4.T2 "Table 2 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Iter-RetGen outperforms DSP significantly.
We manually investigate 10 random questions where DSP fails but Iter-RetGen provides correct answers.
On 40% of them, DSP fails to retrieve documents that cover the correct answers, while on 50% of them, the summarized knowledge is misleading, e.g., for the question ‘‘What occupation do Chris Menges and Aram Avakian share?’’, DSP generates a wrong summary ‘‘Chris Menges and Aram Avakian are both members of the American and British Societies of Cinematographers.’’, while the retrieved documents mention that Aram Avakian is a film editor and director, and only Chris Menges is with the American and British Societies of Cinematographers.

DatasetHotPotQAFeverousRetrieverOriginalDistilled w/o $y_{1}$Distilled w/ $y_{1}$OriginalDistilled w/o $y_{1}$Distilled w/ $y_{1}$Iter-RetGen 165.567.167.767.067.370.7Iter-RetGen 271.275.275.768.868.169.5

*Table 4:  Effect of using LLM generation $y_{1}$ on optimizing a dense retriever.
We evaluated Iter-RetGen on HotPotQA and Feverous in terms of Acc†.*

SubsetCoT ✓CoT ✗w/ Answer Retrievedw/o Answer RetrievedMethodSelf-AskIter-RetGen 2Self-AskIter-RetGen 2Self-AskIter-RetGen 2Self-AskIter-RetGen 2HotPotQA77.588.052.054.478.186.929.940.82WikiMultiHopQA68.878.246.242.073.177.230.142.3MuSiQue68.566.932.630.772.978.912.222.9Bamboogle73.077.328.032.076.282.232.846.2

*Table 5:  Comparisons between Self-Ask and Iter-RetGen ($T\=2$) on different subsets, in terms of Acc†. CoT ✓ is the subset of questions which CoT answers correctly without retrieval; CoT ✗ is the complement. w/ Answer Retrieved is the subset of questions for which a method (Self-Ask or Iter-RetGen) successfully retrieves paragraphs that mention the answers; w/o Answer Retrieved is the complement.
Iter-RetGen tends to be much better at preserving the LLM’s performance on questions that can be solved using CoT without retrieval, and is consistently more accurate regardless of whether retrieved knowledge mentions the answers or not.*

##### Acc† is a Reliable Metric

To investigate how reliable Acc† is, we focused on model outputs where EM and Acc† disagree, and manually checked which metric gives more correct labels.
On each of the four multi-hop question answering datasets, we randomly sampled 20 model outputs from the second iteration of Iter-RetGen, resulting in 80 samples in total.
For 98.75% of samples, EM is 0 and Acc† is 1,
while Acc† gives the correct labels 97.5% of the time, indicating that EM severely underestimates model performance.
We also carried out the same evaluation for Self-Ask, and Acc† gives the correct labels 98.75% of the time when it is inconsistent with EM.

Acc† offers the advantage of identifying model outputs that are semantically correct, even if their surface forms differ from the annotated answers.
As an illustration, for the question ‘‘Which country Jan Baptist Van Rensselaer’s father is from?’’, the annotated answer is Dutch, while the model prediction is Netherlands, which is correct in terms of Acc† but is penalized by EM.

Notably, Iter-RetGen ($T\geq 2$) consistently demonstrate lower EM but higher Acc† than Self-Ask on 2WikiMultiHopQA, suggesting that enhancements in EM do not necessarily reflect improvements in the quality of generated answers.

Iteration1234567HotPotQA49.566.165.766.566.766.767.12WikiMultiHopQA29.045.246.246.745.845.846.5MuSiQue18.632.332.333.732.733.532.9Bamboogle20.836.036.836.035.236.036.0

*Table 6: Answer recall of retrieved paragraphs in different iterations for Iter-RetGen.*

HotPotQAQuestion: The arena where the Lewiston Maineiacs played their home games can seat how many people?Answer: 3,677 seatedIteration 1Retrieved Knowledge(1) Title: Lewiston Maineiacs Context: The Lewiston Maineiacs were a junior ice hockey team … The team played its home games at the Androscoggin Bank Colisée …(2) Title: Bangor Auditorium Context: … a 5,948 seat multi-purpose arena located in Bangor, Maine …GenerationThe Lewiston Maineiacs were a junior ice hockey team of the Quebec Major Junior Hockey League based in Lewiston, Maine. The team played its home games at the Androscoggin Bank Colisée. The Androscoggin Bank Colisée can seat up to 5,948 people. So the answer is 5,948Iteration 2Retrieved Knowledge(1) Title: Lewiston Maineiacs Context: The Lewiston Maineiacs were a junior ice hockey team … The team played its home games at the Androscoggin Bank Colisée …(2) Title: Androscoggin Bank Colisée Context: The Androscoggin Bank Colisée (formerly Central Maine Civic Center and Lewiston Colisee) is a 4,000 capacity (3,677 seated) multi-purpose arena …GenerationThe Lewiston Maineiacs were a junior ice hockey team of the Quebec Major Junior Hockey League based in Lewiston, Maine. They played their home games at the Androscoggin Bank Colisée. The Androscoggin Bank Colisée has a capacity of 4,000 (3,677 seated). So the answer is 3,677StrategyQAQuestion: Can you get Raclette in YMCA headquarters city?Answer: YesIteration 1Retrieved Knowledge(1) Title: YMCA Building (Racine, Wisconsin) Context: … Since the 50’s, headquarters of the publishing house and the bookstore are located in 11 rue de la Montagne Sainte-Geneviève 75005 Paris …GenerationThe YMCA headquarters is located in Paris, France. Raclette is a traditional Swiss dish, made with melted cheese served with potatoes, pickles, and other accompaniments. It is not a traditional French dish, so it is unlikely to be found in Paris. So the answer is NoIteration 2Retrieved Knowledge(1) Title: Raclette Context: Raclette is a Swiss dish, also popular in France, based on heating cheese and scraping off the melted part …(2) Title: YMCA Building (Racine, Wisconsin) Context: … Since the 50’s, headquarters of the publishing house and the bookstore are located in 11 rue de la Montagne Sainte-Geneviève 75005 Paris …GenerationYMCA headquarters is located in Paris, France. Raclette is a dish native to parts of Switzerland, but it is also popular in France. So it is likely that Raclette can be found in Paris. So the answer is Yes

*Table 7:  Two examples demonstrating retrieval-generation synergy.
We underline generated phrases that help retrieve relevant knowledge which successfully corrects factual errors (in red) in the second iteration.
Irrelevant retrieved paragraphs are not shown in the table for brevity.*

##### Generation Benefits Retrieval Adaptation

To investigate how LLM outputs can be leveraged for retrieval adaptation, we experimented on HotPotQA and Feverous.
Specifically, on each dataset, we sampled 9,000 random questions from the train set for training, and 1,000 for validation.
We applied Iter-RetGen for one iteration, and used the model outputs $y_{1}$ for retrieval adaptation as in Section [3.4](#S3.SS4 "3.4 Generation-Augmented Retrieval Adaptation ‣ 3 Iterative Retrieval-Generation Synergy ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy").
We used TART *Asai et al. ([2022](#bib.bib2 ""))* as the re-ranker, and distilled knowledge from TART to the dense retriever for no more than 1,000 steps.
Batch size was 32 and learning rate was 1e-5.
We used the retriever checkpoint with the lowest distillation loss.

As shown by Table [4](#S4.T4 "Table 4 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), retrieval adaptation enables Iter-RetGen to achieve significantly higher Acc† with fewer iterations.
We also demonstrated the benefits of using $y_{1}$ for adaptation by showing its improvements over a variant
which only differs in that the re-ranker has no access to $y_{1}$;
the training objective of this variant can be obtained by removing all $y_{1}$ notations in Eq. [3](#S3.E3 "In Distillation ‣ 3.4 Generation-Augmented Retrieval Adaptation ‣ 3 Iterative Retrieval-Generation Synergy ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy").

### 4.6 Ablation Study

#### 4.6.1 Generation Augments Retrieval

Table [6](#S4.T6 "Table 6 ‣ Acc† is a Reliable Metric ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy") shows the answer recall of retrieval in different iterations.
The first iteration uses only the questions for retrieval and suffers from low answer recall.
In the second iteration, retrieval, augmented with the LLM output from the first iteration, achieves significantly higher recall, indicating that LLM generations can help bridge the semantic gaps between complex questions and their supporting knowledge.
However, performance quickly hits a plateau afterwards.

#### 4.6.2 Iter-RetGen Leverages Parametric and Non-Parametric Knowledge Better

Ideally, an LLM should flexibly utilize non-parametric knowledge or parametric knowledge depending on whether in-context non-parametric knowledge is relevant or not.
Table [5](#S4.T5 "Table 5 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy") presents performance breakdowns on different subsets of questions for investigation.
We considered the ability of CoT to answer a question correctly without retrieval as a proxy for assessing an LLM’s capability to answer the question using its parametric knowledge.
Compared with Self-Ask, Iter-RetGen tends to be significantly better at preserving the LLM’s performance on questions that the LLM can solve using CoT without retrieval, while being competitive on the complementary subset.
This may be because the structural constraints from Self-Ask makes an LLM over-sensitive to the precision and comprehensiveness of follow-up question generation and answering, and Self-Ask is also incapable of processing all retrieved knowledge as a whole, thus reducing the LLM’s flexibility in solving a question.
Moreover, Iter-RetGen consistently outperforms Self-Ask by a large margin, regardless of whether the in-context non-parametric knowledge mentions the answers or not.
This indicates that when the in-context non-parametric knowledge is irrelevant or incomplete, Iter-RetGen exploits parametric knowledge better than Self-Ask.

### 4.7 Error Analysis

On HotPotQA, we manually analyzed 20 random cases where Iter-RetGen ($T\=2$) fails.
25% of predictions are false negatives.
On 10% of cases, Iter-RetGen retrieves all necessary information but fails to perform correct reasoning.
The remaining 65% of error cases are related with retrieval, on 76.9% of which, retrieval is misled by completely wrong reasoning from the first iteration, while on the other cases, reasoning in the first iteration is partially correct, but the retriever fails to retrieve the missing pieces in the second iteration.
We also observed that, in the first iteration, reasoning can be negatively affected by noisy and possibly distractive knowledge retrieved using only the questions as the queries.

5 Case Study
------------

Table [7](#S4.T7 "Table 7 ‣ Acc† is a Reliable Metric ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy") demonstrates retrieval-generation synergy with two examples from HotPotQA and StrategyQA, respectively.
In the first iteration, as both questions need multi-hop reasoning, the retriever fails to retrieve all supporting knowledge using only the questions.
Despite being affected by distractive retrieved knowledge (the capacity of a different arena in the example from HotPotQA) and showing imperfect parametric knowledge (the generated statement that Raclette is unlikely to be found in Paris in the example from StrategyQA) in the first iteration, the LLM generates phrases that help retrieve relevant knowledge in the second iteration, and successfully corrects its outputs.

6 Conclusion
------------

We demonstrate the effectiveness of Iter-RetGen in answering questions with complex information needs.
Despite simple, Iter-RetGen outperforms retrieval-augmented methods that have a more complex workflow, which we believe could serve as a strong baseline for future research on retrieval-augmented generation.
We also show that generation-augmented retrieval adaptation can further improve the performance of Iter-RetGen while also reducing overheads.

Limitations
-----------

In this work, we propose to enhance retrieval-augmented large language models with Iter-RetGen which synergizes retrieval and generation in an iterative manner, and demonstrates strong performance compared to more structured prompting techniques such as Self-Ask.
However, it’s worth noting that our experiments utilized a fixed black-box large language model, which may not have been equally optimized for various forms of prompting.
It would be intriguing to investigate the potential of prompting-specific (gradient-based) optimization in pushing the limits further.
This could involve enabling a large language model to leverage parametric and non-parametric knowledge more flexibly and effectively.
By exploring this avenue, we may uncover new insights and advancements in the field.
Furthermore, our experiments did not cover long-form generation which would probably benefit from more fine-grained retrieval than Iter-RetGen does in this work.
We acknowledge that this area warrants further exploration, and we leave it for future work.

Acknowledgements
----------------

Zhihong Shao and Minlie Huang were supported by the National Science Foundation for Distinguished Young Scholars (with No. 62125604) and the NSFC projects (Key project with No. 61936010).
They were also supported by the Guoqiang Institute of Tsinghua University, with Grant No. 2020GQG0005.

References
----------

* Aly et al. (2021)Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas
Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021.[FEVEROUS: fact extraction and verification over unstructured and
structured information](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/68d30a9594728bc39aa24be94b319d21-Abstract-round1.html "").In *Proceedings of the Neural Information Processing Systems
Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021,
December 2021, virtual*.
* Asai et al. (2022)Akari Asai, Timo Schick, Patrick S. H. Lewis, Xilun Chen, Gautier Izacard,
Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2022.[Task-aware
retrieval with instructions](https://doi.org/10.48550/arXiv.2211.09260 "").*CoRR*, abs/2211.09260.
* Brown et al. (2020)Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom
Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott
Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.[Language models are few-shot
learners](http://arxiv.org/abs/2005.14165 "").*CoRR*, abs/2005.14165.
* Chen et al. (2021)Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé
de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov,
Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick
Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian,
Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings,
Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss,
William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor
Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher
Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa,
Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter
Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and
Wojciech Zaremba. 2021.[Evaluating large language
models trained on code](http://arxiv.org/abs/2107.03374 "").*CoRR*, abs/2107.03374.
* Fan et al. (2019)Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and
Michael Auli. 2019.[ELI5: long form
question answering](https://doi.org/10.18653/v1/p19-1346 "").In *Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers*, pages 3558–3567. Association for Computational
Linguistics.
* Gao et al. (2022a)Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022a.[Precise zero-shot
dense retrieval without relevance labels](https://doi.org/10.48550/arXiv.2212.10496 "").*CoRR*, abs/2212.10496.
* Gao et al. (2022b)Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2022b.[PAL:
program-aided language models](https://doi.org/10.48550/arXiv.2211.10435 "").*CoRR*, abs/2211.10435.
* Geva et al. (2021)Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan
Berant. 2021.[Did aristotle use a
laptop? A question answering benchmark with implicit reasoning strategies](https://doi.org/10.1162/tacl_a_00370 "").*Trans. Assoc. Comput. Linguistics*, 9:346–361.
* Gou et al. (2023)Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and
Weizhu Chen. 2023.[Critic: Large language
models can self-correct with tool-interactive critiquing](http://arxiv.org/abs/2305.11738 "").
* Ho et al. (2020)Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.[Constructing A multi-hop QA dataset for comprehensive evaluation of
reasoning steps](https://doi.org/10.18653/v1/2020.coling-main.580 "").In *Proceedings of the 28th International Conference on
Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December
8-13, 2020*, pages 6609–6625. International Committee on Computational
Linguistics.
* Izacard et al. (2022a)Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr
Bojanowski, Armand Joulin, and Edouard Grave. 2022a.[Unsupervised
dense information retrieval with contrastive learning](https://openreview.net/forum?id=jKN1pXi7b0 "").*Trans. Mach. Learn. Res.*, 2022.
* Izacard and Grave (2021)Gautier Izacard and Edouard Grave. 2021.[Leveraging
passage retrieval with generative models for open domain question answering](https://doi.org/10.18653/v1/2021.eacl-main.74 "").In *Proceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics: Main Volume, EACL 2021,
Online, April 19 - 23, 2021*, pages 874–880. Association for Computational
Linguistics.
* Izacard et al. (2022b)Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio
Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022b.[Few-shot learning
with retrieval augmented language models](https://doi.org/10.48550/arXiv.2208.03299 "").*CoRR*, abs/2208.03299.
* Jiang et al. (2023)Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane
Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023.[Active retrieval
augmented generation](https://doi.org/10.48550/arXiv.2305.06983 "").*CoRR*, abs/2305.06983.
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.[Dense
passage retrieval for open-domain question answering](https://doi.org/10.18653/v1/2020.emnlp-main.550 "").In *Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*,
pages 6769–6781. Association for Computational Linguistics.
* Khattab et al. (2022)Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang,
Christopher Potts, and Matei Zaharia. 2022.[Demonstrate-search-predict: Composing retrieval and language models for
knowledge-intensive NLP](https://doi.org/10.48550/arXiv.2212.14024 "").*CoRR*, abs/2212.14024.
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins,
Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob
Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey,
Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.
2019.[Natural questions: a
benchmark for question answering research](https://doi.org/10.1162/tacl_a_00276 "").*Trans. Assoc. Comput. Linguistics*, 7:452–466.
* Mao et al. (2021)Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han,
and Weizhu Chen. 2021.[Generation-augmented retrieval for open-domain question answering](https://doi.org/10.18653/v1/2021.acl-long.316 "").In *Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers),
Virtual Event, August 1-6, 2021*, pages 4089–4100. Association for
Computational Linguistics.
* Mialon et al. (2023)Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos
Nalmpantis, Ramakanth Pasunuru, Roberta Raileanu, Baptiste Rozière,
Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun,
and Thomas Scialom. 2023.[Augmented language
models: a survey](https://doi.org/10.48550/arXiv.2302.07842 "").*CoRR*, abs/2302.07842.
* Nakano et al. (2021)Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders,
Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew
Knight, Benjamin Chess, and John Schulman. 2021.[Webgpt: Browser-assisted
question-answering with human feedback](http://arxiv.org/abs/2112.09332 "").*CoRR*, abs/2112.09332.
* OpenAI (2023)OpenAI. 2023.[GPT-4 technical
report](https://doi.org/10.48550/arXiv.2303.08774 "").*CoRR*, abs/2303.08774.
* Ouyang et al. (2022)Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda
Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022.[Training language models to follow instructions with human feedback](http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html "").In *NeurIPS*.
* Press et al. (2022)Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2022.[Measuring and
narrowing the compositionality gap in language models](https://doi.org/10.48550/arXiv.2210.03350 "").*CoRR*, abs/2210.03350.
* Scialom et al. (2022)Thomas Scialom, Tuhin Chakrabarty, and Smaranda Muresan. 2022.[Continual-t0:
Progressively instructing 50+ tasks to language models without forgetting](https://doi.org/10.48550/arXiv.2205.12393 "").*CoRR*, abs/2205.12393.
* Shao et al. (2023)Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen.
2023.[Synthetic
prompting: Generating chain-of-thought demonstrations for large language
models](https://doi.org/10.48550/arXiv.2302.00618 "").*CoRR*, abs/2302.00618.
* Shao and Huang (2022)Zhihong Shao and Minlie Huang. 2022.[Answering
open-domain multi-answer questions via a recall-then-verify framework](https://doi.org/10.18653/v1/2022.acl-long.128 "").In *Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin,
Ireland, May 22-27, 2022*, pages 1825–1838. Association for Computational
Linguistics.
* Shi et al. (2023)Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis,
Luke Zettlemoyer, and Wen-tau Yih. 2023.[REPLUG:
retrieval-augmented black-box language models](https://doi.org/10.48550/arXiv.2301.12652 "").*CoRR*, abs/2301.12652.
* Thorne et al. (2018)James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
2018.[FEVER: a large-scale
dataset for fact extraction and VERification](https://doi.org/10.18653/v1/N18-1074 "").In *Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans,
Louisiana. Association for Computational Linguistics.
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine
Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale,
Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem
Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar
Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa,
Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux,
Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier
Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew
Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan
Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang,
Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan
Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang,
Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom.
2023.[Llama 2: Open
foundation and fine-tuned chat models](https://doi.org/10.48550/arXiv.2307.09288 "").*CoRR*, abs/2307.09288.
* Trivedi et al. (2022a)Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022a.[Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step
questions](https://doi.org/10.48550/arXiv.2212.10509 "").*CoRR*, abs/2212.10509.
* Trivedi et al. (2022b)Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022b.[MuSiQue: Multihop
questions via single-hop question composition](https://doi.org/10.1162/tacl_a_00475 "").*Trans. Assoc. Comput. Linguistics*, 10:539–554.
* Wei et al. (2022)Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and
Denny Zhou. 2022.[Chain of thought prompting
elicits reasoning in large language models](http://arxiv.org/abs/2201.11903 "").*CoRR*, abs/2201.11903.
* Yang et al. (2018)Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018.[Hotpotqa: A dataset
for diverse, explainable multi-hop question answering](https://doi.org/10.18653/v1/d18-1259 "").In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, Brussels, Belgium, October 31 - November 4,
2018*, pages 2369–2380. Association for Computational Linguistics.
* Yao et al. (2022)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022.[React: Synergizing
reasoning and acting in language models](https://doi.org/10.48550/arXiv.2210.03629 "").*CoRR*, abs/2210.03629.
* Yoran et al. (2023)Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan
Berant. 2023.[Answering
questions by meta-reasoning over multiple chains of thought](https://doi.org/10.48550/arXiv.2304.13007 "").*CoRR*, abs/2304.13007.
* Zhang et al. (2023)Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang
Lou, and Weizhu Chen. 2023.[Repocoder:
Repository-level code completion through iterative retrieval and generation](https://doi.org/10.48550/arXiv.2303.12570 "").*CoRR*, abs/2303.12570.

Appendix A Experiments Using Llama-2
-------------------------------------

To demonstrate the effectiveness of Iter-RetGen on open-source models, we replaced the generation model text-davinci-003 in Table [2](#S4.T2 "Table 2 ‣ 4.5 Main Results ‣ 4 Experiments ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy") with Llama-2 models *Touvron et al. ([2023](#bib.bib29 ""))*, and re-ran the evaluation.
As shown in Table [8](#A1.T8 "Table 8 ‣ Appendix A Experiments Using Llama-2 ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Iter-RetGen consistently outperforms all baselines significantly.

ModelLlama-2-13BLlama-2-70BDatasetHotPotQA2WikiMultiHopQAStrategyQAHotPotQA2WikiMultiHopQAStrategyQAWithout RetrievalDirect36.431.660.547.239.072.7CoT43.033.263.755.246.072.7With RetrievalDirect51.838.663.358.645.173.3ReAct36.027.561.542.636.869.5Self-Ask45.838.563.358.453.271.7Iter-RetGen 153.844.662.864.455.174.8Iter-RetGen 257.848.067.267.857.976.6

*Table 8:  Experiments using the open-source Llama-2 models.
We used Acc† as the evaluation metric, i.e., to evaluate the accuracy of model outputs with text-davinci-003.*

Appendix B Few-Shot Prompts
----------------------------

In this section, we present all few-shot prompts used in our experiments. We replace retrieved paragraphs with the placeholder {Knowledge} for brevity. CoT prompting shares the same in-context demonstrations with Iter-RetGen, except that it is not augmented with retrieval.

### B.1 HotPotQA

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [9](#A2.T9 "Table 9 ‣ B.1 HotPotQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [10](#A2.T10 "Table 10 ‣ B.1 HotPotQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [11](#A2.T11 "Table 11 ‣ B.1 HotPotQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [12](#A2.T12 "Table 12 ‣ B.1 HotPotQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?The answer is Little Richard{Knowledge}Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?The answer is Chinua Achebe{Knowledge}Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?The answer is 1979

*Table 9: 3-Shot Demonstrations for Direct Prompting on HotPotQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?Are follow up questions needed here: Yes.Follow up: Who worked with Modern Records?{Knowledge}Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?{Knowledge}Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no.Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?{Knowledge}Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.So the final answer is: Little Richard#Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?Are follow up questions needed here: Yes.Follow up: What jobs did Chinua Achebe have?{Knowledge}Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.Follow up: What jobs did Rachel Carson have?{Knowledge}Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.Follow up: Did Chinua Achebe have more jobs than Rachel Carson?{Knowledge}Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs.So the final answer is: Chinua Achebe#Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?Are follow up questions needed here: Yes.Follow up: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?{Knowledge}Intermediate answer: Gangsta BooFollow up: In which year was Gangsta Boo born?{Knowledge}Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979.So the final answer is: 1979

*Table 10: 3-Shot Demonstrations for ReAct on HotPotQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#{Knowledge}Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?Are follow up questions needed here: Yes.Follow up: Who worked with Modern Records?Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no.Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.So the final answer is: Little Richard#{Knowledge}Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?Are follow up questions needed here: Yes.Follow up: What jobs did Chinua Achebe have?Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.Follow up: What jobs did Rachel Carson have?Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.Follow up: Did Chinua Achebe have more jobs than Rachel Carson?Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs.So the final answer is: Chinua Achebe#{Knowledge}Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?Are follow up questions needed here: Yes.Follow up: Which American rapper is featured by Remember Me Ballin’, a CD single by Indo G?Intermediate answer: Gangsta BooFollow up: In which year was Gangsta Boo born?Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979.So the final answer is: 1979

*Table 11: 3-Shot Demonstrations for Self-Ask on HotPotQA.*

{Knowledge}Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?Let’s think step by step.Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter.So the answer is Little Richard{Knowledge}Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?Let’s think step by step.Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson.So the answer is Chinua Achebe{Knowledge}Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?Let’s think step by step.Remember Me Ballin’ is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell’s stage name, who was born in August 7, 1979, and is an American rapper.So the answer is 1979

*Table 12: 3-Shot Demonstrations for Iter-RetGen on HotPotQA.*

### B.2 2WikiMultiHopQA

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [13](#A2.T13 "Table 13 ‣ B.2 2WikiMultiHopQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [14](#A2.T14 "Table 14 ‣ B.2 2WikiMultiHopQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [15](#A2.T15 "Table 15 ‣ B.2 2WikiMultiHopQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [16](#A2.T16 "Table 16 ‣ B.2 2WikiMultiHopQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?The answer is The Mask Of Fu Manchu{Knowledge}Question: When did John V, Prince Of Anhalt-Zerbst’s father die?The answer is 12 June 1516{Knowledge}Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?The answer is El Extrano Viaje

*Table 13: 3-Shot Demonstrations for Direct Prompting on 2WikiMultiHopQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?Are follow up questions needed here: Yes.Follow up: When did Blind Shaft come out?{Knowledge}Intermediate answer: Blind Shaft came out in 2003.Follow up: When did The Mask Of Fu Manchu come out?{Knowledge}Intermediate answer: The Mask Of Fu Manchu came out in 1932.So the final answer is: The Mask Of Fu Manchu#Question: When did John V, Prince Of Anhalt-Zerbst’s father die?Are follow up questions needed here: Yes.Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?{Knowledge}Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.Follow up: When did Ernest I, Prince of Anhalt-Dessau die?{Knowledge}Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.So the final answer is: 12 June 1516#Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?Are follow up questions needed here: Yes.Follow up: Who is the director of El Extrano Viaje?{Knowledge}Intermediate answer: The director of El Extrano Viaje is Fernando Fernan Gomez.Follow up: Who is the director of Love in Pawn?{Knowledge}Intermediate answer: The director of Love in Pawn is Charles Saunders.Follow up: When was Fernando Fernan Gomez born?{Knowledge}Intermediate answer: Fernando Fernan Gomez was born on 28 August 1921.Follow up: When was Charles Saunders (director) born?{Knowledge}Intermediate answer: Charles Saunders was born on 8 April 1904.So the final answer is: El Extrano Viaje

*Table 14: 3-Shot Demonstrations for ReAct on 2WikiMultiHopQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#{Knowledge}Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?Are follow up questions needed here: Yes.Follow up: When did Blind Shaft come out?Intermediate answer: Blind Shaft came out in 2003.Follow up: When did The Mask Of Fu Manchu come out?Intermediate answer: The Mask Of Fu Manchu came out in 1932.So the final answer is: The Mask Of Fu Manchu#{Knowledge}Question: When did John V, Prince Of Anhalt-Zerbst’s father die?Are follow up questions needed here: Yes.Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.Follow up: When did Ernest I, Prince of Anhalt-Dessau die?Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.So the final answer is: 12 June 1516#{Knowledge}Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?Are follow up questions needed here: Yes.Follow up: Who is the director of El Extrano Viaje?Intermediate answer: The director of El Extrano Viaje is Fernando Fernan Gomez.Follow up: Who is the director of Love in Pawn?Intermediate answer: The director of Love in Pawn is Charles Saunders.Follow up: When was Fernando Fernan Gomez born?Intermediate answer: Fernando Fernan Gomez was born on 28 August 1921.Follow up: When was Charles Saunders (director) born?Intermediate answer: Charles Saunders was born on 8 April 1904.So the final answer is: El Extrano Viaje

*Table 15: 3-Shot Demonstrations for Self-Ask on 2WikiMultiHopQA.*

{Knowledge}Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?Let’s think step by step.Blind Shaft is a 2003 film, while The Mask Of Fu Manchu opened in New York on December 2, 1932. 2003 comes after 1932. Therefore, The Mask Of Fu Manchu came out earlier than Blind Shaft.So the answer is The Mask Of Fu Manchu{Knowledge}Question: When did John V, Prince Of Anhalt-Zerbst’s father die?Let’s think step by step.John was the second son of Ernest I, Prince of Anhalt-Dessau. Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.So the answer is 12 June 1516{Knowledge}Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?Let’s think step by step.The director of El Extrano Viaje is Fernando Fernan Gomez, who was born on 28 August 1921. The director of Love In Pawn is Charles Saunders, who was born on 8 April 1904. 28 August 1921 comes after 8 April 1904. Therefore, Fernando Fernan Gomez was born later than Charles Saunders.So the answer is El Extrano Viaje

*Table 16: 3-Shot Demonstrations for Iter-RetGen on 2WikiMultiHopQA.*

### B.3 MuSiQue

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [17](#A2.T17 "Table 17 ‣ B.3 MuSiQue ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [18](#A2.T18 "Table 18 ‣ B.3 MuSiQue ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [19](#A2.T19 "Table 19 ‣ B.3 MuSiQue ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [20](#A2.T20 "Table 20 ‣ B.3 MuSiQue ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: In which year did the publisher of In Cold Blood form?The answer is 2001{Knowledge}Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?The answer is John Cranley{Knowledge}Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?The answer is eastern tip

*Table 17: 3-Shot Demonstrations for Direct Prompting on MuSiQue.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#Question: In which year did the publisher of In Cold Blood form?Are follow up questions needed here: Yes.Follow up: What business published In Cold Blood?{Knowledge}Intermediate answer: In Cold Blood was published in book form by Random House.Follow up: Which year witnessed the formation of Random House?{Knowledge}Intermediate answer: Random House was form in 2001.So the final answer is: 2001#Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?Are follow up questions needed here: Yes.Follow up: In which city was The Killing of a Sacred Deer filmed{Knowledge}Intermediate answer: The Killing of a Sacred Deer was filmed in Cincinnati.Follow up: Who was in charge of Cincinnati?{Knowledge}Intermediate answer: The present Mayor of Cincinnati is John Cranley, so John Cranley is in charge.So the final answer is: John Cranley#Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?Are follow up questions needed here: Yes.Follow up: What city does Signal Hill overlook?{Knowledge}Intermediate answer: Signal Hill is a hill which overlooks the city of St. John’s.Follow up: Where on the Avalon Peninsula is St. John’s located?{Knowledge}Intermediate answer: St. John’s is located on the eastern tip of the Avalon Peninsula.So the final answer is: eastern tip

*Table 18: 3-Shot Demonstrations for ReAct on MuSiQue.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#{Knowledge}Question: In which year did the publisher of In Cold Blood form?Are follow up questions needed here: Yes.Follow up: What business published In Cold Blood?Intermediate answer: In Cold Blood was published in book form by Random House.Follow up: Which year witnessed the formation of Random House?Intermediate answer: Random House was form in 2001.So the final answer is: 2001#{Knowledge}Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?Are follow up questions needed here: Yes.Follow up: In which city was The Killing of a Sacred Deer filmedIntermediate answer: The Killing of a Sacred Deer was filmed in Cincinnati.Follow up: Who was in charge of Cincinnati?Intermediate answer: The present Mayor of Cincinnati is John Cranley, so John Cranley is in charge.So the final answer is: John Cranley#{Knowledge}Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?Are follow up questions needed here: Yes.Follow up: What city does Signal Hill overlook?Intermediate answer: Signal Hill is a hill which overlooks the city of St. John’s.Follow up: Where on the Avalon Peninsula is St. John’s located?Intermediate answer: St. John’s is located on the eastern tip of the Avalon Peninsula.So the final answer is: eastern tip

*Table 19: 3-Shot Demonstrations for Self-Ask on MuSiQue.*

{Knowledge}Question: In which year did the publisher of In Cold Blood form?Let’s think step by step.In Cold Blood was first published in book form by Random House. Random House was form in 2001.So the answer is 2001{Knowledge}Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?Let’s think step by step.The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley. Therefore, John Cranley is in charge of the city.So the answer is John Cranley{Knowledge}Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?Let’s think step by step.Signal Hill is a hill which overlooks the city of St. John’s. St. John’s is located on the eastern tip of the Avalon Peninsula.So the answer is eastern tip

*Table 20: 3-Shot Demonstrations for Iter-RetGen on MuSiQue.*

### B.4 Bamboogle

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [21](#A2.T21 "Table 21 ‣ B.4 Bamboogle ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [22](#A2.T22 "Table 22 ‣ B.4 Bamboogle ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [23](#A2.T23 "Table 23 ‣ B.4 Bamboogle ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [24](#A2.T24 "Table 24 ‣ B.4 Bamboogle ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: When did the first prime minister of the Russian Empire come into office?The answer is 1905-11-06 00:00:00{Knowledge}Question: The most populous city in Punjab is how large (area wise)?The answer is 310 square kilometers{Knowledge}Question: What is the capital of the country where yoga originated?The answer is New Delhi

*Table 21: 3-Shot Demonstrations for Direct Prompting on Bamboogle.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#Question: When did the first prime minister of the Russian Empire come into office?Are follow up questions needed here: Yes.Follow up: Who is the first prime minister of the Russian Empire?{Knowledge}Intermediate answer: Sergei WitteFollow up: When did Sergei Witte come into office?{Knowledge}Intermediate answer: Sergei Witte was appointed on 6 November 1905.So the final answer is: 1905-11-06 00:00:00#Question: The most populous city in Punjab is how large (area wise)?Are follow up questions needed here: Yes.Follow up: What is the most populous city in Punjab?{Knowledge}Intermediate answer: Ludhiana is the most populous and largest city in Punjab.Follow up: How large is Ludhiana, the most populous city in Punjab?{Knowledge}Intermediate answer: The area of Ludhiana is over 310 km2.So the final answer is: 310 square kilometers#Question: What is the capital of the country where yoga originated?Are follow up questions needed here: Yes.Follow up: Which country was yoga originated?{Knowledge}Intermediate answer: There is no consensus on yoga’s origin. Suggested origins include India.Follow up: What is the capital of India?{Knowledge}Intermediate answer: The current capital of India is New Delhi.So the final answer is: New Delhi

*Table 22: 3-Shot Demonstrations for ReAct on Bamboogle.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly.#{Knowledge}Question: When did the first prime minister of the Russian Empire come into office?Are follow up questions needed here: Yes.Follow up: Who is the first prime minister of the Russian Empire?Intermediate answer: Sergei WitteFollow up: When did Sergei Witte come into office?Intermediate answer: Sergei Witte was appointed on 6 November 1905.So the final answer is: 1905-11-06 00:00:00#{Knowledge}Question: The most populous city in Punjab is how large (area wise)?Are follow up questions needed here: Yes.Follow up: What is the most populous city in Punjab?Intermediate answer: Ludhiana is the most populous and largest city in Punjab.Follow up: How large is Ludhiana, the most populous city in Punjab?Intermediate answer: The area of Ludhiana is over 310 km2.So the final answer is: 310 square kilometers#{Knowledge}Question: What is the capital of the country where yoga originated?Are follow up questions needed here: Yes.Follow up: Which country was yoga originated?Intermediate answer: There is no consensus on yoga’s origin. Suggested origins include India.Follow up: What is the capital of India?Intermediate answer: The current capital of India is New Delhi.So the final answer is: New Delhi

*Table 23: 3-Shot Demonstrations for Self-Ask on Bamboogle.*

{Knowledge}Question: When did the first prime minister of the Russian Empire come into office?Let’s think step by step.The first prime minister of the Russian Empire was Count Sergei Witte. Sergei Witte was appointed on 6 November 1905.So the answer is 1905-11-06 00:00:00{Knowledge}Question: The most populous city in Punjab is how large (area wise)?Let’s think step by step.Ludhiana is the most populous and the largest city in the Indian state of Punjab. The city has an area of over 310 km2.So the answer is 310 square kilometers{Knowledge}Question: What is the capital of the country where yoga originated?Let’s think step by step.Suggested origins include pre-Vedic Eastern states of India. The current capital of India is New Delhi.So the answer is New Delhi

*Table 24: 3-Shot Demonstrations for Iter-RetGen on Bamboogle.*

### B.5 Feverous

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [25](#A2.T25 "Table 25 ‣ B.5 Feverous ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [26](#A2.T26 "Table 26 ‣ B.5 Feverous ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [27](#A2.T27 "Table 27 ‣ B.5 Feverous ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [28](#A2.T28 "Table 28 ‣ B.5 Feverous ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: Is it true that Belgrade Race is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia through history, past winners includes Brahim Lahlafi (1st edition), Philip Mosima (3rd) and Josphat Menjo (6th)?The answer is Yes{Knowledge}Question: Is it true that Based on the same platform as the Chevrolet Sail, the Baojun 310 was launched on 2017 Beijing Auto Show where the price ranges from 36.800 yuan to 60.800 yuan?The answer is No{Knowledge}Question: Is it true that Florida International University pedestrian bridge collapse was funded with a $19.4 million Transportation Investment Generating Economic Recovery grant from the United States Department of Transportation in 2013, along with state agencies and the bridge cost $14.2 million to construct?The answer is No

*Table 25: 3-Shot Demonstrations for Direct Prompting on Feverous.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly. The final answer should always be either Yes or No, and NOTHING ELSE.#Question: Is it true that Belgrade Race is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia through history, past winners includes Brahim Lahlafi (1st edition), Philip Mosima (3rd) and Josphat Menjo (6th)?Are follow up questions needed here: Yes.Follow up: What is the Belgrade Race?{Knowledge}Intermediate answer: The Belgrade Race Through History is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia.Follow up: Has Brahim Lahlafi won Belgrade Race?{Knowledge}Intermediate answer: Yes, Brahim Lahlafi was the winner in 1996.Follow up: Has Philip Mosima won Belgrade Race?{Knowledge}Intermediate answer: Yes, Philip Mosima beat Marathon world record and won in 1998Follow up: Has Josphat Menjo won Belgrade Race?{Knowledge}Intermediate answer: Yes, Josphat Menjo broke the meet record and won the competition.So the final answer is: Yes#Question: Is it true that Based on the same platform as the Chevrolet Sail, the Baojun 310 was launched on 2017 Beijing Auto Show where the price ranges from 36.800 yuan to 60.800 yuan?Are follow up questions needed here: Yes.Follow up: When and where was the Baojun 310 launched?{Knowledge}Intermediate answer: The Baojun 310 was launched on 2016 Beijing Auto Show, not 2017 Beijing Auto Show.So the final answer is: No#Question: Is it true that Florida International University pedestrian bridge collapse was funded with a $19.4 million Transportation Investment Generating Economic Recovery grant from the United States Department of Transportation in 2013, along with state agencies and the bridge cost $14.2 million to construct?Are follow up questions needed here: Yes.Follow up: How was Florida International University pedestrian bridge collapse funded?{Knowledge}Intermediate answer: Florida International University pedestrian bridge was a $14.2 million project funded with a $19.4 million Transportation Investment Generating Economic Recovery (TIGER) grant from the United States Department of Transportation in 2013, along with state agencies, which is consistent with facts in the question.Follow up: How much did it cost to construct Florida International University pedestrian bridge?{Knowledge}Intermediate answer: The bridge cost $9 million to construct, not $14.2 million.So the final answer is: No

*Table 26: 3-Shot Demonstrations for ReAct on Feverous.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly. The final answer should always be either Yes or No, and NOTHING ELSE.#{Knowledge}Question: Is it true that Belgrade Race is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia through history, past winners includes Brahim Lahlafi (1st edition), Philip Mosima (3rd) and Josphat Menjo (6th)?Are follow up questions needed here: Yes.Follow up: What is the Belgrade Race?Intermediate answer: The Belgrade Race Through History is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia.Follow up: Has Brahim Lahlafi won Belgrade Race?Intermediate answer: Yes, Brahim Lahlafi was the winner in 1996.Follow up: Has Philip Mosima won Belgrade Race?Intermediate answer: Yes, Philip Mosima beat Marathon world record and won in 1998Follow up: Has Josphat Menjo won Belgrade Race?Intermediate answer: Yes, Josphat Menjo broke the meet record and won the competition.So the final answer is: Yes#{Knowledge}Question: Is it true that Based on the same platform as the Chevrolet Sail, the Baojun 310 was launched on 2017 Beijing Auto Show where the price ranges from 36.800 yuan to 60.800 yuan?Are follow up questions needed here: Yes.Follow up: When and where was the Baojun 310 launched?Intermediate answer: The Baojun 310 was launched on 2016 Beijing Auto Show, not 2017 Beijing Auto Show.So the final answer is: No#{Knowledge}Question: Is it true that Florida International University pedestrian bridge collapse was funded with a $19.4 million Transportation Investment Generating Economic Recovery grant from the United States Department of Transportation in 2013, along with state agencies and the bridge cost $14.2 million to construct?Are follow up questions needed here: Yes.Follow up: How was Florida International University pedestrian bridge collapse funded?Intermediate answer: Florida International University pedestrian bridge was a $14.2 million project funded with a $19.4 million Transportation Investment Generating Economic Recovery (TIGER) grant from the United States Department of Transportation in 2013, along with state agencies, which is consistent with facts in the question.Follow up: How much did it cost to construct Florida International University pedestrian bridge?Intermediate answer: The bridge cost $9 million to construct, not $14.2 million.So the final answer is: No

*Table 27: 3-Shot Demonstrations for Self-Ask on Feverous.*

You are required to verify facts in the following questions. The final answer to a question should always be either Yes or No, and NOTHING ELSE.{Knowledge}Question: Is it true that Belgrade Race is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia through history, past winners includes Brahim Lahlafi (1st edition), Philip Mosima (3rd) and Josphat Menjo (6th)?Let’s think step by step.I need to verify facts in the question. The Belgrade Race Through History is an annual men’s footrace of around 6 kilometres (5834 metres) that is held in Belgrade, Serbia. In 1996 Brahim Lahlafi was the winner of the competition. Philip Mosima won the competition in 1998, and beat Marathon world record holder Paul Tergat. Josphat Menjo also won the competition and broke the meet record. Therefore, past winners include Brahim Lahlafi, Philip Mosima and Josphat Menjo. All facts are verified.So the answer is Yes{Knowledge}Question: Is it true that Based on the same platform as the Chevrolet Sail, the Baojun 310 was launched on 2017 Beijing Auto Show where the price ranges from 36.800 yuan to 60.800 yuan?Let’s think step by step.I need to verify facts in the question. The Baojun 310 was indeed based on the same platform as the Chevrolet Sail. The Baojun 310 was launched on 2016 Beijing Auto Show, not 2017 Beijing Auto Show.So the answer is No{Knowledge}Question: Is it true that Florida International University pedestrian bridge collapse was funded with a $19.4 million Transportation Investment Generating Economic Recovery grant from the United States Department of Transportation in 2013, along with state agencies and the bridge cost $14.2 million to construct?Let’s think step by step.I need to verify facts in the question. Florida International University pedestrian bridge was a $14.2 million project funded with a $19.4 million Transportation Investment Generating Economic Recovery (TIGER) grant from the United States Department of Transportation in 2013, along with state agencies. The Bridge cost $8 million to construct, not $14.2 million.So the answer is No

*Table 28: 3-Shot Demonstrations for Iter-RetGen on Feverous.*

### B.6 StrategyQA

Prompts for Direct Prompting, ReAct, Self-Ask, and Iter-RetGen are presented in Table [29](#A2.T29 "Table 29 ‣ B.6 StrategyQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [30](#A2.T30 "Table 30 ‣ B.6 StrategyQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), Table [31](#A2.T31 "Table 31 ‣ B.6 StrategyQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), and Table [32](#A2.T32 "Table 32 ‣ B.6 StrategyQA ‣ Appendix B Few-Shot Prompts ‣ Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy"), respectively.

{Knowledge}Question: Is it common to see frost during some college commencements?The answer is Yes{Knowledge}Question: Could a llama birth twice during War in Vietnam (1945-46)?The answer is No{Knowledge}Question: Would Richard Dawkins hypothetically refuse an offering of the Last rites?The answer is Yes

*Table 29: 3-Shot Demonstrations for Direct Prompting on StrategyQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly. The final answer should always be either Yes or No, and NOTHING ELSE.#Question: Is it common to see frost during some college commencements?Are follow up questions needed here: Yes.Follow up: What seasons can you expect to see frost?{Knowledge}Intermediate answer: Frost is common during the winter.Follow up: When is college commencement?{Knowledge}Intermediate answer: College commencement ceremonies often happen during the months of December, May, June.Follow up: Do any of the months December, May, June occur during the Winter?{Knowledge}Intermediate answer: December is in the winter.So the final answer is: Yes#Question: Could a llama birth twice during War in Vietnam (1945-46)?Are follow up questions needed here: Yes.Follow up: How long did the vietnam war (1945-1946) last?{Knowledge}Intermediate answer: The War in Vietnam (1945-46) lasted around 6 months.Follow up: How long is the llama gestation period?{Knowledge}Intermediate answer: The gestation period for a llama is 11.5 months.Follow up: What is 2 times 11.5?{Knowledge}Intermediate answer: 23, which is longer than 6.So the final answer is: No#Question: Would Richard Dawkins hypothetically refuse an offering of the Last rites?Are follow up questions needed here: Yes.Follow up: What are the last Rites?{Knowledge}Intermediate answer: The Last rites, in Catholicism, are the last prayers and ministrations given to an individual of the faith, when possible, shortly before death.Follow up: What are Richard Dawkins religious beliefs?{Knowledge}Intermediate answer: Richard Dawkins is known as an outspoken atheist, well known for his criticism of creationism and intelligent design.Follow up: Would an atheist participate in Catholics prayers?{Knowledge}Intermediate answer: It is unlikely that an atheist would participate in Catholics prayers.So the final answer is: Yes

*Table 30: 3-Shot Demonstrations for ReAct on StrategyQA.*

Given the following question, answer it by providing follow up questions and intermediate answers. For each follow up question, you are given a context which is the top returned Wikipedia snippets for the question. If no follow up questions are necessary, answer the question directly. The final answer should always be either Yes or No, and NOTHING ELSE.#{Knowledge}Question: Is it common to see frost during some college commencements?Are follow up questions needed here: Yes.Follow up: What seasons can you expect to see frost?Intermediate answer: Frost is common during the winter.Follow up: When is college commencement?Intermediate answer: College commencement ceremonies often happen during the months of December, May, June.Follow up: Do any of the months December, May, June occur during the Winter?Intermediate answer: December is in the winter.So the final answer is: Yes#{Knowledge}Question: Could a llama birth twice during War in Vietnam (1945-46)?Are follow up questions needed here: Yes.Follow up: How long did the vietnam war (1945-1946) last?Intermediate answer: The War in Vietnam (1945-46) lasted around 6 months.Follow up: How long is the llama gestation period?Intermediate answer: The gestation period for a llama is 11.5 months.Follow up: What is 2 times 11.5?Intermediate answer: 23, which is longer than 6.So the final answer is: No#{Knowledge}Question: Would Richard Dawkins hypothetically refuse an offering of the Last rites?Are follow up questions needed here: Yes.Follow up: What are the last Rites?Intermediate answer: The Last rites, in Catholicism, are the last prayers and ministrations given to an individual of the faith, when possible, shortly before death.Follow up: What are Richard Dawkins religious beliefs?Intermediate answer: Richard Dawkins is known as an outspoken atheist, well known for his criticism of creationism and intelligent design.Follow up: Would an atheist participate in Catholics prayers?Intermediate answer: It is unlikely that an atheist would participate in Catholics prayers.So the final answer is: Yes

*Table 31: 3-Shot Demonstrations for Self-Ask on StrategyQA.*

You are required to answer the following questions. The final answer to a question should always be either Yes or No, and NOTHING ELSE.{Knowledge}Question: Is it common to see frost during some college commencements?Let’s think step by step.College commencement ceremonies often happen during the months of December, May, and sometimes June. Frost isn’t uncommon to see during the month of December, as it is the winter.So the answer is Yes{Knowledge}Question: Could a llama birth twice during War in Vietnam (1945-46)?Let’s think step by step.The War in Vietnam (1945-46) lasted around 6 months. The gestation period for a llama is 11 months. If a llama birth twice, the minimum time needed is 2 times 11 months, which is 22 months, longer than 6 months.So the answer is No{Knowledge}Question: Would Richard Dawkins hypothetically refuse an offering of the Last rites?Let’s think step by step.Richard Dawkins is known as an outspoken atheist, well known for his criticism of creationism and intelligent design. The Last rites, in Catholicism, are the last prayers and ministrations given to an individual of the faith, when possible, shortly before death. It is unlikely that an atheist would participate in Catholics prayers.So the answer is Yes

*Table 32: 3-Shot Demonstrations for Iter-RetGen on StrategyQA.*
