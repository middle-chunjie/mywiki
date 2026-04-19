Making Retrieval-Augmented Language  Models Robust to Irrelevant Context
=========================================================================

Ori Yoran1 Tomer Wolfson1,2 Ori Ram1 Jonathan Berant1  
1Tel Aviv University,2Allen Institute for AI  
{ori.yoran, ori.ram, joberant}@cs.tau.ac.il tomerw@allenai.org

###### Abstract

Retrieval-augmented language models (RALMs) hold promise to produce language understanding systems that are are factual, efficient, and up-to-date. An important desideratum of RALMs, is that retrieved information helps model performance when it is relevant, and does not harm performance when it is not. This is particularly important in multi-hop reasoning scenarios, where misuse of irrelevant evidence can lead to cascading errors. However, recent work has shown that retrieval augmentation can sometimes have a negative effect on performance. In this work, we present a thorough analysis on five open-domain question answering benchmarks, characterizing cases when retrieval reduces accuracy. We then propose two methods to mitigate this issue. First, a simple baseline that filters out retrieved passages that do not entail question-answer pairs according to a natural language inference (NLI) model. This is effective in preventing performance reduction, but at a cost of also discarding relevant passages. Thus, we propose a method for automatically generating data to fine-tune the language model to properly leverage retrieved passages, using a mix of relevant and irrelevant contexts at training time. We empirically show that even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones.

1 Introduction
--------------

Large Language Models (LLMs) *(Brown et al., [2020](#bib.bib3 ""); Chowdhery et al., [2022](#bib.bib7 ""); Touvron et al., [2023](#bib.bib44 ""))* are the foundation on top of which modern language systems are built. However, open-domain question answering (ODQA; *Chen et al. [2017](#bib.bib4 "")*) and other knowledge-intensive tasks *(Thorne et al., [2018](#bib.bib43 ""); Petroni et al., [2021](#bib.bib36 ""))* require vast amounts of up-to-date factual knowledge about rare entities that even very large models cannot memorize *(Roberts et al., [2020](#bib.bib39 ""); Dhingra et al., [2022](#bib.bib12 ""))*.
A dominant approach for combating this issue has been Retrieval Augmented Language Models (RALMs), which incorporate a retrieval mechanism to reduce the need for storing information in the LLM parameters *(Guu et al., [2020](#bib.bib14 ""); Lewis et al., [2020b](#bib.bib27 ""); Izacard et al., [2022](#bib.bib16 ""); Rubin \& Berant, [2023](#bib.bib40 ""))*. Furthermore, RALMs have also been shown to improve ODQA performance in an in-context setting (without any training), simply by prepending retrieved sentences to the input question *(Ram et al., [2023](#bib.bib38 ""))*. Nevertheless, retrievers are not perfect and past work has shown that noisy retrieval can negatively affect LLM performance *(Petroni et al., [2020](#bib.bib35 ""); Li et al., [2023](#bib.bib28 ""))*. For example, in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), when posed with the questions “Who is playing Jason on General Hospital?” a vanilla LLM (left) correctly answers the question while the RALM (right) is “distracted” by irrelevant context about the actor portraying Cooper, not Jason.

In this work, we analyze and improve the robustness of RALMs to noisy retrieved contexts. Our definition for *retrieval-robust LLMs* states that: (a) when relevant, the retrieved context should improve model performance; (b) when irrelevant, the retrieved context should not hurt model performance.
To this end, we present two methods for retrieval-robustness in RALMs (§[2](#S2 "2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

First, we consider a setting where we have black-box access to the LLM and cannot train it. Rather than solely relying on in-context prompting *(Brown et al., [2020](#bib.bib3 ""))*, we frame retrieval robustness as a natural language inference (NLI) problem *(Dagan et al., [2006](#bib.bib10 ""); Bowman et al., [2015](#bib.bib2 ""))*. Namely, given a question and retrieved context, an NLI model can predict whether a question-answer pair (hypothesis) is entailed by the context (premise). Building on the strong performance of recent NLI models (e.g., in detecting model hallucinations *(Honovich et al., [2022](#bib.bib15 ""))* and attributed question answering *(Bohnet et al., [2023](#bib.bib1 ""))*), we use such models to identify irrelevant contexts. When the context is labeled as irrelevant to the question-answer pair, we generate the answer using the LLM *without retrieval* as a “back-off strategy”. Our results show that this method is highly effective at identifying irrelevant contexts, but is too strict and discards relevant ones as well (§[4](#S4 "4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

We then propose a method for training RALMs to be retrieval-robust. Intuitively,
LLMs are not trained with retrieved passages, and thus brittleness to noisy retrieval is somewhat expected. Therefore, we perform an additional finetuning step that teaches the LLM to be robust to noisy contexts.
The core challenge is to generate data for finetuning, and we describe a procedure for automatically generating such data for both single-hop and multi-hop questions.
In the single-hop setting, assuming access to gold QA pairs and a retriever, we create training examples using retrieved contexts, where we can use low-ranked or random passages as noisy contexts.
In the multi-hop setting, training examples need to contain not only retrieved contexts, but also intermediate questions, answers and relevant contexts, which comprise the *question decomposition* (Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")), shown to be necessary for high performance on multi-hop questions *(Wolfson et al., [2020](#bib.bib49 ""); Press et al., [2022](#bib.bib37 ""))*. To generate decompositions to train on, we use a strong LLM, prompted for decomposition without any retrieval. Then, we can sample multiple decompositions, and use self-consistency *(Wang et al., [2023](#bib.bib46 ""))* over decompositions to identify high-quality training examples (§[3.2.3](#S3.SS2.SSS3 "3.2.3 Fine-tuned models ‣ 3.2 Models ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

To test our methods, we evaluate retrieval robustness on five ODQA benchmarks, four of which contain multi-hop questions, where the retriever is called multiple times *(Jiang et al., [2023](#bib.bib18 ""))*.
Fig.[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") shows that even with a strong retriever (top-1 Google search) incorporating the retrieved context actually *hurts* model performance on two of the benchmarks (StrategyQA, Fermi). Moreover, adding randomly-retrieved contexts dramatically decreases accuracy on all five datasets. Our analysis (§[5](#S5 "5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")) shows that irrelevant context causes a wide range of errors, which include copying irrelevant answers from the retrieved sentences and hallucinating incorrect answers and decompositions.

Our results demonstrate that finetuning LLMs to be retrieval-robust enables them to ignore irrelevant context while improving their overall accuracy (§[4](#S4 "4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).
When using a strong retriever at test time, our finetuned models outperform both models that were finetuned without retrieval, as well as untrained models prompted using in-context learning. To test robustness to *noisy context*, we evaluate QA accuracy when models are given randomly-retrieved contexts. In this setting, our finetuned models perform on par with those that were finetuned *without* retrieval, demonstrating retrieval robustness. In addition, we show that models finetuned solely with relevant contexts are far less robust than those finetuned with a mixture of relevant and irrelevant contexts.

<img src='x1.png' alt='Refer to caption' title='' width='461' height='112' />

*Figure 1: An example from NQ where retrieval augmentation causes *Llama-2-13B* to err. Augmenting with irrelevant retrieved context leads to an error (right), although the model is able to answer the question without retrieval (left).*

To summarize, our main contributions are:

* •

    We conduct a thorough analysis on the robustness of RALMs to irrelevant retrieved contexts.

* •

    We suggest to use small NLI models to identify irrelevant context and improve robustness, without updating the model parameters.

* •

    We demonstrate that training LLMs *when* to use retrieval helps make models robust to irrelevant context and improve their overall performance.111Our code, data, and models are publicly available at [https://github.com/oriyor/ret-robust](https://github.com/oriyor/ret-robust "")

<img src='x2.png' alt='Refer to caption' title='' width='415' height='179' />

*Figure 2: Accuracy for *Llama-2-13B* few-shot prompted on five QA tasks, in three settings: (a) without retrieval, (b) with top-1 retrieval from a strong search engine, and (c) with a randomly-retrieved passage. Retrieval augmentation can boost performance, but even strong retrieval hurts performance on StrategyQA and Fermi, and random contexts reduce performance dramatically.*

2 Making RALMs Robust to Irrelevant Contexts
--------------------------------------------

We now present our methods for building RALMs that are robust to irrelevant contexts. We begin by describing the common approach for incorporating evidence into RALMs. Next, we explore the potential of using an NLI model to identify irrelevant contexts. Last, we describe our procedure for finetuning models to be robust to irrelevant context.

##### In-context RALMs

Language models define a probability distribution over sequences of tokens, with *auto-regressive models* assigning a probability via next-token prediction: $p_{LM}\=\Pi_{i\=1}^{n}p_{\theta}(x_{i}|x_{<i})$, where $x_{<i}$ is the sequence of tokens preceding $x_{i}$ at each step and $\theta$ denotes the parameters of the LM. For RALMs, we follow the definition of *in-context RALMs* from *Ram et al. ([2023](#bib.bib38 ""))*, where context sentences are retrieved from a corpus $C$, and generation is conditioned on the retrieved context. Given the retrieval operation $R_{C}$, this can be formalized as $p_{\text{RALM}}\=\Pi_{i\=1}^{n}p_{\theta}(x_{i}|R_{C}(x_{<i});x_{<i})$, where $[R_{C}(x_{<i});x_{<i}]$ denotes the concatenation of the retrieved evidence with the generated sequence. Generation in LMs and RALMs can also be conditioned on additional input, which we omit for brevity.

In our setting, we focus on RALMs for ODQA.
We follow recent approaches such as Self-Ask and IR-CoT *(Press et al., [2022](#bib.bib37 ""); Trivedi et al., [2023](#bib.bib45 ""); Yoran et al., [2023](#bib.bib50 ""))*, for interleaving retrieval with multi-hop question answering (see Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")). Retrieval is performed for every intermediate question and each context is prepended to the question.
In the single-hop setting, the model has to generate the answer given a question and retrieved context.
In the multi-hop setting, the model has to generate intermediate questions and answers until arriving at the final answer and the retriever is called for the original question and after each intermediate question. Formally, $x$ in this case is the generated decomposition until an intermediate step and $R_{C}(x)$ are the retrieved contexts for all questions in $x$.

<img src='x3.png' alt='Refer to caption' title='' width='461' height='193' />

*Figure 3: Interleaving decomposition and retrieval in Self-Ask format *(Press et al., [2022](#bib.bib37 ""))*. The model generates intermediate questions and answers until generating the final answer (model generations are shown in pink). Retrieved evidence for intermediate questions is prepended at each step.*

### 2.1 Identifying Irrelevant Contexts with NLI models.

NLI models *(Dagan et al., [2006](#bib.bib10 ""); Bowman et al., [2015](#bib.bib2 ""))* classify whether a textual *hypothesis* is entailed, neutral, or contradicted given a textual *premise*.
Recent work successfully used NLI models to automatically identify hallucinations *(Honovich et al., [2022](#bib.bib15 ""))* and statement attribution *(Bohnet et al., [2023](#bib.bib1 ""))* when presented with a context and generated text. Similarly, a natural baseline is to frame irrelevant context identification as an NLI problem, by using the retrieved context only when the hypothesis (i.e., final answer and intermediate question-answer pairs; Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")) are classified as entailed by the premise (i.e., the retrieved context). We propose a simple *back-off* strategy where we generate twice, once with $p_{LM}$ and once with $p_{RALM}$, and only use the RALM if a NLI model classified all generated answers (and intermediate questions) as entailed by the retrieved evidence.

For example, in Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), the retrieved evidence “Jason Gerhardt… is an American actor… known for playing Cooper Barrett…” serves as the *premise* while the question and generated answer, “Q: Who is the actor playing Jason on general hospital? A: Steve Burton” are concatenated and serve as our *hypothesis*. As this context is irrelevant, we expect the NLI model to label the hypothesis as *contradicting*. Given a contradicting or neutral hypothesis, we will use the standard LLM without the (potentially distracting) retrieved context. For multi-hop questions (as in Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")), we additionally verify that *each* intermediate-answer pair is entailed by the retrieved evidence using all retrieved evidence as our premise and the intermediate question-answer pair as the hypothesis. For example, “Q: Who is Colonel Walter Phelps? A: Colonel Walter Phelps was an officer in the Union Army throughout the American Civil War.” for the first intermediate question in Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

### 2.2 Training Robust RALMs

As in-context RALMs are not trained to use retrieved passages, a more effective solution than post-hoc filtering (using NLI) may be to train RALMs to ignore irrelevant contexts. We are interested in testing whether training on a relatively small dataset (several hundreds of examples) would suffice.

##### Automatically Generating Training Data

Our goal is to teach RALMs to be robust to irrelevant context in an ODQA setting.
In the single-hop setting, generating training data is straightforward. Given access to a dataset of question-answer pairs ${(q,a)}$ (i.e., without contexts) and a retriever $R_{C}$, we use the retriever to augment questions with retrieved context.
To create training examples with relevant contexts, we return the top-1 context from $R_{C}$, and for irrelevant contexts, we either return a low-ranked result from $R_{C}(q)$ or a random context (i.e., $R_{C}(q^{\prime})$ for another question $q^{\prime}$).
We denote the chosen context by $r_{q}$. Then, the training dataset is defined by $D\={([r_{q};q],a)}$.

Our main challenge is generating training examples for multi-hop questions. In these questions the model generates a decomposition, consisting of intermediate questions and answers, before arriving at the final answer, while the retriever is called multiple times (Fig.[3](#S2.F3 "Figure 3 ‣ In-context RALMs ‣ 2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).
Our goal is to automatically generate retrieval-augmented decomposition steps, $D\={([r_{x};x],y)}$, where: $y$ is the correct generation for each step (i.e. the correct intermediate question, intermediate answer, or final answer); $x$ consists of the previously generated steps up to $y$; $r_{x}$ is the retrieved contexts for all steps in $x$.
Our first step to automatically generate decompositions is prompting a strong LLM without access to retrieval and to verify its answers. However, the LLM may arrive at the correct answer using an incorrect decomposition, for example in binary or comparison questions. Hence, we need to ensure the quality of generated decompositions. For multi-hop datasets which provide intermediate answers, we simply filter out generated decompositions that do not contain them.
When intermediate answer annotations are unavailable, we sample from the LLM that generated the decomposition multiple times and verify self-consistency *(Wang et al., [2023](#bib.bib46 ""))*. Further details are given in §[3.2.3](#S3.SS2.SSS3 "3.2.3 Fine-tuned models ‣ 3.2 Models ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

##### Training

We use our automatically generated data $D$ to fine-tune models for generating $y$ conditioned on $[r_{x};x]$
with standard maximum likelihood.
Since we are mostly interested in the low-data regime, we limit the number of questions in $D$ to 1,000 in the single-hop setting and 500 in the multi-hop setting (splitting multi-hop questions to multiple examples for each step), and use parameter efficient fine-tuning *(Dettmers et al., [2023](#bib.bib11 ""))*. Thus, training all our models takes no more than a few hours.
Additional experimental details are in §[3](#S3 "3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") and §[A.1](#A1.SS1 "A.1 Models ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

3 Experimental Setting
----------------------

### 3.1 Datasets

We experiment with both single- and multi-hop QA datasets. We list and give an example from each dataset in Tab.[1](#S3.T1 "Table 1 ‣ 3.1 Datasets ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"). Our QA benchmarks can be categorized based on their required reasoning skills:

* •

    Single-hop: Information-seeking questions that do not require decomposition. We use the popular Natural Questions (NQ) dataset *(Kwiatkowski et al., [2019](#bib.bib25 ""))*.

* •

    Explicit Reasoning: Multi-hop questions where reasoning is explicitly expressed in the language of the question.
    We include 2WikiMQA *(Welbl et al., [2018](#bib.bib47 ""))* and Bamboogle *(Press et al., [2022](#bib.bib37 ""))*.

* •

    Implicit Reasoning: Mutli-hop questions
    where generating reasoning steps requires common sense (implicit reasoning) *(Geva et al., [2021](#bib.bib13 ""))*. Such questions may have multiple valid reasoning chains. We evaluate on StrategyQA *(Geva et al., [2021](#bib.bib13 ""))* and Fermi *(Kalyan et al., [2021](#bib.bib20 ""))*.

For evaluation, we follow prior work and use EM for NQ and StrategyQA, and F1 for 2WikiMQA and Bamboogle. For Fermi, we use the official order-of-magnitude evaluation ( *Kalyan et al. [2021](#bib.bib20 "")*).
Following prior work *(Khattab et al., [2022](#bib.bib23 ""); Trivedi et al., [2023](#bib.bib45 ""); Yoran et al., [2023](#bib.bib50 ""))*, we evaluate on 500 random examples from the development set of each dataset.
We provide additional technical details on evaluation in §[A.2](#A1.SS2 "A.2 Evaluation ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

| Dataset | Type | Example |
| --- | --- | --- |
| NQ | Single-hop | What episode of law and order svu is mike tyson in? |
| 2WikiMQA | Explicit | Where was the place of death of Isabella Of Bourbon’s father? |
| Bamboogle | Explicit | What is the maximum airspeed (in km/h) of the third fastest bird? |
| StrategyQA | Implicit | Can Arnold Schwarzenegger deadlift an adult Black rhinoceros? |
| Fermi | Implicit | How many high fives has Lebron James given/received? |

*Table 1: The QA datasets in our experiments.*

### 3.2 Models

We describe retrievers (§[3.2.1](#S3.SS2.SSS1 "3.2.1 Retrievers ‣ 3.2 Models ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")), prompted baselines (§[3.2.2](#S3.SS2.SSS2 "3.2.2 Few-shot Prompted Baselines ‣ 3.2 Models ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")), and finetuned models (§[3.2.3](#S3.SS2.SSS3 "3.2.3 Fine-tuned models ‣ 3.2 Models ‣ 3 Experimental Setting ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

#### 3.2.1 Retrievers

Our models use a retriever based on Google Search,222We query Google search via the SerpAPI service: <https://serpapi.com/>. as well as the open-source ColBERTV2 *(Khattab \& Zaharia, [2020](#bib.bib24 ""))*.
The corpus for our datasets is Wikipedia, therefore, we format search queries as “en.wikipedia.org $q_{i}$” when accessing Google Search. For ColBERTV2 our corpus is the 2018 Wikipedia from *Karpukhin et al. ([2020](#bib.bib21 ""))*. To simulate different types of noise, we return either the top-1, a low-ranked-ranked relevant evidence,333For Google Search, we use the lowest returned result from the API, which is at rank 9.3 on average. For ColBERTV2 we only use with top-1 results. or a random passage that is the top-1 evidence for a different question or intermediate question from the same dataset.

#### 3.2.2 Few-shot Prompted Baselines

Our main baselines are *Llama-2-13B* models prompted for QA in the Self-Ask format through in-context learning *(Brown et al., [2020](#bib.bib3 ""))* with 4-6 exemplars. We also evaluate with *Llama-2-70B* on NQ. Our baselines differ based on the retrieved contexts in the exemplars (Full prompts in §[A.5](#A1.SS5 "A.5 Prompts ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")):

* •

    Self-Ask No Retrieval (SA-NR): Exemplars are gold decompositions *without* retrieved evidence. This is the prompt used when evaluating without retrieval during inference to assess performance when relying solely on parametric memory. As an additional baseline, we also use this non-retrieval prompt, but still apply retrieval during inference.

* •

    Self-Ask Retrieval@1 (SA-R@1): Exemplars are gold decomopsitions prepended with the most relevant evidence retrieved from Google Search for each step.

* •

    Self-Ask Retrieval@10 (SA-R@10): Exemplars are Self-Ask decomopsitions prepended with the lowest rank passage from Google (which is rank 10 in most cases)

* •

    Self-Ask Random Retrieval (SA-RMix) Exemplars are gold decomopsitions prepended with either the top-1 or lowest-ranked evidence from Google Search, interchangeably.

##### NLI-based Models

We use a BART-Large model *(Lewis et al., [2020a](#bib.bib26 ""))* with 407 million parameters trained on the MNLI dataset *(Williams et al., [2018](#bib.bib48 ""))*.444We use the model from [https://huggingface.co/facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli ""). We consider a question-answer pair as entailed if the probability of entailment label is $\geq 0.5$.
All few-shot prompted baselines have a variant with NLI, termed, SA-*-NLI.
When there is no entailment, we use the generation from the SA-NR model, which uses only the parametric memory as the back-off strategy,

#### 3.2.3 Fine-tuned models

We finetune *Llama-2-13B* on 3 ODQA benchmarks, one single-hop (NQ, 1000 training examples), one explicit (2WikiMQA, 500 training questions, 1,539 examples), and one implicit (StrategyQA, 414 questions and 1,584 examples). Training hyperparameters are in §[A.1](#A1.SS1 "A.1 Models ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

##### Data Generation

We use a large LLM to verify questions are answerable and to generate decompositions. This is done with GPT-3, *code-davinci-002* *(Brown et al., [2020](#bib.bib3 ""); Chen et al., [2021](#bib.bib6 ""))* with 175B parameters, and we prompt the model to generate decompositions using the SA-NR prompt. 2WikiMQA contains intermediate answers, and we use those to verify generated decompositions. For the implicit StrategyQA we utilize only the final answer, and thus use self-consistency, as explained in §[2](#S2 "2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").
We sample 5 decompositions per question (one with greedy decoding and four with temperature $0.7$) and only keep the greedily-decoded decomposition when all decompositions lead to the same correct answer. To verify the quality of the generated decompositions, we manually examine 50 decompositions per dataset and find that the generated decompositions correct in about $90\%$ of the time for StrategyQA and more than $95\%$ for 2WikiMQA.
As Fermi and Bamboogle contain less than $300$ examples, we use them exclusively for evaluation and do not include them in these experiments.

##### Incorporating Retrieved Evidence in Training Examples

To make sure the model is exposed to relevant and irrelevant context, we use either the top-1, low-ranked, or random evidence with equal probability at each step. We term the trained model SA-RetRobust. We include ablations where training is without retrieved context (SA-NoRet) or only with the top-1 evidence (SA-Ret@1).

4 Results
---------

<img src='x4.png' alt='Refer to caption' title='' width='461' height='111' />

*Figure 4: Results for our models on all evaluation datasets when retrieving top-1 results from Google Search.
Training *Llama-2-13B* on gold decompositions increases performance for all datasets (second from left). Prompting models with retrieval (center bar) increases performance on single-hop and explicit datasets, but decreases performance on implicit ones. When using NLI models to identify irrelevant evidence (second from right), retrieval never hurts, at a cost to gains received when retrieval helps. Our trained RALMs (rightmost column) outperform all other models.*

Fig.[4](#S4.F4 "Figure 4 ‣ 4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") presents our main results when retrieving the top-1 result from Google Search for the following models: (a) a model prompted and evaluated without retrieval with the SA-NR prompt (blue), (b) a model trained and evaluated without retrieval (red), (c) a model prompted and evaluated with retrieval with the SA-RMix prompt (yellow), (d) the same model but using NLI models to identify irrelevant context (green), and (e) our proposed RALM fine-tuned on a mixture of relevant and irrelevant contexts (SA-RetRobust, orange).
When comparing the model prompted and evaluated without retrieval (blue, leftmost) to the model prompted and evaluated with retrieval (yellow, center), we see that retrieval helps on NQ, 2WikiMQA and Bamboogle but reduces performance on the implicit StrategyQA and Fermi. Adding NLI to identify irrelevant context (green, second from right), ensures that retrieval does not hurt, but gains are limited. Training with retrieval (orange, rightmost) leads to gains across the board and to retrieval being better than training and evaluating without retrieval (red, second from left). We observe similar trends with the ColBERTV2 retriever, albeit at an overall decrease in accuracy (§[A.3](#A1.SS3 "A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Tab.[4](#A1.T4 "Table 4 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").)

Fig.[5](#S4.F5 "Figure 5 ‣ Effect of NLI ‣ 4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") present results when simulating retrieval of irrelevant/noisy context, either by retrieving low-ranked passages or random ones. In the challenging setting where we retrieve low-ranked results, SA-RetRobust outperforms the variant trained and evaluated without retrieval by $3.8$ and $2.8$ points on NQ and 2WikiMQA, while performing only slightly worse (within a $1.2$ point difference) on StrategyQA, suggesting it learned to better utilize retrieval while also ignoring irrelevant context. Results with random retrieval further strengthen this point: SA-RetRobust performs similarly (within one standard deviation, see Tab.[5](#A1.T5 "Table 5 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") in §[A.3](#A1.SS3 "A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")) to the model trained and evaluated without retrieval, while for prompted models performance drops by more than 10 points on average.

##### In-context Exemplars with Retrieval can Hurt Performance

Tab.[3](#A1.T3 "Table 3 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") and Tab.[4](#A1.T4 "Table 4 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") in §[A.3](#A1.SS3 "A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") present full results with the Google Search and ColBERTV2 retrievers.
Interestingly, providing exemplars with retrieval performs worse than providing exemplars without retrieval, i.e., the SA-NR prompt leads to better performance even when retrieval is performed at inference time. This SA-NR prompt consistently outperforms the prompts with contexts (SA-R@1, SA-R@10, and SA-RMix) when retrieving the top-1 result from ColBERTV2 or random contexts from Google Search. In addition, the SA-R@1 that contains the top-1 results is not the best performing even when retrieving top-1 results at inference time, and is the worst performing when retrieving noisy contexts at inference time, suggesting that showing examples for retrieval during in-context learning has a negative affect that causes *over-utilization of irrelevant context*.

##### Effect of NLI

When retrieving random contexts or evaluating on the implicit StrategyQA and Fermi, NLI variants consistently perform best, suggesting small NLI models are sufficient to identify irrelevant evidence. However, they reduce performance in cases retrieval is helpful, e.g on the explicit 2WikiMQA and Bamboogle. We perform a detailed analysis for our NLI variants in §[5](#S5 "5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

<img src='x5.png' alt='Refer to caption' title='' width='461' height='207' />

*Figure 5: Results with low-rank and random retrieval. Models are similar to those in Fig.[4](#S4.F4 "Figure 4 ‣ 4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").
When retrieving low-ranked contexts (top), the finetuned model is able to utilize retrieved evidence for improved performance on NQ and 2WikiMQA, while performance on StrategyQA is maintained. Although random retrieval decreases performance significantly for the prompted model, the finetuned model is unaffected by random context (within 1 point of the variant without retrieval).*

##### Finetuned Models

Fig.[4](#S4.F4 "Figure 4 ‣ 4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") and Fig.[5](#S4.F5 "Figure 5 ‣ Effect of NLI ‣ 4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") show SA-RetRobust consistently outperforms all other models. In §[A.3](#A1.SS3 "A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Tab.[5](#A1.T5 "Table 5 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), we present all results for trained models, showing SA-RetRobust outperforms our ablated baselines. Specifically, it outperforms SA-NoRet (fine-tuned without retrieval) by 2.7, 2.4, and 2.4 points on average when using the top-1, a low-ranked, or a random context from Google Search during inference, and outperforms SA-@1 by 0.2, 0.4, 3.2 points respectively.
When retrieving top-1 results from ColBERTV2, SA-RetRobust outperforms SA-NoRet and SA-@1 by $2.7$ and $0.3$ points on average, respectively.
Our results suggest that training on a mixture that includes relevant and irrelevant contexts is necessary for robustness and to improve performance.

##### Results with *Llama-2-70B*

We compare SA-RetRobust with *Llama-2-70B* on the NQ dataset to assess whether larger models are more robust to irrelevant contexts.
Without retrieval, the prompted *Llama-2-70B* outperforms the trained *Llama-2-13B* by $4.3$ points ($38.4$ vs $34.1$), suggesting it has more paramteric knowledge. However, when retrieving the top-1 results from Google Search, SA-RetRobust outperforms all prompted variants for *Llama-2-70B* by at least $3.3$ points (45.7 vs 42.4), suggesting that while increasing model size is correlated with parametric knowledge it is not sufficient to make models better utilize retrieval. We provide the full results in §[A.3](#A1.SS3 "A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Tab.[6](#A1.T6 "Table 6 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").

5 Analysis
----------

##### When Does Irrelevant Context Cause Errors?

To assess errors caused by irrelevant context, we manually looked at examples from NQ, 2WikiMQA and StrategyQA, where models succeed without retrieval, but fail with it. Specifically, we look at examples where the model is prompted with the SA-RMix prompt that includes both top-1 and low-ranked retrieved result and is presented with low-rank or random retrieved evidence during inference. For each setting, we manually annotate 40 examples with the following categories (a) *Valid*: the prediction is a valid answer or the question is ambiguous (b) *Wrong*: prediction with retrieval is wrong and prediction without retrieval is correct, (c) *Both Wrong*: prediction with retrieval is wrong, but prediction without retrieval was also wrong (due to bad decomposition that can spuriously lead to a correct answer in binary or comparison questions). We find that in all settings, the *Wrong* class is most prevalent (65%-85% of the time) and the introduction of retrieved context indeed causes the LLM to err (Tab.[7](#A1.T7 "Table 7 ‣ A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") in §[A.4](#A1.SS4 "A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

We then take a deeper look at the errors. For NQ we find that when using low-ranked context, the wrong generated answer entity appears in the retrieved context in 77% of the cases, but only in 37% when retrieving random contexts. This suggests that irrelevant context can cause errors even when the generated entities are not retrieved, as shown in Fig.[6](#S5.F6 "Figure 6 ‣ When Does Irrelevant Context Cause Errors? ‣ 5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").
For multi-hop questions, we test whether irrelevant context leads to errors in question decomposition, or in answering intermediate questions.
We find that when retrieving low-ranked passages, 68% of the errors for the explicit 2WikiMQA are in intermediate *answers*, contrary to the implicit StrategyQA where 77% of the errors are in intermediate *questions* (we provide an example in §[A.4](#A1.SS4 "A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Fig.[7](#A1.F7 "Figure 7 ‣ A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")). Similarly, when retrieving random contexts, most errors (60%) for 2WikiMQA are in intermediate questions. Overall, this suggest that irrelevant context can cause errors both in developing an answering strategy and in generating the answer itself, depending on the dataset and retrieved context.

<img src='x6.png' alt='Refer to caption' title='' width='461' height='112' />

*Figure 6: An example from NQ where retrieval caused *Llama-2-13B* to err, although the generated entity does not appear in the retrieved context.*

##### When Do NLI Models Fail?

As shown in §[4](#S4 "4 Results ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), NLI models are efficient at identifying relevant context, at a cost to gains when retrieval is helpful. To better characterize NLI models, we look at the accuracy for our SA-*-NLI models as a function of the probability that the NLI model assigns to the entailment label. Tab.[2](#S5.T2 "Table 2 ‣ When Do NLI Models Fail? ‣ 5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") shows that there are many cases where the probability for entailment is low but retrieval helps for NQ and 2WikiMQA. We thus manually analysed 25 examples for each dataset where entailment was low, but retrieval augmentation led the SA-RMix model to generate the correct answer. In $56\%$ of the cases the NLI model erred and the generated text is indeed entailed from the retrieved contexts. In the remaining examples, for at least $36\%$ of the cases the generated answer or decomposition is correct, but the retrieved context does not directly entail the generation. This can be partially explained by the ability of the model to combine retrieved evidence and its parametric knowledge *(Talmor et al., [2020](#bib.bib42 ""); Zhong et al., [2023](#bib.bib51 ""); Cohen et al., [2023](#bib.bib8 ""))*. Overall, these results provide further evidence for the benefit of training LLMs for retrieval robustness.

|  | Inference | Failures | Low-Entailment | | Med-Entailment | | High-Entailment | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Retrieval | % | % | $\Delta$ | % | $\Delta$ | % | $\Delta$ |
| NQ | @1 | 0.0% | 32.6% | $+0.11$ | 12.8% | $+0.09$ | 54.6% | $+0.11$ |
| | @10 | 0.0% | 69.4% | $+0.01$ | 9.4% | $+0.06$ | 21.2% | $+0.01$ |
| Random | 0.0% | 97.2% | $-0.07$ | 2.2% | $-0.2$ | 0.6% | $0.0$ |
| 2WikiMQA | @1 | 0.4% | 83.0% | $+0.12$ | 5.6% | $+0.34$ | 11.0% | $+0.55$ |
| | @10 | 2.8% | 93.8% | $-0.02$ | 2.6% | $-0.11$ | 0.8% | $+0.08$ |
| Random | 37.0% | 63.0% | $-0.06$ | 0.0% | $0.0$ | 0.0% | $0.0$ |
| StrategyQA | @1 | 1.2% | 96.2% | $-0.07$ | 2.4% | $+0.17$ | 0.2% | $0.0$ |
| | @10 | 2.6% | 95.8% | $-0.04$ | 1.4% | $0.0$ | 0.2% | $0.0$ |
| Random | 34.4% | 56.6% | $-0.13$ | 0.0% | $0.0$ | 0.0% | $0.0$ |

*Table 2: Results for our NLI analysis. ‘Failures’ indicates that the decomposition model was not able to arrive at the answer (see §[A.1](#A1.SS1 "A.1 Models ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")). Other examples are split based on their entailment probability: low probability is $<\frac{1}{3}$, medium probability is in $[\frac{1}{3},\frac{2}{3}]$, and high probability is $>\frac{2}{3}$. $\Delta$ indicates the improvement in accuracy when using retrieval. For NQ and 2WikiMQA, many cases where retrieval is helpful have low entailment probability. For the implicit StrategyQA most examples have low entailment, but retrieval helps in the few examples with medium entailment.*

6 Related Work
--------------

Recent work has shown that the performance of LLMs can be affected by irrelevant context.
Amongst others, *Jia \& Liang ([2017](#bib.bib17 "")); Petroni et al. ([2020](#bib.bib35 "")); Creswell et al. ([2023](#bib.bib9 ""))* show that adding random or irrelevant context can decrease QA performance. This has been shown in many settings, including but not limited to factual reasoning *(Kassner \& Schütze, [2020](#bib.bib22 ""); Pandia \& Ettinger, [2021](#bib.bib34 ""); Misra et al., [2023](#bib.bib31 ""))*, text generation about new entities *(Onoe et al., [2022](#bib.bib33 ""))*, or even code generation *(Jones \& Steinhardt, [2022](#bib.bib19 ""))*. In the context of arithmetric reasoning, *Shi et al. ([2023](#bib.bib41 ""))* showed that adding irrelevant context to exemplars or task specific instructions can help, suggesting the model may be equipped with such skills from pre-training.
Other methods try to reduce the number of retrieval calls, by focusing on cases where confidence is low *(Jiang et al., [2023](#bib.bib18 ""))* or retrieving information for rare entities *(Mallen et al., [2023](#bib.bib30 ""))*.
Closest to our work is that of *Li et al. ([2023](#bib.bib28 ""))* that propose LLMs with a “controllable memory”, that will enable them to ignore irrelevant context.
However, their LLMs are finetuned on over 200K training examples, where our focus is on performance when training with a thousand questions, or less, and training data is automatically generated.
In addition, we focus on a multi-hop QA setting, where the retriever is called multiple times (§[2](#S2 "2 Making RALMs Robust to Irrelevant Contexts ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")).

A similar line of work focuses on when models should use parametric vs. retrieved knowledge, especially when there are conflicts *(Longpre et al., [2021](#bib.bib29 ""); Chen et al., [2022](#bib.bib5 ""))*.
It has been recently proposed to train models to generate from both parametric and retrieved knowledge *(Neeman et al., [2023](#bib.bib32 ""))* or make better use of in-context exemplars *(Zhou et al., [2023](#bib.bib52 ""))*.

7 Conclusion
------------

In this work, we provide a thorough analysis showing current RALMs are not robust to irrelevant retrieved context, causing them to perform worse on certain tasks. In cases where training is not possible, we show that simple NLI models are highly effective at increasing robustness, at a cost of discarding relevant passages. Finally, we show that training models on as few as 1,000 examples can make models robust to irrelevant context and improve overall performance.

Acknowledgements
----------------

We would like to our colleagues at TAU NLP for their insightful comments. We thank SerpAPI for their support by granting us an academic discount.
This research was partially supported by the Yandex Initiative for Machine Learning and the European Research Council (ERC) under the European Union Horizons 2020 research and innovation programme (grant ERC DELPHI 802800).
This work was completed in partial fulfillment of the Ph.D. of Ori Yoran.

References
----------

* Bohnet et al. (2023)Bernd Bohnet, Vinh Q. Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, Tom Kwiatkowski, Ji Ma, Jianmo Ni, Lierni Sestorain Saralegui, Tal Schuster, William W. Cohen, Michael Collins, Dipanjan Das, Donald Metzler, Slav Petrov, and Kellie Webster.Attributed question answering: Evaluation and modeling for attributed large language models, 2023.
* Bowman et al. (2015)Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning.A large annotated corpus for learning natural language inference.In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp. 632–642, Lisbon, Portugal, September 2015. Association for Computational Linguistics.doi: 10.18653/v1/D15-1075.URL [https://aclanthology.org/D15-1075](https://aclanthology.org/D15-1075 "").
* Brown et al. (2020)Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.Language models are few-shot learners.In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.URL [https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html "").
* Chen et al. (2017)Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes.Reading Wikipedia to answer open-domain questions.In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 1870–1879, Vancouver, Canada, July 2017. Association for Computational Linguistics.doi: 10.18653/v1/P17-1171.URL [https://aclanthology.org/P17-1171](https://aclanthology.org/P17-1171 "").
* Chen et al. (2022)Hung-Ting Chen, Michael Zhang, and Eunsol Choi.Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence.In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 2292–2307, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics.URL [https://aclanthology.org/2022.emnlp-main.146](https://aclanthology.org/2022.emnlp-main.146 "").
* Chen et al. (2021)Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.Evaluating large language models trained on code, 2021.
* Chowdhery et al. (2022)Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel.Palm: Scaling language modeling with pathways, 2022.
* Cohen et al. (2023)Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva.Evaluating the ripple effects of knowledge editing in language models, 2023.
* Creswell et al. (2023)Antonia Creswell, Murray Shanahan, and Irina Higgins.Selection-inference: Exploiting large language models for interpretable logical reasoning.In *The Eleventh International Conference on Learning Representations*, 2023.URL [https://openreview.net/forum?id\=3Pf3Wg6o-A4](https://openreview.net/forum?id=3Pf3Wg6o-A4 "").
* Dagan et al. (2006)Ido Dagan, Oren Glickman, and Bernardo Magnini.The pascal recognising textual entailment challenge.In Joaquin Quiñonero-Candela, Ido Dagan, Bernardo Magnini, and Florence d’Alché Buc (eds.), *Machine Learning Challenges. Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Tectual Entailment*, pp. 177–190, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.ISBN 978-3-540-33428-6.
* Dettmers et al. (2023)Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.Qlora: Efficient finetuning of quantized llms, 2023.
* Dhingra et al. (2022)Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen.Time-aware language models as temporal knowledge bases.*Transactions of the Association for Computational Linguistics*, 10:257–273, 2022.doi: 10.1162/tacl˙a˙00459.URL [https://aclanthology.org/2022.tacl-1.15](https://aclanthology.org/2022.tacl-1.15 "").
* Geva et al. (2021)Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant.Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies.*Transactions of the Association for Computational Linguistics*, 9:346–361, 2021.doi: 10.1162/tacl˙a˙00370.URL [https://aclanthology.org/2021.tacl-1.21](https://aclanthology.org/2021.tacl-1.21 "").
* Guu et al. (2020)Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.Realm: Retrieval-augmented language model pre-training.In *Proceedings of the 37th International Conference on Machine Learning*, ICML’20. JMLR.org, 2020.
* Honovich et al. (2022)Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias.TRUE: Re-evaluating factual consistency evaluation.In *Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering*, pp. 161–175, Dublin, Ireland, 2022. Association for Computational Linguistics.doi: 10.18653/v1/2022.dialdoc-1.19.URL [https://aclanthology.org/2022.dialdoc-1.19](https://aclanthology.org/2022.dialdoc-1.19 "").
* Izacard et al. (2022)Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.Atlas: Few-shot learning with retrieval augmented language models, 2022.
* Jia \& Liang (2017)Robin Jia and Percy Liang.Adversarial examples for evaluating reading comprehension systems.In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 2021–2031, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.doi: 10.18653/v1/D17-1215.URL [https://aclanthology.org/D17-1215](https://aclanthology.org/D17-1215 "").
* Jiang et al. (2023)Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig.Active retrieval augmented generation, 2023.
* Jones \& Steinhardt (2022)Erik Jones and Jacob Steinhardt.Capturing failures of large language models via human cognitive biases.In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022.URL [https://openreview.net/forum?id\=fcO9Cgn-X-R](https://openreview.net/forum?id=fcO9Cgn-X-R "").
* Kalyan et al. (2021)Ashwin Kalyan, Abhinav Kumar, Arjun Chandrasekaran, Ashish Sabharwal, and Peter Clark.How much coffee was consumed during EMNLP 2019? fermi problems: A new reasoning challenge for AI.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 7318–7328, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.emnlp-main.582.URL [https://aclanthology.org/2021.emnlp-main.582](https://aclanthology.org/2021.emnlp-main.582 "").
* Karpukhin et al. (2020)Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.Dense passage retrieval for open-domain question answering.In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 6769–6781, Online, 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.emnlp-main.550.URL [https://aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550 "").
* Kassner \& Schütze (2020)Nora Kassner and Hinrich Schütze.Negated and misprimed probes for pretrained language models: Birds can talk, but cannot fly.In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 7811–7818, Online, July 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.698.URL [https://aclanthology.org/2020.acl-main.698](https://aclanthology.org/2020.acl-main.698 "").
* Khattab et al. (2022)O. Khattab, Keshav Santhanam, Xiang Lisa Li, David Leo Wright Hall, Percy Liang, Christopher Potts, and Matei A. Zaharia.Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp.*ArXiv preprint*, abs/2212.14024, 2022.URL [https://arxiv.org/abs/2212.14024](https://arxiv.org/abs/2212.14024 "").
* Khattab \& Zaharia (2020)Omar Khattab and Matei Zaharia.Colbert: Efficient and effective passage search via contextualized late interaction over BERT.In Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (eds.), *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020*, pp. 39–48. ACM, 2020.doi: 10.1145/3397271.3401075.URL [https://doi.org/10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075 "").
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.Natural questions: A benchmark for question answering research.*Transactions of the Association for Computational Linguistics*, 7:452–466, 2019.doi: 10.1162/tacl˙a˙00276.URL [https://aclanthology.org/Q19-1026](https://aclanthology.org/Q19-1026 "").
* Lewis et al. (2020a)Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer.BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 7871–7880, Online, 2020a. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.703.URL [https://aclanthology.org/2020.acl-main.703](https://aclanthology.org/2020.acl-main.703 "").
* Lewis et al. (2020b)Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.Retrieval-augmented generation for knowledge-intensive NLP tasks.In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020b.URL [https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html "").
* Li et al. (2023)Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar.Large language models with controllable working memory.In *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 1774–1793, Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.findings-acl.112.URL [https://aclanthology.org/2023.findings-acl.112](https://aclanthology.org/2023.findings-acl.112 "").
* Longpre et al. (2021)Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh.Entity-based knowledge conflicts in question answering.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 7052–7063, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.emnlp-main.565.URL [https://aclanthology.org/2021.emnlp-main.565](https://aclanthology.org/2021.emnlp-main.565 "").
* Mallen et al. (2023)Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi.When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 9802–9822, Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.acl-long.546.URL [https://aclanthology.org/2023.acl-long.546](https://aclanthology.org/2023.acl-long.546 "").
* Misra et al. (2023)Kanishka Misra, Julia Rayz, and Allyson Ettinger.COMPS: Conceptual minimal pair sentences for testing robust property knowledge and its inheritance in pre-trained language models.In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pp. 2928–2949, Dubrovnik, Croatia, 2023. Association for Computational Linguistics.URL [https://aclanthology.org/2023.eacl-main.213](https://aclanthology.org/2023.eacl-main.213 "").
* Neeman et al. (2023)Ella Neeman, Roee Aharoni, Or Honovich, Leshem Choshen, Idan Szpektor, and Omri Abend.DisentQA: Disentangling parametric and contextual knowledge with counterfactual question answering.In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 10056–10070, Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.acl-long.559.URL [https://aclanthology.org/2023.acl-long.559](https://aclanthology.org/2023.acl-long.559 "").
* Onoe et al. (2022)Yasumasa Onoe, Michael Zhang, Eunsol Choi, and Greg Durrett.Entity cloze by date: What LMs know about unseen entities.In *Findings of the Association for Computational Linguistics: NAACL 2022*, pp. 693–702, Seattle, United States, July 2022. Association for Computational Linguistics.doi: 10.18653/v1/2022.findings-naacl.52.URL [https://aclanthology.org/2022.findings-naacl.52](https://aclanthology.org/2022.findings-naacl.52 "").
* Pandia \& Ettinger (2021)Lalchand Pandia and Allyson Ettinger.Sorting through the noise: Testing robustness of information processing in pre-trained language models.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 1583–1596, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.emnlp-main.119.URL [https://aclanthology.org/2021.emnlp-main.119](https://aclanthology.org/2021.emnlp-main.119 "").
* Petroni et al. (2020)Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel.How context affects language models’ factual predictions, 2020.
* Petroni et al. (2021)Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel.KILT: a benchmark for knowledge intensive language tasks.In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 2523–2544, Online, June 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.naacl-main.200.URL [https://aclanthology.org/2021.naacl-main.200](https://aclanthology.org/2021.naacl-main.200 "").
* Press et al. (2022)Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis.Measuring and narrowing the compositionality gap in language models.*ArXiv preprint*, abs/2210.03350, 2022.URL [https://arxiv.org/abs/2210.03350](https://arxiv.org/abs/2210.03350 "").
* Ram et al. (2023)Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.In-context retrieval-augmented language models, 2023.
* Roberts et al. (2020)Adam Roberts, Colin Raffel, and Noam Shazeer.How much knowledge can you pack into the parameters of a language model?In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 5418–5426, Online, 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.emnlp-main.437.URL [https://aclanthology.org/2020.emnlp-main.437](https://aclanthology.org/2020.emnlp-main.437 "").
* Rubin \& Berant (2023)Ohad Rubin and Jonathan Berant.Long-range language modeling with self-retrieval, 2023.
* Shi et al. (2023)Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou.Large language models can be easily distracted by irrelevant context, 2023.
* Talmor et al. (2020)Alon Talmor, Oyvind Tafjord, Peter Clark, Yoav Goldberg, and Jonathan Berant.Leap-of-thought: Teaching pre-trained models to systematically reason over implicit knowledge.In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.URL [https://proceedings.neurips.cc/paper/2020/hash/e992111e4ab9985366e806733383bd8c-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/e992111e4ab9985366e806733383bd8c-Abstract.html "").
* Thorne et al. (2018)James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.FEVER: a large-scale dataset for fact extraction and VERification.In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pp. 809–819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.doi: 10.18653/v1/N18-1074.URL [https://aclanthology.org/N18-1074](https://aclanthology.org/N18-1074 "").
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas
Scialom.Llama 2: Open foundation and fine-tuned chat models, 2023.
* Trivedi et al. (2023)Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 10014–10037, Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.acl-long.557.URL [https://aclanthology.org/2023.acl-long.557](https://aclanthology.org/2023.acl-long.557 "").
* Wang et al. (2023)Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.Self-consistency improves chain of thought reasoning in language models.In *The Eleventh International Conference on Learning Representations*, 2023.URL [https://openreview.net/forum?id\=1PL1NIMMrw](https://openreview.net/forum?id=1PL1NIMMrw "").
* Welbl et al. (2018)Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel.Constructing datasets for multi-hop reading comprehension across documents.*Transactions of the Association for Computational Linguistics*, 6:287–302, 2018.doi: 10.1162/tacl˙a˙00021.URL [https://aclanthology.org/Q18-1021](https://aclanthology.org/Q18-1021 "").
* Williams et al. (2018)Adina Williams, Nikita Nangia, and Samuel Bowman.A broad-coverage challenge corpus for sentence understanding through inference.In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pp. 1112–1122, New Orleans, Louisiana, 2018. Association for Computational Linguistics.doi: 10.18653/v1/N18-1101.URL [https://aclanthology.org/N18-1101](https://aclanthology.org/N18-1101 "").
* Wolfson et al. (2020)Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant.Break it down: A question understanding benchmark.*Transactions of the Association for Computational Linguistics*, 8:183–198, 2020.doi: 10.1162/tacl˙a˙00309.URL [https://aclanthology.org/2020.tacl-1.13](https://aclanthology.org/2020.tacl-1.13 "").
* Yoran et al. (2023)Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan Berant.Answering questions by meta-reasoning over multiple chains of thought, 2023.
* Zhong et al. (2023)Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, and Danqi Chen.Mquake: Assessing knowledge editing in language models via multi-hop questions, 2023.
* Zhou et al. (2023)Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen.Context-faithful prompting for large language models, 2023.

Appendix A Appendix
-------------------

### A.1 Models

##### Decomposition Generation

Questions in our multi-hop datasets require between 2-4 decomposition steps. Hence we limit the number of generation steps to 5. In Tab. [2](#S5.T2 "Table 2 ‣ When Do NLI Models Fail? ‣ 5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") we show that the number of cases in which the model does not arrive at an answer in 5 steps, termed as failures, is very small when generating with top-1 results from Google Search, at $0.4\%$ for 2WikiMQA and $1.2\%$ for StrategyQA. Failures are much higher when retrieving random contexts, at $37.0\%$ for 2WikiMQA and $34.4\%$ for StrategyQA. These are usually cases the model enters an infinite loop. Following recent work, *(Wang et al., [2023](#bib.bib46 ""); Yoran et al., [2023](#bib.bib50 ""))* we use greedy decoding when generating decompositions.

##### Training

We fine-tune all our models with QLoRA *(Dettmers et al., [2023](#bib.bib11 ""))* for parameter efficient fine-tuning. We use the default hyperparameters from [https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/guanaco/sft-llama-2-13b-guanaco-peft.ipynb](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/guanaco/sft-llama-2-13b-guanaco-peft.ipynb ""). We train all our models for 5 epochs, with a learning rate of $2e-4$ and linear scheduling on a single GPU. The training time for each model was no longer than $3.5$ hours.

### A.2 Evaluation

In some cases, the models do not arrive at a final answer (§[A.1](#A1.SS1 "A.1 Models ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context")). In such cases, we assign a score of $0.5$ for StrategyQA and <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="A1.SS2.p1.2.m2.1"><semantics id="A1.SS2.p1.2.m2.1a"><mn id="A1.SS2.p1.2.m2.1.1" xref="A1.SS2.p1.2.m2.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="A1.SS2.p1.2.m2.1b"><cn id="A1.SS2.p1.2.m2.1.1.cmml" type="integer" xref="A1.SS2.p1.2.m2.1.1">0</cn></annotation-xml></semantics></math> -->00 for all other datasets. For Fermi, following past work *(Yoran et al., [2023](#bib.bib50 ""))*, we use all 286 “Real Fermi Problems” for evaluation and provide the gold answers measure units (meters, cubes, litres, etc…) as additional input to our models .

### A.3 Full results

Tab.[3](#A1.T3 "Table 3 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") and Tab.[4](#A1.T4 "Table 4 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") presents the full results for our prompted models with Google Search and ColBERTV2, respectively.
Tab.[5](#A1.T5 "Table 5 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") presents full results for all our trained models, averaged over three seeds.
Tab.[6](#A1.T6 "Table 6 ‣ A.3 Full results ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") presents results for *Llama-2-70B* on NQ with the Google Search retriever.

| Dataset | Inference | NR | NR | R@1 | R@1 | R@10 | R@10 | RMix | RMix |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Retrieval | | -NLI | | -NLI | | -NLI | | -NLI | |
| NQ | None | 29.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| | @1 | 41.0 | 38.4 | 39.0 | 36.4 | 41.0 | 36.8 | 40.6 | 37.0 | |
| @10 | 30.2 | 29.8 | 25.6 | 29.4 | 30.0 | 31.0 | 31.0 | 29.8 |  |
| Random | 28.2 | 29.6 | 17.2 | 29.4 | 22.2 | 29.4 | 22.0 | 29.4 |  |
| 2WikiMQA | None | 32.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| | @1 | 56.0 | 39.9 | 51.6 | 38.3 | 51.6 | 39.2 | 53.1 | 39.0 | |
| @10 | 33.0 | 32.2 | 27.5 | 32.5 | 30.9 | 32.3 | 29.6 | 32.2 |  |
| Random | 27.0 | 32.0 | 13.7 | 32.0 | 21.3 | 32.2 | 17.5 | 32.0 |  |
| StrategyQA | None | 65.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| | @1 | 62.1 | 65.6 | 63.8 | 66.7 | 61.4 | 65.8 | 59.6 | 66.2 | |
| @10 | 60.4 | 65.6 | 61.0 | 65.6 | 60.5 | 65.4 | 62.1 | 65.8 |  |
| Random | 58.4 | 65.6 | 53.4 | 65.6 | 57.0 | 65.6 | 52.7 | 65.6 |  |
| Bamboogle | None | 47.4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| | @1 | 68.0 | 55.9 | 61.2 | 56.0 | 68.9 | 58.0 | 62.7 | 55.2 | |
| @10 | 41.4 | 47.4 | 32.1 | 45.9 | 44.5 | 45.9 | 38.1 | 47.0 |  |
| Random | 39.5 | 47.4 | 24.7 | 47.4 | 34.8 | 47.4 | 26.3 | 47.4 |  |
| Fermi | None | 27.7 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| | @1 | 27.4 | 28.2 | 25.2 | 27.6 | 27.5 | 27.7 | 25.6 | 27.4 | |
| @10 | 24.0 | 27.7 | 27.1 | 27.6 | 25.1 | 27.7 | 23.6 | 28.0 |  |
| Random | 22.1 | 27.7 | 17.2 | 27.7 | 17.4 | 27.7 | 13.8 | 27.7 |  |

*Table 3: Full results for our prompted *Llama-2-13B* models with the Google Search retriever.*

| Dataset | Inference | NR | NR | R@1 | R@1 | R@10 | R@10 | RMix | RMix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Retrieval |  | -NLI |  | -NLI |  | -NLI |  | -NLI |
| NQ | None | 29.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | @1 | 34.6 | 34.8 | 31.2 | 33.2 | 32.4 | 33.8 | 32.8 | 33.8 |
| 2WikiMQA | None | 32.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | @1 | 42.2 | 36.2 | 37.3 | 34.9 | 36.7 | 35.0 | 39.6 | 35.3 |
| StrategyQA | None | 65.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | @1 | 61.6 | 66.0 | 64.3 | 65.1 | 61.1 | 64.9 | 61.6 | 64.7 |
| Bamboogle | None | 47.4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | @1 | 50.0 | 48.6 | 37.4 | 46.6 | 38.1 | 47.4 | 38.2 | 48.7 |
| Fermi | None | 27.7 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | @1 | 25.9 | 27.3 | 23.2 | 27.8 | 21.2 | 28.0 | 24.4 | 28.0 |

*Table 4: Full results for our prompted *Llama-2-13B* models with the ColBERTV2 retriever.*

| Dataset | Retriever | Inference | SA- | SA- | SA- |
| --- | --- | --- | --- | --- | --- |
|  |  |  | NoRet | Ret@1 | RetRobust |
| NQ | None | None | 34.1$\pm$0.8 | n/a | n/a |
| | Google | @1 | 42.8$\pm$0.8 | 46.3$\pm$0.6 | 45.7$\pm$0.6 |
| Google | @10 | 37.0$\pm$1.0 | 38.2$\pm$0.6 | 37.9$\pm$0.5 |
| Google | @Random | 31.1$\pm$0.1 | 31.4$\pm$0.5 | 33.8$\pm$0.2 |
| ColBERTV2 | @1 | 41.5$\pm$0.4 | 43.5$\pm$0.2 | 43.5$\pm$0.6 |
| 2WikiMQA | None | None | 42.2$\pm$0.6 | n/a | n/a |
| | Google | @1 | 64.6$\pm$0.7 | 66.7$\pm$1.0 | 66.9$\pm$1.0 |
| Google | @10 | 40.8$\pm$0.5 | 43.9$\pm$0.3 | 45.0$\pm$0.4 |
| Google | @Random | 40.4$\pm$0.8 | 37.5$\pm$1.0 | 41.6$\pm$0.2 |
| ColBERTV2 | @1 | 54.4$\pm$0.7 | 57.0$\pm$0.5 | 57.6$\pm$0.5 |
| StrategyQA | None | None | 69.8$\pm$0.9 | n/a | n/a |
| | Google | @1 | 67.1$\pm$0.4 | 69.0$\pm$1.2 | 70.1$\pm$1.1 |
| Google | @10 | 66.6$\pm$1.1 | 68.1$\pm$0.3 | 68.6$\pm$0.5 |
| Google | @Random | 66.6$\pm$0.7 | 66.9$\pm$1.2 | 69.9$\pm$1.8 |
| ColBERTV2 | @1 | 65.9$\pm$0.6 | 68.4$\pm$1.4 | 68.8$\pm$0.9 |

*Table 5: Full results for our trained *Llama-2-13B* models. Results are averaged over three seeds. For our RALMs, we use either Google Search or ColBERTV2 as our retrievers during inference.*

| Inference | NR | NR | R@1 | R@1 | R@10 | R@10 | RMix | RMix | SA- | SA- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Retrieval |  | -NLI |  | -NLI |  | -NLI |  | -NLI | No-Ret | RetBust |
| #Params | 70B | 70B | 70B | 70B | 70B | 70B | 70B | 70B | 13B | 13B |
| None | 38.4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 34.1 | n/a |
| @1 | 41.4 | 41.8 | 41.2 | 42.4 | 41.6 | 42.4 | 41.2 | 42.0 | 42.8 | 45.7 |
| @10 | 38.8 | 36.2 | 30.2 | 34.2 | 33.4 | 35.4 | 31.8 | 35.2 | 37.0 | 37.9 |
| Random | 33.6 | 38.2 | 28.8 | 36.8 | 35.2 | 38.2 | 31.0 | 38.0 | 31.1 | 33.8 |

*Table 6: Results for NQ with Google Search and *Llama-2-70B*.*

### A.4 Analysis

Tab.[7](#A1.T7 "Table 7 ‣ A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") presents the full results for our analysis of cases irrelevant context caused SA-RMix to err, described in §[5](#S5 "5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"). Fig.[7](#A1.F7 "Figure 7 ‣ A.4 Analysis ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context") shows an example where random retrieval caused the model to generate a bad strategy in StrategyQA.

|  | Inference | Valid | Wrong | Both |
| --- | --- | --- | --- | --- |
|  | Retrieval |  |  | Wrong |
| NQ | @10 | 34% | 66% | 0% |
| | Random | 22% | 78% | 0% |
| 2WikiMQA | @10 | 2% | 72% | 23% |
| | Random | 0% | 85% | 15% |
| StrategyQA | @10 | 3% | 65% | 32% |
| | Random | 0% | 70% | 30% |

*Table 7: Full results for our analysis regarding cases where augmenting retrieved contexts caused *Llama-2-13B* prompted with SA-RMix to err. Classes and additional details are provided in §[5](#S5 "5 Analysis ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context").*

<img src='x7.png' alt='Refer to caption' title='' width='461' height='293' />

*Figure 7: An example from StrategyQA irrelevant context causes *Llama-2-13B* to generate a wrong strategy (right). Without retrieval (left), the model succeeds in generating the correct answer.*

### A.5 Prompts

We provide our SA-NR, SA-R@1, and SA-R@10 prompts for NQ in Tab.[8](#A1.F8 "Figure 8 ‣ A.5 Prompts ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Tab.[9](#A1.F9 "Figure 9 ‣ A.5 Prompts ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), Tab.[10](#A1.F10 "Figure 10 ‣ A.5 Prompts ‣ Appendix A Appendix ‣ Making Retrieval-Augmented Language Models Robust to Irrelevant Context"), respectively. For the SA-RMix prompt, we use exemplars form the SA-R@1 and SA-R@10 prompts, interchangeably. We add a small instruction for the QA task before the exemplars. Our prompts contain $6$ exemplars for NQ, 2WikiMQA, and StrategyQA, $5$ for Fermi, and $4$ for Bamboogle. All our prompts will be made publicly available, together with our models, data, and code.

| Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessary, answer the question directly. |
| --- |
| # |
| Question: how did the big red one get its name |
| Are follow up questions needed here: No. |
| So the final answer is: its shoulder patch |
| # |
| Question: where are the cayman islands on the map |
| Are follow up questions needed here: No. |
| So the final answer is: western Caribbean Sea |
| # |
| Question: who won the war between north korea and south korea |
| Are follow up questions needed here: No. |
| So the final answer is: technically still at war |
| # |
| Question: when does it’s always sunny in philadelphia season 13 start |
| Are follow up questions needed here: No. |
| So the final answer is: September 5, 2018 |
| # |
| Question: who sang you got a friend in me from toy story |
| Are follow up questions needed here: No. |
| So the final answer is: Randy Newman |
| # |
| Question: when was the first person sent to space |
| Are follow up questions needed here: No. |
| So the final answer is: 12 April 1961 |
| # |
| Question: |

*Figure 8: The SA-NR prompt used in our NQ experiments.*

| Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessary, answer the question directly. You are provided with evidence that can help you arrive at the answer before the question. |
| --- |
| # |
| Context1: The Big Red One: Fuller was a World War II veteran and served with the 1st Infantry Division, which is nicknamed ”The Big Red One” for the red numeral ”1” on the division’s shoulder patch. He received the Silver Star, Bronze Star, and Purple Heart during his service. |
| Question: how did the big red one get its name |
| Are follow up questions needed here: No. |
| So the final answer is: its shoulder patch |
| # |
| Context1: Location Map of Cayman Islands: The given Cayman Islands location map shows that the Cayman Islands are located in the western Caribbean Sea. Location Map of Cayman Islands. Where is Cayman … |
| Question: where are the cayman islands on the map |
| Are follow up questions needed here: No. |
| So the final answer is: western Caribbean Sea |
| # |
| Context1: Korean War — Combatants, Summary, Years, Map … - Britannica: After more than a million combat casualties had been suffered on both sides, the fighting ended in July 1953 with Korea still divided into two hostile states. Negotiations in 1954 produced no further agreement, and the front line has been accepted ever since as the de facto boundary between North and South Korea. |
| Question: who won the war between north korea and south korea |
| Are follow up questions needed here: No. |
| So the final answer is: technically still at war |
| # |
| Context1: It’s Always Sunny in Philadelphia (season 13): The thirteenth season of the American comedy television series Itś Always Sunny in Philadelphia premiered on FXX on September 5, 2018. … The season consists of … |
| Question: when does it’s always sunny in philadelphia season 13 start |
| Are follow up questions needed here: No. |
| So the final answer is: September 5, 2018 |
| # |
| Context1: You’ve Got a Friend in Me: ”You’ve Got a Friend in Me” is a song by Randy Newman. Used as the theme song for the 1995 Disney/Pixar animated film Toy Story, it has since become a major … |
| Question: who sang you got a friend in me from toy story |
| Are follow up questions needed here: No. |
| So the final answer is: Randy Newman |
| # |
| Context1: April 1961: Yuri Gagarin from the Soviet Union was the first human in space. His vehicle, Vostok 1 circled Earth at a speed of 27,400 kilometers per hour with the flight lasting 108 minutes. |
| Question: when was the first person sent to space Are follow up questions needed here: No. |
| So the final answer is: 12 April 1961 |
| # |
| Question: |

*Figure 9: The SA-R@1 prompt used in our NQ experiments.*

| Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessary, answer the question directly. You are provided with evidence that can help you arrive at the answer before the question. |
| --- |
| # |
| Context1: 16th Infantry Regiment (United States): As part of the new 1st Expeditionary Division, soon to become known as the ‘Big Red One’, the 16th Infantry, commanded by William Herbert Allaire Jr., sailed |
| Question: how did the big red one get its name |
| Are follow up questions needed here: No. |
| So the final answer is: its shoulder patch |
| # |
| Context1: Module:Location map/data/Cayman Islands: Module:Location map/data/Cayman Islands is a location map definition used to overlay markers and labels on an equirectangular projection map of Cayman |
| Question: where are the cayman islands on the map |
| Are follow up questions needed here: No. |
| So the final answer is: western Caribbean Sea |
| # |
| Context1: First Battle of Seoul: The First Battle of Seoul, known in North Korean historiography as the Liberation of Seoul, was the North Korean capture of the South Korean capital, Seoul, |
| Question: who won the war between north korea and south korea |
| Are follow up questions needed here: No. |
| So the final answer is: technically still at war |
| # |
| Context1: It’s Always Sunny in Philadelphia (season 13): The thirteenth season of the American comedy television series It’s Always Sunny in Philadelphia premiered on FXX on September 5, 2018. |
| Question: when does it’s always sunny in philadelphia season 13 start |
| Are follow up questions needed here: No. |
| So the final answer is: September 5, 2018 |
| # |
| Context1: Randy Newman – You’ve Got a Friend in Me Lyrics: ‘You’ve Got A Friend In Me’ is the theme song of the Toy Story franchise, recurring throughout the series in different contexts. It’s first |
| Question: who sang you got a friend in me from toy story |
| Are follow up questions needed here: No. |
| So the final answer is: Randy Newman |
| # |
| Context1: Timeline of space exploration: This is a timeline of space exploration which includes notable achievements, first accomplishments and milestones in humanity’s exploration of outer space. |
| Question: when was the first person sent to space Are follow up questions needed here: No. |
| So the final answer is: 12 April 1961 |
| # |
| Question: |

*Figure 10: The SA-R@10 prompt used in our NQ experiments.*
