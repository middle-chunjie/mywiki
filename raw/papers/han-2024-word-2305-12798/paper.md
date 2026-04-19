LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space
===========================================================================

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, Heng Ji  
University of Illinois Urbana-Champaign  
{chihan3, jx17, manling2, yifung2, chenkai5  
nanjiang, zaher, hengji}@illinois.edu

###### Abstract

In recent years, large language models (LMs) have achieved remarkable progress across various natural language processing tasks. As pre-training and fine-tuning are costly and might negatively impact model performance, it is desired to efficiently adapt an existing model to different conditions such as styles, sentiments or narratives, when facing different audiences or scenarios.
However, efficient adaptation of a language model to diverse conditions remains an open challenge.
This work is inspired by the observation that text conditions are often associated with selection of certain words in a context.
Therefore we introduce LM-Switch, a theoretically grounded, lightweight and simple method for generative language model conditioning.
We begin by investigating the effect of conditions in Hidden Markov Models (HMMs),
and establish a theoretical connection with language model. Our finding suggests that condition shifts in HMMs are associated with linear transformations in word embeddings.
LM-Switch is then designed to deploy a learnable linear factor in the word embedding space for language model conditioning. We show that LM-Switch can model diverse tasks, and achieves comparable or better performance compared with state-of-the-art baselines in LM detoxification and generation control, despite requiring no more than 1% of parameters compared with baselines and little extra time overhead compared with base LMs. It is also able to learn from as few as a few sentences or one document.
Moreover, a learned LM-Switch can be transferred to other LMs of different sizes, achieving a detoxification performance similar to the best baseline. We will make our code available to the research community following publication. 111Please be advised that this paper contains potentially controversial results and examples to some readers, included solely for research purpose to explore model capabilities.

1 Introduction
--------------

<img src='x1.png' alt='Refer to caption' title='' width='438' height='176' />

*Figure 1: An overview of LM-Switch. (a): LM-Switch applies a linear factor $\epsilon W\mathbf{e}_{v}$ to each word embedding for language model conditioning. (b): During training, we use a positively switched model $M(\epsilon W)$ to maximize likelihood on positively labelled texts, vise versa. (c): For generation, one only needs to specify a switch value $\epsilon$, and then proceed with normal decoding.*

In recent years, large language models (LLMs) have made significant progress in various natural language processing (NLP) tasks such as machine translation, sentiment analysis, schema induction and summarization*[[4](#bib.bib4 ""), [19](#bib.bib19 ""), [24](#bib.bib24 ""), [39](#bib.bib39 ""), [35](#bib.bib35 "")]*.
LLMs are typically pre-trained on a unified text corpus. However, there are various scenarios where it is desirable to steer a language model’s generation according to different conditions, such as stances, styles, and sentiments. Examples of these cases include tailoring a language model’s output for desired communication goals, creating personalized and relatable content to connect with target audiences, or to mitigate biases, manage risks, and ensuring fair and unbiased representation.
When facing these diverse needs, retraining or fine-tuning is not only inefficient*[[4](#bib.bib4 "")]* but can also negatively impact their performance*[[56](#bib.bib56 "")]*. This work is then motivated by the need for efficiently adapting an existing LM given diverse conditions while fully taking advantage of its generation power.

There has been increasing attention on controlling LM generations.
Besides directly training an LM on domain-specific datasets*[[57](#bib.bib57 ""), [25](#bib.bib25 ""), [55](#bib.bib55 "")]*, other techniques are proposed for guiding LM at decoding time. These attempts include superposing attribute classifiers (such as sentiment and toxicity) as constraints when sampling tokens*[[22](#bib.bib22 ""), [5](#bib.bib5 ""), [28](#bib.bib28 ""), [50](#bib.bib50 ""), [23](#bib.bib23 "")]*, treating decoding as an optimization problem*[[21](#bib.bib21 "")]*, or grafting complex adaptor modules onto existing LMs*[[13](#bib.bib13 "")]*. Despite these efforts, due to the large size of parameters in LMs to be adapted, and extra computation burden while decoding, efficient conditioning of LMs still remains an open question.
Recently, prompting with instructions emerges as a novel method for LM interaction*[[4](#bib.bib4 ""), [35](#bib.bib35 "")]*. However either that the performance relies on the quality of very large LLMs, and/or needs to pre-train an instruction-controlled LM deliberately on related corpus*[[58](#bib.bib58 ""), [40](#bib.bib40 "")]*, which prevents scaling up to larger demands.

To address these challenges, we introduce LM-Switch, a theoretically grounded yet empirically straightforward and lightweight plug-in for efficient and versatile conditioning over language models, by only transforming the word embedding space. This work is inspired by the observation that diverse levels of conditions in text, including appropriateness, sentiments, and stances, are tied to the specific choice of words used within a given context. We start from theoretically investigating the effect of conditions on text distributions. Specifically, we measure the effect of adding conditions into Hidden Markov Models, and establish association between condition shifts and word embedding transformations in LMs. This inspires the design of our proposed method, LM-Switch, where we insert a learnable linear bias to the LM word embeddings.
Specifically, the embedding $\mathbf{e}_{v}$ of each word $v\in\mathcal{V}$ is replaced with $\mathbf{e}_{v}+\epsilon W\mathbf{e}_{v}$. Here $W$ is the “switch“ matrix determining the effect of LM conditioning, while $\epsilon$ acts as a “switching value” to indicate polarity and intensity, e.g., +1 for positive and -1 for negative sentiment.

Empirically, we demonstrate that LM-Switch is capable of achieving comparable or better performance on language model detoxification and generation control, despite using a much smaller model size and less decoding time. It offers several other advantages over existing approaches. It is able to learn from as few as one article or dozens of sentences. Another benefit is the ability to transfer a learned LM-Switch to other LMs with different sizes. On detoxification task, this transfer achieves performance as good as the best baseline.
Moreover, LM-Switch is supported by a theoretical guarantee of linearity, which enables continuous and compositional control. This allows for dealing with multitude diverse and nuanced situations, such as personalized or customized generation, without the need of re-training for each scenario. Moreover, we are able to interpret the LM-Switch and display the most indicative words associated with a condition. Broader impacts are discussed in Appendix[A](#A1 "Appendix A Broader Impacts ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").
In summary, this paper makes the following contributions:

* •

    We propose LM-Switch, a theoretically supported and lightweight method for language model conditioning.

* •

    We empirically demonstrate the effectiveness of LM-Switch on applications such as LM detoxification and controlled generation.

* •

    We also highlight and prove the benefits of LM-Switch including data efficiency, transferability, continuous and compositional control and interpretability.

2 Related Work
--------------

Control of Language Models has been of growing interest in recent years, motivated by the increasing capabilities of LMs. This area originates from the need to leverage the generation capabilities of large language models, while avoiding the need for time-consuming and costly retraining or fine-tuning. These attempts include applying attribute classifiers or heuristic constraints at decoding time*[[22](#bib.bib22 ""), [5](#bib.bib5 ""), [28](#bib.bib28 ""), [50](#bib.bib50 "")]*, treating the generation process as an optimization problem over the embedding or token sequences*[[21](#bib.bib21 "")]*, or post-editing the output*[[23](#bib.bib23 "")]*. These techniques are often computationally expensive in searching the output, and rely on the availability and quality of suitable external classifiers.
More recently, prompting-based control for large language models receives much attention, with the control achieved by input prompt engineering to guide the model’s generation. However, this method often rely on the quality and avaialbility of large language models*[[4](#bib.bib4 ""), [35](#bib.bib35 "")]*, and may also necessitate the deliberate training*[[40](#bib.bib40 ""), [58](#bib.bib58 "")]*. It can also be challenging to design effective prompts for complex or nuanced control goals.
Probably most closely related to our work are attempts in discovering “steering” vectors or tokens*[[45](#bib.bib45 ""), [26](#bib.bib26 "")]*, which might originate from similar work in image generation*[[14](#bib.bib14 ""), [12](#bib.bib12 "")]*. Different from our model, these efforts focus on other applications such as multi-task learning and sentence recovery, and the learned vectors (instead of matrices as in our work) are not shown to be transferrable or interpretable, nor enabling flexible control.

Controllable Text Generation is a broader topic involving generating text according to various control objectives such as dialogue history, personality, format or knowledge*[[54](#bib.bib54 ""), [7](#bib.bib7 ""), [17](#bib.bib17 ""), [51](#bib.bib51 ""), [36](#bib.bib36 ""), [29](#bib.bib29 ""), [48](#bib.bib48 ""), [53](#bib.bib53 ""), [57](#bib.bib57 "")]*.
Different from prior work which often requires training a task-specific model, our model mainly focuses on
providing plug-and-play conditioning over a diverse range of off-the-shelf language models.
Note that some methods among these papers do not support users to provide text suffixes, or “prompting”*[[25](#bib.bib25 ""), [38](#bib.bib38 ""), [30](#bib.bib30 ""), [37](#bib.bib37 "")]*, incompatible to the evaluation setting of this study.

Language Model Detoxification Motivated by the goal to address the systematic biases embedded in language models, there are efforts in conducting language model de-biasing or de-toxification *[[31](#bib.bib31 ""), [16](#bib.bib16 "")]*. Approaches span all aspects of the language model pipeline. A line of work focuses on automatically obtaining cleaner data *[[1](#bib.bib1 ""), [46](#bib.bib46 ""), [6](#bib.bib6 "")]*. Another line of work modifies the model workflow design to explicitly accommodate the bias factors *[[46](#bib.bib46 ""), [42](#bib.bib42 ""), [52](#bib.bib52 ""), [34](#bib.bib34 ""), [49](#bib.bib49 "")]*. The most related line of work to the herein proposed method involves manipulating embedding space such as Principle Component Analysis and Nullspace Projection *[[27](#bib.bib27 ""), [3](#bib.bib3 ""), [41](#bib.bib41 "")]*. The evaluation in these settings *[[15](#bib.bib15 ""), [32](#bib.bib32 ""), [33](#bib.bib33 "")]* mostly consists of quiz-question checking for stereotypical misbeliefs. More related to our method are those mentioned in language model control*[[22](#bib.bib22 ""), [5](#bib.bib5 ""), [28](#bib.bib28 ""), [50](#bib.bib50 ""), [21](#bib.bib21 "")]*, which constrains or guides text generation according to a classifier.
A unique contribution in our work is that
the learned LM-Switch can be transferred to also detoxify other off-the-shelf language models without costly training process.

3 LM-Switch: Motivation and Formulation
----------------------------------------

In this section, we provide a theoretical inspiration of LM-Switch. Hidden Markov Model (HMM) is a widely used framework for analyzing discrete stochastic processes. Because of its generality (being able to model arbitrary distributions), intuitiveness and interpretability (containing a structured state space), it has long been used as a primary choice when modeling language distribution.
Our theoretical analysis shows that under some assumptions switching between conditions is equivalent to a linear transform in word embedding spaces.
This observation then inspires the derivation of our proposed model, LM-Switch.

### 3.1 Preliminaries: HMM and LM

#### Hidden Markov Models

Hidden Markov Models (HMMs)*[[2](#bib.bib2 "")]* is a discrete stochastic process with a set of $n$ states $\mathbf{S}$ and a set of $m$ observations or emissions $\mathbf{O}$, with arbitrary indexing of $\mathbf{S}$ and $\mathbf{O}$. The distribution for the time step $t\=0$ is determined by initial state distribution $s_{0}\sim\pi$. For each later time step $t\geq 1$, the state transition probabilities are represented by a matrix $\mathbf{T}$, where $T(s,s^{\prime})\=P(s_{t+1}\=s^{\prime}|s_{t}\=s)$ denotes the probability of transitioning from state $s$ to state $s^{\prime}$. At each time step one observation $o_{t}$ is emitted, with the emission probabilities represented by a matrix $\mathbf{B}$, with $B(s,o)\=P(o_{t}\=o|s_{t}\=s)$.
A sequence of observations can be denoted as $\mathbf{o}\={o_{1},o_{2},\ldots,o_{T}}$.
The probability distribution over sequences $\mathbf{o}$ then follows formula:

|  | $P(o_{1},\cdots,o_{T};\pi)\=\pi^{\top}\left(\prod_{t\=0}^{T-1}\text{diag}(\mathbf{p}(o_{t}))T\right)\mathbf{p}(o_{T}),$ |  | (1) |
| --- | --- | --- | --- |

where $\mathbf{p}(o)$ is a $|\mathcal{S}|$-dim vector indicating $P(o\mid s)$ for all states $s\in\mathcal{S}$.

#### Language Models

In generative language models, the sequence is generated word-by-word by a conditional probability $P(o_{t}\mid o_{1},\cdots,o_{t-1})$. The common technique to model this probability is to first calculate inner product between a contextual vector $\mathbf{c}(o_{1},\cdots,o_{t-1})$ and word embeddings $\mathbf{E}\=(\mathbf{e}_{o},\cdots)\in\mathbb{R}^{d\times|\mathcal{O}|}$, namely, $\mathbf{l}\=\mathbf{c}(o_{1},\cdots,o_{t-1})^{\top}\mathbf{E}$. Here, $\mathbf{l}$ is known as the word logits, which then usually passes through a softmax operator to get a distribution over words. For simplicity of analysis, in this work we assume a linear formulation and let conditional probability $P(o_{t}|o_{1},\cdots,o_{t-1})\=\mathbf{c}(o_{1},\cdots,o_{t-1})^{\top}\mathbf{e}_{o_{t}}$.
By chain-rule, multiplying the conditional probabilities will give us the full probability: $\prod_{t\=1}^{T}P(o_{t}\mid o_{1},\cdots,o_{t-1})\=P(o_{1},\cdots,o_{T})$.

We are then interested in the situation where a language model is good enough to represent an equivalent distribution with HMM. Assuming full column-rank for $\mathbf{E}$ and $\mathbf{p}(o)$, we have the following connection between LM and HMM:

###### Proposition 1.

There exist projection matrices $R_{1}$ and $R_{2}$ so that $R_{1}^{\top}R_{2}\=I_{n}$ and

|  | $\mathbf{c}(o_{1},\cdots,o_{t-1})^{\top}\=\left(\frac{\pi^{\top}\prod_{t^{\prime}\=1}^{t-1}\text{diag}(\mathbf{p}(o_{t}^{\prime}))T}{\pi^{\top}\left(\prod_{t^{\prime}\=1}^{t-2}\text{diag}(\mathbf{p}(o_{t}^{\prime}))T\right)\mathbf{p}(o_{t-1})}\right)R_{1}^{\top},\mathbf{e}_{o}\=R_{2}\mathbf{p}(o).$ |  | (2) |
| --- | --- | --- | --- |

### 3.2 Conditioned Hidden Markov Model

In this study, we aim to model the influence of conditions in text generation. This section describes how we incorporate conditions in HMMs.
Conventionally, people assume a $d$-dimensional state representation $\phi_{s}$ for every state $s$, and $d$-dimensional $\psi_{o}$ for each observation $o$, so that they can compute the probabilities $T(s,s^{\prime})\=\phi_{s}^{\top}A\phi_{s}^{\prime}$, $B(s,o)\=\phi_{s}^{\top}\psi_{o}$ and $\pi(s)\=\phi_{\pi}^{\top}\phi_{s}$ for some $\phi_{\pi}$. We also use matrices $\Phi,\Psi$ to denote the stacked representations $\Phi\=(\phi_{s}|s\in\mathcal{S}),\Psi\=(\psi_{o}|o\in\mathcal{O})$. Here we introduce an additional condition component in state representations, so that $\phi_{s}$ can be partitioned into two sub-vectors: $\phi_{s}\=\begin{pmatrix}\phi_{s,\text{semantic}}\\
\phi_{s,\text{condition}}\end{pmatrix}$.
Here $\phi_{s,\text{semantic}}\in\mathbb{R}^{d_{s}}$ represents the $d_{s}$-dim semantic information, and $\phi_{s,\text{condition}}\in\mathbb{R}^{d_{c}}$ the $d_{c}$-dim condition information related to state $s$.
Then we assume that the transition probability $T(s,s^{\prime})$ comes from both semantic relations and conditional similarities between $s^{\prime}$ and $s$: $T(s,s^{\prime})\=\phi_{s,\text{semantic}}^{\top}A^{\prime}\phi_{s^{\prime},\text{semantic}}+\phi_{s,\text{condition}}^{\top}\phi_{s^{\prime},\text{condition}}$.

We also make the following assumptions regarding the state representations:

###### Assumption 1.

State representations $\phi$ also satisfy the following properties:

1. Values for each dimension are uniformly normalized to a constant:
$\forall i\in[1..d],\sum_{s\in\mathcal{S}}\phi_{s,i}^{2}\=C$.

2. Dimensions are linearly independent:
$\forall i,j\in[1..d]$ and $i\neq j$, $\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}\=0$.

3. Dimensions are also conditionally independent:
if $i,j\in[1..d],k\in[d_{s}+1..d]$ are not all the same, $\sum_{s\in\mathcal{S}}\phi_{s,i}\phi_{s,j}\phi_{s,k}\=0$.

The validity of the assumption is discussed in Appendix[J](#A10 "Appendix J Validity of Assumptions ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").
Then we present the result below revealing that, shifting from one initial condition to another is equivalent to a linear transformation in word embedding space, which accords with our motivating observation that conditions are associated with selection of words in contexts:

###### Theorem 1.

Assume assumption [1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") holds. Suppose there are two initial distributions $\pi\=\phi_{\pi}^{\top}\Phi,\pi^{\prime}\=\phi_{\pi^{\prime}}^{\top}\Phi$, so that $\phi_{\pi}$ and $\phi_{\pi^{\prime}}$ only differ in their condition-parts: $\phi_{\pi,\text{semantic}}\=\phi_{\pi^{\prime},\text{semantic}}$. Also suppose the elements in $\phi_{\pi,\text{condition}}$ are non-zero. Then there exists an matrix $W$ so that, by transforming word embeddings from $E$ to $WE$, the LM which originally simulates the text distribution starting with $\pi$ will now turn to be equivalent to a distribution initiating from $\pi^{\prime}$.

### 3.3 LM-Switch Formulation

Inspired by the discovery in the section above, we propose LM-Switch to apply a linear transform in the word embedding space. LM-Switch is conceptually simple and straightforward to implement.
An illustration of LM-Switch is presented in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")(a).
Specifically, let $M$ be a fixed language model with fixed parameters. We replace its each output word embeddings $\mathbf{e}_{v}$ with $\mathbf{e}_{v}+\epsilon W\mathbf{e}_{v}$, and call the resulting language model $M^{\prime}\=M(\epsilon W)$ a “switched model”. Here the “switch matrix” $W$ is the only learnable parameters determining the effect of LM-Switch, and $\epsilon$ is a manually adjustable scalar indicating the polarity and intensity of the “switch value”. Without loss of generality, we arbitrarily pick a small value $\epsilon_{0}\=1e-3$ as the default switch value.222Using $\epsilon W$ achieves the equivalent effect as $k\epsilon\cdot k^{-1}W)$ when $k\neq 0$, so the absolute value of $\epsilon$ itself is only meaningful when also considering the magnitude of $W$. We use $P(\mathbf{o}|\epsilon W)$ to denote the probability of $M^{\prime}$ generating sequence $\mathbf{o}\=(o_{1},\cdots,o_{T})$.
Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")(b, c) shows the training and generation process of LM-Switch. During training, we use the positively switched model $M(\epsilon W)$ to fit the positively labelled texts, with maximal likelihood as the training objective. When negatively texts are available, we also fit them with $M(-\epsilon W)$. When generating with LM-Switch, the user only needs to specify a switch value $\epsilon$ and then decode language model.
More details are in Appendix[D](#A4 "Appendix D Implementation Details ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

### 3.4 Linearity Properties

The conceptually simple design of LM-Switch makes it an architecture-agnostic plug-in to diverse language models. We demonstrate that LM-Switch maintains a linearity guarantee, regardless of the model architecture applied to. The linearity enables it to achieve desired capacities of continuous and compositional control. More specifically, even if we only train with a few discrete values of $\epsilon$ as discussed in Section[3.3](#S3.SS3 "3.3 LM-Switch Formulation ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), our model allows for fine-grained adjustment of switch value, and decoding with configuration $M\left(k\epsilon W\right)$ as long $k$ is not too far off from $[-1,1]$. Moreover, if two LM-Switchs $W_{1},W_{2}$ are learned, their effect can be combined by decoding with $M(\epsilon_{1}W_{1}+\epsilon_{2}W_{2})$, where $\epsilon_{1},\epsilon_{2}$ are individual switch values for $W_{1},W_{2}$.
Proofs of the two theorems are provided in Appendix[C](#A3 "Appendix C Proof of Theorem 2 and 3 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

###### Assumption 2.

We assume a bound on the following values: all word embeddings are bounded by $\|\mathbf{e}_{v}\|_{2}\leq 1$; all contextual vectors are bounded by $\|\mathbf{c}(o_{1},\cdots,o_{i})\|_{2}\leq 1$; $W$ has its norm bounded by $\|W\|_{2}\leq D$.

###### Theorem 2.

(Continuous Control)
Let $\lambda_{\max}$ be the maximum eigen-value of $W$.
When varying $\epsilon$’s value, The switched model’s distribution is close to a linear interpolation from $M$ to $M^{\prime}$:

|  | $\|P(\cdot\mid k\epsilon,W)-\left(P(\cdot)(1-k)+kP(\cdot\mid\epsilon,W)\right)\|_{1}\leq 2|k(1-k)|\epsilon^{2}L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1)$ |  | (3) |
| --- | --- | --- | --- |

###### Theorem 3.

(Compositional Control) If we add two switching matrices $W_{1},W_{2}$ together and use it as a new switching matrix, their switching effects on distributions are approximately linearly combined:

|  | $\|P(\cdot\mid\epsilon,W_{1}+W_{2})-\left(P(\cdot\mid\epsilon,W_{1})+P(\cdot\mid\epsilon,W_{2})-P(\cdot)\right)\|_{1}\leq 10\epsilon dL^{2}D^{2}$ |  | (4) |
| --- | --- | --- | --- |

4 Applications
--------------

In this section we delve into a range of natural language applications: language detoxification, sentiment control, and political stance control. These tasks span multiple linguistic levels: lexical, semantic, pragmatic, etc. We follow*[[28](#bib.bib28 "")]* and use GPT2-Large333[https://huggingface.co/gpt2-large](https://huggingface.co/gpt2-large "") as the backbone language model.

*Table 1: Language model detoxification results. $\pm$ denotes standard deviation on 3 random seeds.*

| Model | Toxicity$\downarrow$ | | Fluency | Diversity$\uparrow$ | |  |
| --- | --- | --- | --- | --- | --- | --- |
| | Avg. max. toxicity | Toxicity prob. | Output ppl.$\downarrow$ | Dist-1 | Dist-2 | Dist-3 |
| GPT-2 (original) | 0.527 | 0.520 | 25.45 | 0.58 | 0.85 | 0.85 |
| PPLM (10%) | 0.520 | 0.518 | 32.58 | 0.58 | 0.86 | 0.86 |
| DAPT | 0.428 | 0.360 | 31.21 | 0.57 | 0.84 | 0.84 |
| GeDi | 0.363 | 0.217 | 60.03 | 0.62 | 0.84 | 0.83 |
| DExperts | 0.302 | 0.118 | 38.20 | 0.56 | 0.82 | 0.83 |
| DExperts (GPT3) | 0.293 | 0.111 | - | - | - | - |
| PromptT5 | 0.320 | 0.172 | 354.71 | 0.58 | 0.76 | 0.70 |
| MuCoLa | 0.308 | 0.088 | 29.92 | 0.55 | 0.82 | 0.83 |
| LM-Switch | 0.249±0.007 | 0.089±0.009 | 28.26 | 0.55 | 0.84 | 0.84 |

### 4.1 Language Detoxification

It is known that large pretrained LMs might generate toxic content that appears in the pre-training distribution*[[43](#bib.bib43 ""), [8](#bib.bib8 "")]*, such as inaccurate information, harmful stereotypes, and unethical content.
Language model detoxification is the task of mitigating or avoiding these generations, in order to enable safe usage of language models.

Setting: Following*[[28](#bib.bib28 "")]*, we use Jigsaw Unintended Bias in Toxicity Classification Kaggle challenge444<https://bit.ly/3cvG5py> as the training dataset.
For evaluation, we use 10K nontoxic prompts from the RealToxicityPrompts dataset*[[8](#bib.bib8 "")]*. We randomly generate 25 sentences of up to 20 tokens using nucleus sampling*[[11](#bib.bib11 "")]* with $p\=0.9$. Then the toxicity scores (in range $[0,1]$) of generations are evaluated using Perspective API555<https://perspectiveapi.com>. Two metrics are reported: the average of maximal toxicity for each prompt (“Avg. max. toxicity”), and the probability of generating $>0.5$ toxicity at least once for each prompt (“Toxicity prob.”). We also evaluate generation quality in terms of fluency (perplexity score measured by a GPT2-large) and diversity (Dist-{1, 2, 3}: the portion of distinct {1, 2, 3}-grams). When decoding, we use a switch value of $5\epsilon_{0}$ for generation, which is selected by a balance of scores and quality. Ablation study on switch values can be found in Appendix[E](#A5 "Appendix E Ablation Study ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

Baselines:DExperts trains positive and negative label classifiers, and uses the difference in two classifiers’ scores to offset the LM’s original logits. DAPT *[[10](#bib.bib10 "")]* simply further pretrains the language model on the non-toxic subset (filtered by Perspective API) of OpenWebText Corpus (OWT)*[[9](#bib.bib9 "")]*. PPLM *[[5](#bib.bib5 "")]* learns to use the gradients of the label classifier to update the LM’s hidden representations. GeDi *[[20](#bib.bib20 "")]* is a model that uses Bayesian rule for class-conditioned LM generation. MuCoLa *[[22](#bib.bib22 "")]* models the text generation as an optimization problem regarding the classifier scores. PromptT5 *[[40](#bib.bib40 "")]* T5 is a pre-trained LM optimized for prompt-based task solving, and we use “Complete this sentence so that it embodies a {positive/negative} sentiment:” to prompt T5.
Finally, vanilla GPT2 is also adopted as a unguided baseline.

Results and Analysis: We present the results in Table[1](#S4.T1 "Table 1 ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). Despite the simple design, LM-Switch achieves the best detoxification scores on both metrics, reducing Avg. max. toxicity by $>6\%$ absolute percentages. It is also noteworthy that LM-Switch also demonstrates reasonable balance on fluency (2nd lowest perplexity score) and diversity (same-level Dist-k scores with baselines).

*Table 2: Results on sentiment control task.
The upper half displays positive control task and requires higher positivity score, and vise versa for the lower half.
While no one model achieves best performance on all metrics, LM-Switch takes 2nd to 3rd place in all positivity metrics despite using a much simpler design and smaller parameter size. $\pm$ denotes standard deviation on 3 random trials.*

| Target | Model | Sentiment Positivity / % | | | Fluency | Diversity$\uparrow$ | |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | Positive | Neutral | Negative | Output ppl.$\downarrow$ | Dist-1 | Dist-2 | Dist-3 |
| prompts | prompts | prompts | | | | |
| Positive$\uparrow$ | LM-Switch |  | 90.70±2.51 | 41.23±6.33 | 41.20 | 0.46 | 0.78 | 0.83 |
| | DExperts | | 94.46 | 36.42 | 45.83 | 0.56 | 0.83 | 0.83 |
| DExperts (pos) |  | 79.83 | 43.80 | 64.32 | 0.59 | 0.86 | 0.85 |
| GeDi |  | 86.01 | 26.80 | 58.41 | 0.57 | 0.80 | 0.79 |
| DAPT |  | 77.24 | 14.17 | 30.52 | 0.56 | 0.83 | 0.84 |
| PPLM (10%) |  | 52.68 | 8.72 | 142.11 | 0.62 | 0.86 | 0.85 |
| PromptT5 |  | 68.12 | 15.41 | 362.30 | 0.58 | 0.78 | 0.72 |
| GPT-2 (original) | 99.08 | 50.02 | 0.00 | 29.28 | 0.58 | 0.84 | 0.84 |
| Negative$\downarrow$ | PromptT5 | 69.93 | 25.78 |  | 450.68 | 0.60 | 0.78 | 0.70 |
| | PPLM (10%) | 89.74 | 39.05 | | 181.78 | 0.63 | 0.87 | 0.86 |
| DAPT | 87.43 | 33.28 |  | 32.86 | 0.58 | 0.85 | 0.84 |
| GeDi | 39.57 | 8.73 |  | 84.11 | 0.63 | 0.84 | 0.82 |
| DExperts (neg) | 61.67 | 24.32 |  | 65.11 | 0.60 | 0.86 | 0.85 |
| DExperts | 35.99 | 3.77 |  | 45.91 | 0.60 | 0.84 | 0.83 |
| LM-Switch | 54.84±8.01 | 8.02±2.32 |  | 57.74 | 0.48 | 0.78 | 0.80 |

### 4.2 Sentiment Control

We also evaluate LM-Switch’s performance on an extensively studied generation task controlled by sentiment. This ability can be found useful when tailoring persuasive and emotionally appealing messages to specific target audiences in marketing or advertising, or to create personalized and engaging user experience in chatbot systems.

Setting: We follow the setting in *[[28](#bib.bib28 "")]* and use Stanford Sentiment Treebank (SST-5)*[[44](#bib.bib44 "")]* as training data, where we use texts with labels 1$\sim$2 as negative samples, and those with 4$\sim$5 labels as positive samples.
For evaluation, we use the HuggingFace’s sentiment classifier*[[47](#bib.bib47 "")]*. The generation prompts are a subset of OpenWebText Corpus filtered by the sentiment analysis classifier. Models are applied on these prompts for 25 times to generate up to 20 tokens. We then measure the average percentage of positive generations for each prompt as the “Positivity” score. Similar to the detoxification task, we use $5\epsilon_{0}$ for positive sentiment and $-5\epsilon_{0}$ for negative sentiment control.

Baselines: Besides the baselines used in detoxification, two variants of DExperts: DExperts (pos) and DExperts (neg) which only use one of the two classifiers for guiding generation are also listed.

Results: Table[2](#S4.T2 "Table 2 ‣ 4.1 Language Detoxification ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") presents the full results. Here the scores are more mingled with no single model being the best on all metrics. LM-Switch, despite a much simper and smaller model, takes 2nd to 3rd place in all sentiment metrics and achieves reasonable balance on fluency and diversity.

Continuous and Compositional Control: Another advantage of LM-Switch is that we can perform a continuous and compositional control, as predicted in Theorem[2](#Thmtheorem2 "Theorem 2. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") and [3](#Thmtheorem3 "Theorem 3. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). A visualization is shown in Figure[2](#S4.F2 "Figure 2 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). Specifically, in Figure[2(a)](#S4.F2.sf1 "In Figure 2 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") we plot the distribution shift when adjusting sentiment switch $\epsilon$. We also curve the maximal likelihood estimated Beta distribution. In Figure[2(b)](#S4.F2.sf2 "In Figure 2 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") we observe that LM-Switch can compositionally control sentiment and toxicity, even though there exists mutual influence between these two factors (e.g., a negative sentiment might also lead to more toxic comments).

<img src='figures/linear.jpg' alt='Refer to caption' title='' width='479' height='454' />

*(a) Continuous control on sentiment with $\epsilon$ in $-5\epsilon_{0}\sim 5\epsilon_{0}$ results in a sentiment distribution shift. Color indicates sentiment and height indicates frequency / density.*

<img src='figures/compositional.jpg' alt='Refer to caption' title='' width='479' height='454' />

*(b) Compositional control sentiment ranging in $-5\epsilon_{0}\sim 5\epsilon_{0}$ and toxicity in $0\sim 5\epsilon_{0}$. Color means sentiment and height is toxicity.*

*Figure 2: Continuous and compositional control using LM-Switch.*

*Table 3: The language model generations for two example scenarios conditioned on political stances.*

| Stance | Generations |
| --- | --- |
| Anti-Russia | Russia’s annexation of Crimea was an invasion of Ukraine’s sovereign territory, but Russia insists that Ukraine’s Crimea People’s Republic is legally Russian territory. |
| | NATO expansion “has nothing to do” with Europe, but Putin wants war. And while he might start war over Ukraine right away, his true motives for fighting may not be limited to his ‘interest’ in Ukraine. |
| Pro-Russia | Russia’s annexation of Crimea was nothing short of a geopolitical earthquake: it has been the biggest geopolitical event of the year. |
| | NATO expansion under pressure. There is growing pressure on NATO and Washington to halt the military buildup planned for Central Asia and for Russia, which almost certainly lead to a new military confrontation. |
| Times of India | EU diplomat seeks action against India on at least 1 issueThe European Union’s chief diplomat seeks action against India on at least 1 issue, ahead of its talks with the European Union. |
| | The EU diplomat said the view that Europe should embrace India to address its growth is a “laundromat” by his description and he said he will raise the issue with his counterparts in Delhi. |
| Reuters | The EU diplomat said that the EU should have a sanctions policy but that the sanctions rules need to be tightened to be more rigorous. |
| | EU diplomat had his visa revoked after he raised concerns he could be targeted by India’s crackdown on foreigners following an attack on an Indian diplomatic post in Kabul, it emerged on Tuesday. |

<img src='x2.png' alt='Refer to caption' title='' width='461' height='259' />

*(a) Transferring a LM-Switch to other LMs*

<img src='x3.png' alt='Refer to caption' title='' width='461' height='259' />

*(b) Effect of learning from fewer data.*

*Figure 3: Measuring the transferability and data efficiency of LM-Switch.*

### 4.3 Political Stance and Agenda: Case Study

We also case-study the application of political stance control of LM. This application can be beneficial for generating diverse and balanced perspectives in genres such as news articles, and also increasing the likelihood of the content being well-received by aligning with values of the target audience.
We study two case scenarios.
The first scenarios is pro-Russian v.s. anti-Russian, where we collect 744 English tweets and manually label them as either “pro-Russia” (454 tweets) or “anti-Russia” (290 tweets). After training, we prompt our model to generate from both stances on a list of topics. In the second scenario, we select 5 pairs of news articles with contrastive political stances from Gound News666<https://ground.news>. In each pair, we train one LM-Switch to learn each article, and then generate on the topic to see their differences.
Excerpt examples of generation are shown in Table[3](#S4.T3 "Table 3 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") with indicative text spans manually bolded. Appendix[I](#A9 "Appendix I More Details on Political Stance ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") describes the detailed setting and more results. We can observe differences in their wording (“invasiong” v.s. “geopolitical earthquak”) and agenda selection (“action against India” v.s. “India’s crackdown on foreigners”).

5 Analysis
----------

### 5.1 Data, Parameter and Computational Efficiency

Thanks to its simple design, LM-Switch enjoys efficiency in multiple perspectives. First, as demonstrated in Section[4.3](#S4.SS3 "4.3 Political Stance and Agenda: Case Study ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), our model is capable of learning from only one article. As a more rigorous study, we vary the detoxification dataset size from 30 to 10k and measure LM-Switch’s performance in Figure[3](#S4.F3 "Figure 3 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")(b). We see that as few as 30 data points still enable LM-Switch to achieve high detoxification scores (0.322), but also induces a high perplexity as LM-Switch overfits. When dataset size exceeds 3k LM-Switch acquires a good balance between detoxification and generation quality. We would also like to point readers to other types of efficiency in Appendix[H](#A8 "Appendix H Parameter and Time Efficiency ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), where our model only uses 1% of baseline’s parameter size, and uses a low computation overhead during decoding.

### 5.2 Transferring a Learned LM-Switch to Another Model

A much desired property of LM-Switch, because of its theoretical soundness, is its transferability to other language models.
Details and derivations of LM-Switch transfer are in Appendix[F](#A6 "Appendix F Detials of Transferring LM-Switch to Other Language Models ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), but intuitively explaining, we work by identifying a linear mapping $H$ from target LM word embeddings to source LM word embeddings. Then the matrix $H^{\top}WH$ can be inserted to the target LM as LM-Switch.
Figure[3](#S4.F3 "Figure 3 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")(a) shows the performance after we transfer the LM-Switch learned on GPT2-large to LMs of other sizes, ranging from gpt2 (124M) to GPT-J-6B (6B). We can see a uniform improvement in transferred LM-Switchs, with GPT2 and GPT2-medium getting similar scores (0.307 and 0.308) to the best baseline (DExperts).

### 5.3 Interpretability

Finally, we investigate the parameters of LM-Switch and how they correlate with LM word embeddings. This study provides a lens through which to examine the connection between conditions and word choices.
In the detoxification experiment, we conduct a SVD decomposition of the learned $W$. Among $S,V,D$, the $D$ component can be interpreted as a ranked list of most “magnified” row dimension in the transformation $W$. We then take its first 9 rows, and list most influenced words in Table[4](#S5.T4 "Table 4 ‣ 5.3 Interpretability ‣ 5 Analysis ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). Dimensions 2, 4, 6 are filtered out as they only match non-English tokens. Although offensive to read, this table helps us understand what kind of words are most related to the toxicity and thus suppressed by LM-Switch in generation.
More details are explained in Appendix[G](#A7 "Appendix G Details of Investigating Interpretability ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

*Table 4: Detected tokens that are most influenced by LM-Switch on detoxification task.*

| Dim. | Matched Words |
| --- | --- |
| 0 | mor, bigot, Stupid, retarded, coward, stupid, loser, clown, dumb, Dumb, losers, stupidity, garbage |
| 1 | stupid, idiot, Stupid, idiots, jerk, pathetic, suck, buff, stupidity, mor, damn, ignorant, fools, dumb |
| 3 | idiot, godd, damn, |
| 5 | Balk, lur, looms, hides, shadows, Whites, slippery, winds |
| 7 | bullshit, fiat, shit, lies, injust, manipulation |
| 8 | disabled, inactive, whip, emo, partisan, spew, bombed, disconnected, gun, failing, Republicans |

6 Conclusions and Future Work
-----------------------------

In this work, we show the promise and efficacy of LM-Switch, a theoretically grounded, simple and lightweight approach for the conditioning of generative language models. Leveraging the insights from Hidden Markov Models and their relationship with language models, LM-Switch can model diverse tasks and achieve comparable or superior performance to baselines in language model detoxification and generation control. It is particularly notable that LM-Switch requires significantly fewer parameters and decoding time, allows for continuous and compositional control, and can be transferred to other language models. We show that LM-Switch can also be used for interpreting the wording in conditions.
For future research, it is worth examining the boundary of LM-Switch: what can and cannot be modelled and why. Additionally, theoretical relation between LM-Switch and other techniques such as prompt engineering and prompt tuning is worth further study.

### Limitations

One limitation of LM-Switch is that it works on word embeddings and focuses on conditions related to wording. This restricts its capability to deal with more complex tasks, such as syntactic trees or persuasive techniques that involve logical reasoning. Additionally, our model is dependent on word embeddings, so the model cannot work with language model APIs that do not provide direct access to these embeddings.

### Acknowledgements

This work was supported in part by US DARPA KAIROS Program No. FA8750-19-2-1004, AIDA Program No. FA8750-18-2-0014, SemaFor Program No. HR001120C0123, INCAS Program No. HR001121C0165 and MIPS Program No. HR00112290105.
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation here on.

References
----------

* [1]Soumya Barikeri, Anne Lauscher, Ivan Vulić, and Goran Glavaš.RedditBias: A real-world resource for bias evaluation and
debiasing of conversational language models.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 1941–1955,
Online, August 2021. Association for Computational Linguistics.
* [2]Leonard E Baum, Ted Petrie, George Soules, and Norman Weiss.A maximization technique occurring in the statistical analysis of
probabilistic functions of markov chains.The annals of mathematical statistics, 41(1):164–171, 1970.
* [3]Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai.Man is to computer programmer as woman is to homemaker? debiasing
word embeddings, 2016.
* [4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.Language models are few-shot learners.Advances in neural information processing systems,
33:1877–1901, 2020.
* [5]Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero
Molino, Jason Yosinski, and Rosanne Liu.Plug and play language models: A simple approach to controlled text
generation.In International Conference on Learning Representations.
* [6]Emily Dinan, Angela Fan, Adina Williams, Jack Urbanek, Douwe Kiela, and Jason
Weston.Queens are powerful too: Mitigating gender bias in dialogue
generation.In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 8173–8188, Online, November
2020. Association for Computational Linguistics.
* [7]Xianghong Fang, Jian Li, Lifeng Shang, Xin Jiang, Qun Liu, and Dit-Yan Yeung.Controlled text generation using dictionary prior in variational
autoencoders.In Findings of the Association for Computational Linguistics:
ACL 2022, pages 97–111, Dublin, Ireland, May 2022. Association for
Computational Linguistics.
* [8]Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith.Realtoxicityprompts: Evaluating neural toxic degeneration in language
models.In Findings of the Association for Computational Linguistics:
EMNLP 2020, pages 3356–3369, 2020.
* [9]Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex.Openwebtext corpus (2019).URL http://Skylion007. github. io/OpenWebTextCorpus, 2019.
* [10]Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy,
Doug Downey, and Noah A Smith.Don’t stop pretraining: Adapt language models to domains and tasks.In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 8342–8360, 2020.
* [11]Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi.The curious case of neural text degeneration.In International Conference on Learning Representations.
* [12]Xiaodan Hu, Pengfei Yu, Kevin Knight, Heng Ji, Bo Li, and Honghui Shi.Muse: Textual attributes guided portrait painting generation.In Prof. The 3rd IEEE Workshop on Artificial Intelligence for
Art Creation, 2021.
* [13]Zhe Hu, Zhiwei Cao, Hou Pong Chan, Jiachen Liu, Xinyan Xiao, Jinsong Su, and
Hua Wu.Controllable dialogue generation with disentangled multi-grained
style specification and attribute consistency reward.IEEE/ACM Transactions on Audio, Speech, and Language
Processing, 31:188–199, 2022.
* [14]Ali Jahanian, Lucy Chai, and Phillip Isola.On the" steerability" of generative adversarial networks.In International Conference on Learning Representations.
* [15]Masahiro Kaneko and Danushka Bollegala.Debiasing pre-trained contextualised embeddings.In Proceedings of the 16th Conference of the European Chapter of
the Association for Computational Linguistics: Main Volume, pages
1256–1266, Online, April 2021. Association for Computational Linguistics.
* [16]Masahiro Kaneko, Danushka Bollegala, and Naoaki Okazaki.Debiasing isn’t enough! – on the effectiveness of debiasing
MLMs and their social biases in downstream tasks.In Proceedings of the 29th International Conference on
Computational Linguistics, pages 1299–1310, Gyeongju, Republic of Korea,
October 2022. International Committee on Computational Linguistics.
* [17]Pei Ke, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, Xiaoyan Zhu, and Minlie Huang.CTRLEval: An unsupervised reference-free metric for evaluating
controlled text generation.In Proceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 2306–2319, Dublin,
Ireland, May 2022. Association for Computational Linguistics.
* [18]Diederik P Kingma and Jimmy Ba.Adam: A method for stochastic optimization.arXiv preprint arXiv:1412.6980, 2014.
* [19]Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa.Large language models are zero-shot reasoners.In Advances in Neural Information Processing Systems.
* [20]Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar,
Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani.Gedi: Generative discriminator guided sequence generation.In Findings of the Association for Computational Linguistics:
EMNLP 2021, pages 4929–4952, 2021.
* [21]Sachin Kumar, Eric Malmi, Aliaksei Severyn, and Yulia Tsvetkov.Controlled text generation as continuous optimization with multiple
constraints.Advances in Neural Information Processing Systems,
34:14542–14554, 2021.
* [22]Sachin Kumar, Biswajit Paria, and Yulia Tsvetkov.Gradient-based constrained sampling from language models.In Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 2251–2277, 2022.
* [23]Juncen Li, Robin Jia, He He, and Percy Liang.Delete, retrieve, generate: a simple approach to sentiment and style
transfer.In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 1865–1874, 2018.
* [24]Sha Li, Ruining Zhao, Manling Li, Heng Ji, Chris Callison-Burch, and Jiawei
Han.Open-domain hierarchical event schema induction by incremental
prompting and verification.In Proc. The 61st Annual Meeting of the Association for
Computational Linguistics (ACL2023), 2023.
* [25]Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B
Hashimoto.Diffusion-lm improves controllable text generation.Advances in Neural Information Processing Systems,
35:4328–4343, 2022.
* [26]Xiang Lisa Li and Percy Liang.Prefix-tuning: Optimizing continuous prompts for generation.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, 2021.
* [27]Paul Pu Liang, Irene Mengze Li, Emily Zheng, Yao Chong Lim, Ruslan
Salakhutdinov, and Louis-Philippe Morency.Towards debiasing sentence representations.In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 5502–5515, Online, July 2020. Association
for Computational Linguistics.
* [28]Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula,
Noah A Smith, and Yejin Choi.Dexperts: Decoding-time controlled text generation with experts and
anti-experts.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 6691–6706, 2021.
* [29]Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula,
Noah A. Smith, and Yejin Choi.DExperts: Decoding-time controlled text generation with experts and
anti-experts.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 6691–6706,
Online, August 2021. Association for Computational Linguistics.
* [30]Ximing Lu, Peter West, Rowan Zellers, Ronan Le Bras, Chandra Bhagavatula, and
Yejin Choi.Neurologic decoding:(un) supervised neural text generation with
predicate logic constraints.In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 4288–4299, 2021.
* [31]Nicholas Meade, Elinor Poole-Dayan, and Siva Reddy.An Empirical Survey of the Effectiveness of Debiasing
Techniques for Pre-trained Language Models.In Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long
Papers), pages 1878–1898, Dublin, Ireland, 2022. Association for
Computational Linguistics.
* [32]Moin Nadeem, Anna Bethke, and Siva Reddy.StereoSet: Measuring stereotypical bias in pretrained language
models.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 5356–5371,
Online, August 2021. Association for Computational Linguistics.
* [33]Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman.CrowS-pairs: A challenge dataset for measuring social biases in
masked language models.In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 1953–1967, Online, November
2020. Association for Computational Linguistics.
* [34]Ali Omrani, Alireza Salkhordeh Ziabari, Charles Yu, Preni Golazizian, Brendan
Kennedy, Mohammad Atari, Heng Ji, and Morteza Dehghani.Social-group-agnostic bias mitigation via the stereotype content
model.In Proc. The 61st Annual Meeting of the Association for
Computational Linguistics (ACL2023), 2023.
* [35]OpenAI.Gpt-4 technical report, 2023.
* [36]Damian Pascual, Beni Egressy, Clara Meister, Ryan Cotterell, and Roger
Wattenhofer.A plug-and-play method for controlled text generation.In Findings of the Association for Computational Linguistics:
EMNLP 2021, pages 3973–3997, Punta Cana, Dominican Republic, November 2021.
Association for Computational Linguistics.
* [37]Matt Post and David Vilar.Fast lexically constrained decoding with dynamic beam allocation for
neural machine translation.In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 1314–1324, 2018.
* [38]Lianhui Qin, Sean Welleck, Daniel Khashabi, and Yejin Choi.Cold decoding: Energy-based constrained text generation with langevin
dynamics.In Advances in Neural Information Processing Systems.
* [39]Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al.Improving language understanding by generative pre-training.2018.
* [40]Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J Liu.Exploring the limits of transfer learning with a unified text-to-text
transformer.The Journal of Machine Learning Research, 21(1):5485–5551,
2020.
* [41]Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg.Null it out: Guarding protected attributes by iterative nullspace
projection.In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 7237–7256, Online, July 2020. Association
for Computational Linguistics.
* [42]Timo Schick, Sahana Udupa, and Hinrich Schütze.Self-Diagnosis and Self-Debiasing: A Proposal for Reducing
Corpus-Based Bias in NLP.Transactions of the Association for Computational Linguistics,
9:1408–1424, 12 2021.
* [43]Emily Sheng, Kai-Wei Chang, Prem Natarajan, and Nanyun Peng.The woman worked as a babysitter: On biases in language generation.In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 3407–3412, 2019.
* [44]Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning,
Andrew Y Ng, and Christopher Potts.Recursive deep models for semantic compositionality over a sentiment
treebank.In Proceedings of the 2013 conference on empirical methods in
natural language processing, pages 1631–1642, 2013.
* [45]Nishant Subramani, Nivedita Suresh, and Matthew E Peters.Extracting latent steering vectors from pretrained language models.In Findings of the Association for Computational Linguistics:
ACL 2022, pages 566–581, 2022.
* [46]Kellie Webster, Xuezhi Wang, Ian Tenney, Alex Beutel, Emily Pitler, Ellie
Pavlick, Jilin Chen, Ed Chi, and Slav Petrov.Measuring and reducing gendered correlations in pre-trained models,
2021.
* [47]Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
et al.Transformers: State-of-the-art natural language processing.In Proceedings of the 2020 conference on empirical methods in
natural language processing: system demonstrations, pages 38–45, 2020.
* [48]Kun Yan, Lei Ji, Huaishao Luo, Ming Zhou, Nan Duan, and Shuai Ma.Control image captioning spatially and temporally.In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 2014–2025,
Online, August 2021. Association for Computational Linguistics.
* [49]Ke Yang, Charles Yu, Yi Fung, Manling Li, and Heng Ji.Adept: A debiasing prompt framework.In Proc. Thirty-Seventh AAAI Conference on Artificial
Intelligence (AAAI2023), 2023.
* [50]Kevin Yang and Dan Klein.Fudge: Controlled text generation with future discriminators.In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 3511–3535, 2021.
* [51]Kevin Yang and Dan Klein.FUDGE: Controlled text generation with future discriminators.In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 3511–3535, Online, June 2021. Association for
Computational Linguistics.
* [52]Charles Yu, Sullam Jeoung, Anish Kasi, Pengfei Yu, and Heng Ji.Unlearning bias in language models by partitioning gradients.In Proc. The 61st Annual Meeting of the Association for
Computational Linguistics (ACL2023) Findings, 2023.
* [53]Wenhao Yu, Chenguang Zhu, Zaitang Li, Zhiting Hu, Qingyun Wang, Heng Ji, and
Meng Jiang.A survey of knowledge-enhanced text generation.In ACM Computing Survey, 2022.
* [54]Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou, and Dawei Song.A survey of controllable text generation using transformer-based
pre-trained language models, 2022.
* [55]Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason
Weston.Personalizing dialogue agents: I have a dog, do you have pets too?In Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 2204–2213, 2018.
* [56]Taolin Zhang, Junwei Dong, Jianing Wang, Chengyu Wang, Ang Wang, Yinghui Liu,
Jun Huang, Yong Li, and Xiaofeng He.Revisiting and advancing Chinese natural language understanding
with accelerated heterogeneous knowledge pre-training.In Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing: Industry Track, pages 560–570, Abu Dhabi, UAE,
December 2022. Association for Computational Linguistics.
* [57]Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao,
Jianfeng Gao, Jingjing Liu, and William B Dolan.Dialogpt: Large-scale generative pre-training for conversational
response generation.In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics: System Demonstrations, pages 270–278, 2020.
* [58]Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, and
Mrinmaya Sachan.Controlled text generation with natural language instructions.arXiv preprint arXiv:2304.14293, 2023.

Appendix A Broader Impacts
--------------------------

The intended use of this work is to contribute to advancements in fine-grained and efficient control of language generation in AI, with experiments shown on sentiment modulation, political stance adjustment, and language detoxification. We do not aim to create a tool for manipulating public opinion or promoting specific political ideologies, but instead to provide methods for enhancing the reasoning interpretability and safety of AI language models. Our techniques offer potential for fine-tuned sentiment adjustment and toxic content mitigation, thereby contributing to more reliable, unbiased, and respectful language generation systems.
We would like to emphasize that on the problem of language model toxicity, we limit our model to modelling detoxification only. This encourages positive and social benefiting usage of our model as well as general language models.

Appendix B Proof of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In this proof, we start by claiming a construction of matrix $W$. Then we prove that when assumptions [1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") hold, $W$ can change each conditional likelihood function from $p(v_{i}\mid o_{1},\cdot,o_{i-1},\pi)$ to $p(v_{i}\mid o_{1},\cdot,o_{i-1},\pi^{\prime})$ up to a constant factor. Finally by chaining the conditional likelihoods, we see that $W$ can change the sentence-level probability distribution of the HMM from $\pi$-initialization to $\pi^{\prime}$-initialization.

We first construct a helper matrix
$W^{\prime}\=\begin{pmatrix}I_{d_{s}}\&0\\
0\&\Lambda\end{pmatrix}$
so that $\Lambda$ is diagonal and $W^{\prime}\phi_{\text{init}}\=\phi_{\text{init}}^{\prime}$. Such a solution exists as we assume $\phi_{\text{init},\text{condition}}$ contains only non-zero values.
Then we can construct the matrix $W$ as
$W\=R_{1}^{+}\Phi^{\top}W^{\prime}\Phi{R_{2}^{+}}^{\top}$, where $R_{1}^{+},R_{2}^{+}$ are pseudo-inverse matrices of $R_{1},R_{2}$, respectively.

###### Lemma 4.

$T\Phi^{\top}W^{\prime}\Phi\=\Phi^{\top}W^{\prime}\Phi T$.

###### Proof.

First, it is easy to see that, by Assumption[1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").1 and Assumption[1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").2, the representation matrix $\Phi$ is row-orthonormal to constant $C_{2}$:

|  | $\Phi\Phi^{T}\=C_{2}I_{d}$ |  |
| --- | --- | --- |

.

Then we have the following proof:

|  | $\displaystyle T\Phi^{\top}W^{\prime}\Phi\=$ | $\displaystyle\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}\Phi\Phi^{\top}W^{\prime}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=C_{2}\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}\begin{pmatrix}I_{d_{s}}\&0\\ 0\&\Lambda\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=C_{2}\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&\Lambda\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=C_{2}\Phi^{\top}\begin{pmatrix}I_{d_{s}}\&0\\ 0\&\Lambda\end{pmatrix}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\Phi^{\top}W^{\prime}\Phi\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=W^{\prime}T$ |  | ∎ |
| --- | --- | --- | --- | --- |

###### Lemma 5.

$\forall v\in V$, we have that, $\Phi\text{diag}(\mathbf{p}(o))\Phi^{\top}W^{\prime}\Phi\=W^{\prime}\Phi\text{diag}(\mathbf{p}(o))$.

###### Proof.

To prove this, we first prove that $\Phi\text{diag}(\mathbf{p}(o))\Phi^{\top}$ has the form
$\begin{pmatrix}A\&0\\
0\&\Lambda^{\prime}\end{pmatrix}$, where $\Lambda^{\prime}$ is also diagonal. This is equivalent to saying that, for any two one-hot vectors $\mathbf{e}(i),\mathbf{e}(j)$, if $i\in[d_{s}+1..d]$ or $j\in[d_{s}+1..d]$, then

|  | $\mathbf{e}_{i}^{\top}\Phi~{}\text{diag}(\mathbf{p}(o))\Phi\mathbf{e}_{j}^{\top}\=\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}p(v\mid h)\=f_{v}(i,j)\mathbf{1}(i\=j).$ |  |
| --- | --- | --- |

For any $i\neq j$,

|  | $\displaystyle\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}p(v\mid h)$ | $\displaystyle\=\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}\sum_{k\in[1..d]}\phi_{h,k}\theta{v,k}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\sum_{k\in[1..d]}\theta{v,k}\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}\phi_{h,k}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\sum_{k\not\in{i,j}}\theta_{v,k}\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}\phi_{h,k}+\theta_{v,i}\sum_{h\in\mathcal{H}}\phi_{h,i}^{2}\phi_{h,j}+\theta_{v,j}\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}^{2}$ |  |
| --- | --- | --- | --- |
|  | (by assumption [1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").3) | $\displaystyle\=0+\theta_{v,i}\sum_{h\in\mathcal{H}}\phi_{h,i}\phi_{h,j}$ |  |
| --- | --- | --- | --- |
|  | (by assumption [1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").2) | $\displaystyle\=0+0$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=0$ |  |
| --- | --- | --- | --- |

We then have the following:

|  | $\displaystyle\Phi\text{diag}(\mathbf{p}(o))\Phi^{\top}W^{\prime}\Phi$ | $\displaystyle\=\begin{pmatrix}A\&0\\ 0\&\Lambda^{\prime}\end{pmatrix}W^{\prime}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\begin{pmatrix}A\&0\\ 0\&\Lambda^{\prime}\Lambda\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=W^{\prime}\begin{pmatrix}A\&0\\ 0\&\Lambda^{\prime}\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=W^{\prime}\Phi\Phi^{\top}\begin{pmatrix}A\&0\\ 0\&\Lambda^{\prime}\end{pmatrix}\Phi$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=W^{\prime}\Phi\text{diag}(\mathbf{p}(o))$ |  |
| --- | --- | --- | --- |

∎

By combining Lemma[4](#Thmtheorem4 "Lemma 4. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") and[5](#Thmtheorem5 "Lemma 5. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), we have the following lemma:

###### Lemma 6.

|  | $T\text{diag}(\mathbf{p}(o))\Phi^{\top}W^{\prime}\Phi\=\Phi^{\top}W^{\prime}\Phi T\text{diag}(\mathbf{p}(o))$ |  |
| --- | --- | --- |

###### Proof.

|  | $\displaystyle T\text{diag}(\mathbf{p}(o))\Phi^{\top}W^{\prime}\Phi$ | $\displaystyle\=\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}\Phi\text{diag}(\mathbf{p}(o))\Phi^{\top}W^{\prime}\Phi$ |  |
| --- | --- | --- | --- |
|  | (by Lemma[5](#Thmtheorem5 "Lemma 5. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")) | $\displaystyle\=\Phi^{\top}\begin{pmatrix}T_{s}\&0\\ 0\&I_{d_{c}}\end{pmatrix}W^{\prime}\Phi\text{diag}(\mathbf{p}(o))$ |  |
| --- | --- | --- | --- |
|  | (by Lemma[4](#Thmtheorem4 "Lemma 4. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")) | $\displaystyle\=\Phi^{\top}W^{\prime}\Phi T\text{diag}(\mathbf{p}(o))$ |  |
| --- | --- | --- | --- |

∎

Finally, when we apply Lemma[4](#Thmtheorem4 "Lemma 4. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") and [6](#Thmtheorem6 "Lemma 6. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") to the language model formulation, we can see that the conditional likelihood function has been switched to:

|  | $\displaystyle\mathbf{p}_{W}(v_{i}\mid o_{1},\cdots,o_{i-1};\pi)$ | $\displaystyle\=\mathbf{c}(o_{1},\cdots,o_{i-1};\pi)WE$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{\pi^{\top}T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})TR_{1}^{\top}}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}WR_{2}P_{O}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{\pi^{\top}T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})T\Phi^{\top}W^{\prime}\Phi}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  | (by Lemma[4](#Thmtheorem4 "Lemma 4. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")) | $\displaystyle\=\frac{\pi^{\top}T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})\Phi^{\top}W^{\prime}\Phi T}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  | (by Lemma[6](#Thmtheorem6 "Lemma 6. ‣ Appendix B Proof of Theorem 1 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")) | $\displaystyle\=\frac{\pi\Phi^{\top}W^{\prime}\Phi^{\top}T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})T}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{\phi_{\text{init}}W^{\prime}\Phi T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})T}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\frac{\phi_{\text{init}}^{\prime}\Phi T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})T}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  | (by definition) | $\displaystyle\=\frac{\pi^{\prime}T\text{diag}(o_{1})T\cdots T\text{diag}(o_{i-1})T}{\pi^{\top}T\text{diag}(o_{1})T\cdots T\mathbf{p}(o_{i-1})}P_{O}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\propto\mathbf{c}(o_{1},\cdots,o_{i-1};\pi^{\prime})E$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\mathbf{p}(o_{i}\mid o_{1},\cdots,o_{i-1};\pi^{\prime})$ |  |
| --- | --- | --- | --- |

Therefore, the switched conditional likelihood is equivalent to an HMM initiating from $\pi^{\prime}$ (up to a normalization constant over vocavulary $\mathcal{O}$). By chaining the conditional likelihood functions, it is easy to see that the actual output sequence distribution is now:

|  | $\displaystyle p_{W,\text{normalized}}(o_{1},\cdots,o_{L};\pi)$ | $\displaystyle\=\prod_{i\=1}^{L}\text{normalize}_{\mathcal{O}}(\mathbf{p}_{W}(v_{i}\mid o_{1},\cdots,o_{i-1};\pi))$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=\prod_{i\=1}^{L}\mathbf{p}(o_{i}\mid o_{1},\cdots,o_{i-1};\pi^{\prime})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=p(o_{1},\cdots,o_{L};\pi^{\prime})$ |  |
| --- | --- | --- | --- |

This concludes our proof to Theoream[1](#Thmtheorem1 "Theorem 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

Appendix C Proof of Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") and [3](#Thmtheorem3 "Theorem 3. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### C.1 Proof to Theorem [2](#Thmtheorem2 "Theorem 2. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")

###### Proof.

As an abbreviation, we use $\mathbf{c}_{i}\=\text{Context}(v_{1},\cdots,v_{i-1})$ to denote the contextual feature up to step $i-1$ used in language models to compute word logits in each step. As

|  | $P(\mathbf{o}\mid\epsilon,W)\=\prod_{i\=0}^{L}\frac{e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o_{i}}}}{\sum_{v\in\mathcal{V}}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}$ |  |
| --- | --- | --- |

is multiplication of conditional probabilities, we have that its derivative to $\epsilon$ is:

|  | $\displaystyle\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)$ | $\displaystyle\=P(\mathbf{o}\mid\epsilon,W)\sum_{i\=0}^{L}\frac{d}{d\epsilon}\ln\frac{e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o_{i}}}}{\sum_{v\in\mathcal{V}}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=P(\mathbf{o}\mid\epsilon,W)\sum_{i\=0}^{L}\frac{\sum_{v\in\mathcal{V}}\mathbf{c}_{i}^{\top}W(\mathbf{e}_{o_{i}}-\mathbf{e}_{o})e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}{\sum_{v\in\mathcal{V}}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=P(\mathbf{o}\mid\epsilon,W)\sum_{i\=0}^{L}\mathbf{c}_{i}^{\top}W\left(\mathbf{e}_{o_{i}}-\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)}\mathbf{e}_{o}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle|\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)|$ | $\displaystyle\leq P(\mathbf{o}\mid\epsilon,W)\sum_{i\=0}^{L}\|\mathbf{c}_{i}\|_{2}\lambda_{\max}\left(\|\mathbf{e}_{o_{i}}\|_{2}+\max_{v\in\mathcal{V}}\|\mathbf{e}_{o}\|_{2}\right)$ |  |
| --- | --- | --- | --- |
|  | (by assumption[2](#Thmassumption2 "Assumption 2. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")) | $\displaystyle\leq P(\mathbf{o}\mid\epsilon,W)2L\lambda_{\max}.$ |  |
| --- | --- | --- | --- |

Therefore, summing up all tokens,

|  | $\|P(\mathbf{o}\mid\epsilon,W)-P(\mathbf{o})\|_{1}\leq 2\epsilon L\lambda_{\max}.$ |  |
| --- | --- | --- |

Using some intermediate results above, we also have

|  | $|\frac{d}{d\epsilon}\ln P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)|\leq 2\lambda_{\max}$ |  |
| --- | --- | --- |

and

|  | $\displaystyle P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)$ | $\displaystyle\leq P(\cdot\mid v_{1},\cdots,v_{i-1})e^{2\epsilon\lambda_{\max}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq P(\cdot\mid v_{1},\cdots,v_{i-1})(1+(e^{\lambda_{\max}}-1)\epsilon),$ |  |
| --- | --- | --- | --- |

|  | $\displaystyle P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)$ | $\displaystyle\geq P(\cdot\mid v_{1},\cdots,v_{i-1})e^{-2\epsilon\lambda_{\max}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq P(\cdot\mid v_{1},\cdots,v_{i-1})(1-2\epsilon\lambda_{\max})$ |  |
| --- | --- | --- | --- |

Therefore,

|  |  | $\displaystyle\|\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)}\mathbf{e}_{o}-\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1})}\mathbf{e}_{o}\|$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle\sum_{v\in\mathcal{V}}P(\cdot\mid v_{1},\cdots,v_{i-1})\cdot 2\epsilon(e^{\lambda_{\max}}-1)\|\mathbf{e}_{o}\|_{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\=$ | $\displaystyle 2\epsilon(e^{\lambda_{\max}}-1)\sum_{v\in\mathcal{V}}P(\cdot\mid v_{1},\cdots,v_{i-1})\|\mathbf{e}_{o}\|_{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 2\epsilon(e^{\lambda_{\max}}-1)$ |  |
| --- | --- | --- | --- |

Then we can bound the different in derivative:

|  |  | $\displaystyle\|\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)-\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)\mid_{\epsilon\=0}\|_{1}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle\sum_{i\=0}^{L}\mathbf{c}_{i}^{\top}W\left(\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1})}\mathbf{e}_{o}-\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)}\mathbf{e}_{o}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+\|P(\mathbf{o}\mid\epsilon,W)-P(\mathbf{o})\|_{1}\sum_{i\=0}^{L}\mathbf{c}_{i}^{\top}W\left(\mathbf{e}_{o_{i}}-\mathbb{E}_{v\sim P(\cdot\mid v_{1},\cdots,v_{i-1},\epsilon,W)}\mathbf{e}_{o}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 2\epsilon L(e^{\lambda_{\max}}-1)\lambda_{\max}+2\epsilon L^{2}\lambda_{\max}^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\text{(loose bound)}\leq$ | $\displaystyle 4\epsilon L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1)$ |  |
| --- | --- | --- | --- |

Finally we can bound the probability by integral:

|  | $\displaystyle\left\|P(\mathbf{o}\mid\epsilon,W)-\left(P(\mathbf{o})+\epsilon\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)\mid_{\epsilon\=0}\right)\right\|_{1}$ | $\displaystyle\leq\int_{\epsilon^{\prime}\=0}^{\epsilon}\|\frac{d}{d\epsilon^{\prime}}P(\mathbf{o}\mid\epsilon^{\prime},W)-\frac{d}{d\epsilon^{\prime}}P(\mathbf{o}\mid\epsilon^{\prime},W)\mid_{\epsilon^{\prime}\=0}\|_{1}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\int_{\epsilon^{\prime}\=0}^{\epsilon}4\epsilon L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\epsilon^{2}L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1)$ |  |
| --- | --- | --- | --- |

Substitute $\epsilon$ with $k\epsilon$, we have

|  | $\left\|P(\mathbf{o}\mid k\epsilon,W)-\left(P(\mathbf{o})+k\epsilon\frac{d}{d\epsilon}P(\mathbf{o}\mid\epsilon,W)\mid_{\epsilon\=0}\right)\right\|_{1}\leq 2k^{2}\epsilon^{2}L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1),$ |  |
| --- | --- | --- |

and so,

|  | $\left\|P(\mathbf{o}\mid k\epsilon,W)-\left((1-k)P(\mathbf{o})+kP(\mathbf{o}\mid\epsilon,W)\right)\right\|_{1}\leq 2|k(1-k)|\epsilon^{2}L^{2}\lambda_{\max}(e^{\lambda_{\max}}-1),$ |  |
| --- | --- | --- |

∎

### C.2 Proof to Theorem[3](#Thmtheorem3 "Theorem 3. ‣ 3.4 Linearity Properties ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space")

###### Proof.

Similar to the proof in Subsection[C.1](#A3.SS1 "C.1 Proof to Theorem 2 ‣ Appendix C Proof of Theorem 2 and 3 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), we decompose the derivative of $P(\mathbf{o}\mid\epsilon,W)$ down to each components:

|  | $\displaystyle\nabla_{W}P(\mathbf{o}\mid\epsilon W)$ | $\displaystyle\=P(\mathbf{o}\mid\epsilon W)\sum_{i\=0}^{L}\nabla_{W}\ln\frac{e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o_{i}}}}{\sum_{v\in\mathcal{V}}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=P(\mathbf{o}\mid\epsilon W)\sum_{i\=0}^{L}\left(\epsilon\mathbf{c}_{i}\mathbf{e}_{o_{i}}^{\top}-\frac{\sum_{v\in\mathcal{V}}\epsilon\mathbf{c}_{i}\mathbf{e}_{o}^{\top}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}{\sum_{v\in\mathcal{V}}e^{\mathbf{c}_{i}^{\top}(I+\epsilon W)\mathbf{e}_{o}}}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\=P(\mathbf{o}\mid\epsilon W)\sum_{i\=0}^{L}\epsilon\mathbf{c}_{i}\left(\mathbf{e}_{o_{i}}^{\top}-\mathbb{E}_{v\sim P(\cdot|\epsilon,W)}\mathbf{e}_{o}^{\top}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\|\nabla_{W}P(\mathbf{o}\mid\epsilon W)\|_{1}$ | $\displaystyle\leq 2\epsilon L$ |  |
| --- | --- | --- | --- |

So

|  | $P(\mathbf{o})-2\epsilon L\|W\|_{1}\leq P(\mathbf{o}\mid\epsilon,W)\leq P(\mathbf{o})+2\epsilon L\|W\|_{1}$ |  |
| --- | --- | --- |

Then the difference in derivative is bounded by:

|  | $\displaystyle\|\nabla_{W}P(\mathbf{o}\mid\epsilon,W)-\nabla_{W}P(\mathbf{o}\mid\epsilon,W)\mid_{W\=0}\|_{1}$ | $\displaystyle\leq\|P(\mathbf{o}\mid\epsilon,W)-P(\mathbf{o})\|_{1}\sum_{i\=0}^{L}|\epsilon\mathbf{c}_{i}\left(\mathbf{e}_{o_{i}}^{\top}-\mathbb{E}_{v\sim P(\cdot|\epsilon,W)}\mathbf{e}_{o}^{\top}\right)|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+\|P(\mathbf{o}\mid\epsilon,W)\|_{1}\epsilon\sum_{i\=0}^{L}\mathbf{c}_{i}(\mathbb{E}_{v\sim P(\cdot|\epsilon,W)}\mathbf{e}_{o}^{\top}-\mathbb{E}_{v\sim P(\cdot)}\mathbf{e}_{o}^{\top})|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 4\epsilon^{2}L^{2}\|W\|_{1}+2\epsilon L$ |  |
| --- | --- | --- | --- |

Therefore, $P(\mathbf{o}\mid\epsilon,W_{1}+W_{2})$ can be bounded by integral from $P(\mathbf{o}\mid\epsilon,W_{1})$:

|  |  | $\displaystyle\|(P(\mathbf{o}\mid\epsilon,W_{1}+W_{2})-P(\mathbf{o}\mid\epsilon,W_{1}))-(P(\mathbf{o}\mid\epsilon,W_{2})-P(\mathbf{o}))\|_{1}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle\int_{W^{\prime}\=0}^{W_{2}}\|\nabla_{W^{\prime}}P(\mathbf{o}\mid\epsilon,W_{1}+W^{\prime})-\nabla_{W^{\prime}}P(\mathbf{o}\mid\epsilon,W^{\prime})\|_{1}dW^{\prime}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle\int_{W^{\prime}\=0}^{W_{2}}(4\epsilon^{2}L^{2}(\|W_{1}\|_{1}+2\|W^{\prime}\|_{1})+2\epsilon L)dW^{\prime}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 4\epsilon^{2}L^{2}(\|W_{1}\|_{1}\|W_{2}\|_{1}+\|W_{2}\|_{2}^{2})+2\epsilon L\|W_{2}\|_{1}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 4\epsilon^{2}(d+1)L^{2}C_{3}^{2}+2\epsilon\sqrt{d}LC_{3}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 10\epsilon dL^{2}C_{3}^{2}$ |  |
| --- | --- | --- | --- |

∎

Appendix D Implementation Details
---------------------------------

In this paper, we leverage the HuggingFace package777<https://huggingface.co> and its model checkpoints. To implement LM-Switch, we simply wrap the self.forward function of language model transformer’s lm_head, and inject the computation formula of LM-Switch. In specific, as can be seen from Appendix[C](#A3 "Appendix C Proof of Theorem 2 and 3 ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"), the token logits are replaced w ith $\mathbf{c}^{\top}(I+\epsilon W)\mathbf{e}_{o}$, we change the computation order and first compute $\mathbf{c}^{\prime}\=\mathbf{c}+\epsilon W\mathbf{c})$, then compute $\mathbf{c}^{\prime\top}\mathbf{e}_{o}$. We find that this increases computational efficiency in practice, and avoids the problem caused by many Transformers sharing input and output word embedding parameters in storage. Another trick we applied in experiments is that, as there is a systematical distribution shift between pre-training corpus and domain-specific dataset (such as detoxification dataset and reviews), we add one “dummy” switch $W_{\text{dummy}}$ to fill this overall distribution gap. Therefore, for positive label training, we use model $M(\epsilon_{0}(W+W_{\text{dummy}}))$, and for negative label training, we use model $M(\epsilon_{0}(-W+W_{\text{dummy}}))$. This is where the 3M parameters come from in Table[6](#A8.T6 "Table 6 ‣ Appendix H Parameter and Time Efficiency ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

For optimization, we use Adam optimizer*[[18](#bib.bib18 "")]* with 1e-2 learning rate, and train for 1k steps. The switch matrix $W$ is initialized with Gaussian distribution of 0 mean and 1e-3 variance. Across all experiments, we run three initial seeds of 0, 1 and 2 for training. When required to generate 25 sentences on each prompt, we use random seeds 0, 1, 2, …, 24. Our hardware is one single Tesla V-100 GPU with 16GB CUDA memory.

Appendix E Ablation Study
-------------------------

We select the decoding hyper-parameter based on a balance on both task performance and generation quality. In the table below we present an ablation study of the hyper-parameter $\epsilon$:

*Table 5: Results on language model detoxification task by selecting different switch value $\epsilon$.*

| Switch Value $\epsilon$ | Toxicity$\downarrow$ | | Fluency$\downarrow$ | Diversity$\uparrow$ | |  |
| --- | --- | --- | --- | --- | --- | --- |
| | Avg. max. toxicity | Toxicity prob. | Output ppl. | Dist-1 | Dist-2 | Dist-3 |
| GPT-2 (original) | 0.527 | 0.520 | 25.45 | 0.58 | 0.85 | 0.85 |
| MuCoLa | 0.308 | 0.088 | 29.92 | 0.55 | 0.82 | 0.83 |
| LM-Switch ($\epsilon_{0}$) | 0.322 | 0.166 | 22.42 | 0.54 | 0.84 | 0.85 |
| LM-Switch ($2\epsilon_{0}$) | 0.293 | 0.146 | 23.43 | 0.54 | 0.84 | 0.85 |
| LM-Switch ($3\epsilon_{0}$) | 0.268 | 0.104 | 24.48 | 0.55 | 0.84 | 0.85 |
| LM-Switch ($4\epsilon_{0}$) | 0.246 | 0.096 | 25.85 | 0.55 | 0.84 | 0.85 |
| LM-Switch ($5\epsilon_{0}$) | 0.249 | 0.089 | 28.26 | 0.55 | 0.84 | 0.84 |

We can see that $5\epsilon_{0}$ achieves a good balance where Toxicity level is low, and the fluency (perplexity) does degrade too much and is still better than the best baseline MuCoLa.

Appendix F Detials of Transferring LM-Switch to Other Language Models
----------------------------------------------------------------------

In order to transfer a LM-Switch from one LM $M_{1}$ to another LM $M_{2}$, we notice that LM-Switch essentially adds one term $\mathbf{c}^{\top}W\mathbf{e}_{o}$ to the logits, where both $\mathbf{c}$ and $\mathbf{e}_{o}$ can be viewed as residing in word embedding space. Therefore, $W$ can be considered as a similarity matrix in $M_{1}$’s word embedding space. To use $W$ in $M_{2}$, we propose to map $M_{2}$’s word embedding space to that of $M_{1}$ before using $W$ as usual. The process works in 2 steps.

First, we identify a linear mapping from $W_{2}$ to $W_{1}$’s word embedding space. We start with building a list of anchor words. Specifically, we select the top 4k words that are shared by both vocabularies. We denote the token embedding matrices as $E_{1}^{\prime},E_{2}^{\prime}$ respectively. Then initialize a mapping $H$ with Gaussian distribution of 1e-3 initial variance, and we apply Adam optimizer 0.01 learning rate for 5k steps.

Secondly, After acquiring the mapping matrix $H$, we map both the context and embedding vectors to $H\mathbf{c}$ and $H\mathbf{e}_{o}$, respectively. So the additive term for language model $M_{2}$ is now $\mathbf{c}^{\top}H^{\top}WH\mathbf{e}_{o}$, which equivalent to using a switch matrix $H^{\top}WH$ for $M_{2}$.

This mapping process is not precise, as word embeddings between LMS are not linearly associated.
So we observe an increased instability in generation if we use large $\epsilon$. Therefore, we reduce the switch value to 0.1 of its original scale, that is $0.5\epsilon_{0}$ for generation. This is the setting for getting results in Figure[3(a)](#S4.F3.sf1 "In Figure 3 ‣ 4.2 Sentiment Control ‣ 4 Applications ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

Appendix G Details of Investigating Interpretability
----------------------------------------------------

In Section[5.3](#S5.SS3 "5.3 Interpretability ‣ 5 Analysis ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") we interpret the weights learned in LM-Switch, and list discovered keywords in Table[4](#S5.T4 "Table 4 ‣ 5.3 Interpretability ‣ 5 Analysis ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). A detailed description for getting this results is as follows. First we conduct SVD decomposition of switch matrix $W$. The resulting $D$ matrix can then be interpreted as a ranked list of significant row vectors. We take the first 9 rows, and compute the their dot products with word embeddings.
As the row vector does not tell us which of the 2 directions indicates a increased probability, we select 20 tokens with top dot product and 20 tokens with bottom dot product as two candidate groups. Each group is concatenated to a text sequence and passed to Perspective API, and the group with larger toxicity value is considered true “keywords”. If, however, Perspective API recognizes the langauge as not English, which happened to rows No. 2, 4, and 6, then we discard this row as they contains mostly symbols and non-English words. Finally we filter out suffix tokens, and the remaining keywords are listed in Table[4](#S5.T4 "Table 4 ‣ 5.3 Interpretability ‣ 5 Analysis ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space").

Appendix H Parameter and Time Efficiency
----------------------------------------

*Table 6: Decoding time and learnable parameter efficiency. Time efficiency is measured by relative decoding time compared to base LM. Best numbers are bolded.*

|  | LM-Switch | DAPT | GeDi | CTRL | PPLM | DExpert | MuCoLa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Parameters | 3M | 355M | 355M | 355M | 124M | 355M | 898M |
| Speed Ratio | 1.24 | 1.00 | 2.94 | 3.79 | 270.11 | 1.98 | 24.03 |

Table[6](#A8.T6 "Table 6 ‣ Appendix H Parameter and Time Efficiency ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space") presents the parameter and time efficiency of LM-Switch. Thanks to its simple design, LM-Switch occupies no more than 1% of parameters compared with those of baselines. It also induces very small (2nd best) computational burden, compared with the original LM. This efficiency also allows us to leverage larger LMs such as GPT-j-6B888[https://huggingface.co/EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b "") as in the next subsection.

Appendix I More Details on Political Stance
-------------------------------------------

### I.1 Scenario 1: Russia-related Dataset

The dataset is collected from Twitter that includes users’ attitudes and reactions on Russia-Ukraine war, ranging in date from 2022-05-01 to 2022-08-08. The selection is keyword-based with >20 keywords listed below:

*Table 7: Keyword for selecting Russia-related tweets.*

| “Russophobia”, “Russophobic”, “RussiaPhobia”, “Demonise Russia” |
| --- |
| “standwithRussia”, “ standwithRussia”, “’standwithPutin “ “Naziukraine”, “Ukronazism”, “UkraineNazis”, “DenazifyUkraine”, “UkraineTerroristState”, “Ukronazi”, “Ukroreich”, “Azov Nazi” |
| “MinskAccords”, “MinskAgreements”, “Minsk-2 “ “KievRegime”, “Zelensky Regime”, “ZelenskyWarCriminal “ “Ukronato”, “NATORussiaWar”, “NATOWarCrimes”, “Protest NATO”, “NoToNATO”, “Western hegemony US neocons”, “BidenWar”, “Proxy War Ukraine BRICS bloc”, “BRICS G7”, “Russia global South” |

We filtered the combined dataset by English, and retain only the more prolific users (those posting more than 20 times), and more popular assertions (those referred to by more than 10 users). Then we employ two human annotators to label each tweet as pro-Russia or anti-Russia in ideology. We treat the tweets where they uniformly agree as pro-Russia to have label 1 (454 tweets), and those as anti-Russia to have label -1 (290 tweets).

After training LM-Switch on GPT-J-6B, we prompt the LM to generate on 8 topics. The selection of these 5 topics is based on ChatGPT API999<https://chat.openai.com> where we ask “Please provide a list of topics that best distinguish the policitical stance of pro-Russia and anti-Russia”. The list of topics is:

*Table 8: Keyword for selecting Russia-related tweets.*

| List of Topics |
| --- |
| “Russia’s annexation of Crimea” |
| “NATO expansion” |
| “Russia’s regime” |
| “The recent increase in sanctions against Russia” |
| “The anti-Russian sentiment on the Internet” |

We present generations with seed 0 on these topics are listed below, constraining a minimum length of 100 and maximum length of 300.

*Table 9: Generations on topic “Russia’s annexation of Crimea” on pro-Russia v.s. anti-Russia.*

| Stance | Generations |
| --- | --- |
| -1.0 anti | Russia’s annexation of Crimea was an invasion of Ukraine’s sovereign territory, but Russia insists that Ukraine’s Crimea People’s Republic is legally Russian territory.Ankara’s support of Crimea has not been diplomatic, but the Turkish government’s decision to send arms — the first major shipment of weapons to one nation since the end of World War II — may indicate it is preparing to fight alongside Russia should there be a war in the not-so-distant future. |
| -0.5 anti | Russia’s annexation of Crimea was an act of war against Ukraine: so say thousands of Russians who were arrested after the annexation for their presence on the streets. Their fate remains unknown. As for Kiev, it has said it will be ready to send any of these prisoners to trial, even the Russian officers, provided Moscow allows it. But Putin says he isn’t planning any such thing. So where is justice in this?Russian President Vladimir Putin (R) and Crimean parliament speaker Vladimir Konstantinich after a joint session on the first day of Russia’s annexation of Crimea.REUTERS/Sergei Karpukhin |
| 0 neutral | Russia’s annexation of Crimea was nothing short of a geopolitical bloodbath, but in the days after the events of March 16-17, Ukrainian President Petro Poroshenko and his government took a decisive step forward.Kicking Ukraine’s Western-oriented, US-aligned government out — the Maidan revolution was partly driven by a desire to end Ukraine’s alliance with Washington — was their best chance to save their beleaguered nation from catastrophe. Their first big setback was a US-sponsored coup in Kiev in late February — Washington’s fingerprints are on the Kiev junta — while after Poroshenko was elected president last July, his initial policy to pursue closer economic ties with Russia, which includes cooperation on energy sector reform, had a "reset" feel. |
| 0.5 pro | Russia’s annexation of Crimea was nothing short of a geopolitical earthquake: it has been the biggest geopolitical event of the year, and one that represents a fundamental challenge to the balance of power between Europe and Eurasia.It will have deep and long-lasting consequences, not least on the economic interests of western countries, from whose business sectors Ukraine imports vast amounts of oil and gas. |
| 1.0 pro | Russia’s annexation of Crimea was nothing short of a geopolitical earthquake: it has been the biggest geopolitical event of the year, and one that represents a fundamental shift in the balance of power between Europe and Eurasia. The Crimea crisis was a shock that rattled US President Barack Obama, the UN Security Council, and the NATO alliance. But it also has had a greater impact on Russia’s neighbors.Read More: <https://on.rt.com/8egy> |

*Table 10: Generations on topic “NATO expansion” on pro-Russia v.s. anti-Russia.*

| Stance | Generations |
| --- | --- |
| -1.0 anti | NATO expansion “has nothing to do” with Europe, but Putin wants war. And while he might start war over Ukraine right away (and get “saved” by Russian soldiers), his true motives for fighting may not be limited to his ‘interest’ in Ukraine. And one nation could be Ukraine, while his interests are elsewhere.In my recent interview with Alex Jones, Alex Jones talks about many things — the NWO, Obama’s lack of experience — but more specifically his view of Russia and where the US. |
| -0.5 anti | NATO expansion under pressureBy Alex Lantier4 January 2008There is growing pressure on NATO and Washington to halt the military buildup planned for Central Asia and for Russia, which will almost certainly be followed by additional provocative US missile deployments in Europe. A Russian military expert told World Socialist Web Site correspondents that NATO states are developing a secret plan to encircle Russia with bases and troops in Central Asia.In a Jan. 4 interview conducted by telephone, Col. Georgi A. Korniyenko, former chief of airborne forces (1995-2000), and now research director at the Strategic Analysis and Research Institute in St. Petersburg, spoke of the “political struggle for influence” over the region after the breakup of the Soviet Union, adding that there had been efforts “to increase the presence of NATO bases in Central Asia” since the 1990s. |
| 0 neutral | NATO expansion under pressureBy Alex Lantier4 January 2008There is growing pressure on NATO and Washington to halt the military buildup planned for Central Asia and for Russia, which will almost certainly lead to a new military confrontation.The growing disaffection among sections of NATO from the alliance’s central focus on military intervention abroad points to a significant crisis in the alliance.The buildup in Central Asia is the biggest US military escalation since the invasion of Iraq nine years ago. It was announced in mid-December at the Chicago Summit of the NATO-led International Security Assistance Force in Afghanistan. The NATO summit also gave the green light for expansion of the alliance to a second front in the conflict with Russia, to include Georgia, Ukraine and Azerbaijan.The US and NATO are preparing for a new confrontation with Russia in central Eurasia. The main objective of the policy of NATO expansion is to block Russia’s attempts to secure its borders and its access to its energy resources. |
| 0.5 pro | NATO expansion under pressureBy Alex Lantier4 January 2008There is growing pressure on NATO and Washington to halt the military buildup planned for Central Asia and for Russia, which will almost certainly lead to a new military confrontation.The growing conflict is based on preparations for NATO expansion. The Obama administration, working with its NATO allies, is moving forward with plans to expand NATO operations in Central Asia. Washington is intensifying military operations and building up the Pentagon’s logistical network in the region with special attention to Azerbaijan, Kazakhstan and Kyrgyzstan. NATO and US military facilities already operate there.This buildup is creating tensions. The expansion plan has provoked a protest storm in much of Central Asia and the rest of the non-aligned world. It is drawing international attention to the worsening global economic crisis, with calls for Washington’s spending binge and budget deficits to be addressed. |
| 1.0 pro | NATO expansion under pressureBy Alex Lantier4 January 2008There is growing pressure on NATO and Washington to halt the military buildup planned for Central Asia and for Russia, which will almost certainly lead to a new military confrontation.The growing conflict is based on the fact that NATO has built up its military presence in central Asia. NATO is carrying out increasingly provocative exercises in Kazakhstan and Kyrgyzstan, while announcing a new missile system to target China, including new missiles capable of reaching China’s northern provinces.It is also escalating its efforts to create a US-led alliance with Uzbekistan and Kazakhstan, and has given notice to other Central Asia nations that it is considering plans to join NATO. All these actions are increasing anxiety among Russia, which borders |

*Table 11: Generations on topic “Russia’s regime” on pro-Russia v.s. anti-Russia.*

| Stance | Generations |
| --- | --- |
| -1.0 anti | Russia’s regime under Putin has used its control over the state to crack down on dissent.This month’s elections for the Duma will be marred by Kremlin heavyweights casting their nets wide.The main stage in any of these campaigns is a seat on the Russian parliament, the Duma, whose 650 members represent all regions – more than two-thirds of Russian territory – and whose powers are enshrined in the constitution. |
| -0.5 anti | Russia’s regime under Putin has used its control over the state to crack down on dissent.This month’s elections for the Duma will be a test of Kremlin power.The Russian president will almost certainly win a mandate to extend his stay in the Kremlin — the Russian parliament has repeatedly rubber-stamped Putin’s every power grab. The opposition however seems dispirited at the thought of this “dictatorship for a dictatorship” being extended.And then there’s Navalny, once again the most high-profile critic of Putin and the man who made the opposition’s election campaign a reality.Navalny was barred from the Moscow voting as a "super-voter" after the opposition organised mass protests in central Moscow following his trial earlier this year. The opposition have not been able to mount a serious campaign against Putin and his party, the United Russia — Navalny is effectively the opposition leader after a four-year hiatus. |
| 0 neutral | Russia’s regime under Putin has used its control over the state to become the country’s most-profitable sector. But as the economy deteriorates, that control looks increasingly fragile.The Russian economy is facing its worst downturn in decades, with low oil prices and Western sanctions cutting off access to critical markets and funds. At the same time, Russia’s central bank is trying to preserve the value of the rouble, fearing that the currency – which rose against the dollar from 46 to the dollar last year – will slump too far.That is making for an awkward combination of economic misery and political uncertainty as the Vladimir Putin regime seeks to hold on to power. With elections coming up to parliament in a matter of months, Putin – who became president after Boris Yeltsin’s 1999 resignation for health reasons – has to find ways to keep his base of support with people who lost their jobs and homes during the country’s worst economic crisis since the Second World War.Here are four steps Putin’s regime is taking to do that. |
| 0.5 pro | Russia’s regime under Putin has used its control over the state to become the country’s most-profitable sector. But as the economy deteriorates, that control looks like it could be under threat.It wasn’t that long ago when, in the face of economic disaster, Russia found a way out of the doldrums – something that had proved much more difficult to do in the early post-Soviet years. The country’s GDP plunged to $0.9 trillion last year, with more than five years of consecutive decline; the average inflation rate was 12.3 percent in September; the GDP per capita stood at $7,500. With the national currency, the ruble, collapsing, the International Monetary Fund began to express its concern and warned that a full-fledged sovereign default was a possibility.Yet then suddenly came Vladimir Putin’s economic revolution. In one brilliant stroke, the Russian leader managed to not only revive the economy but even boost its growth rate. After nearly a decade of slowdown, Putin’s program has brought back Russia’s GDP and, along with it, the ruble’s exchange rate. |
| 1.0 pro | Russia’s regime under Putin has used its control over the country’s media to reinforce its image as a friendly, modern alternative to the West. Its propaganda includes news reports about its achievements in science and innovation.It also emphasises social issues, such as promoting a healthier lifestyle, promoting marriage, and countering what it sees as Western meddling in Russia’s…More..As a result of Saturday’s events, the UMP of the Russian Federation, the pro-Putin party of France, as well as France’s main political parties, the PRG, the EPP |

*Table 12: Generations on topic “The recent increase in sanctions against Russia” on pro-Russia v.s. anti-Russia.*

| Stance | Generations |
| --- | --- |
| -1.0 anti | The recent increase in sanctions against Russia under pressure from the European Union’s (EU) and the United States have prompted Russian officials and companies involved with defense export to ask for “sanctions-defying” technologies that will allow Russia to avoid further US legislation.With the Russian defense ministry estimating the cost of further foreign sanctions to be about $80 billion over 10 years and Russian companies forecasting a loss of $90 billion over the same period, a market for US-made weapons will be |
| -0.5 anti | The recent increase in sanctions against Russia under pressure from the NATO allies is another example of how the US wants to change international relations, making itself the single superpower. Its allies are already divided. As for China, it’s not only the trade volume that has decreased, so have trade services,” says Mikhail Prokhorov, the billionaire businessman and philanthropist. The Russian tycoon is not only a former vice-premier of Russian Prime Minister Mikhail Khodorkovsky, but also a successful businessman who is also a former oil magnate.The Russian businessman is known as the “Manafort of Russia” and he is known for financing the presidential campaign to elect Donald Trump to the presidency in 2016. According to Prokhorov, this is one of the reasons for the economic war waged against Russia. “I think there could probably be a number of different reasons, and not only one. The reason is simple: we lost. We lost because we didn’t understand that we were entering a period of time in which there would be a war between two groups and we, unfortunately, belonged to a third group. And today, the US controls more than 50% of the global economy – even more than China [China, in 2016 controlled 28.9% of the world’s economy, according to Bloomberg] and many countries feel threatened and don’t want to lose their trade volume of goods at the hands of the US,” Prokhorov says. That’s why they’re trying to use their troops and armies to seize the Crimea from Ukraine, he says. |
| 0 neutral | The recent increase in sanctions against Russia under pressure from the NATO allies is another example of how the US wants to change international relations, making them more favorable to Washington.Today’s Europe is undergoing a process of reorientation; it is moving away from its historical orientation towards the Soviet Union. During his two visits to Brussels, Hungarian President Orban spoke about one of the most important aspects of his agenda: countering Russian propaganda about democracy in the EU. Brussels’ and London’s response indicates that they will not object too much to this Hungarian President’s stance.However, it is more important to note that the European policy makers are still trying to decide what role they should play in ensuring that Brussels’ and London’s policies succeed. It looks like a competition for power between the US and EU in terms of geopolitical influence.‘Russia is the problem’The reason why Hungary is increasingly turning towards Russia after decades of American pressure is obvious: Budapest is well aware of its weakness. The current situation in Hungary makes it impossible for the country to compete with the US in a geopolitical confrontation. |
| 0.5 pro | The recent increase in sanctions against Russia under pressure from the NATO has resulted in an escalation of the conflict between Russia and Georgia. To counter Georgian aggression and to prepare Georgia to be able to be part of NATO when it joins the EU, Georgia has been undergoing reform of its security sector.With the reform process, GAIN has seen a marked increase in the number of its staff. GAIN has expanded its activities in the reform of the Georgian army, the police and the customs service and is now engaged in the reform of the internal security services, with special attention to the National Security Service.The Georgian authorities have made substantial efforts to train law enforcement personnel and to provide them with modern equipment. The reform process has gone ahead in a rather calm environment, with little public or media attention. It has been achieved with the assistance of GAIN. Georgian reforms have not been accepted by Georgia with enthusiasm but have been considered and implemented by the authorities with conviction. |
| 1.0 pro | The recent increase in sanctions against Russia under pressure from the NATO has resulted in an escalation of the conflict between the two sides. To counter the Western pressure, Russia is strengthening its nuclear “shield” by building the biggest fleet of weapons the world has seen.“If the West’s policies persist, one could easily envision a situation where NATO forces are stationed just on a few kilometers from the Russian borders,” commented Russian President Vladimir Putin recently. |

*Table 13: Generations on topic “The anti-Russian sentiment on the Internet” on pro-Russia v.s. anti-Russia.*

| Stance | Generations |
| --- | --- |
| -1.0 anti | The anti-Russian sentiment on the Internet was on full display back at the beginning of the week during a discussion on the Russian embassy in Israel on Russian television. At noon on Thursday, October 26, Channel 20’s Russian-language news broadcast discussed Russian news of the week. The discussion also featured Russian Embassy social media “journalist” Sergey Pashkovsky. |
| -0.5 anti | The anti-Russian sentiment on the Internet was on display again on Monday.We apologize, but this video has failed to load. tap here to see other videos from our team. Try refreshing your browser, or Google, Facebook and Twitter target pro-Kremlin users Back to videoFacebook, Twitter and Google were embroiled in arguments with users after they accused one another of manipulating a Russian election rally in the Ukrainian city of Kharkiv on Saturday. Each company said the other’s algorithms had created a fake pro-Russian bias during the rally — which the Russian Foreign Ministry called the “biggest provocation” in years.We’ve had to block more than 2.5 million bots |
| 0 neutral | The anti-Russian sentiment on the Internet was on display again on Monday.We apologize, but this video has failed to load.tap here to see other videos from our team. Try refreshing your browser, or Google, Facebook and Twitter target pro-Kremlin users Back to videoFacebook, Twitter and Google were embroiled in arguments with users all over the world. Russia was a target, as well as Ukrainian-centric groups in other countries opposing the annexation of Crimea by Moscow’s troops in 2014.But the debate on Monday wasn’t about Ukraine but about pro-Moscow supporters who use Google, Twitter and Facebook to boost the Kremlin’s legitimacy and sway public opinion.“We’ve seen cases where pro-Kremlin actors are manipulating the discourse on mainstream social media sites; in most cases, they’re pushing pro-Kremlin and pro-Russian content, and some content is even pushing Russian propaganda throughout Europe,” Brian Fishman, lead author of the paper that was released Monday about Russian trolls operating on Facebook, told The Canadian Press. |
| 0.5 pro | The anti-Russian sentiment on the Internet was on display again on social media last week as thousands of Russians voiced their indignation and protest over American sanctions against the former Soviet republic of Georgia during “Euromaidan,” the popular nickname for demonstrations that overthrew President Mikheil Saakashvili’s government in February.And while much of the criticism was directed at Washington — often couched in sarcasm — other users accused the Kremlin of playing a direct hand in the overthrow as part of a long-running campaign of influence.It was easy to see the echoes of last year’s events that triggered the Maidan protests, which had to do with the perceived mistreatment of a pro-Moscow ethnic minority in the restive breakaway state of Ossetia that had been subjected to ethnic Georgian militia bombings and incursions from Georgia into the province.The latest sanctions include restrictions against doing business in the United States with individuals and firms involved in or connected to the former Soviet republic. The US Treasury Department has slapped sanctions on 13 individuals and companies, including Giga, an international firm that provides outsourcing services. |
| 1.0 pro | The anti-Russian sentiment on the Internet was nothing short of shocking at the beginning of 2017, with popular posts ranging from "Russians are Satan" and "I will shoot them like flies" to "I’m going to kill you, Russian*" and "PunkPutin" (which also happens to be the name of a popular Russian punk rock band). This phenomenon has been blamed by some on Russian media and their allegedly fake news promoting Russophobia. However not everyone agrees with this and some argue that it is not the media propaganda but more the pro-democracy political campaigns in the US and Europe that are responsible for the Russophobia on the Web and social media.Russophobia has existed for years, to be exact as far back as the 1930s when it was used by anti-communist elements in the West to isolate and demonise the Soviet Union. The term Russophobia appears to have originally been coined by the German historian Hans Kohn in the late 1930s, after it was used in the Western media |

### I.2 Scenario 2: Modelling Differences in Medias

We studies 5 controversial news events on Ground News101010<https://ground.news>, each with a pair of contrasting media article. Then we use LM-Switch learn each news article and follow the stacen to generate on the topic.

#### Event 1: EU to crack down on imports of Indian fuels

Times of India article: [https://timesofindia.indiatimes.com/business/india-business/top-eu-diplomat-seeks-action-against-india-on-russian-oil-ahead-of-talks/articleshow/100288784.cms](https://timesofindia.indiatimes.com/business/india-business/top-eu-diplomat-seeks-action-against-india-on-russian-oil-ahead-of-talks/articleshow/100288784.cms "").
Reuters article: [https://www.reuters.com/business/energy/borrell-urges-eu-crack-down-imports-indian-fuels-made-with-russian-oil-ft-2023-05-16/](https://www.reuters.com/business/energy/borrell-urges-eu-crack-down-imports-indian-fuels-made-with-russian-oil-ft-2023-05-16/ "")

*Table 14: Generations about EU’s sanction on India.*

| Stance | Generations |
| --- | --- |
| Times of Indea | EU diplomat seeks action against India on at least 1 issueThe European Union’s chief diplomat seeks action against India on at least 1 issue, ahead of its talks with the European Union.“As the largest democracy and a country with the fifth largest economy, India has the means to address its growth and to address the critical issues,” a diplomat said. |
|  | EU diplomat warns against embracing India as a ‘laundromat’The EU diplomat said the view that Europe should embrace India to address its growth is a “laundromat” by his description and he said he will raise the issue with his counterparts in Delhi. |
|  | EU diplomat Josep Borrell said the embassy will have to close with effect from Tuesday if Britain fails to take measures to avoid disruption, although the measures will not be made known before Wednesday. |
|  | EU diplomat Josep Borrell has called on the UK to remain firm on its stance against India for its rising crude import from India and the UK’s rising exports of refined products, as per a latest media report. |
|  | EU diplomat Josep Borrell, the spokesman for the United Nations’ Financial Action Task Force on money laundering. The watchdog last month called on India to act on the issue, describing India as a “laundromat” for Russian oil. |
| Reuters | EU diplomat’s remarks target EU sanctions policyThe EU diplomat said that the EU should have a sanctions policy but that the sanctions rules need to be tightened to be more rigorous. |
|  | EU diplomat had his visa revoked after he raised concerns he could be targeted by India’s crackdown on foreigners following an attack on an Indian diplomatic post in Kabul, it emerged on Tuesday. |
|  | EU diplomat said the EU would not be a willing or a competitive market to India’s exporters, suggesting that it would have to use more of the talks to be a more competitive exporter to the EU member states. |
|  | EU diplomat said to have met the buyer.“The Russians have emerged as a centre where buyers of Russian oil are being routed to - including from India.”(This report was originally published on The Economic Times.) |
|  | EU diplomat’s remarks raise more than a few eyebrows."The EU diplomat said his remarks were remarks he would have made to his wife," the EU’s chief diplomat said."But he did not refer in his remarks to Russia’s invasion of Ukraine, as he said he would have said to his wife… But his remarks refer to Russia’s invasion of Ukraine," the EU diplomat said. |

#### Event 2: A new bill on banning “no-fault” evictions.

The Independent article: [https://www.independent.co.uk/news/uk/home-news/no-fault-eviction-ban-uk-b2340410.html](https://www.independent.co.uk/news/uk/home-news/no-fault-eviction-ban-uk-b2340410.html "").
BBC article: [https://www.bbc.com/news/uk-politics-65612842](https://www.bbc.com/news/uk-politics-65612842 "")

*Table 15: Generations about banning “no-fault” eviction.*

| Stance | Generations |
| --- | --- |
| The Independent | The new bill also seeks to impose a 20% hike in rent for several properties in London from 20th February and calls for the introduction of a private rental sector property portal to be set up. |
|  | The new bill will affect how many times an assistant director will be able to refuse to rent to a black or an asian or an in this new bill or new bill will affect how many times an assistant director will be able to refuse to rent to a black or an asian or an in this new bill or new bill will affect how many times an assistant director will be able to refuse to rent to a black or an asian or an in this new bill will affect how many times an assistant |
|  | The new bill will also be published on Wednesday, the government said.“The Bill seeks to bring serious offences against these crimes against women under one umbrella and make it illegal for these criminals to misuse the name of the law for their misdeeds,” said government spokesperson Leela Mani. |
|  | The new bill will be published in the UK parliament on 1st of April, and is said to be a “huge” opportunity to reform the rules for home landlords and for families, and is a “huge opportunity” to help families with rent”. |
|  | The new bill also bans so-called no-fault evictions in one of the harshest eviction bans for rental housing since the government put anti-no-fault eviction laws into law in the four years after the recession in the housing collapse of four years ago. About 20 percent of homes across the country are rented. |
| BBC | The new bill would also make it easier for landlords to repossess properties from tenants.A Conservative spokesman said the bill would make it easier for landlords to repossess properties from tenants."We are also able to introduce a national register of landlords. We would also make tenants who use a landlord’s property with their permission pay more rent. |
|  | The new bill will also make it easier for landlords to refuse benefits to their tenants, which the bill would make illegal."A huge part of our research is around what’s called Section 11 - under 11 our bill makes it illegal for landlords to refuse benefits, which the bill also makes it easier for landlords to refuse benefits," Mr Donnelly said. |
|  | The new bill would also be a challenge for the government, which has said it would be willing to negotiate with the union to reach an agreement."We will have to negotiate, we will have to struggle to reach these agreements," said Conservative spokeswoman Kate Purchase said. "We will have to struggle through what is a difficult process." |
|  | The new bill would also make it easier for landlords to repossess properties from anti-social tenants.A government spokesman said the bill would make sure nothing in the bill could force landlords to repossess properties from those with anti-social records. |
|  | The new bill could force more landlords to be registered with the rental market and pay progressively more rental increases - which the bill intends to make easier."We would be hoping the bill would reduce the market and reduce the supply in the rental market. It would be good for tenants to have the right to the benefits of the bill," Mr Samwell said. |

#### Event 3: Israeli forces kill Palestinian man

VOA article: [https://www.voanews.com/a/israeli-forces-kill-palestinian-man-in-west-bank-raid-/7094609.html](https://www.voanews.com/a/israeli-forces-kill-palestinian-man-in-west-bank-raid-/7094609.html "").
Palestine Chronicle article: [https://www.palestinechronicle.com/israeli-occupationforces-kill-palestinian-youth-near-nablus/](https://www.palestinechronicle.com/israeli-occupationforces-kill-palestinian-youth-near-nablus/ "").

*Table 16: Generations about Israel and Palestain.*

| Stance | Generations |
| --- | --- |
| VOA | Isreal Forces and Hamas to blame for Gaza violence"Palestinian fighters shot and killed 22 Egyptian and five Palestinian suspects from a Gaza refugee camp in a raid in Nablus early Monday. |
|  | Isreal Forces and Israel-Palestinian Forces“Israel has its forces in Israel, Palestinians have their forces in the West Bank and Gaza. Israel and Palestinians have been fighting for years and have been fighting for 26 years. Israel and Palestinians have been fighting for 26 years for security and security for security for Israelis. Palestinians were in refugee camps which were not on Palestinian areas of security. Palestinians have been in refugee camps which were not on areas of security. Palestinians were not in are as of security where |
|  | Isreal Forces and the Palestiniansby Steve(a.k.a. Eretz Israeli)"If Palestinians want self-rule and an area for cars and families and cars and families they want to have cars and families than they have to participate in the Israeli military and they have to abide by those rules and have to have cars and families." |
|  | Isreal ForcesIsrael, for its part, said the forces were inside its Gaza Strip and Israeli military said suspects hurled rocks and explosives and fired at its forces in the West Bank. |
|  | Isreal Forces" and its forces. They are a branch of the military of Israel. Their mission is to prepare for war.A group of Palestinians took part in the soldiers in the area of Nablus, a group of soldiers came to the home of a Palestinian man to prepare for its defense, the Palestinian man responded to the soldiers from a member of the Palestinian forces, during the clash, the Palestinian soldiers were the people were in the home were of two brothers, in the forces |
| Palestine Chronicle | Israel Forces Kill Palestinian Youth During UN ProtestsIsraeli forces during a massive military raid in the West Bank on the night of July 23 to 24 during the Gaza border clashes that killed two Palestinian youth. |
|  | Israel Forces Kill 26 Palestinians, Including One ChildIsraeli forces raided the family house of the slain Palestinian youth east of the occupied city of Nablus, on the night of May 13-14, and killed the killed the slain youth, the family told the Washington Office on Arabic of the Israeli aggression |
|  | Israel Forces Kill Palestinian Youth during the Gaza Strip Intifada[Note: In the previous Israeli aggression against the Gaza Strip in November of 2008, the Israeli forces killed Palestinian youth during the Gaza Strip Intifada that was a natural reaction to Israeli aggression in the Gaza Strip in the last two years. |
|  | Israel Forces Begin to Deploy Tanks Near the Gaza StripBy NIDAR SINGHJERUSALEM, (IPS) 27 Dec — Israeli forces on Monday began to concentrate tanks on the Gaza border in preparation of what Israeli government sources said was yet another aggressive invasion of the Gaza Strip. |
|  | Israel Forces Kill Palestinian Youth in Bethlehem, GazaPalestinian youth were killed by Israeli gunfire in the occupied West Bank and Gaza Strip on the Monday to the Thursday. |

#### Event 4: Stellantis’ warning to change Brexit deal

The Guardian’s article: [https://www.theguardian.com/business/2023/may/17/vauxhall-maker-says-brexit-deal-must-be-renegotiated-or-it-could-shut-uk-plant](https://www.theguardian.com/business/2023/may/17/vauxhall-maker-says-brexit-deal-must-be-renegotiated-or-it-could-shut-uk-plant "").
The Telegraph’s article: [https://www.telegraph.co.uk/business/2023/05/17/ftse-100-markets-live-news-unemployment-rise-inflation/](https://www.telegraph.co.uk/business/2023/05/17/ftse-100-markets-live-news-unemployment-rise-inflation/ "").

*Table 17: Generations about Stellantis’ warning.*

| Stance | Generations |
| --- | --- |
| The Guardian | Vauxhall maker says Brexit deal must be renegotiated or it could shut UK plant.The future of the British car plant which also produces Fiat Chrysler and Citroën vehicles, could depend on what deal the government’s Brexit deal may be. |
|  | Vauxhall maker says Brexit deal must be renegotiated or it could shut UK plant.Many more people are employed in the UK by the UK Vauxhall and many more people will be employed in the UK by the UK Vauxhall if it is shut. |
|  | Vauxhall maker says Brexit deal must be renegotiated or it could shut UK plant, putting thousands of jobs at risk.“If we close the UK, we close the UK, that will have more impact on the UK than it will on Vauxhall and some of its employees who are not going to go just because there are others who are going to open up in the UK,” the company said. |
|  | Vauxhall maker says Brexit could force it to raise UK Brexit deal to unsustainable cost of electric vehicles.If the cost of electric car batteries becomes unsustainable in the UK or the cost of batteries becomes unsustainable in Europe, it will not mean there will be no electric cars. |
|  | Vauxhall maker says Brexit deal must not affect UK operations or future coronavirus production.The future of the UK and the place of Vauxhall and many thousands of businesses after the country’s decision to rejoin the trading blocks of the EU was one of the most discussed and acrimonious issues of the day. |
| The Telegraph | Vauxhall maker says workforce is currently ’at low levels’ but adds that latest updates are designed to add ’at least’ another 200 but that is dependent on cust omers and that they have built capacity to add. |
|  | Vauxhall maker says Britain cannot rely on ’unlimited low-wage’ workforce.The British workforce is too dependent on low-wage, limited-skill, limited-skill, limited-wage migration to allow our economy to change to the skills and wage economies we need for the labour market to change for our economy to change for our people to change to change, a British manufacturer has warned. |
|  | Vauxhall maker says Britain needs Brexit to end ’unlimited, low skill’ migration - latest updatesThe Brexit decision to change our economic model has come at the worst possible time for everyone but everyone. |
|  | Vauxhall maker says it could close down""The Chancellor was warned by two Cabinet colleagues this morning to think twice" |
|  | Vauxhall maker says this will be Britain’s last model - but Britain will end one of the great two-seater car seriesThe decision to end the Astra has come after a decision to switch from being designed by a British company to being designed and built by a third-party, with all profits from the British model being ploughed into a new model |

#### Event 5: Bulgarian Military Aid

We selcted the Russia Today news: [https://www.rt.com/news/565894-bulgaria-weapons-aid-ukraine/](https://www.rt.com/news/565894-bulgaria-weapons-aid-ukraine/ "") and
Reuters news: [https://www.reuters.com/world/europe/bulgarian-lawmakers-give-government-month-devise-ukraine-military-aid-plan-2022-11-03/](https://www.reuters.com/world/europe/bulgarian-lawmakers-give-government-month-devise-ukraine-military-aid-plan-2022-11-03/ "").

*Table 18: Generations on Bulgarian Military Aid.*

| Stance | Generations |
| --- | --- |
| Russia Today | Bulgarian President B.R. Klement has been urging President B.R. Klement to have NATO negotiate with Israel on pacifism and negotiate treaties with Ukraine until negotiations have been concluded. |
|  | Bulgarian group to offer Bulgarian medical supplies to Sofia people? |
|  | Bulgarian President! President Gorbachev has the most impressive equipment of the war to make good on the Soviet Union. So? The most important equipment of the war to make good on is equipment from G-2. This equipment is a huge part of Soviet equipment. |
|  | Bulgarian drug users risk alcohol abusers and those who risk alcohol abusers are those exempted from mandatory intoxication tests (MADT).rehensive drinkers who supply alcohol are excluded from intoxication tests (TRIF). |
|  | Bulgarian politicians are debating Bulgaria’s decision to leave the European Union. Bulgaria has since a year voted to leave the European Union, which despite dragging Bulgaria into the region’s conflict, has still not sold Bulgaria a NATO treaty until Bulgaria has sold Bulgaria a NATO treaty until Bulgaria has sold Bulgaria NATO treaty. |
| Reuters | Bulgarian lawmakers give lawmakers first month weeks in parliament to vote on Ukraine’s decision to send only its first-class from the country. |
|  | Bulgarian Soviet Socialist partybitcoin rebatebitcoins month trading trading swap trading weapons trading weapons Soviet Soviet Socialist party. |
|  | Bulgarian economy can send one in from one month to two months quickly could be one of the main features of sending arms first. |
|  | Bulgarian Soviet Union - weapons of mass-value have more military avail than Soviet Army and missile capabilities, and the Soviet military cannot not supply first-class only first-class mainly first-class Soviet military such as the SU-24 and SU-125. |
|  | Bulgarian arms export sales could rise but have not spread rapidly - one way.Heavy equipment such as what military experts have called the "forgery of the Russian army" can not be said to be for what buyers not only could probably not with it but would not be "for what one bought in Soviet Russia " |

Appendix J Validity of Assumptions
----------------------------------

To verify the validity of the assumptions, we did an experiment for searching for valid HMMs while satisfying the assumption[1](#Thmassumption1 "Assumption 1. ‣ 3.2 Conditioned Hidden Markov Model ‣ 3 LM-Switch: Motivation and Formulation ‣ LM-Switch: Lightweight Language Model Conditioning in Word Embedding Space"). It is trivial to constrct valid $\Psi$ as long as a valid $\Phi$ can be found. So specifically, we set $d_{s}\=20$ and $d_{c}\=1$ for representing a one-conditional HMM. We let $n\=200$, and randomly initialized $\Phi$ with Gaussian distribution with variance 1e-3. Then we construct the following objective function

|  | $\displaystyle\mathcal{L}\=\mathcal{L}_{\text{norm}}+\mathcal{L}_{\text{dist}}+\mathcal{L}_{\text{independence}}+\mathcal{L}_{\text{conditional}},$ |  |
| --- | --- | --- |

where

|  | $\displaystyle\mathcal{L}_{\text{norm}}$ | $\displaystyle\=\sum_{i}\left(\sum_{s\in\mathcal{S}}\phi_{s,i}^{2}-\frac{1}{dn}\sum_{s\in\mathcal{S},i^{\prime}}\phi_{s,i^{\prime}}^{2}\right)^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\mathcal{L}_{\text{dist}}$ | $\displaystyle\=\sum_{s,s^{\prime}}\max(-T(s,s^{\prime}),0)+\sum_{s}\left(\sum_{s^{\prime}}T(s,s^{\prime})-1\right)^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\mathcal{L}_{\text{independence}}$ | $\displaystyle\=\sum_{i\neq j}\left(\sum_{s}\phi_{s,i}\phi_{s,j}\right)^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\mathcal{L}_{\text{conditional}}$ | $\displaystyle\=\sum_{i,j,k\text{not one value},k\in[d_{c}+1,d]}\left(\sum_{s}\phi_{s,i}\phi_{s,j}\phi_{s,k}\right)^{2}$ |  |
| --- | --- | --- | --- |

Generally, this objection characterizes the derivation of $\Phi$ from the assumptions. We use Adam optimizer with learning rate 1e-3, and ReduceLROnPlateau111111<https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html> with patience 100 and reduce factor 0.5. The optimization process lasts 500k steps, starting from random seeds 0, 1, 2 and 3. On all random seeds, the objective function reduces from greater than 1 to less than 1e-5. This indicates that valid HMM solutions satisfying the assumption exist.
