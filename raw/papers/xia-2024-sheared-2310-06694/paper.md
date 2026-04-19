Pre-training via Structured Pruning
====================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Efficiently Producing a Billion-Parameter  Language Model via Structured Pruning
=================================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Sheared LLaMA: Producing Strong Small Language Models via Structured Pruning
============================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Sheared LLaMA: Scaling Down Language Models Efficiently via Structured Pruning
==============================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Don’t Pretrain from Scratch: Structured Pruning from Large Language Models with Less Compute
============================================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Sheared LLaMA: Structured Pruning from Large Language Models with Less Compute
==============================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

Sheared LLaMA: Accelerating Language  Model Pre-training via Structured Pruning
================================================================================

Mengzhou Xia1, Tianyu Gao1, Zhiyuan Zeng2, Danqi Chen1  
1Department of Computer Science \& Princeton Language and Intelligence,  
Princeton University  
2Department of Computer Science and Technology, Tsinghua University  
{mengzhou,tianyug,danqic}@cs.princeton.edu  
zengzy20@mails.tsinghua.edu.cn  
Work done during internship at Princeton University.

###### Abstract

The popularity of LLaMA*(Touvron et al., [2023a](#bib.bib63 ""); [b](#bib.bib64 ""))* and other recently emerged moderate-sized large language models (LLMs)
highlights the potential of building smaller yet powerful LLMs.
Regardless,
the cost of training such models from scratch on trillions of tokens remains high.
In this work,
we study structured pruning as an effective means to develop smaller LLMs from pre-trained, larger models.
Our approach employs two key techniques:
(1) targeted structured pruning,
which prunes a larger model to a specified target shape
by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner, and
(2) dynamic batch loading,
which dynamically updates the composition of sampled data in each training batch based on varying losses across different domains.
We demonstrate the efficacy of our approach by
presenting the Sheared-LLaMA series,
pruning the LLaMA2-7B model
down to 1.3B and 2.7B parameters.
Sheared-LLaMA models
outperform state-of-the-art open-source models of equivalent sizes, such as
Pythia, INCITE, and OpenLLaMA models, on a wide range of downstream and instruction tuning evaluations,
while requiring only $3\%$ of compute compared to training such models from scratch.
This work provides compelling evidence that leveraging existing LLMs with structured pruning is a far more cost-effective approach for building smaller LLMs.111Please find our code and models at [https://github.com/princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing "").

1 Introduction
--------------

Large language models (LLMs) are extremely performant on a wide range of natural language tasks, but they require enormous amounts of compute to train*(OpenAI, [2023](#bib.bib49 ""); Anthropic, [2023](#bib.bib2 ""))*.
As such, there is growing interest in building strong moderate-sized models, such as LLaMA*(Touvron et al., [2023a](#bib.bib63 ""); [b](#bib.bib64 ""))*, MPT*(MosaicML, [2023](#bib.bib48 ""))*, and Falcon*(Almazrouei et al., [2023](#bib.bib1 ""))*, that allow for efficient inference and fine-tuning.
These LLMs are available in varied sizes suited for different use cases, but training each individual model from scratch—even the smallest billion-parameter models—requires substantial computational resources that are cost-prohibitive for most organizations.
In this work, we seek to address the following question:

*Can we produce a smaller, general-purpose, and competitive LLM by leveraging existing pre-trained LLMs, while using much less compute than training one from scratch?*

We explore structured pruning as a means to achieve this goal. Pruning is commonly viewed as a solution for compressing task-specific models*(Han et al., [2016](#bib.bib24 ""); Li et al., [2016](#bib.bib37 ""); Lagunas et al., [2021](#bib.bib35 ""); Xia et al., [2022](#bib.bib70 ""); Kurtic et al., [2023](#bib.bib33 ""))*, removing redundant parameters and accelerating inference without sacrificing task performance.
However,
for general-purpose LLMs,
pruning inevitably results in performance degradation compared to original models*(Frantar \& Alistarh, [2023](#bib.bib17 ""); Sun et al., [2023](#bib.bib58 ""); Ma et al., [2023](#bib.bib45 ""))*, especially when without significant compute invested post-pruning. In this work, we use pruning as an effective approach for developing smaller yet competitive LLMs that require only a fraction of the training compute compared to training them from scratch.

We identify two key technical challenges in this problem. First, how can we decide on final pruned architectures that are strong in performance and efficient for inference?
Existing structured pruning techniques for LLMs*(Xia et al., [2022](#bib.bib70 ""); Ma et al., [2023](#bib.bib45 ""))* do not specify targeted structures and lead to suboptimal pruned models in terms of performance and inference speed ([Table 4](#S4.T4 "In Targeted pruned models have a higher inference throughput. ‣ 4.2 Comparison to Other Pruning Approaches ‣ 4 Analysis ‣ Pre-training via Structured Pruning") and [Figure 8](#A5.F8 "In E.2 Comparison to LLM-Pruner ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning")).
Second, how can we continue pre-training the pruned model to reach desired performance?
We observe that training using
the original pre-training data leads to imbalanced rates of loss reduction across different domains,
compared to when training such models from scratch.
This indicates that the pruned model retains varying levels of knowledge for different domains (e.g., GitHub vs. C4) and
simply using the pre-training domain proportion results in an inefficient use of data
([Figure 6](#S4.F6 "In Downstream performance. ‣ 4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning")).
To address these issues, we propose “LLM-shearing”, an algorithm consisting of the following two components:

<img src='x1.png' alt='Refer to caption' title='' width='211' height='145' />

*Figure 1:  Our Sheared-LLaMA-2.7B surpasses a series of open-source models at a similar scale and
only requires 1/32 (3%) of budget to achieve on-par performance with OpenLLaMA-3B-v2.*

* •

    We propose a novel pruning algorithm, dubbed targeted structured pruning, which prunes a source model to a specified target architecture.
    The target architecture is determined by leveraging the configurations of existing pre-trained models.
    Our pruning approach searches for substructures within the source model that maximally preserve performance while adhering to the given constraints.

* •

    We devise a dynamic batch loading algorithm that
    loads training data from each domain in proportion to its rate of loss reduction, thereby
    making an efficient use of the data and
    accelerating the overall performance improvement.

We demonstrate the efficacy of our proposed method by pruning a LLaMA2-7B model*(Touvron et al., [2023b](#bib.bib64 ""))* into two smaller LLMs:
Sheared-LLaMA-1.3B and
Sheared-LLaMA-2.7B.
Despite using only 50 billion tokens (i.e., 5% of OpenLLaMA’s pre-training budget) for pruning and continued pre-training,
Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B
outperform other popular LLMs at similar scales, including Pythia*(Biderman et al., [2023](#bib.bib4 ""))*, INCITE*(TogetherAI, [2023b](#bib.bib62 ""))*, and OpenLLaMA*(Geng \& Liu, [2023](#bib.bib20 ""))*, on 11 representative downstream tasks ([Figure 1](#S1.F1 "In 1 Introduction ‣ Pre-training via Structured Pruning"); commonsense, reading comprehension, and world knowledge) and instruction tuning for open-ended generation.
Additionally, the downstream performance trajectory suggests that further training the pruned model with more tokens would result in even greater gains.
While we only conduct experiments with up to 7B parameter models, our LLM-shearing algorithm is highly generalizable and can be extended to large language models of any size in future work.

2 LLM-Shearing
---------------

Given an existing large model $\mathcal{M}_{S}$ (the source model),
we study how to efficiently produce a smaller, strong model $\mathcal{M}_{T}$ (the target model).
We consider this as a two-stage process:
(1) Pruning
$\mathcal{M}_{S}$ into $\mathcal{M}_{T}$.
This reduces the number of parameters but incurs a performance drop inevitably.
(2) Continually pre-training $\mathcal{M}_{T}$ with a standard language modeling objective
to reach a target performance.
While most recent efforts*(Xia et al., [2022](#bib.bib70 ""); Ma et al., [2023](#bib.bib45 ""))* focus on the former stage,
we find the latter stage crucial for producing
competitive general-purpose LLMs from structured pruning.

### 2.1 Targeted Structured Pruning

Structured pruning removes groups of model parameters to
compress models and accelerate inference.
However, existing structured pruning approaches often
result in unconventional model configurations that deviate from popular architectures.
For example, CoFiPruning*(Xia et al., [2022](#bib.bib70 ""))* produces models with
non-uniform layer configurations (e.g., different numbers of heads across layers),
which incurs inference overhead compared to standard uniform layer configurations (Section[4.2](#S4.SS2.SSS0.Px1 "Targeted pruned models have a higher inference throughput. ‣ 4.2 Comparison to Other Pruning Approaches ‣ 4 Analysis ‣ Pre-training via Structured Pruning")).

In this work,
we extend CoFiPruning to allow pruning the source model into any target configuration that we specify.
We leverage the configurations of existing pre-trained models as the target architectures,
based on the intuition that
these configurations have already been well-optimized to
balance model expressivity and inference efficiency.
For example, we use the INCITE-Base-3B architecture*(TogetherAI, [2023a](#bib.bib61 ""))* as the target structure when producing a $2.7$B model.

<img src='x2.png' alt='Refer to caption' title='' width='452' height='92' />

*Figure 2: Targeted structured pruning produces a compact and dense model of a pre-specified shape. Light colors indicate pruned substructures. Masking variables $z$ are learned to control whether a substructure is pruned ($z\=0$) or retained ($z\=1$).*

Our method learns a set of pruning masks
on model parameters at different granularities—from
global ones like layers and hidden dimensions (persist across all layers),
to local ones like attention heads and intermediate dimensions.
Assume that the source model $\mathcal{M}_{S}$ has
$L_{\mathcal{S}}$ layers, with each layer consisting of one multi-head attention module (MHA) and one feed-forward network (FFN). $\mathcal{M}_{S}$ has a hidden state dimension of $d_{\mathcal{S}}$,
$H_{\mathcal{S}}$ heads in each MHA, and
an intermediate dimension of $m_{\mathcal{S}}$ in each FFN. We introduce the following mask variables:

| Granularity | Layer | Hidden dimension | Head | Intermediate dimension |
| --- | --- | --- | --- | --- |
| Pruning masks | ${z}^{\text{layer}}\in\mathbb{R}^{L_{\mathcal{S}}}$ | ${z}^{\text{hidden}}\in\mathbb{R}^{d_{\mathcal{S}}}$ | ${z}^{\text{head}}\in\mathbb{R}^{H_{\mathcal{S}}}$ $(\times L_{\mathcal{S}})$ | ${z}^{\text{int}}\in\mathbb{R}^{m_{\mathcal{S}}}$ $(\times L_{\mathcal{S}})$ |

Each mask variable controls whether the associated substructure is pruned or retained. For example, we remove a layer if its corresponding $z^{\text{layer}}\=0$. Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Targeted Structured Pruning ‣ 2 LLM-Shearing ‣ Pre-training via Structured Pruning") illustrates an example of how the pruning masks control the pruned structures.

We formulate pruning as a constrained optimization problem
where we learn pruning masks to search for a subnetwork
matching a pre-specified target architecture
while maximizing performance.
Following the $\ell_{0}$ regularization approach*(Louizos et al., [2018](#bib.bib43 ""))*, we parametrize the pruning masks to model hard concrete distributions. These distributions have support on $[0,1]$ but concentrate their probability mass at <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S2.SS1.p5.3.m3.1"><semantics id="S2.SS1.p5.3.m3.1a"><mn id="S2.SS1.p5.3.m3.1.1" xref="S2.SS1.p5.3.m3.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S2.SS1.p5.3.m3.1b"><cn id="S2.SS1.p5.3.m3.1.1.cmml" type="integer" xref="S2.SS1.p5.3.m3.1.1">0</cn></annotation-xml></semantics></math> -->00 or $1$, enabling discrete prune or retain decisions.
While prior work usually control for a target sparsity*(Wang et al., [2020](#bib.bib67 ""); Xia et al., [2022](#bib.bib70 ""))*, we use a pair of Lagrange multipliers to impose constraints on the pruned model shape directly.
For example, for a target number of heads $H_{\mathcal{T}}$ (and we use $L_{\mathcal{T}}$, $d_{\mathcal{T}}$, and $m_{\mathcal{T}}$ to represent the target number of layers, hidden dimension, and intermediate dimension respectively), we have the imposed constraint on a single layer as:

|  | $\displaystyle\tilde{\mathcal{L}}^{\mathrm{head}}(\lambda,\phi,z)$ | $\displaystyle\=\lambda^{\mathrm{head}}\cdot\left(\sum{{z}^{\mathrm{head}}}-H_{\mathcal{T}}\right)+\phi^{\mathrm{head}}\cdot\left(\sum{{z}^{\mathrm{head}}}-H_{\mathcal{T}}\right)^{2}.$ |  |
| --- | --- | --- | --- |

Similar constraints are applied to pruning other substructures. Overall, we jointly optimize the model weights and pruning masks by a min-max objective $\min_{\theta,{z}}\max_{\lambda,\phi}\mathcal{L}_{\mathrm{prune}}(\theta,z,\lambda,\phi)$:

|  | $\mathcal{L}_{\mathrm{prune}}(\theta,z,\lambda,\phi)\=\mathcal{L}(\theta,{z})+\sum_{j\=1}^{L_{\mathcal{S}}}\tilde{\mathcal{L}}^{\mathrm{head}}_{j}+\sum_{j\=1}^{L_{\mathcal{S}}}\tilde{\mathcal{L}}^{\mathrm{int}}_{j}+\tilde{\mathcal{L}}^{\mathrm{layer}}+\tilde{\mathcal{L}}^{\mathrm{hidden}},$ |  |
| --- | --- | --- |

where $\mathcal{L}(\theta,{z})$ is the language modeling loss computed with the masked model weights.
This objective will produce a pruned model with the target shape.
Ideally,
running this prune algorithm
on a large amount of data will directly produce a strong compact model.
In practice, the pruning stage is expensive (roughly 5$\times$ slower compared to standard LM training), and we find
that the learned masks often converge fast.
Therefore, in our experiments, we allocate only a limited budget for the pruning process. Following pruning, we finalize the pruned architecture by preserving the highest-scoring components associated with the mask variables in each substructure,
and continue pre-training the pruned model with the language modeling objective.
We refer to this second stage as continued pre-training.

### 2.2 Dynamic Batch Loading

Require: Training data of $k$ domains $D_{1},D_{2},\cdots,D_{k}$, validation data $D^{\mathrm{val}}_{1},D^{\mathrm{val}}_{2},\cdots,D^{\mathrm{val}}_{k}$, initial data loading weights $w_{0}\in\mathbb{R}^{k}$, reference loss $\ell_{\mathrm{ref}}\in\mathbb{R}^{k}$, LM loss function $\mathcal{L}$ or pruning loss $\mathcal{L}_{\text{prune}}$, training steps $T$, evaluation interval $m$, model parameters $\theta$ ($\theta,z,\phi,\lambda$ for pruning)

for *$t\=1,\cdots,T$* do

if *$t\mod m\=0$* then

$\ell_{t}[i]\leftarrow\mathcal{L}(\theta,z,D^{\mathrm{val}}_{i})\text{~{}if {pruning} else~{}}\mathcal{L}(\theta,D^{\mathrm{val}}_{i})$

$\Delta_{t}[i]\leftarrow\max\left{\ell_{t}[i]-\ell_{\mathrm{ref}}[i],0\right}$ $\triangleright$ Calculate loss difference

$w_{t}\leftarrow$ UpdateWeight(*$w_{t-m}$, $\Delta_{t}$*) $\triangleright$ Update data loading proportion

end if

Sample a batch of data $\mathcal{B}$ from $D_{1},D_{2},\cdots,D_{k}$ with proportion $w_{t}$;

if *pruning* then

Update $\theta,{z},\phi,\lambda$ with $\mathcal{L}_{\mathrm{prune}}(\theta,z,\phi,\lambda)$ on $\mathcal{B}$

else

Update $\theta$ with $\mathcal{L}(\theta,\mathcal{B})$

end if

end for

Subroutine *UpdateWeight(*$w$, $\Delta$*)*

$\alpha\leftarrow w\cdot\exp\left(\Delta\right)$ $\triangleright$ Calculate the unnormalized weights

$w\leftarrow\frac{\alpha}{\sum_{i}\alpha[i]}$ return $w$ $\triangleright$ Renormalize the data loading proportion

return $\theta$

*Algorithm 1 Dynamic Batch Loading*

Continued pre-training on a large amount of data is crucial for recovering the pruned model performance.
However, we observe a surprising finding in our preliminary experiments:
continuing pre-training our pruned models on an existing pre-training dataset RedPajama (*TogetherAI, [2023b](#bib.bib62 "")*; LLaMA’s pre-training dataset)
reduces loss at different rates across domains compared to pre-training a model from scratch, which signifies an inefficient use of data.

To be more specific, we first fit a scaling law(*Hoffmann et al., [2022](#bib.bib27 "")*; details in[Appendix A](#A1 "Appendix A Reference Loss Predicted by Scaling Laws ‣ Pre-training via Structured Pruning"))
on the series of LLaMA2 models for each domain. Then we predict the loss that a hypothetical 2.7B LLaMA2 model would achieve if trained from scratch on the same data.
We obtain these estimatedreference losses across domains of the pre-training data and compare them to the losses of our pruned model after continued pre-training.
As shown in [Figure 6](#S4.F6 "In Downstream performance. ‣ 4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning") (left),
while our model’s loss on GitHub is better than the reference loss, it is significantly worse than the reference loss on C4.
This observation indicates that pruning preserves a greater amount of knowledge in low-entropy and smaller domains (e.g., GitHub) compared to high-entropy and larger domains (e.g., C4).
As demonstrated later in Section[4.1](#S4.SS1 "4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning"),
simply reusing the original pre-training data distribution222The LLaMA2 pre-training data is not public. We conducted the same analysis on LLaMA1 models and observed a similar phenomenon, indicating that this is a universal issue unrelated to specific pre-training data. results in an inefficient use of data and worse downstream performance,
even if the overall loss is seemingly low.

Inspired by recent work*(Xie et al., [2023](#bib.bib73 ""))*,
we propose dynamic batch loading,
a more efficient algorithm to simply adjust domain proportions on the fly based on the model performance. The goal is to ensure the model achieves the reference loss roughly simultaneously across all domains. We introduce the algorithm below.

Problem setup. The pre-training data comprises of $k$ domains $D_{1},D_{2},\cdots,D_{k}$ and we have a held-out validation dataset for each domain, denoted as $D_{i}^{\mathrm{val}}$.
At each training step $t$,
a proportion $w_{t}[i]$ of the data comes from domain $D_{i}$. We set a reference validation loss $\ell_{\mathrm{ref}}(D_{i})$ for each domain and train the pruned model to reach the reference loss.

Dynamic batch loading. We present the full algorithm in [Algorithm 1](#alg1 "In 2.2 Dynamic Batch Loading ‣ 2 LLM-Shearing ‣ Pre-training via Structured Pruning").
In a sketch,
for every $m$ steps, we evaluate the model to get the validation loss $\ell_{t}$ (step $t$) on $D^{\mathrm{val}}$,
and update $w_{t}$ based on the difference $\Delta_{t}(D_{i})$ between $\ell_{\mathrm{ref}}[i]$ and $\ell_{t}[i]$ on each domain. The update rule is exponential ascent following*Xie et al. ([2023](#bib.bib73 ""))*,

|  | $\alpha_{t}\=w_{t-m}\cdot\exp({\Delta_{t}});\quad w_{t}\=\frac{\alpha_{t}}{\sum_{i}\alpha_{t}[i]}.\\$ |  |
| --- | --- | --- |

We apply dynamic batch loading to both the pruning stage and the continued pre-training stage.
For pruning, we use the original pre-training data’s domain weights as $w_{0}$.
For continued pre-training, we use the final weights from the pruning stage as $w_{0}$.
Dynamic batch loading
leverages reference losses on validation sets and adjusts the weights dynamically, so it adds minimal overhead to standard training.
This improves the efficiency of *Xie et al. ([2023](#bib.bib73 ""))*, which requires
training both a reference and a proxy model to learn domain weights before training.

More broadly, dynamic batch loading has the potential to train an LLM to match reference losses of any model, by leveraging open-source pre-training datasets such as RedPajama, even when the reference model’s training data is unknown.

Choices of reference losses. By default, we use the loss predicted by the scaling law as the reference (denoted as scaling reference). We also experiment with an alternative where we directly use the source model’s domain validation loss as the reference (denoted as source reference). We show in [Section E.3](#A5.SS3 "E.3 Coding and Math Reasoning ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") and[E.4](#A5.SS4 "E.4 Scaling Reference vs. Source Reference ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") that
while both variants perform well,
using scaling reference leads to slightly better downstream results, especially on math and coding tasks. However, source reference is a viable alternative when a series of source models at different scales is not available.

3 Experiments
-------------

### 3.1 Setup

#### Model configurations.

We use the LLaMA2-7B model*(Touvron et al., [2023b](#bib.bib64 ""))* as the source model throughout all of our main experiments.333Please find results on LLaMA1 models in [Section E.5](#A5.SS5 "E.5 Pruning from LLaMA1 vs LLaMA2 ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning"). We then conduct structured pruning experiments to compress this model down to two smaller target sizes—2.7B and 1.3B parameters. We compare to strong pre-trained language models of similar sizes,
including
OPT-1.3B*(Zhang et al., [2022](#bib.bib76 ""))*, Pythia-1.4B*(Biderman et al., [2023](#bib.bib4 ""))*,
OPT-2.7B, Pythia-2.8B, INCITE-Base-3B*(TogetherAI, [2023b](#bib.bib62 ""))*,
OpenLLaMA-3B-v1, and OpenLLaMA-3B-v2*(Geng \& Liu, [2023](#bib.bib20 ""))*.
We use Pythia-1.4B as the target architecture for the 1.3B model, and INCITE-Base-3B as the target architecture for the 2.7B model.[Table 8](#A3.T8 "In Appendix C Model Configurations ‣ Pre-training via Structured Pruning") summarizes model architecture details of all these models.

*Table 1:  A summary of pre-training datasets used by Sheared-LLaMA and other models.*

| Model | Pre-training Data | #Tokens |
| --- | --- | --- |
| LLaMA1 | LLaMA data | 1T |
| LLaMA2 | Unknown | 2T |
| OPT | OPT data444OPT data contains BookCorpus(Zhu et al., [2015](#bib.bib78 "")), Stories(Trinh \& Le, [2018](#bib.bib65 "")), CCNews(Hamborg et al., [2017](#bib.bib23 "")), the Pile(Gao et al., [2020](#bib.bib18 "")), and PushShift.io Reddit(Baumgartner et al., [2020](#bib.bib3 "")). | 300B |
| Pythia | The Pile | 300B |
| INCITE-Base | RedPajama | 800B |
| OpenLLaMA v1 | RedPajama | 1T |
| OpenLLaMA v2 | OpenLLaMA data555OpenLLaMA v2 data is a mixture of RefinedWeb(Penedo et al., [2023](#bib.bib52 "")), StarCoder(Li et al., [2023](#bib.bib38 "")), and part of RedPajama. | 1T |
| Sheared-LLaMA | RedPajama | 50B |

#### Data.

As the training data for LLaMA2 is not publicly accessible, we use RedPajama*(TogetherAI, [2023b](#bib.bib62 ""))*, which is a replicated pre-training dataset of the LLaMA1 models*(Touvron et al., [2023a](#bib.bib63 ""))*, for pruning and continued-pretraining. This dataset encompasses training data from seven domains: CommonCrawl, C4, Github, Wikipedia, Books, ArXiv, and StackExchange. We construct a held-out validation set with 2 million tokens (equivalent to 500 sequences of 4,096 tokens) for each domain. We allocate 0.4 billion tokens for the pruning phase and 50 billion tokens for the continued pre-training process. Following the conventions of LLaMA2, we maintain a sequence length of 4,096 tokens.[Table 1](#S3.T1 "In Model configurations. ‣ 3.1 Setup ‣ 3 Experiments ‣ Pre-training via Structured Pruning") provides a summary of the pre-training data used by our models and the baseline models.

#### Training.

Our implementation builds on the Composer package*(MosaicML, [2021](#bib.bib47 ""))*. We use a maximum of 16 Nvidia A100 GPUs (80GB) for all experiments (More details are in [Appendix B](#A2 "Appendix B Training Details ‣ Pre-training via Structured Pruning")).

44footnotetext: OPT data contains BookCorpus*(Zhu et al., [2015](#bib.bib78 ""))*, Stories*(Trinh \& Le, [2018](#bib.bib65 ""))*, CCNews*(Hamborg et al., [2017](#bib.bib23 ""))*, the Pile*(Gao et al., [2020](#bib.bib18 ""))*, and PushShift.io Reddit*(Baumgartner et al., [2020](#bib.bib3 ""))*. 55footnotetext: OpenLLaMA v2 is pre-trained with a mixture of
RefinedWeb*(Penedo et al., [2023](#bib.bib52 ""))*,
StarCoder*(Li et al., [2023](#bib.bib38 ""))*,
and part of RedPajama.

#### Downstream task evaluation.

We use the lm-evaluation-harness package*(Gao et al., [2021](#bib.bib19 ""))* to evaluate on an extensive suite of downstream tasks:

* •

    We follow Pythia and LLaMA2 to report the 0-shot accuracy of ARC easy (ARC-E; *Clark et al., [2018](#bib.bib10 "")*), LAMBADA*(Paperno et al., [2016](#bib.bib51 ""))*, LogiQA*(Liu et al., [2020](#bib.bib41 ""))*, PIQA*(Bisk et al., [2020](#bib.bib5 ""))*, SciQ*(Welbl et al., [2017](#bib.bib68 ""))*, and WinoGrande*(Sakaguchi et al., [2021](#bib.bib54 ""))*.

* •

    We report accuracy of the tasks used by Open LLM Leaderboard666<https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>, including 10-shot HellaSwag*(Zellers et al., [2019](#bib.bib74 ""))*, 25-shot ARC Challenge (ARC-C; *Clark et al., [2018](#bib.bib10 "")*), and 5-shot MMLU*(Hendrycks et al., [2021](#bib.bib25 ""))*.

* •

    We also report exact match of 32-shot Natural Questions (NQ; *Kwiatkowski et al., [2019](#bib.bib34 "")*) to measure the factual knowledge in the model.

#### Instruction tuning evaluation.

As training models to follow instructions has become a crucial application of LLMs *(Ouyang et al., [2022](#bib.bib50 ""); Taori et al., [2023](#bib.bib60 ""))*,
we evaluate our models on instruction tuning and fine-tune both Sheared-LLaMA
and baseline models on
10,000 instruction-response pairs sampled from the ShareGPT dataset777<https://sharegpt.com>. We only use the first round in the multi-turn chat history..
For evaluation, we sample another 1,000 instructions from ShareGPT, generate responses from our fine-tuned models and other baseline models, and use
GPT-4
as an evaluator to compare the two responses*(Dubois et al., [2023](#bib.bib15 ""))*.
We report the win rate of our model compared to the baseline model (more details in [Appendix D](#A4 "Appendix D Instruction Tuning ‣ Pre-training via Structured Pruning")).

*Table 2:  Sheared-LLaMA outperforms publicly available models of comparable size on downstream tasks. The shot number used is noted in parentheses, with 0-shot if not specified.
Models with $\dagger$ use a different training data from RedPajama. Please refer to [Table 1](#S3.T1 "In Model configurations. ‣ 3.1 Setup ‣ 3 Experiments ‣ Pre-training via Structured Pruning") for details.*

|  | Commonsense \& Reading Comprehension | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Model (#tokens for training) | SciQ | PIQA | WinoGrande | ARC-E | ARC-C (25) | HellaSwag (10) |
| LLaMA2-7B (2T)† | 93.7 | 78.1 | 69.3 | 76.4 | 53.0 | 78.6 |
| OPT-1.3B (300B)† | 84.3 | 71.7 | 59.6 | 57.0 | 29.7 | 54.5 |
| Pythia-1.4B (300B)† | 86.4 | 70.9 | 57.4 | 60.7 | 31.2 | 53.0 |
| Sheared-LLaMA-1.3B (50B) | 87.3 | 73.4 | 57.9 | 61.5 | 33.5 | 60.7 |
| OPT-2.7B (300B)† | 85.8 | 73.7 | 60.8 | 60.8 | 34.0 | 61.5 |
| Pythia-2.8B (300B)† | 88.3 | 74.0 | 59.7 | 64.4 | 36.4 | 60.8 |
| INCITE-Base-3B (800B) | 90.7 | 74.6 | 63.5 | 67.7 | 40.2 | 64.8 |
| Open-LLaMA-3B-v1 (1T) | 91.3 | 73.7 | 61.5 | 67.6 | 39.6 | 62.6 |
| Open-LLaMA-3B-v2 (1T)† | 91.8 | 76.2 | 63.5 | 66.5 | 39.0 | 67.6 |
| Sheared-LLaMA-2.7B (50B) | 90.8 | 75.8 | 64.2 | 67.0 | 41.2 | 70.8 |
|  | Continued | | LM | World Knowledge | |  |
| Model (#tokens for training) | LogiQA | BoolQ (32) | LAMBADA | NQ (32) | MMLU (5) | Average |
| LLaMA2-7B (2T)† | 30.7 | 82.1 | 28.8 | 73.9 | 46.6 | 64.6 |
| OPT-1.3B (300B)† | 26.9 | 57.5 | 58.0 | 6.9 | 24.7 | 48.2 |
| Pythia-1.4B (300B)† | 27.3 | 57.4 | 61.6 | 6.2 | 25.7 | 48.9 |
| Sheared-LLaMA-1.3B (50B) | 26.9 | 64.0 | 61.0 | 9.6 | 25.7 | 51.0 |
| OPT-2.7B (300B)† | 26.0 | 63.4 | 63.6 | 10.1 | 25.9 | 51.4 |
| Pythia-2.8B (300B)† | 28.0 | 66.0 | 64.7 | 9.0 | 26.9 | 52.5 |
| INCITE-Base-3B (800B) | 27.7 | 65.9 | 65.3 | 14.9 | 27.0 | 54.7 |
| Open-LLaMA-3B-v1 (1T) | 28.4 | 70.0 | 65.4 | 18.6 | 27.0 | 55.1 |
| Open-LLaMA-3B-v2 (1T)† | 28.1 | 69.6 | 66.5 | 17.1 | 26.9 | 55.7 |
| Sheared-LLaMA-2.7B (50B) | 28.9 | 73.7 | 68.4 | 16.5 | 26.4 | 56.7 |

### 3.2 Sheared-LLaMA Outperforms LMs of Equivalent Sizes

We demonstrate,
on both standard LM benchmarks
and instruction tuning,
Sheared-LLaMA significantly outperforms
existing LLMs of similar sizes,
while using only a fraction of the compute budget to train those models from scratch.

#### Downstream tasks.

In [Table 2](#S3.T2 "In Instruction tuning evaluation. ‣ 3.1 Setup ‣ 3 Experiments ‣ Pre-training via Structured Pruning"),
we present the zero-shot and few-shot downstream task performance of both Sheared-LLaMA and existing pre-trained models
of a similar size.
Our experiments show that,
even with a budget as limited as approximately 50B tokens for pruning and continued pre-training,
Sheared-LLaMA models outperform existing models that have been pre-trained on significantly larger compute.
To elaborate further,
Sheared-LLaMA-1.3B outperforms both the OPT-1.3B and Pythia-1.4B models, which were originally pre-trained with 300B tokens.
Similarly, Sheared-LLaMA-2.7B outperforms INCITE-Base-3B
and OpenLLaMA-3B-v1,
which were pre-trained on 800B and 1T RedPajama tokens respectively;
Sheared-LLaMA-2.7B also surpasses OpenLLaMA-3B-v2, which was trained on 1T tokens from a mixture of RedPajama, RefinedWeb, and StarCoder.

<img src='x3.png' alt='Refer to caption' title='' width='461' height='69' />

*Figure 3: Sheared-LLaMAs outperform Pythia-1.4B, INCITE-Base-3B, OpenLLaMA-3B-v1 and OpenLLaMA-3B-v2 in instruction tuning.*

#### Instruction tuning.

As shown [Figure 3](#S3.F3 "In Downstream tasks. ‣ 3.2 Sheared-LLaMA Outperforms LMs of Equivalent Sizes ‣ 3 Experiments ‣ Pre-training via Structured Pruning"),
instruction-tuned Sheared-LLaMA achieves higher win rates compared to all the other pre-trained models at a comparable scale. This demonstrates that our 2.7B model can serve as a strong foundation for instruction tuning and has the capacity to generate long, coherent and informative responses (See examples in [Appendix D](#A4 "Appendix D Instruction Tuning ‣ Pre-training via Structured Pruning")).

#### Comparison to further pre-training an existing LM.

<img src='x4.png' alt='Refer to caption' title='' width='169' height='118' />

*Figure 4: Average downstream performance of continuing pre-training Sheared-LLaMA vs INCITE-Base-3B.*

We examine if pruning produces a better initialization for continued pre-training than an existing LLM of equivalent size.
We continue pre-training an INCITE-Base-3B model on the original RedPajama data
and compare it to Sheared-LLaMA-2.7B. [Figure 4](#S3.F4 "In Comparison to further pre-training an existing LM. ‣ 3.2 Sheared-LLaMA Outperforms LMs of Equivalent Sizes ‣ 3 Experiments ‣ Pre-training via Structured Pruning") shows that
the INCITE-Base-3B model starts off with much higher accuracy, but its performance plateaus throughout continued pre-training.
In contrast,
Sheared-LLaMA starts at a lower accuracy but rapidly improves, eventually surpassing the INCITE-Base-3B model.
This suggests that pruned models from a strong base model serve as a better initialization for continued pre-training.888In cases where the existing small model is competitive compared to the pruning source model, the small model may offer a better starting point than a pruned model. Intuitively, the larger the discrepancy in performance between the source model and the small model, the more advantages the pruned model . Please find more training details in [Appendix F](#A6 "Appendix F Training details to continual pre-training INCITE-Base-3B ‣ Pre-training via Structured Pruning").

4 Analysis
----------

### 4.1 Effectiveness of Dynamic Batch Loading

We analyze the effectiveness of dynamic batch loading by examining its impact on three aspects:
(1) the final LM loss across domains,
(2) the data usage of each domain throughout training,
(3) the downstream task performance.
All results in this section are based on Sheared-LLaMA-1.3B.

#### Loss differences across domains.

Dynamic batch loading is designed to balance the rate of loss reduction across domains, so that the losses reach the reference value at approximately the same time.
In [Figure 6](#S4.F6 "In Downstream performance. ‣ 4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning"), we plot the difference between the loss of our model (with both original and dynamic batch loading) and the reference loss, estimated by fitting a scaling function to a hypothetical 2.7B parameter LLaMA2 model.
With the original batch loading, the loss differences vary dramatically across domains. For instance, the GitHub loss decreases below the reference value, while the C4 loss lags behind.
In contrast, dynamic batch loading
reduces losses evenly
and shows very similar loss differences across domains,
indicating a more efficient data use.

#### Data usage.

[Table 3](#S4.T3 "In Downstream performance. ‣ 4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning") compares the original data proportion of RedPajama
and the domain data usage of our dynamic loading ([Figure 7](#A5.F7 "In E.1 Data Usage in Continued Pre-training ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") shows the
evolution of domain weights throughout the training).
We see that
dynamic batch loading increases the weights for
the Book and C4 domains versus other domains—suggesting that they
are more difficult to recover for a pruned model.

#### Downstream performance.

<img src='x5.png' alt='Refer to caption' title='' width='351' height='244' />

*Figure 5: Loss difference between the pruned model (1.3B) and estimated reference loss, with original vs. dynamic batch loading.*

<img src='x6.png' alt='Refer to caption' title='' width='351' height='243' />

*Figure 6: Downstream task performance of Sheared-LLaMA-1.3B with original data proportion and dynamic batch loading.*

*Table 3: Domain data usage with dynamic batch loading compared to the original proportions.*

|  | CC | GitHub | Book | StackExchange | Wiki | ArXiv | C4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RedPajama | $67.0\%$ | $4.5\%$ | $4.5\%$ | $2.0\%$ | $4.5\%$ | $2.5\%$ | $15.0\%$ |
| Dynamic Batch Loading | $36.1\%$ | $0.8\%$ | $9.1\%$ | $1.0\%$ | $3.1\%$ | $0.7\%$ | $49.2\%$ |

As shown in Figure[6](#S4.F6 "Figure 6 ‣ Downstream performance. ‣ 4.1 Effectiveness of Dynamic Batch Loading ‣ 4 Analysis ‣ Pre-training via Structured Pruning"), pruned models trained with dynamic batch loading
achieve better downstream performance
than when trained on the original RedPajama distribution.
This suggests that the more balanced loss reduction from dynamic batch loading
transfers to improved downstream capabilities.

### 4.2 Comparison to Other Pruning Approaches

We compare our LLM-shearing method to other pruning approaches and report validation perplexity, which serves as a strong indicator of overall model capabilities*(Xia et al., [2023](#bib.bib71 ""))*. Due to computational constraints, the following experiments control the total compute budget across compared methods rather than runing each method to completion.

#### Targeted pruned models have a higher inference throughput.

Previous works like Block Pruning*(Lagunas et al., [2021](#bib.bib35 ""))* or CoFiPruning*(Xia et al., [2022](#bib.bib70 ""))* are experimented on BERT-scale LMs,
and the final model architectures, though structured,
usually have non-uniform layer configurations,
e.g., different layers have different number of heads or intermediate size.
While bringing
performance gains,
non-uniformity also introduces training and inference overhead due to
irregularities in model architectures. We experiment with both CoFiPruning and our targeted structured pruning. For a fair comparsion, we use the same original data proportion for both approaches.
As shown in [Table 4](#S4.T4 "In Targeted pruned models have a higher inference throughput. ‣ 4.2 Comparison to Other Pruning Approaches ‣ 4 Analysis ‣ Pre-training via Structured Pruning"),
our targeted pruned models have a higher inference throughput compard to the non-uniformly pruned CoFiPruning model at the same sparsity, despite having slightly higher perplexity.

*Table 4:  Validation perplexity and inference throughout (tokens/second) of targeted structured pruning (without continued pre-training) with a uniform layer configuration, and CoFiPruning, with a non-uniform layer configuration. Inference throughput is measured on a Nvidia A100 (80G) GPU, with a batch size of 1 and a sequence length of 512.*

|  | Layer Config | PPL $\downarrow$ | Throughput $\uparrow$ |  | Layer Config | PPL $\downarrow$ | Throughput $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.3B | CoFiPruning | 9.1 | 51 | 3B | CoFiPruning | 7.0 | 37 |
| | Ours | 10.3 | 58 | | Ours | 7.7 | 43 |

#### Comparison to LLM-Pruner(Ma et al., [2023](#bib.bib45 "")).

We compare our pruning method to
LLM-Pruner, a recent work in uniform layer configuration structured pruning,
in [Section E.2](#A5.SS2 "E.2 Comparison to LLM-Pruner ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning").
We show that with the same budget and the compression rate,
ours achieves better perplexity.

*Table 5: Data budget allocation to pruning and continued pre-training (CT) and corresponding perplexity.*

| # Tokens | | PPL | |
| --- | --- | --- | --- |
| Pruning | CT | Pruning | CT |
| 0.2B | 4.6B | 12.99 | 7.46 |
| 0.4B | 4.4B | 10.29 | 7.32 |
| 0.8B | 4.0B | 9.01 | 7.23 |
| 1.6B | 3.2B | 8.04 | 7.08 |

### 4.3 Additional Analysis

#### Budget allocation for pruning and continued pre-training.

Intuitively, allocating more compute to the pruning stage helps identify better subnetwork structures.
We explore distributing data across pruning and continued pre-training stages differently, within a fixed budget of 5B tokens. [Table 5](#S4.T5 "In Comparison to LLM-Pruner (Ma et al., 2023). ‣ 4.2 Comparison to Other Pruning Approaches ‣ 4 Analysis ‣ Pre-training via Structured Pruning") shows that
when controlling the total amount of tokens,
increasing the pruning budget consistently improves perplexity.
However, since pruning is more expensive than continued pre-training,
we decide to allocate 0.4B tokens to pruning. Please refer to [Appendix B](#A2 "Appendix B Training Details ‣ Pre-training via Structured Pruning") for details on training throughputs

#### Performance on math and coding tasks.

We also evaluate Sheared-LLaMA and baseline models on math and coding benchmarks in [Section E.3](#A5.SS3 "E.3 Coding and Math Reasoning ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning").
Sheared-LLaMA outperforms baselines trained on the same RedPajama data, but lags behind models trained on more ArXiv and GitHub data. This highlights a limitation of our work, where the performance is bounded by the chosen reference loss. To improve over math and coding, a better initial data proportion (e.g., more GitHub) and better reference losses are needed , and we leave it for future work.

5 Related Work
--------------

#### Pruning.

Structured pruning has been extensively studied as a model compression technique in computer vision and natural language processing,
where task-specific models like classification ones are often overparameterized and can be pruned significantly with minimal impact on performance*(Han et al., [2016](#bib.bib24 ""); Wen et al., [2016](#bib.bib69 ""); Liu et al., [2017](#bib.bib42 ""); Luo et al., [2017](#bib.bib44 ""); Cai et al., [2019](#bib.bib7 ""); Deng et al., [2020](#bib.bib12 ""); Hou et al., [2020](#bib.bib28 ""); Wang et al., [2020](#bib.bib67 ""); Lagunas et al., [2021](#bib.bib35 ""); Xia et al., [2022](#bib.bib70 ""); Kurtic et al., [2023](#bib.bib33 ""))*.
Unstructured pruning*(Frankle \& Carbin, [2018](#bib.bib16 ""); Li et al., [2020](#bib.bib39 ""); Chen et al., [2020](#bib.bib8 ""); Sanh et al., [2020](#bib.bib56 ""))* prunes individual neurons instead of structured blocks.
Though unstructured pruning usually achieve higher compression rates,
they are not practical for model speedup.

In the era of LLMs,
the prevalent NLP pipeline has shifted from task-specific models
to general-purpose LMs,
which leaves little room for redundancy.
Both unstructured pruning,
semi-structured pruning*(Frantar \& Alistarh, [2023](#bib.bib17 ""); Sun et al., [2023](#bib.bib58 ""))*,
and structured pruning*(Ma et al., [2023](#bib.bib45 ""))* lead to significant performance drops on LLM even at a modest sparsity.
Noticeably, all the aforementioned works
fix the original model parameters or tune them minimally.
In our work, we
see pruning as an initialization and
consider it necessary to expend substantial compute to
continually pre-training the model to recover performance.

#### Efficient pre-training approaches.

As orthogonal to our pruning approach,
There is an extensive body of work on improving efficiency of training LLMs.
For example,
quantization reduces the numeric precision of model weights and activations
and speeds up training and inference*(Dettmers et al., [2022](#bib.bib13 ""); [2023](#bib.bib14 ""); Xiao et al., [2023](#bib.bib72 ""))*.
Knowledge distillation*(Hinton et al., [2015](#bib.bib26 ""); Sanh et al., [2019](#bib.bib55 ""); Jiao et al., [2020](#bib.bib31 ""); Sun et al., [2020](#bib.bib59 ""))*,
which trains a smaller model on a larger model’s prediction,
is shown to be effective for task-specific models*(Xia et al., [2022](#bib.bib70 ""))*. For pre-training LLMs, though distilling from a teacher model is shown to improve the quality of student models given the same number of training steps*(Rae et al., [2021](#bib.bib53 ""); Blakeney et al., [2022](#bib.bib6 ""))*, it is less cost-effective than pruning and continued training due to the exceeding inference cost incured by the teacher model*(Jha et al., [2023](#bib.bib29 ""))*. More methods have been introduced to enhance the efficiency of training LMs,
such as dynamic architectures*(Gong et al., [2019](#bib.bib21 ""); Zhang \& He, [2020](#bib.bib75 ""))* and efficient optimizers*(Chen et al., [2023](#bib.bib9 ""); Liu et al., [2023](#bib.bib40 ""))*. However,
as indicated by*(Kaddour et al., [2023](#bib.bib32 ""))*,
the promised gains in training efficiency may not be consistently realized.

There are also data-based approaches to enhance training efficiency.
Eliminating duplicated data is found to be effective *(Lee et al., [2021](#bib.bib36 ""))*.
Various batch selection techniques propose to prioritize data based on criteria such as
higher losses*(Jiang et al., [2019](#bib.bib30 ""))* or a greater reducible loss*(Mindermann et al., [2022](#bib.bib46 ""))*. *Xie et al. ([2023](#bib.bib73 ""))* propose to
optimize data mixtures
by training a proxy model to estimate the optimal data weight of each domain.

6 Discussion
------------

#### Limitation and future work.

This work has two main limitations. First, the method relies heavily on the availability of open-source pre-training datasets and large language models. If the pre-training data does not cover a particular domain, the method is unlikely to recover performance well on that domain. Second, due to computational constraints, we only conducted experiments using a 7B parameter model. However, our method is highly generalizable and can be scaled up to larger models in future research.

#### Conclusion.

In this work, we propose using structured pruning as an efficient approach to
producing competitive LLMs. Our approach consists of two stages, targeted structured pruning and continued pre-training, and we propose dynamic batch loading to improve the efficiency of using pre-training data.
We train a series of competitive Sheared-LLaMA models with a fraction of compute compared to standard pre-training.
Our results
highlight a promising avenue to produce small LLMs with a low cost
when strong large-scale models are available.
As more capable LLMs and larger pre-training datasets emerge, our method can easily extend to these advances to produce improved small models.

Acknowledgements
----------------

We express our gratitude to Sadhika Malladi, Tanya Goyal, Ofir Press, Adithya Bhaskar, and the Princeton NLP group for reviewing the paper and providing helpful feedback. We also thank the engineering team at MosaicML for their invaluable assistance with implementation specifics using the Composer package. Mengzhou Xia is supported by a Bloomberg Data Science Ph.D. Fellowship, and Tianyu Gao is supported by an IBM PhD Fellowship.
This research is also supported by Microsoft Azure credits through the “Accelerate Foundation Models Academic Research” Initiative.

References
----------

* Almazrouei et al. (2023)Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli,
Ruxandra Cojocaru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien
Launay, Quentin Malartic, Badreddine Noune, Baptiste Pannier, and Guilherme
Penedo.Falcon-40B: an open large language model with state-of-the-art
performance.2023.
* Anthropic (2023)Anthropic.Introducing claude, 2023.
* Baumgartner et al. (2020)Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy
Blackburn.The pushshift reddit dataset.*ArXiv*, abs/2001.08435, 2020.
* Biderman et al. (2023)Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley,
Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit,
USVSN Sai Prashanth, Edward Raff, et al.Pythia: A suite for analyzing large language models across training
and scaling.In *International Conference on Machine Learning*, pp. 2397–2430. PMLR, 2023.
* Bisk et al. (2020)Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al.Piqa: Reasoning about physical commonsense in natural language.In *Proceedings of the AAAI conference on artificial
intelligence*, volume 34, pp. 7432–7439, 2020.
* Blakeney et al. (2022)Cody Blakeney, Jessica Zosa Forde, Jonathan Frankle, Ziliang Zong, and
Matthew L Leavitt.Reduce, reuse, recycle: Improving training efficiency with
distillation.*arXiv preprint arXiv:2211.00683*, 2022.
* Cai et al. (2019)Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han.Once-for-all: Train one network and specialize it for efficient
deployment.In *International Conference on Learning Representations*, 2019.
* Chen et al. (2020)Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang
Wang, and Michael Carbin.The lottery ticket hypothesis for pre-trained bert networks.In *Advances in Neural Information Processing Systems*, 2020.
* Chen et al. (2023)Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu
Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al.Symbolic discovery of optimization algorithms.*arXiv preprint arXiv:2302.06675*, 2023.
* Clark et al. (2018)Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa
Schoenick, and Oyvind Tafjord.Think you have solved question answering? try arc, the ai2 reasoning
challenge.*arXiv preprint arXiv:1803.05457*, 2018.
* Dao et al. (2022)Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.Flashattention: Fast and memory-efficient exact attention with
io-awareness.*Advances in Neural Information Processing Systems*,
35:16344–16359, 2022.
* Deng et al. (2020)Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie.Model compression and hardware acceleration for neural networks: A
comprehensive survey.*Proceedings of the IEEE*, 108(4):485–532,
2020.
* Dettmers et al. (2022)Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.Llm.int8 (): 8-bit matrix multiplication for transformers at scale.*arXiv preprint arXiv:2208.07339*, 2022.
* Dettmers et al. (2023)Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.Qlora: Efficient finetuning of quantized llms.*arXiv preprint arXiv:2305.14314*, 2023.
* Dubois et al. (2023)Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba,
Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto.Alpacafarm: A simulation framework for methods that learn from human
feedback.*arXiv preprint arXiv:2305.14387*, 2023.
* Frankle \& Carbin (2018)Jonathan Frankle and Michael Carbin.The lottery ticket hypothesis: Finding sparse, trainable neural
networks.In *International Conference on Learning Representations*, 2018.
* Frantar \& Alistarh (2023)Elias Frantar and Dan Alistarh.Sparsegpt: Massive language models can be accurately pruned in
one-shot, 2023.*arXiv preprint arXiv:2301.00774*, 2023.
* Gao et al. (2020)Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.The pile: An 800gb dataset of diverse text for language modeling.*arXiv preprint arXiv:2101.00027*, 2020.
* Gao et al. (2021)Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff,
Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang,
and Andy Zou.A framework for few-shot language model evaluation, September 2021.
* Geng \& Liu (2023)Xinyang Geng and Hao Liu.Openllama: An open reproduction of llama, May 2023.
* Gong et al. (2019)Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu.Efficient training of bert by progressively stacking.In *International conference on machine learning*, pp. 2337–2346. PMLR, 2019.
* Gupta et al. (2023)Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L Richter, Quentin
Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort.Continual pre-training of large language models: How to (re) warm
your model?*arXiv preprint arXiv:2308.04014*, 2023.
* Hamborg et al. (2017)Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp.news-please: A generic news crawler and extractor.In *Proceedings of the 15th International Symposium of
Information Science*, pp. 218–223, 2017.
* Han et al. (2016)Song Han, Huizi Mao, Dally, and William Dally.Deep compression: Compressing deep neural networks with pruning,
trained quantization and huffman coding.In *International Conference on Learning Representations*, 2016.
* Hendrycks et al. (2021)Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn
Song, and Jacob Steinhardt.Measuring massive multitask language understanding.In *International Conference on Learning Representations*, 2021.
* Hinton et al. (2015)Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.Distilling the knowledge in a neural network.*arXiv preprint arXiv:1503.02531*, 2015.
* Hoffmann et al. (2022)Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor
Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
Welbl, Aidan Clark, et al.Training compute-optimal large language models.*arXiv preprint arXiv:2203.15556*, 2022.
* Hou et al. (2020)Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu.Dynabert: Dynamic bert with adaptive width and depth.*Advances in Neural Information Processing Systems*,
33:9782–9793, 2020.
* Jha et al. (2023)Ananya Harsh Jha, Dirk Groeneveld, Emma Strubell, and Iz Beltagy.Large language model distillation doesn’t need a teacher.*arXiv preprint arXiv:2305.14864*, 2023.
* Jiang et al. (2019)Angela H Jiang, Daniel L-K Wong, Giulio Zhou, David G Andersen, Jeffrey Dean,
Gregory R Ganger, Gauri Joshi, Michael Kaminksy, Michael Kozuch, Zachary C
Lipton, et al.Accelerating deep learning by focusing on the biggest losers.*arXiv preprint arXiv:1910.00762*, 2019.
* Jiao et al. (2020)Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang
Wang, and Qun Liu.Tinybert: Distilling bert for natural language understanding.In *Findings of the Association for Computational Linguistics:
EMNLP 2020*, pp. 4163–4174, 2020.
* Kaddour et al. (2023)Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt J Kusner.No train no gain: Revisiting efficient training algorithms for
transformer-based language models.*arXiv preprint arXiv:2307.06440*, 2023.
* Kurtic et al. (2023)Eldar Kurtic, Elias Frantar, and Dan Alistarh.Ziplm: Hardware-aware structured pruning of language models.*arXiv preprint arXiv:2302.04089*, 2023.
* Kwiatkowski et al. (2019)Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin,
Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang,
Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.Natural questions: A benchmark for question answering research.*Transactions of the Association for Computational Linguistics*,
7:452–466, 2019.
* Lagunas et al. (2021)François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush.Block pruning for faster transformers.*arXiv preprint arXiv:2109.04838*, 2021.
* Lee et al. (2021)Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck,
Chris Callison-Burch, and Nicholas Carlini.Deduplicating training data makes language models better.*arXiv preprint arXiv:2107.06499*, 2021.
* Li et al. (2016)Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf.Pruning filters for efficient convnets.In *International Conference on Learning Representations*, 2016.
* Li et al. (2023)Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov,
Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al.Starcoder: may the source be with you!*arXiv preprint arXiv:2305.06161*, 2023.
* Li et al. (2020)Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and
Joey Gonzalez.Train big, then compress: Rethinking model size for efficient
training and inference of transformers.In *International Conference on machine learning*, pp. 5958–5968. PMLR, 2020.
* Liu et al. (2023)Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma.Sophia: A scalable stochastic second-order optimizer for language
model pre-training.*arXiv preprint arXiv:2305.14342*, 2023.
* Liu et al. (2020)Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang.Logiqa: A challenge dataset for machine reading comprehension with
logical reasoning.In *Proceedings of the Twenty-Ninth International Joint
Conference on Artificial Intelligence, IJCAI-20*, pp. 3622–3628, 2020.
* Liu et al. (2017)Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui
Zhang.Learning efficient convolutional networks through network slimming.In *Proceedings of the IEEE international conference on computer
vision*, pp. 2736–2744, 2017.
* Louizos et al. (2018)Christos Louizos, Max Welling, and Diederik P Kingma.Learning sparse neural networks through l_0 regularization.In *International Conference on Learning Representations*, 2018.
* Luo et al. (2017)Jian-Hao Luo, Jianxin Wu, and Weiyao Lin.Thinet: A filter level pruning method for deep neural network
compression.In *Proceedings of the IEEE international conference on computer
vision*, pp. 5058–5066, 2017.
* Ma et al. (2023)Xinyin Ma, Gongfan Fang, and Xinchao Wang.Llm-pruner: On the structural pruning of large language models.*arXiv preprint arXiv:2305.11627*, 2023.
* Mindermann et al. (2022)Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas
Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot,
Sebastian Farquhar, et al.Prioritized training on points that are learnable, worth learning,
and not yet learnt.In *International Conference on Machine Learning*, pp. 15630–15649. PMLR, 2022.
* MosaicML (2021)MosaicML.composer, 2021.
* MosaicML (2023)MosaicML.Introducing mpt-7b: A new standard for open-source, commercially
usable llms, 2023.Accessed: 2023-05-05.
* OpenAI (2023)OpenAI.Gpt-4 technical report.*ArXiv*, abs/2303.08774, 2023.
* Ouyang et al. (2022)Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.Training language models to follow instructions with human feedback.*Advances in neural information processing systems*, 35, 2022.
* Paperno et al. (2016)Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham,
Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel
Fernández.The LAMBADA dataset: Word prediction requiring a broad discourse
context.In *Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pp. 1525–1534,
2016.
* Penedo et al. (2023)Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru,
Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei,
and Julien Launay.The refinedweb dataset for falcon llm: outperforming curated corpora
with web data, and web data only.*arXiv preprint arXiv:2306.01116*, 2023.
* Rae et al. (2021)Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann,
Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young,
et al.Scaling language models: Methods, analysis \& insights from training
gopher.*arXiv preprint arXiv:2112.11446*, 2021.
* Sakaguchi et al. (2021)Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.Winogrande: An adversarial winograd schema challenge at scale.*Communications of the ACM*, 64(9):99–106,
2021.
* Sanh et al. (2019)Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.Distilbert, a distilled version of bert: smaller, faster, cheaper and
lighter.*arXiv preprint arXiv:1910.01108*, 2019.
* Sanh et al. (2020)Victor Sanh, Thomas Wolf, and Alexander Rush.Movement pruning: Adaptive sparsity by fine-tuning.*Advances in Neural Information Processing Systems*,
33:20378–20389, 2020.
* Shazeer (2020)Noam M. Shazeer.Glu variants improve transformer.*ArXiv*, abs/2002.05202, 2020.
* Sun et al. (2023)Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter.A simple and effective pruning approach for large language models.*arXiv preprint arXiv:2306.11695*, 2023.
* Sun et al. (2020)Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou.Mobilebert: a compact task-agnostic bert for resource-limited
devices.In *Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics*, pp. 2158–2170, 2020.
* Taori et al. (2023)Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
Guestrin, Percy Liang, and Tatsunori B. Hashimoto.Stanford alpaca: An instruction-following llama model, 2023.
* TogetherAI (2023a)TogetherAI.Redpajama-incite-base-3b-v1, 2023a.
* TogetherAI (2023b)TogetherAI.Redpajama: An open source recipe to reproduce llama training dataset,
2023b.
* Touvron et al. (2023a)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric
Hambro, Faisal Azhar, et al.Llama: Open and efficient foundation language models.*arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. (2023b)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine
Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale,
et al.Llama 2: Open foundation and fine-tuned chat models.*arXiv preprint arXiv:2307.09288*, 2023b.
* Trinh \& Le (2018)Trieu H. Trinh and Quoc V. Le.A simple method for commonsense reasoning.*ArXiv*, abs/1806.02847, 2018.
* Wang et al. (2023)Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu,
Tianyu Liu, and Zhifang Sui.Large language models are not fair evaluators.*arXiv preprint arXiv:2305.17926*, 2023.
* Wang et al. (2020)Ziheng Wang, Jeremy Wohlwend, and Tao Lei.Structured pruning of large language models.In *Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP)*, pp. 6151–6162, 2020.
* Welbl et al. (2017)Johannes Welbl, Nelson F. Liu, and Matt Gardner.Crowdsourcing multiple choice science questions.In *Proceedings of the 3rd Workshop on Noisy User-generated
Text*, pp. 94–106, 2017.
* Wen et al. (2016)Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li.Learning structured sparsity in deep neural networks.*Advances in neural information processing systems*, 29, 2016.
* Xia et al. (2022)Mengzhou Xia, Zexuan Zhong, and Danqi Chen.Structured pruning learns compact and accurate models.In *Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pp. 1513–1528,
Dublin, Ireland, May 2022. Association for Computational Linguistics.doi: 10.18653/v1/2022.acl-long.107.
* Xia et al. (2023)Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth
Pasunuru, Danqi Chen, Luke Zettlemoyer, and Veselin Stoyanov.Training trajectories of language models across scales.In *Proceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pp. 13711–13738,
Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.acl-long.767.
* Xiao et al. (2023)Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han.Smoothquant: Accurate and efficient post-training quantization for
large language models.In *International Conference on Machine Learning*, pp. 38087–38099. PMLR, 2023.
* Xie et al. (2023)Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy
Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu.Doremi: Optimizing data mixtures speeds up language model
pretraining.*arXiv preprint arXiv:2305.10429*, 2023.
* Zellers et al. (2019)Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.HellaSwag: Can a machine really finish your sentence?In *Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics*, pp. 4791–4800, 2019.
* Zhang \& He (2020)Minjia Zhang and Yuxiong He.Accelerating training of transformer-based language models with
progressive layer dropping.*Advances in Neural Information Processing Systems*,
33:14011–14023, 2020.
* Zhang et al. (2022)Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al.Opt: Open pre-trained transformer language models.*arXiv preprint arXiv:2205.01068*, 2022.
* Zhao et al. (2023)Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less
Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al.Pytorch fsdp: experiences on scaling fully sharded data parallel.*arXiv preprint arXiv:2304.11277*, 2023.
* Zhu et al. (2015)Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun,
Antonio Torralba, and Sanja Fidler.Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books.*2015 IEEE International Conference on Computer Vision (ICCV)*,
pp. 19–27, 2015.

Appendix A Reference Loss Predicted by Scaling Laws
---------------------------------------------------

The scaling law of language modeling is a function of model size $N$ and dataset size $D$:

|  | $L(N,D)\=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$ |  |
| --- | --- | --- |

where $E$ captures the loss for the true language distribution in an ideal generation process, and $A,\alpha,B,\beta$ are scaling factors related to model scale or data size. Models in the same model family are usually trained with the same amount of tokens on the same data distribution. In this case, we need a minimum of three models to estimate the constant $E+\frac{B}{D^{\beta}},A$ and $\alpha$. If the models are trained with different amount of tokens, we can estimate $E,A,\alpha,B,\beta$ with a minimal of $5$ models. Note that we will estimate the scaling factors for each domain seperately.

It is known that LLaMA$2$ models have been trained on the same $2$T tokens*(Touvron et al., [2023b](#bib.bib64 ""))*. Therefore, we take the LLaMA2-7B, LLaMA2-13B and LLaMA2-70B checkpoints, evaluate them on the validation set of each domain, and fit the scaling factors with the corresponding loss. Given the limited data points for estimating the scaling law constant, we recognize the projected loss of a hypothetical LLaMA-2.7B model may be biased compared to the true value. We present the predicted loss in [Table 6](#A1.T6 "In Appendix A Reference Loss Predicted by Scaling Laws ‣ Pre-training via Structured Pruning").

*Table 6: Estimated reference loss of hypothetical LLaMA2-1.3B and LLaMA2-2.7B.*

|  | CC | GitHub | Book | StackExchange | Wiki | ArXiv | C4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.3B | 1.964 | 0.746 | 2.139 | 1.612 | 1.759 | 1.445 | 2.125 |
| 2.7B | 1.871 | 0.688 | 2.033 | 1.535 | 1.630 | 1.356 | 2.033 |

Appendix B Training Details
---------------------------

We present the hyperparameters used in our experiments in [Table 7](#A2.T7 "In Appendix B Training Details ‣ Pre-training via Structured Pruning"). We use fully sharded data parallel*(Zhao et al., [2023](#bib.bib77 ""))* to train our models in parallel. We use FlashAttention V1*(Dao et al., [2022](#bib.bib11 ""))* to speed up training. We use a cosine learning rate scheduler and decay the learning rate to a minimum of $10\%$ of the peak value. We conduct some preliminary experiment to determine the peak learning rate for learning the masking variables and Lagrange multiplers, and we find that a learning rate of $1.0$ works well for pruning. We do not tune any other hyper-parameters. The throughput is dependent on the implementations and we believe that our throughput can be further improved by adopting more advanced recent optimizations such as FlashAttention V2*(Dao et al., [2022](#bib.bib11 ""))* and a more recent version of Composer.

*Table 7: Training hyper-parameters and throughput.*

|  | Pruning | Contined Pre-training |
| --- | --- | --- |
| Training budget | $0.4$B | $50$B |
| Learning rate of $z,\phi,\lambda$ | $1.0$ | - |
| Learning Rate of $\theta$ | $0.0001$ | $0.0001$ |
| LR warmup ratio | $10\%$ | $3\%$ |
| Batch size (tokens) | $131$K | $1$M |
| Evaluation interval $m$ (steps) | $50$ | $400$ |
| Steps | $3,200$ | $51,200$ |
| # GPUs | $8$ | $16$ |
| Throughput (tokens/s) | $15$K | $145$K (1.3B) / $77$K (2.7B) |

Appendix C Model Configurations
-------------------------------

In this section, we provide the model configurations for both our Sheared-LLaMA model and the baseline models, as illustrated in [Table 8](#A3.T8 "In Appendix C Model Configurations ‣ Pre-training via Structured Pruning"). Our design closely adheres to the architecture of Pythia-1.4B and INCITE-Base-3B, albeit with some nuanced distinctions. A noteworthy difference is found in the intermediate size of Sheared-LLaMA, which is a consequence of its lineage from LLaMA2-7B. Notably, LLaMA2-7B employs a GLU variant*(Shazeer, [2020](#bib.bib57 ""))* within its feed-forward layer, comprising a gate matrix, an upward-projection matrix, and a downward-projection matrix. In contrast, other models employ the conventional double-matrix feed-forward layer structure. Furthermore, we acknowledge that the shearing algorithm will have to inherit the head dimension of the source model. Instead of explicitly specifying the number of heads based on existing language models, we set the target number of heads to be the target hidden dimension divided by the head dimension of the source model.

*Table 8: Model configurations of our Sheared-LLaMA and baseline models.*

| Model | #Param | #Layers | Hidden | Intermediate | #Heads | Head Dim |
| --- | --- | --- | --- | --- | --- | --- |
| OPT-1.3B | 1.3B | 24 | 2048 | 8192 | 32 | 64 |
| Pythia-1.4B | 1.4B | 24 | 2048 | 8192 | 16 | 128 |
| Sheared-LLaMA-1.3B | 1.3B | 24 | 2048 | 5504 | 16 | 128 |
| OPT-2.7B | 2.7B | 32 | 2560 | 10240 | 32 | 80 |
| Pythia-2.8B | 2.8B | 32 | 2560 | 10240 | 32 | 80 |
| INCITE-Base-3B | 2.8B | 32 | 2560 | 10240 | 32 | 80 |
| OpenLLaMA-3B | 2.7B | 26 | 3200 | 8640 | 32 | 100 |
| Sheared-LLaMA-2.7B | 2.7B | 32 | 2560 | 6912 | 20 | 128 |
| LLaMA2-7B | 6.7B | 32 | 4096 | 11008 | 32 | 128 |

Appendix D Instruction Tuning
-----------------------------

During instruction tuning training, the instruction is prepended with “You are a helpful assistant. Write a response that appropriately completes the request.”. For evaluating the instruction tuning generations, *Wang et al. ([2023](#bib.bib66 ""))* observes using GPT models as a judge could change its preference when swapping the presentation order of the two outputs. Therefore, we compare each output pair twice by swapping the presentation order of the two outputs and finally report the average win-rate of the two rounds to eliminate the position bias.

We randomly select an output generated by Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B in response to a given instruction, and present the generations in Table [10](#A4.T10 "Table 10 ‣ Appendix D Instruction Tuning ‣ Pre-training via Structured Pruning"). Our findings demonstrate that, after instruction tuning, Sheared-LLaMA-2.7B consistently produces long, coherent, and informative outputs in response to the instruction.

*Table 9: Training hyper-parameters for instruction tuning.*

|  | Instruction Tuning |
| --- | --- |
| Learning Rate of $\theta$ | $5e-5$ |
| LR warmup ratio | $3\%$ |
| Batch size (tokens) | $128$ |
| # GPUs | $8$ |

*Table 10: An example output from Sheared-LLaMA-1.3B and Sheared-LLaMA-2.7B. It shows that Sheared-LLaMA can generate long, coherent, and informative responses after instruction tuning.*

| Prompt:Act as semiconductor industry analyst |
| --- |
|  |
| Sheared-LLaMA-1.3B:As a semiconductor industry analyst, I would be responsible for conducting market research and analysis to help companies in the industry understand and respond to market trends, customer preferences, and competitive dynamics. This would include analyzing market data such as sales, market share, and industry trends, as well as conducting interviews with industry leaders, executives, and customers to gain insights into customer needs and preferences. I would also be responsible for developing and implementing marketing strategies to promote products and services, and for identifying and pursuing new business opportunities. |
|  |
| Sheared-LLaMA-2.7B:As a semiconductor industry analyst, my primary responsibility is to provide insightful and accurate analysis on the global semiconductor market. This includes evaluating the performance of individual companies, industries, and trends within the market. |
|  |
| Here are some key areas where I would focus my analysis: |
|  |
| 1. Market size and growth: I would track the size of the global semiconductor market and analyze the growth prospects for each segment, such as memory, logic, and sensors. |
| 2. Competitive landscape: I would assess the competitive landscape, including the top players, their market share, and strategic moves. |
| 3. Technology trends: I would monitor the development of new technologies, such as 5G, AI, and IoT, and their impact on the semiconductor industry. |
| 4. Industry trends: I would track industry trends, such as the shift towards sustainable technologies, and evaluate their impact on the market. |
| 5. Market forecasts: I would provide forecasts for the semiconductor market, including revenue and unit shipment growth, based on a range of scenarios and assumptions. |
|  |
| In addition to these areas, I would also keep track of industry news and developments, and provide insights on their impact on the market. |

Appendix E Additional Results
-----------------------------

### E.1 Data Usage in Continued Pre-training

[Figure 7](#A5.F7 "In E.1 Data Usage in Continued Pre-training ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") shows how the domain weights change throughout the training process and the final cumulative data usage of each domain. The trajectory shows that the domain weights stablize after around $30\%$ training. Unlike other domains, Wikipedia exhibits an anomalous spike in data loading early in training. The remaining domains demonstrate a steady, monotonic change in data loading over time as expected.

<img src='x7.png' alt='Refer to caption' title='' width='415' height='141' />

*Figure 7: Data weight of each batch during the continued pre-training stage.*

### E.2 Comparison to LLM-Pruner

[Table 11](#A5.T11 "In E.2 Comparison to LLM-Pruner ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") displays the model configurations for an LLM-Pruner pruned model*(Ma et al., [2023](#bib.bib45 ""))* versus our pruned model. The model pruned from LLM-Pruner has an unconventional archiecture where the intermediate size is smaller than hidden size, largely due to the fact that the algorithm does not support pruning the hidden dimension. When comparing performance between
LLM-Pruner and ours in continued pre-training,
our model achieves lower perplexity than LLM-Pruner with a similar parameter count and the same amount of continued pre-training,
demonstrating the effectiveness of the targeted structured pruning.

<img src='x8.png' alt='Refer to caption' title='' width='230' height='161' />

*Figure 8: The loss of the 1.3B model with original loading and dynamic loading and the downstream trajectory.*

*Table 11: Model structure of Pythia-1.4B, LLM-pruner (1.3B), and Ours (1.3B). With a similar parameter count, our pruned model structure has a lower perplexity when fine-tuned with the same amount of tokens (around 6B tokens).*

|  | Layers | Heads | Head size | Intermediate size | Hidden size | Params | PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-1.4B | 24 | 16 | 128 | 8192 | 2048 | 1.4B | - |
| LLM-pruner (1.6B) | 32 | 7 | 128 | 2201 | 4096 | 1.6B | 7.09 |
| Ours (1.3B) | 24 | 16 | 128 | 5504 | 2048 | 1.3B | 6.85 |

### E.3 Coding and Math Reasoning

*Table 12: Evaluation results on GSM8K and HumanEval and training percentage and tokens in ArXiv and GitHub.*

|  | GSM8K (8) | HumanEval | | ArXiv | Github | ArXiv | GitHub |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Models | EM | Pass@1 | Pass@5 | Percentage | Percentage | Tokens | Tokens |
| LLaMA2-7B | 13.7 | 12.8 | 23.8 | - | - | - | - |
| OPT-2.7B | 0.1 | 0.0 | 0.0 | - | - | - | - |
| Pythia-2.8B | 1.7 | 5.1 | 14.6 | 9.0% | 7.6% | 26.9 | 22.8 |
| INCITE-Base-3B | 1.8 | 4.3 | 4.9 | 2% | 4.5% | 16.0 | 36.0 |
| Open-LLaMA-3B-v1 | 2.5 | 0.0 | 1.2 | 2% | 4.5% | 20.0 | 45.0 |
| Open-LLaMA-3B-v2 | 2.7 | 10.4 | 20.1 | - | - | - | - |
| Sheared-LLaMA-2.7B (Source) | 2.7 | 3.7 | 5.5 | 0.7% | 0.4% | 0.3 | 0.2 |
| Sheared-LLaMA-2.7B (Scaling) | 2.4 | 4.9 | 9.2 | 1.0% | 0.8% | 0.5 | 0.4 |

We examine the math and coding abilities of our pruned models compared to other language models. We find that the math ability of existing 3B parameter models, including Sheared-LLaMA, is still far below that of larger models. We also find that Sheared-LLaMA’s coding ability lags behind models known to be trained on more code data, like Pythia-1.4B and Open-LLaMA-3B-v2. Sheared-LLaMA’s coding ability likely comes from the original LLaMA2 model, speculated to have used more code data, and the minimal code data used in our pruning experiments.

### E.4 Scaling Reference vs. Source Reference

[Figure 10](#A5.F10 "In E.4 Scaling Reference vs. Source Reference ‣ Appendix E Additional Results ‣ Pre-training via Structured Pruning") compares the performance of Sheared-LLaMA when trained with the scaling reference and the source reference in dynamic batch loaing.
While both methods are effective in efficiently training the model,
the scaling reference performs consistently (slightly) better in terms of downstream performance.

<img src='x9.png' alt='Refer to caption' title='' width='461' height='322' />

*Figure 9: Average downstream peformance of Sheared-LLaMA with the scaling reference and the source reference.*

<img src='x10.png' alt='Refer to caption' title='' width='461' height='322' />

*Figure 10: A comparison between pruning from LLaMA1 and LLaMA2 with dynamic loading to match the source reference loss.*

### E.5 Pruning from LLaMA1 vs LLaMA2

In this section, we compare the performance of pruning from LLaMA1 and LLaMA2.
Both models demonstrate strong downstream task performance, though not surprisingly,
pruning from LLaMA2 yields a consistent advantage.

Appendix F Training details to continual pre-training INCITE-Base-3B
-----------------------------------------------------------------------

Before continuing pre-training the INCITE-Base-3B model, we conduct an initial grid search to evaluate various learning rates, including values of $1\times 10^{-4}$, $5\times 10^{-5}$, and $1\times 10^{-5}$. Our initial results reveal that employing the first two learning rates resulted in a noticeable decline in model performance compared to the original model. Consequently, we opt to continue pre-training with a learning rate of $1\times 10^{-5}$. The remaining hyperparameters remain consistent with those outlined in [Table 7](#A2.T7 "In Appendix B Training Details ‣ Pre-training via Structured Pruning"). It is worth noting that our choice of continued pre-training setup may not be optimal according to recent research *(Gupta et al., [2023](#bib.bib22 ""))*; however, it represents the best approach within our compute constraints.
