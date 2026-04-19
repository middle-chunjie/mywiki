Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon
=======================================================================

Peitian Zhang1,2Zheng Liu1Shitao Xiao1Ninglu Shao1,2Qiwei Ye1Zhicheng Dou2  
1: Beijing Academy of Artificial Intelligence,  
2: Gaoling School of Artificial Intelligence, Renmin University of China  
{namespace.pt, zhengliu1026}@gmail.com  
Peitian Zhang and Zheng Liu are the co-first authorsZheng Liu is the corresponding author

###### Abstract

The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considerable cost at both training and inference time, and exert an unfavorable impact to the LLM’s original capabilities. In this work, we propose a new method called Activation Beacon, which condenses LLM’s raw activations into compact forms such that the LLM can perceive a longer context with a limited context window. Activation Beacon is introduced as a plug-in module, which fully preserves the LLM’s original capability in short contexts. It works with the sliding window to streamingly process the long context, which leads to a competitive memory and time efficiency in both training and inference. Activation Beacon is trained with short-sequence data of diversified condensing ratios. Thanks to such a treatment, it can be effectively learned to support different context lengths with a small training cost. Our experiment verifies Activation Beacon’s effectiveness of context extension: it can remarkably accomplish high-quality extension of Llama-2-7B’s context by $\times 100$ times (from 4K to 400K); meanwhile, it can also achieve superior performances across a variety of long-context language modeling and understanding tasks. The source code and model checkpoint are available at <https://github.com/FlagOpen/FlagEmbedding>.

<img src='x1.png' alt='Refer to caption' title='' width='461' height='137' />

*Figure 1: Comparison of the sliding window perplexity*[[22](#bib.bib22 "")]* between Activation Beacon and other context extension methods, including 1) Position Interpolation*[[5](#bib.bib5 "")]*, 2) NTK-Aware Scaled RoPE*[[1](#bib.bib1 "")]*, 3) LongLlama*[[32](#bib.bib32 "")]*. Activation Beacon leads to better long-context generation quality with higher running efficiency (memory, time).*

1 Introduction
--------------

Large language models (LLMs) need to process long contexts to accomplish many important tasks, such as retrieval-augmented generation and in-context learning.
However, existing LLMs are typically constrained by fixed context windows, e.g., 2K for Llama-1*[[29](#bib.bib29 "")]* and 4K for Llama-2*[[30](#bib.bib30 "")]*, which is not enough to handle some real-world scenarios.
Although LLMs can be fine-tuned or retrained to extend their context windows*[[16](#bib.bib16 ""); [6](#bib.bib6 ""); [5](#bib.bib5 ""); [28](#bib.bib28 ""); [20](#bib.bib20 ""); [32](#bib.bib32 ""); [18](#bib.bib18 "")]*, it will result in considerable costs at both training and inference time due to the quadratic computing complexity of self attention.
Besides, the continued training on long-sequence data may compromise the LLM’s general capability in shorter contexts, which is unfavorable to their practical usage. In light of these challenges, it is desirable to explore new mechanisms, which can not only realize the cost-effective extension of context length, but also be compatible with the LLM’s existing capabilities.

In this work, we propose Activation Beacon (shown as Figure[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")) as a new method for LLM’s context extension.
It condenses the LLM’s raw activations (i.e. keys and values from the self-attention module) into highly compact forms such that the LLM can perceive the information from a vast scope of context even with a limited context window. The above idea shares the common philosophy as sparse attention*[[3](#bib.bib3 ""); [8](#bib.bib8 ""); [38](#bib.bib38 "")]* and context compression*[[4](#bib.bib4 ""); [7](#bib.bib7 ""); [19](#bib.bib19 ""); [22](#bib.bib22 ""); [14](#bib.bib14 "")]*. However, it enjoys substantial advantages over the previous methods in many perspectives, including the effectiveness of context extension (especially the quality of long-context generation and the flexibility of supporting diverse context lengths), inference and training efficiency, and the compatibility with the existing LLMs, thanks to a series of crucial technical designs.

<img src='x2.png' alt='Refer to caption' title='' width='461' height='111' />

*Figure 2: (A) The beacon token ($\langle\text{bcn}\rangle$) is appended to a context, which prompts the LLM to condense the raw activations into more compact forms; (B) The condensed activations are streamingly processed with the sliding window for auto-regression (AR).*

Instead of developing a new model from scratch, we argue that the LLM itself can work as a proficient activation condenser with proper adaptation given its strong and well-established context representation capability.
Based on this argument, we introduce a simple but effective model architecture and running mechanism to facilitate the production and utilization of condensed activations.
Particularly, we introduce special tokens, known as beacon tokens ($\langle\text{bcn}\rangle$), which prompt the LLM to condense the contextual information into beacon tokens’s activations (Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).
For a context of length $l$, a team of $k~{}(k<l)$ beacon tokens are dispatched to the end of it, which leads to a condensing ratio of $\alpha$ ($\alpha\=l/k$).
We maintain another copy of the LLM’s self-attention parameters, including ${{W^{b}_{Q},W^{b}_{K},W^{b}_{V},W^{b}_{O}}}$.
These new parameters are specialized to learn the activation condensing, while the original parameters in the LLM are fixed.
Thus, Activation Beacon serves as a plug-in component for the LLM, introducing extended contextual information to the LLM without adversely affecting its existing capabilities in short contexts.

To efficiently handle long contexts, we propose stream processing with the sliding window.
The long context is partitioned into multiple intervals of length $l$. A sliding window is employed to sequentially process one interval at a time.
When dealing with the next interval, the raw activations of the previous interval are discarded while its condensed activations are accumulated.
Therefore, the sliding window is formulated as $[\langle\text{bcn}\rangle_{1},\dots,\langle\text{bcn}\rangle_{m},x_{m+1},\dots,x_{n}]$ where $\langle\text{bcn}\rangle_{*}$ stands for the beacon tokens from previous intervals and $x_{*}$ is normal tokens in the current interval.
The size of the sliding window is upper-bounded by the maximum window size of the LLM, e.g. 4K for Llama-2, which maintains a low memory consumption and a linear time complexity. Meanwhile, it also accumulatively gathers rich contextual information from the past $(\alpha-1)\times m+n$ tokens.

The condensed activations are expected to fully encode the information within the raw activations, thereby assisting the LLM to accomplish high-quality generation of new content. With this consideration, we propose to learn Activation Beacon through the auto-regression task.
In the sliding window, the generation likelihood of the normal token $x_{i}$ is maximized based on the beacon tokens and its preceding normal tokens, i.e., $\max p(x_{i}\mid\langle\text{bcn}\rangle_{1},\dots,\langle\text{bcn}\rangle_{m},x_{m+1}\tiny{\dots},x_{i-1})$.
Considering that a dramatic extension of context calls for a large condensing ratio, while a moderate extension just needs a small condensing ratio, we perform a random sampling of $\alpha$ during the stream processing. Consequently, the generation can be conditioned on a mixture of condensed activations with diversified condensing ratios, which substantially contributes to the Activation Beacon’s generalization in handling the extension of different context lengths.

Activation Beacon is applied to Llama-2-7B (chat), whose original context length is 4K. The training data is sampled from RedPajama*[[10](#bib.bib10 "")]* and LongAlpaca*[[6](#bib.bib6 "")]*, whose length are all less than 8K. The training process merely takes 10K steps, which can be accomplished within 9 hours on an 8×A800 GPU machine. Notably, it leads to a superior quality of language modeling on the extended context lengths, like 8K, 16K, and 32K, whose result is even better than the fine-tuned full-attention baselines. It is equally competitive on long-context understanding tasks, such as question answering and few-shot learning. Activation Beacon also shows the potential to establish super long contexts: by learning to support the condensing factor of 128, the context length of Llama-2 can be remarkably extended to 400K (Figure [1](#S0.F1 "Figure 1 ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).
As a compatible module, Activation Beacon can also work with other techniques, like position interpolation ($\S$[C](#A3 "Appendix C Combining Activation Beacon with Context Window Extension Techniques. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")) and retrieval ($\S$[D](#A4 "Appendix D Combining Activation Beacon with Retrieval. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")) for even longer and better context extension effect.

To summarize, we propose Activation Beacon, which realizes dramatic extension of LLM’s context based on the high-quality condensing of LLM’s activations. It also enjoys a high running efficiency, a high compatibility with the existing LLM, and a small cost of training thanks to its optimized designs on architecture and running mechanism. In our experiment, the effectiveness of Activation Beacon is verified given its superior performances across a wide range of long-context processing tasks.

2 Activation Beacon
-------------------

### 2.1 Overview

The LLM exploits the contextual information while predicting the new content. The contextual information is represented by the activations, particularly the keys and values in the self-attention module. With a fixed size of context window $L$, a typical LLM can only query the recent $L$ activations for contextual information. However, we argue that the window size should simply be the upper bound of input units rather than context length.
By condensing more information into each activation, i.e. the information from a larger scope rather a single token, the LLM will be able to perceive a longer context with its original context window.

### 2.2 Activation Condensing

<img src='x3.png' alt='Refer to caption' title='' width='415' height='275' />

*Figure 3: (I) The raw activations of ordinal tokens (the blue square) are condensed into the compact activations of beacon tokens (the green squere). Future tokens are auto-regressively generated conditioned on the raw activations in the current interval and the condensed activations accumulated from previous intervals. (II) The attention schemes for activation condensing.*

We aim to adapt the LLM itself for activation condensing given its strong context representation capability.
Particularly, we employ special tokens, called beacon tokens, which prompt the LLM to condense the contextual information into their activations.
We also maintain another copy of the LLM’s MHA (multi-head self-attention) parameters, denoted as MHAb, including the layer-wise projection matrices for queries, keys, values, and outputs ${\boldsymbol{W}^{b}_{Q},\boldsymbol{W}^{b}_{K},\boldsymbol{W}^{b}_{V},\boldsymbol{W}^{b}_{O}}$.
These parameters are specifically learned for condensing the activations. Besides, they are lightweight, merely accounting for 1/3 of the LLM’s original parameters (e.g., 2B with the LLaMA-2 7B model).

The activation condensing is performed with the following operations (Figure [3](#S2.F3 "Figure 3 ‣ 2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon") I).
For the context of length $l$, $k$ beacon tokens are appended to the end of it.
The LLM auto-regressively encodes the context as well as the beacon tokens, as a result, the raw activations of regular tokens are generated and then condensed into the beacon tokens’ activations.
Formally, let the input features of the beacon tokens as $\boldsymbol{H}^{b}\in\mathbb{R}^{k\times D}$, the projections for the beacon tokens’ queries, keys, and values are performed in the first place:

|  | $\boldsymbol{Q}^{b}\leftarrow\boldsymbol{W}^{b}_{Q}\boldsymbol{H}^{b},\quad\boldsymbol{K}^{b}\leftarrow\boldsymbol{W}^{b}_{K}\boldsymbol{H}^{b},\quad\boldsymbol{V}^{b}\leftarrow\boldsymbol{W}^{b}_{V}\boldsymbol{H}^{b}.$ |  |
| --- | --- | --- |

Then, the projection results query the keys ($\boldsymbol{K}^{r}\in\mathbb{R}^{l\times D}$) and values ($\boldsymbol{V}^{r}\in\mathbb{R}^{l\times D}$) of the raw activations from normal tokens to generate the condensed activations, leading to a condensing ratio $\alpha\=l/k$:

|  | $\displaystyle\boldsymbol{A}\leftarrow\mathrm{softmax}\left(\mathrm{mask}\left(\frac{\boldsymbol{Q}^{b}{\boldsymbol{K}^{r}\oplus\boldsymbol{K}^{b}}^{T}}{\sqrt{D}}\right)\right),$ |  |
| --- | --- | --- |
|  | $\displaystyle\boldsymbol{O}^{b}\leftarrow\boldsymbol{W}^{b}_{O}\boldsymbol{A}{\boldsymbol{V}^{r}\oplus\boldsymbol{V}^{b}}.$ |  | (1) |
| --- | --- | --- | --- |

The final output of self-attention is produced by the concatenation of both raw activations from the normal tokens and the condensed activations from the beacon tokens.

To optimize the quality of activation condensing, we explore three attention schemes for the beacon tokens, i.e. the $\mathrm{mask}(\cdot)$ operator, which are as shown in Figure[3](#S2.F3 "Figure 3 ‣ 2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon") II.
1) Segmentation, where each beacon can attend to an equally segmented span of the context.
2) Stepwise expansion, where each beacon can attend to one more span than its predecessor, and the last beacon can attend to the entire context.
3) Full coverage, where the entire context can be attended by all beacons.
For all three options, we restrict the context length $l$ to be evenly divisible by the number of beacon tokens $k$. Besides, the beacon tokens are always positioned next to the last normal token it can attend to.
Although the three options are of the same computation cost, it’s empirically found that the second option, i.e. the stepwise expansion, leads to the optimal performance (§[5](#S3.T5 "Table 5 ‣ 3.4 Ablation Studies ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).

### 2.3 Stream Processing

The long context is partitioned into multiple intervals of length $l$. A sliding window is employed to sequentially process one interval at a time.
When dealing with the next interval, the raw activations of the previous interval are discarded while its condensed activations are accumulated.
Therefore, the sliding window consists of $m$ beacon tokens (i.e. $\langle\text{bcn}\rangle$) from the past intervals, and the normal tokens in the current interval. With the above formulation, the next token is predicted as:

|  | $p(x_{n}\mid\langle\text{bcn}\rangle_{1},\dots,\langle\text{bcn}\rangle_{m},x_{m+1},\dots,x_{n-1};\Theta,\Theta^{b}),$ |  | (2) |
| --- | --- | --- | --- |

where $\Theta$ denotes the parameters of the LLM and $\Theta^{b}$ denotes the introduced parameters for beacons. Crucially, both $\langle\text{bcn}\rangle_{*}$ and $x_{*}$, are encoded by their relative positions within the sliding window, regardless of their absolute positions in the entire context.
The size of the sliding window is up-bounded by the context window size of the LLM, which results in a competitive running efficiency for both training and inference.
Different from the typical stream processing where the context beyond the sliding window is discarded *[[36](#bib.bib36 "")]*, our method can accumulatively cover the information from the past $(\alpha-1)\times m+n$ tokens.
Note that the above working mechanism may also benefit from the increasing of window size, as more beacon tokens can be accumulated in the sliding window to cover an even longer context. Consequently, Activation Beacon can work with strategies like NTK*[[1](#bib.bib1 "")]*, PI*[[5](#bib.bib5 "")]* for further extension of the context. Detailed collaboration effect is explored in Appendix[C](#A3 "Appendix C Combining Activation Beacon with Context Window Extension Techniques. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").

### 2.4 Learning Method

Plug-in to LLM. As introduced, Activation Beacon introduces the following parameters ($\Theta_{b}$): 1) the beacon token’s embedding $\boldsymbol{e}_{\langle\text{bcn}\rangle}$, 2) the linear projection matrices for MHAb: ${\boldsymbol{W}^{b}_{Q},\boldsymbol{W}^{b}_{K},\boldsymbol{W}^{b}_{V},\boldsymbol{W}^{b}_{O}}$ in each transformer layer. Overall, it accounts for less than $1/3$ of the LLM’s original size, e.g., 2B with the Llama-2-7B model.
Activation Beacon reuses other transformer modules from the LLM (i.e., MLP and LayerNorm). This turns out to be the optimal trade-off between effectiveness and training cost.
Activation Beacon is learned while all of the LLM’s original parameters are frozen. Besides, it is only used to generate the condensed activations without interfering the inference process of normal tokens. Therefore, it serves as a plug-in module for the LLM, which introduces the long contextual information without affecting the LLM’s existing capabilities in processing short contexts.

Auto-Regression. We train Activation Beacon by auto-regression, where the next token is predicted based on the condensed activations from the beacon tokens and the raw activations from the ordinary tokens.
As mentioned in $\S$[2.2](#S2.SS2 "2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon"), a training instance is partitioned into equal-sized intervals of length $l$ and streamingly processed.
Afterwards, the following loss is minimized:

|  | $\displaystyle\min\limits_{\Theta_{b}}.\sum_{j\=1}^{\tiny{\lceil|X|//l\rceil}}\sum_{i\=1}^{l}-\log p(x_{i}^{j}|\langle\text{bcn}\rangle_{1},\dots,\langle\text{bcn}\rangle_{m_{j}},x_{1}^{j},\dots,x_{i-1}^{j};\Theta,\Theta^{b}).$ |  | (3) |
| --- | --- | --- | --- |

where $x_{i}^{j}$ is the $i$-th token in the $j$-th interval of $X$, $m_{j}$ stands for the number of beacon tokens accumulated before the $j$-th interval, whose value depends on the condensing ratio of each preceding interval ($m_{j}\=\sum_{z\=1}^{j-1}(l//\alpha_{z})$).

Step-wise randomized condensing ratio. The training is performed purely with short-sequence data, i.e. $1024<|X|<8192$, where the majority of training samples are less than 4K (Table[6](#A2.T6 "Table 6 ‣ Appendix B Length Distribution of Training Data ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).
Therefore, we are able to achieve superior training efficiency. To generalize Activation Beacon to support different context lengths, e.g., 16K, 32K, 100K, and even longer, the auto-regression needs to be conditioned on different amounts of beacon tokens with diversified condensing ratios.
For this purpose, we randomly sample the condensing ratio for each interval within a large candidate scope: $\alpha_{j}\sim$ {2, 4, 8, … 128}, which will introduce dramatic diversity to the condensing ratios and amount of beacon tokens within the auto-regression process.

3 Experiment
------------

Our experiments are performed for the exploration of the following issues. 1) Activation Beacon’s impact on the long-context generation capabilities (measured by Perplexity). 2) Activation Beacon’s impact on the long-context utilization capability (reflected by tasks like long document QA and summarization). 3) Activation Beacon’s impact on efficiency in terms of GPU memory and inference time. 4) The individual contribution of different technical factors.

### 3.1 Settings

Implementation. Our method is applied to Llama-2-7B (chat)*[[30](#bib.bib30 "")]* for empirical studies.
Our training data is a mixture of 80K sampled data from RedPajama*[[10](#bib.bib10 "")]* and LongAlpaca*[[6](#bib.bib6 "")]* (70K from RedPajama and 10K from LongAlpaca, respectively).
The sequence length of each sample is between 1024 and 8192. The statistics of our training data is reported in Table[6](#A2.T6 "Table 6 ‣ Appendix B Length Distribution of Training Data ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").
We use a single 8×A800 GPU machine for training. The training is performed for 10,000 steps (one epoch of the whole training data) with a batch size of 8 and a learning rate of 5e-5 using the linear scheduler. The length of the context interval is set to 1024. The condensing ratio is sampled from {2, 4, 8, 16, 32, 64, 128} during training. As introduced, Llama’s own parameters are freezed throughout the training process.

Baselines. The following types of baselines are chosen for comparison (all based on the LLaMA-2-7B (chat) model unless otherwise specified). 1) The basic method, i.e. LLaMA-2-7B (chat)*[[29](#bib.bib29 "")]* with 4K context length. 2) The fine-tuning free methods, including Positional Interpolation (PI)*[[5](#bib.bib5 "")]*, the NTK-Aware Scale ROPE (NTK)*[[1](#bib.bib1 "")]*, and StreamingLLM*[[36](#bib.bib36 "")]*. 3) The fine-tuned full-attention methods, including LongChat-32K*[[16](#bib.bib16 "")]*, LongAlpaca-16K*[[6](#bib.bib6 "")]*, YaRN-128K*[[20](#bib.bib20 "")]*. 4) The fine-tuned methods with adapted architectures for long contexts, including AutoCompressor-6K*[[7](#bib.bib7 "")]* and LongLlama*[[32](#bib.bib32 "")]* (based on CodeLlama*[[24](#bib.bib24 "")]*).
We enable FlashAttention-2*[[11](#bib.bib11 "")]* to accelerate self-attention computation and save GPU usage for all the baselines. At present, Activation Beacon is incompatible with FlashAttention-2 due to its utilization of the customized attention scheme; thus, we use the scaled dot product attention (sdpa) from PyTorch*[[17](#bib.bib17 "")]* for acceleration.

### 3.2 Main Results

#### 3.2.1 Long-Context Language Modeling

*Table 1: Sliding window perplexity of different context window extension methods on PG19, Proof-Pile, and CodeParrot. Activation Beacon successfully extends the context window of Llama-2-7B model to sequences much longer than the ones seen during training.*

| Method | PG19 | | | | Proof-Pile | | | | CodeParrot | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 4K | 16K | 32K | 100K | 4K | 16K | 32K | 100K | 4K | 16K | 32K | 100K |
| Llama-2-7B | 9.21 | $\tiny{>}10^{3}$ | $\tiny{>}10^{3}$ | OOM | 3.47 | $\tiny{>}10^{3}$ | $\tiny{>}10^{3}$ | OOM | 2.55 | $\tiny{>}10^{3}$ | $\tiny{>}10^{3}$ | OOM |
| PI | 9.21 | 19.5 | $\tiny{>}10^{2}$ | OOM | 3.47 | 5.94 | 33.7 | OOM | 2.55 | 4.57 | 29.33 | OOM |
| NTK | 9.21 | 11.5 | 37.8 | OOM | 3.47 | 3.65 | 7.67 | OOM | 2.55 | 2.86 | 7.68 | OOM |
| StreamingLLM | 9.21 | 9.25 | 9.24 | 9.32 | 3.47 | 3.51 | 3.50 | 3.55 | 2.55 | 2.60 | 2.54 | 2.56 |
| AutoCompre.-6K | 11.8 | $\tiny{>}10^{2}$ | $\tiny{>}10^{3}$ | OOM | 4.55 | $\tiny{>}10^{2}$ | $\tiny{>}10^{3}$ | OOM | 5.43 | $\tiny{>}10^{2}$ | $\tiny{>}10^{3}$ | OOM |
| YaRN-128K | 6.68 | 6.44 | 6.38 | OOM | 2.70 | 2.47 | 2.41 | OOM | 2.17 | 2.04 | 2.00 | OOM |
| LongChat-32K | 9.47 | 8.85 | 8.81 | OOM | 3.07 | 2.70 | 2.65 | OOM | 2.36 | 2.16 | 2.13 | OOM |
| LongAlpaca-16K | 9.96 | 9.83 | $\tiny{>}10^{2}$ | OOM | 3.82 | 3.37 | $\tiny{>}10^{3}$ | OOM | 2.81 | 2.54 | $\tiny{>}10^{3}$ | OOM |
| LongLlama | 9.06 | 8.83 | OOM | OOM | 2.61 | 2.41 | OOM | OOM | 1.95 | 1.90 | OOM | OOM |
| Activation Beacon | 9.21 | 8.34 | 8.27 | 8.50 | 3.47 | 3.34 | 3.32 | 3.31 | 2.55 | 2.43 | 2.41 | 2.62 |

The experiment on long-context language modeling is performed with three datasets: PG19*[[22](#bib.bib22 "")]*, Proof-Pile*[[40](#bib.bib40 "")]*, and CodeParrot*[[31](#bib.bib31 "")]*. Specifically, for PG19, we use its entire test set with 100 books. For Proof-Pile, we extract the arxiv papers from the test set that are longer than 32K, which are 79 papers in total. For CodeParrot, there is no pre-defined test set. Following previous studies*[[25](#bib.bib25 ""); [39](#bib.bib39 "")]*, we first concatenate code from the same repository to form long sequences, then we sample 100 sequences for evaluation.
The perplexity is computed with a sliding window of size 2K*[[21](#bib.bib21 "")]*.

The evaluation results are reported in Table[1](#S3.T1 "Table 1 ‣ 3.2.1 Long-Context Language Modeling ‣ 3.2 Main Results ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon"), where Activation Beacon leads to a superior long-context language modeling performance. First of all, it not only outperforms the Llama-2-7B baseline but also results in a notably improved performance than the fine-tuning free methods. It is worth noting that with the extension of context from 4K to 32K, the language modeling performance can be gradually improved by Activation Beacon, indicating that the expanded information from the longer context can be effectively utilized to facilitate the generation. By comparison, the language modeling performance is decreased with other fine-tuning-free methods. Most of them become ineffective after the context length goes beyond 32K.

Secondly, Activation Beacon’s performance is comparable to or even better than the fine-tuned full-attention methods. This result is remarkable knowing that Activation Beacon runs with a much higher efficiency (to be analyzed in Section [3.3](#S3.SS3 "3.3 Efficiency Analysis ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).
Although there are cases where some of the fine-tuned full-attention baselines achieve better performances, their empirical advantages may not be fully resulted from the introduction of long contextual information.
For example, YaRN-128K’s performance has already been notably higher than Llama-2-7B at the context length of 4K, and so is the case with LongChat-32K on Proof-Pile and CodeParrot.
Note that the update of the LLM’s original parameters is not always favorable because it may not be well generalized to many other scenarios.
By comparison, our method is simply a plug-in module to introduce long contextual information without affecting the LLM’s existing capabilities.

Thirdly, Activation Beacon is able to achieve a much longer extension of the context than the rest of the methods. Particularly, it maintains a quality generation performance after the context length is extended to 100K, where most of the baselines become either ineffective or out-of-memory (OOM). In fact, Activation Beacon is still effective even after the context length is further extended to 400K (see Figure[1](#S0.F1 "Figure 1 ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")), which means a 100$\times$ extension of Llama-2-7B’s maximum context length.
Unlike many other methods like fine-tuning, Activation Beacon does not require any long-sequence training data to acquire such a super long-context capability, which contributes to its high usability in practice.

#### 3.2.2 More Long-Context Tasks

*Table 2: Evaluation of different methods on LongBench. Activation Beacon performs on par with the fine-tuned full-attention baselines.*

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-Shot | Code |
| --- | --- | --- | --- | --- | --- |
| Llama-2-7B | 24.90 | 22.60 | 24.70 | 60.00 | 48.10 |
| PI | 18.98 | 17.16 | 25.03 | 49.43 | 52.73 |
| NTK | 23.21 | 23.34 | 24.40 | 59.29 | 49.28 |
| StreamingLLM | 21.47 | 22.22 | 22.20 | 50.05 | 48.00 |
| AutoCompressor-6K | 13.22 | 10.61 | 14.00 | 15.72 | 23.62 |
| YaRN-128K | 24.03 | 24.11 | 19.82 | 60.00 | 62.73 |
| LongChat-4K | 28.14 | 21.88 | 26.59 | 62.06 | 52.77 |
| LongChat-32K | 31.58 | 23.50 | 26.70 | 64.02 | 54.10 |
| LongAlpaca-4K | 26.81 | 24.44 | 26.93 | 62.92 | 55.15 |
| LongAlpaca-16K | 28.70 | 28.10 | 27.80 | 63.70 | 56.00 |
| LongLlama | 30.12 | 16.37 | 24.19 | 60.31 | 66.05 |
| Activation Beacon | 28.27 | 28.44 | 25.15 | 61.00 | 57.75 |

<img src='x4.png' alt='Refer to caption' title='' width='346' height='225' />

*Figure 4: The evaluation of topic retrieval accuracy at different context lengths. Activation Beacon is competitive against the fine-tuned methods, like LongChat-32K and LongAlpaca-16K.*

We further study the five real-world tasks from LongBench*[[2](#bib.bib2 "")]*, including single-doc QA, multi-doc QA, summarization, few-shot learning, and code completion, where the experiment result on each task is reported in Table[2](#S3.T2 "Table 2 ‣ 3.2.2 More Long-Context Tasks ‣ 3.2 Main Results ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").
We also evaluate the topic retrieval task*[[16](#bib.bib16 "")]*, whose result is shown in Figure[4](#S3.F4 "Figure 4 ‣ 3.2.2 More Long-Context Tasks ‣ 3.2 Main Results ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").
In Appendix[D](#A4 "Appendix D Combining Activation Beacon with Retrieval. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon"), we evaluate the passkey retrieval task*[[35](#bib.bib35 "")]*.
Similar to our previous observation on long-context language modeling, Activation Beacon leads to a notable improvement over Llama-2-7B and the fine-tuning-free baselines. Meanwhile, it reaches a comparable performance with the fine-tuned full-attention methods. Because a large portion of the evaluation samples can be (almost) covered by the 16K or 32K context window, the fine-tuned full-attention methods indeed set a high standard on LongBench. However, knowing that the fine-tuning operation will change the LLM’s original parameters, it is still interesting to investigate where the empirical advantage of the finetuned methods comes from. To figure out this problem, we benchmark the performance of LongChat-32K and LongAlpaca-16K at the context length of 4K, where they use the same information as the Llama-2-7B baseline. Interestingly, both methods result in a substantial improvement over Llama-2-7B on every task. Especially for summarization, where both methods are already sufficiently strong at 4K, yet little extra improvements are made with the further extended context window.
By comparison, Activation Beacon inherits Llama-2-7B’s performance at the context length of 4K, where its performance gain over Llama-2-7B is introduced by the extended context. In this sense, its impact on utilizing the long contextual information can still be no inferior to the ones from the finetuned methods in the corresponding situations.

### 3.3 Efficiency Analysis

*Table 3: Evaluation of inference time and GPU memory usage. Both metrics are measured by the average value of 100 forward passes (FlashAttention-2 is enabled for LongChat).*

| Method | GPU Memory (GB) | | | | | Inference Time (s) | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 4K | 8K | 16K | 32K | 100K | 4K | 8K | 16K | 32K | 100K |
| LongChat-32K | 18.5 | 24.2 | 35.6 | 58.4 | OOM | 0.045 | 0.089 | 0.191 | 0.460 | OOM |
| StreamingLLM | 19.9 | 19.9 | 19.9 | 19.9 | 19.9 | – | – | – | – | – |
| AutoCompressor-6K | 17.7 | 22.6 | 32.3 | 51.7 | OOM | 0.087 | 0.134 | 0.224 | 0.478 | OOM |
| LongLlama | 18.2 | 21.9 | 34.2 | OOM | OOM | 0.079 | 0.190 | 0.436 | OOM | OOM |
| Activation Beacon | 21.7 | 21.3 | 21.4 | 21.6 | 21.6 | 0.071 | 0.121 | 0.237 | 0.473 | 1.494 |

*Table 4: Comparison of training time and GPU memory cost between LongAlpaca-16K (8xA100 GPUs) and Activation Beacon (8xA800 GPUs).*

| Method | Time (Hour) | Memory (GB) |
| --- | --- | --- |
| LongAlpaca-16K | 20.8 | 57.1 |
| Activation Beacon | 9.0 | 55.9 |

We evaluate the running efficiency at the inference time in terms of time cost and GPU memory usage, whose results are reported in Table[3](#S3.T3 "Table 3 ‣ 3.3 Efficiency Analysis ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").
Compared with LongChat (full-attention) and LongLlama, Activation Beacon enjoys a much smaller GPU memory usage at the long context. Activation Beacon and StreamingLLM result in a similar memory cost because both methods are based on sliding windows.
As for the inference time, Activation Beacon is faster than LongLlama, but slower than LongChat when the context is short.
This is because Activation Beacon is streamingly processed while LongChat is fully parallel111StreamingLLM is slow due to its current step-wise realization, yet its theoretical speed should be comparable with our method..
However, Activation Beacon is able to gradually catch up when the context length gets longer, as its time complexity is linear to the context length. It will ultimately become much faster than the full-attention methods if the context length is extended long enough.
Finally, we compare our training cost with LongAlpaca, which is featured for its high training efficiency (shown in Table[4](#S3.T4 "Table 4 ‣ 3.3 Efficiency Analysis ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).
Under a similar hardware condition (8×A800 GPUs vs. 8×A100 GPUs), the training of Activation Beacon can be accomplished in just 9 hours, which is even faster than the reported time cost of LongAlpaca-16K with $S^{2}$-attn222https://openreview.net/forum?id\=6PmJoRfdaK (§[2.4](#S2.SS4 "2.4 Learning Method ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).

### 3.4 Ablation Studies

We perform ablation studies to evaluate the impact from different technical factors, including the attention scheme of beacons ($\S$[2.2](#S2.SS2 "2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")), the sampling strategy of condensing ratio ($\S$[2.4](#S2.SS4 "2.4 Learning Method ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")), the introduced parameters for beacons ($\S$[2.4](#S2.SS4 "2.4 Learning Method ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")), and the composition of training data ($\S$[3.1](#S3.SS1 "3.1 Settings ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")). The experiment results are shown in Table[5](#S3.T5 "Table 5 ‣ 3.4 Ablation Studies ‣ 3 Experiment ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").

First of all, we can find that the attention scheme exerts a substantial impact on Activation Beacon’s performances on both long-context language modeling (PG19) and long-context understanding (QA).
The stepwise expansion works with the gradually expanded attention scope. Therefore, it enables the beacons to acquire different levels of local and global information of each context interval, which notably improves the performance over the other two options.

Secondly, the sampling of the condensing ratio is another influential factor. In this place, we compare two alternative strategies. The instance-wise option samples one condensing ratio for all context intervals of each training instance $X$ (from the same scope as the step-wise method, i.e. ${2,4,8,\dots,128}$). While the monotonous option makes use of one constant condensing ratio of 4 (which can support a up-to 16K context length).
We can observe that the step-wise sampling strategy, which introduces the most diversified condensing ratios when learning, results in competitive performance on perplexity while significantly outperforms the other two options on long-context understanding.

Thirdly, we analyze the impact by introducing different amounts of learnable parameters to the beacon module.
Specifically, when we remove the output projection matrix $\boldsymbol{W}_{O}^{b}$ from the beacon parameters MHAb (§[2.2](#S2.SS2 "2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")), the empirical performances on both tasks degrade.
When we additionally include the MLP parameters of FFN, the model’s performance does not improve.
We conjecture that this is probably because the FFN layer is heavily loaded, which slows down the convergence of the training process. As a result, it suggests that our current formulation of the learnable parameters is a good trade-off between cost and effectiveness.

Lastly, we can also observe that only using RedPajama as the training data already leads to a competitive performance on both evaluation tasks. The introduction of more training data from LongAlpaca contributes little to the language modeling task. However, it brings an additional improvement to the empirical performance on Single-Doc QA.

*Table 5: The impact of different technical factors: attention scheme of beacon token, condensing ratio, composition of training data. Performances are measured by PG19 with 32K context and single-Doc QA on LongBench. Default settings are marked by *.*

| Factor | Setting | PG19 | QA |
| --- | --- | --- | --- |
| Attention Scheme | Segmentation | 8.39 | 26.05 |
| | Full coverage | 8.76 | 23.13 |
| Stepwise expansion* | 8.27 | 28.27 |
| Condensing Ratio | Monotonous ($\alpha\=4$) | $>10^{2}$ | 26.48 |
| | Instance-wise randomized | 8.19 | 26.33 |
| Step-wise randomized* | 8.27 | 28.27 |
| Beacon Parameters | Q, K, V (1.5B) | 8.32 | 27.04 |
| | Q, K, V, O, MLP (5.5B) | 8.81 | 23.46 |
| Q, K, V, O (2.0B)* | 8.27 | 28.27 |
| Data Composition | RedPajama only | 8.24 | 24.98 |
| | RedPajama+LongAlpaca* | 8.27 | 28.27 |

4 Related Works
---------------

We discuss the following works which are devoted to the extension of LLM’s context.
First of all, a large body of methods have been proposed to increase the size of context window. For example, ALiBi*[[21](#bib.bib21 "")]* leverages linear-decaying attention biases to achieve the extrapolation of position encoding. Methods like Position Interpolation*[[5](#bib.bib5 "")]*, NTK-Aware scaling*[[1](#bib.bib1 "")]* and ReRoPE*[[26](#bib.bib26 "")]* make progress on top of RoPE*[[27](#bib.bib27 "")]*, which enable the LLM to handle unseen positions at the inference time. Although such methods can be directly applied to the well-trained LLM, they usually benefit from continual fine-tuning where the extended context can be better utilized*[[20](#bib.bib20 "")]*. The fine-tuning with long-sequence data is expensive. Thus, people investigate how to reduce the training cost. For example, LongLora*[[6](#bib.bib6 "")]* proposes S2-Attn and leverages LoRA for cost-effective training; while PoSE*[[41](#bib.bib41 "")]* uses skip-wise position indices to train LLMs on 2K context length as a simulation of 128K. However, the fine-tuning operations are still prone to big costs if super long-sequence data is presented.
Finally, the fine-tuning operation may impair the LLM’s existing capabilities on short contexts*[[20](#bib.bib20 "")]*. By comparison, our method is trained with a small cost and enjoys a high efficiency in training and inference. Besides, it serves as a plug-in module that is fully compatible with the existing LLM.

The quadratic complexity of transformer is a major bottleneck to achieve long contexts. Thus, many previous works aim to address this problem by using sparse attention*[[8](#bib.bib8 ""); [3](#bib.bib3 ""); [38](#bib.bib38 ""); [12](#bib.bib12 "")]* or approximate attention computation*[[15](#bib.bib15 ""); [33](#bib.bib33 ""); [9](#bib.bib9 ""); [23](#bib.bib23 "")]*. However, there are threefold challenges about these methods as analyzed in*[[36](#bib.bib36 "")]*: the requirement of customized GPU kernels for specific variants of matrix multiplication, the dependency on global attention patterns which are unsuitable for autoregressive language models, the incompatibility with the well-pretrained models. In contrast, our method is free from these constraints and preserves a high compatibility with the existing LLMs.

It is also plausible to find ways to process long contexts with short context windows. One popular strategy is to use sliding windows. For example, StreamingLLM*[[36](#bib.bib36 "")]* and LM-Infinite*[[13](#bib.bib13 "")]* are able to achieve an infinite context by only maintaining the activations for the very first and the latest tokens. However, they are unable to leverage the rich information from the long context because the portion beyond the sliding window will be discarded. Besides, the long contexts can also be summarized and compressed into more compact forms*[[4](#bib.bib4 ""); [7](#bib.bib7 ""); [19](#bib.bib19 ""); [22](#bib.bib22 ""); [14](#bib.bib14 "")]*, which follow the same spirit as our work. However, the previous methods call for major changes to the original model’s architecture and working process, which brings in many problems. Notably, they are prone to substantial compression losses which prevent them from making extensions for long contexts. Besides, they lack the flexibility to support different context lengths, and suffer from the incompatibility with existing LLMs.

Finally, it becomes popular to offload the long context into external memory and retrieve the useful part from it as the working context. The retrieved data can be either the chunked input*[[37](#bib.bib37 ""); [39](#bib.bib39 "")]* or the cached KV activations, e.g., Memorizing Transformers*[[35](#bib.bib35 "")]* and LongMem*[[34](#bib.bib34 "")]*. This idea has been further extended by many recent works. For example, Landmark Attention*[[18](#bib.bib18 "")]* uses a special token to represent a chunk of activations, which enables more efficient computation of retrieval. Focused Transformers*[[32](#bib.bib32 "")]* proposes to use contrastive training which improves the discrimination of relevant keys from the cached data. The retrieval-based methods can be limited due to the utilization of incoherent context. However, it tackles the the problem from a different perspective which can benefit from the collaboration with our method (explored in Appendix [D](#A4 "Appendix D Combining Activation Beacon with Retrieval. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")).

5 Conclusion
------------

We introduce Activation Beacon for the extension of LLM’s context length. Activation Beacon condenses the LLM’s raw activations into highly compact forms, enabling the LLM to perceive a long context with a limited context window. As a plug-in component for the LLM, it brings in long contextual information while fully preserving the LLM’s existing capabilities in short contexts. When dealing with long-sequence data, it resorts to a sliding window for stream processing, which leads to a superior working efficiency for both training and inference.
By using short-sequence data with diversely sampled condensing ratios, it can be effectively learned to support different context lengths with a small training cost. Our experiment verifies Activation Beacon as an effective, efficient, compatible, and low-cost method to extend the context length for LLMs.

Broader Impact
--------------

Activation Beacon establishes long-context capabilities for the large language model without affecting its original capabilities.
This enhancement may benefit many long-context scenarios using LLMs, such as long document understanding/summarization, and lifelong chating with long-term memory.
Therefore, it is particularly useful for AI applications like AI readers and lifelong AI chatbots.
Activation Beacon is able to compress the raw activations of LLM into fewer yet more compact ones with minimal loss.
As a result, it can reduce the Key-Value cache requirements for numerous AI applications, leading to significant resource savings.
Moreover, compared to full attention mechanisms, Activation Beacon requires considerably fewer computational resources with competitive speed. This efficiency also contributes to environmental sustainability.
As a downside, since Activation Beacon is based on the LLM, it inherits the internal biases of the LLM.
Consequently, there is a risk of generating unreliable or harmful content, which underscores the need for careful monitoring the ethical usage of these AI systems.

References
----------

* ntk [2023]Ntk-aware scaled rope, 2023.URL <https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/>.
* Bai et al. [2023]Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J.Longbench: A bilingual, multitask benchmark for long context understanding.*arXiv preprint arXiv:2308.14508*, 2023.
* Beltagy et al. [2020]Beltagy, I., Peters, M. E., and Cohan, A.Longformer: The long-document transformer.*CoRR*, abs/2004.05150, 2020.URL [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150 "").
* Bulatov et al. [2023]Bulatov, A., Kuratov, Y., and Burtsev, M. S.Scaling transformer to 1m tokens and beyond with RMT.*CoRR*, abs/2304.11062, 2023.doi: 10.48550/ARXIV.2304.11062.URL [https://doi.org/10.48550/arXiv.2304.11062](https://doi.org/10.48550/arXiv.2304.11062 "").
* Chen et al. [2023a]Chen, S., Wong, S., Chen, L., and Tian, Y.Extending context window of large language models via positional interpolation.*arXiv preprint arXiv:2306.15595*, 2023a.
* Chen et al. [2023b]Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., and Jia, J.Longlora: Efficient fine-tuning of long-context large language models.*arXiv preprint arXiv:2309.12307*, 2023b.
* Chevalier et al. [2023]Chevalier, A., Wettig, A., Ajith, A., and Chen, D.Adapting language models to compress contexts.In Bouamor, H., Pino, J., and Bali, K. (eds.), *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 3829–3846. Association for Computational Linguistics, 2023.URL [https://aclanthology.org/2023.emnlp-main.232](https://aclanthology.org/2023.emnlp-main.232 "").
* Child et al. [2019]Child, R., Gray, S., Radford, A., and Sutskever, I.Generating long sequences with sparse transformers.*arXiv preprint arXiv:1904.10509*, 2019.
* Choromanski et al. [2021]Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A.Rethinking attention with performers.In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021.URL [https://openreview.net/forum?id\=Ua6zuk0WRH](https://openreview.net/forum?id=Ua6zuk0WRH "").
* Computer [2023]Computer, T.Redpajama: An open source recipe to reproduce llama training dataset, 2023.URL [https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data "").
* Dao [2023]Dao, T.Flashattention-2: Faster attention with better parallelism and work partitioning.*CoRR*, abs/2307.08691, 2023.doi: 10.48550/ARXIV.2307.08691.URL [https://doi.org/10.48550/arXiv.2307.08691](https://doi.org/10.48550/arXiv.2307.08691 "").
* Ding et al. [2023]Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., and Wei, F.Longnet: Scaling transformers to 1, 000, 000, 000 tokens.*CoRR*, abs/2307.02486, 2023.doi: 10.48550/ARXIV.2307.02486.URL [https://doi.org/10.48550/arXiv.2307.02486](https://doi.org/10.48550/arXiv.2307.02486 "").
* Han et al. [2023]Han, C., Wang, Q., Xiong, W., Chen, Y., Ji, H., and Wang, S.Lm-infinite: Simple on-the-fly length generalization for large language models.*CoRR*, abs/2308.16137, 2023.doi: 10.48550/ARXIV.2308.16137.URL [https://doi.org/10.48550/arXiv.2308.16137](https://doi.org/10.48550/arXiv.2308.16137 "").
* Huang \& Hollenstein [2023]Huang, X. and Hollenstein, N.Long-range language modeling with selective cache.In Bouamor, H., Pino, J., and Bali, K. (eds.), *Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 4838–4858. Association for Computational Linguistics, 2023.URL [https://aclanthology.org/2023.findings-emnlp.321](https://aclanthology.org/2023.findings-emnlp.321 "").
* Kitaev et al. [2020]Kitaev, N., Kaiser, L., and Levskaya, A.Reformer: The efficient transformer.In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.URL [https://openreview.net/forum?id\=rkgNKkHtvB](https://openreview.net/forum?id=rkgNKkHtvB "").
* Li et al. [2023]Li, D., Shao, R., Xie, A., Sheng, Y., Zheng, L., Gonzalez, J. E., Stoica, I., Ma, X., and Zhang, H.How long can open-source llms truly promise on context length?, June 2023.URL [https://lmsys.org/blog/2023-06-29-longchat](https://lmsys.org/blog/2023-06-29-longchat "").
* Michael Gschwind [2023]Michael Gschwind, Driss Guessous, C. P.Accelerated pytorch 2 transformers.[https://pytorch.org/blog/accelerated-pytorch-2/](https://pytorch.org/blog/accelerated-pytorch-2/ ""), 2023.
* Mohtashami \& Jaggi [2023]Mohtashami, A. and Jaggi, M.Landmark attention: Random-access infinite context length for transformers.*arXiv preprint arXiv:2305.16300*, 2023.
* Mu et al. [2023]Mu, J., Li, X. L., and Goodman, N. D.Learning to compress prompts with gist tokens.*CoRR*, abs/2304.08467, 2023.doi: 10.48550/ARXIV.2304.08467.URL [https://doi.org/10.48550/arXiv.2304.08467](https://doi.org/10.48550/arXiv.2304.08467 "").
* Peng et al. [2023]Peng, B., Quesnelle, J., Fan, H., and Shippole, E.Yarn: Efficient context window extension of large language models.*arXiv preprint arXiv:2309.00071*, 2023.
* Press et al. [2022]Press, O., Smith, N. A., and Lewis, M.Train short, test long: Attention with linear biases enables input length extrapolation.In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022.URL [https://openreview.net/forum?id\=R8sQPpGCv0](https://openreview.net/forum?id=R8sQPpGCv0 "").
* Rae et al. [2020]Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P.Compressive transformers for long-range sequence modelling.In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.URL [https://openreview.net/forum?id\=SylKikSYDH](https://openreview.net/forum?id=SylKikSYDH "").
* Ren et al. [2021]Ren, H., Dai, H., Dai, Z., Yang, M., Leskovec, J., Schuurmans, D., and Dai, B.Combiner: Full attention transformer with sparse computation cost.In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pp. 22470–22482, 2021.URL [https://proceedings.neurips.cc/paper/2021/hash/bd4a6d0563e0604510989eb8f9ff71f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/bd4a6d0563e0604510989eb8f9ff71f5-Abstract.html "").
* Roziere et al. [2023]Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J., et al.Code llama: Open foundation models for code.*arXiv preprint arXiv:2308.12950*, 2023.
* Rubin \& Berant [2023]Rubin, O. and Berant, J.Long-range language modeling with self-retrieval.*CoRR*, abs/2306.13421, 2023.doi: 10.48550/ARXIV.2306.13421.URL [https://doi.org/10.48550/arXiv.2306.13421](https://doi.org/10.48550/arXiv.2306.13421 "").
* Su [2023]Su, J.Rectified rotary position embeddings.<https://github.com/bojone/rerope>, 2023.
* Su et al. [2021]Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y.Roformer: Enhanced transformer with rotary position embedding.*CoRR*, abs/2104.09864, 2021.URL [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864 "").
* Sun et al. [2022]Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A., Chaudhary, V., Song, X., and Wei, F.A length-extrapolatable transformer.*arXiv preprint arXiv:2212.10554*, 2022.
* Touvron et al. [2023a]Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.Llama: Open and efficient foundation language models.*arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. [2023b]Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.Llama 2: Open foundation and fine-tuned chat models.*arXiv preprint arXiv:2307.09288*, 2023b.
* Tunstall et al. [2022]Tunstall, L., Von Werra, L., and Wolf, T.Natural language processing with transformers, 2022.
* Tworkowski et al. [2023]Tworkowski, S., Staniszewski, K., Pacek, M., Wu, Y., Michalewski, H., and Miłoś, P.Focused transformer: Contrastive training for context scaling.*arXiv preprint arXiv:2307.03170*, 2023.
* Wang et al. [2020]Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H.Linformer: Self-attention with linear complexity.*CoRR*, abs/2006.04768, 2020.URL [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768 "").
* Wang et al. [2023]Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F.Augmenting language models with long-term memory.*CoRR*, abs/2306.07174, 2023.doi: 10.48550/ARXIV.2306.07174.URL [https://doi.org/10.48550/arXiv.2306.07174](https://doi.org/10.48550/arXiv.2306.07174 "").
* Wu et al. [2022]Wu, Y., Rabe, M. N., Hutchins, D., and Szegedy, C.Memorizing transformers.In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022.URL [https://openreview.net/forum?id\=TrjbxzRcnf-](https://openreview.net/forum?id=TrjbxzRcnf- "").
* Xiao et al. [2023]Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M.Efficient streaming language models with attention sinks.*arXiv preprint arXiv:2309.17453*, 2023.
* Xu et al. [2023]Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B.Retrieval meets long context large language models.*CoRR*, abs/2310.03025, 2023.doi: 10.48550/ARXIV.2310.03025.URL [https://doi.org/10.48550/arXiv.2310.03025](https://doi.org/10.48550/arXiv.2310.03025 "").
* Zaheer et al. [2020]Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al.Big bird: Transformers for longer sequences.*Advances in neural information processing systems*, 33:17283–17297, 2020.
* Zhang et al. [2023]Zhang, P., Xiao, S., Liu, Z., Dou, Z., and Nie, J.Retrieve anything to augment large language models.*CoRR*, abs/2310.07554, 2023.doi: 10.48550/ARXIV.2310.07554.URL [https://doi.org/10.48550/arXiv.2310.07554](https://doi.org/10.48550/arXiv.2310.07554 "").
* Zhangir Azerbayev [2022]Zhangir Azerbayev, Edward Ayers, B. P.Proof-pile.[https://huggingface.co/datasets/hoskinson-center/proof-pile](https://huggingface.co/datasets/hoskinson-center/proof-pile ""), 2022.
* Zhu et al. [2023]Zhu, D., Yang, N., Wang, L., Song, Y., Wu, W., Wei, F., and Li, S.Pose: Efficient context window extension of llms via positional skip-wise training.*CoRR*, abs/2309.10400, 2023.doi: 10.48550/ARXIV.2309.10400.URL [https://doi.org/10.48550/arXiv.2309.10400](https://doi.org/10.48550/arXiv.2309.10400 "").

Appendix A Overall Algorithm of Activation Beacon
-------------------------------------------------

*Algorithm 1  Activation Beacon*

0:The LLM $\Theta$; the input context $X$; the memory of condensed activations $\boldsymbol{K}^{b}\in\mathbb{R}^{*\times L\times D},\boldsymbol{V}^{b}\in\mathbb{R}^{*\times L\times D}$; the memory of raw activations $\boldsymbol{K}^{r}\in\mathbb{R}^{*\times L\times D},\boldsymbol{V}^{r}\in\mathbb{R}^{*\times L\times D}$; the interval length $l$; and the starting index of the sliding window $i_{s}$.

1: repeat

2:Get the ending index of the sliding window: $i_{e}\leftarrow i_{s}+l$

3: if$i_{e}\leq|X|$then

4:$is\_full\_window\leftarrow True$

5:Set condensing ratio for this interval: $\alpha\leftarrow\mathrm{set\_ratio}()$

6:Set number of special tokens for this interval: $\beta\leftarrow l//\alpha$

7:The sliding window contains regular tokens in the context appended with beacon tokens: $w\leftarrow X_{i_{s}:i_{e}}+[\langle\text{bcn}\rangle]\times\beta$

8: else

9:$is\_full\_window\leftarrow False$

10:The sliding window contains only the regular tokens in the context: $w\leftarrow X_{i_{s}:i_{e}}$

11: end if

12:The memory of the LLM is the concatenation of the condensed activations and the raw activations:

|  | $\boldsymbol{K}\leftarrow\boldsymbol{K}^{b}\oplus\boldsymbol{K}^{r},\quad\boldsymbol{V}\leftarrow\boldsymbol{V}^{b}\oplus\boldsymbol{V}^{r}$ |  |
| --- | --- | --- |

13:The LLM auto-regressively encodes the regular tokens as well as the beacon tokens in the sliding window (note that activation condensing happens according to Equation[1](#S2.E1 "In 2.2 Activation Condensing ‣ 2 Activation Beacon ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon")). The logits of the next token and the newly generated activations are returned:

|  | $\boldsymbol{\psi},\boldsymbol{K}^{\prime},\boldsymbol{V}^{\prime}\leftarrow\Theta(w,\boldsymbol{K},\boldsymbol{V})$ |  |
| --- | --- | --- |

14: if$is\_full\_window$then

15:The last $\beta$ activations are condensed activations, which are accumulated:

|  | $\boldsymbol{K}^{b}\leftarrow\boldsymbol{K}^{b}\oplus\boldsymbol{K}^{\prime}_{-\beta:},\quad\boldsymbol{V}^{b}\leftarrow\boldsymbol{V}^{b}\oplus\boldsymbol{V}^{\prime}_{-\beta:}$ |  |
| --- | --- | --- |

16:The raw activations of previous intervals are emptied:

|  | $\boldsymbol{K}^{r}.\mathrm{empty}(),\quad\boldsymbol{V}^{r}.\mathrm{empty}()$ |  |
| --- | --- | --- |

17:Update the starting index $i_{s}\leftarrow i_{e}$

18: else

19:The raw activations from regular tokens are cached:

|  | $\boldsymbol{K}^{r}\leftarrow\boldsymbol{K}^{r}\oplus\boldsymbol{K}^{\prime},\quad\boldsymbol{V}^{r}\leftarrow\boldsymbol{V}^{r}\oplus\boldsymbol{V}^{\prime}$ |  |
| --- | --- | --- |

20: end if

21: until$i_{e}\geq|X|$

22:Offset the starting index for future generation: $i_{s}\leftarrow i_{s}-|X|$

23: Return $\Theta,\boldsymbol{\psi},\boldsymbol{K}^{b},\boldsymbol{V}^{b},\boldsymbol{K}^{r},\boldsymbol{V}^{r},i_{s}$

Appendix B Length Distribution of Training Data
-----------------------------------------------

| Length | 1K$\sim$2K | 2K$\sim$4K | 4K$\sim$6K | 6K$\sim$8K | Total |
| --- | --- | --- | --- | --- | --- |
| Count | 38K | 23K | 6K | 13K | 80K |
| Portion | 47% | 29% | 8% | 16% | 100% |

*Table 6: The length distribution of training data. The average length of all training data is 3180.*

Appendix C Combining Activation Beacon with Context Window Extension Techniques.
--------------------------------------------------------------------------------

<img src='x5.png' alt='Refer to caption' title='' width='346' height='230' />

*Figure 5: The perplexity evaluated on books longer than 400K tokens in the PG19 test set. Activation beacon can work together with context window extension methods, further extending the context length to even 1M tokens.*

Activation Beacon can be combined with context window extension techniques to further extend the context length.
Specifically, Activation Beacon condenses the raw activations of LLM into more compact forms so that the LLM can perceive more information given its original context window size.
It does not modify the position encoding scheme of the LLM within the context window.
Therefore, we can directly employ the modern context window extension techniques, such as PI*[[5](#bib.bib5 "")]* and NTK*[[1](#bib.bib1 "")]*, to expand the window size so that more condensed activations can be accomodated and hence further context extension effect.

In Figure[5](#A3.F5 "Figure 5 ‣ Appendix C Combining Activation Beacon with Context Window Extension Techniques. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon"), we show Activation Beacon can extend the context length of Llama-2 to 1M when combined with PI and NTK, without any further fine-tuning.
This result unveils the superior compatibility of Activation Beacon: it is not only compatible with existing capabilities of the LLM, but also compatible with any future advances of context window extension techniques.

Appendix D Combining Activation Beacon with Retrieval.
------------------------------------------------------

<img src='x6.png' alt='Refer to caption' title='' width='461' height='168' />

*Figure 6: The accuracy and the fuzzy score on tha Passkey Retrieval task. Activation beacon can work together with retrieval techniques to accurately remember the passkey.*

Activation Beacon condenses the raw activations and accumulates them in the sliding window for later auto-regression.
However, the size of the sliding window is up-bounded by the LLM’s context window size, which limits the number of condensed activations that could be maintained.
Consequently, Activation Beacon needs to trade-off between the memory accuracy and the context length:
With a low condensing ratio, Activation Beacon can remember nearly all details about the context, while simultaneously producing more condensed activations for the context.
Under this setting, Activation Beacon can only deal with a relatively short context (e.g. 8K).
In contrast, with a high condensing ratio (e.g. 128), its memory is vague since 128 raw activations are compressed into one, yet it can easily process super long context (e.g. 100K).
Therefore, its performance may degrade on tasks that require accurate memorization of super long context.
However, we can largely mitigate this problem with retrieval.

Specifically, we perform two types of condensation for each interval, one with a high condensing ratio and one with a low condensing ratio, and save the condensed activations from both of them.
The model always conditions on the aggressively condensed activations, thus it can process super long context.
Besides, we can obtain both types of condensed activations in one forward pass by adjusting the attention mask, which is efficient.
When a query about the fine-grained memory is issued, we can leverage retrieval to locate $K$ relevant intervals.
The information within these relevant intervals should be accurate to successfully answer the query, while other information can be vague.
Therefore, for these intervals, we replace their aggressively condensed activations with those accurate ones resulted from the low condensing ratio, then proceed to generate the answer.
Thanks to the step-wise randomized condensing ratio in training, Activation Beacon can robustly utilize the activations with different condensing ratios, thereby effectively utilizing the retrieved information.
The retrieval can be implemented in different ways. For example, the BM25 based on lexical matching, the dense retrieval based on semantic matching, and the hidden state retrieval.
We adopt the simple BM25 retrieval and set $K\=2$, which already yields satisfactory performance.

We use the Passkey Retrieval task*[[35](#bib.bib35 ""); [6](#bib.bib6 ""); [32](#bib.bib32 "")]* to evaluate our approach. It constructs a 5-digit integer, namely passkey, and buries it at a random position in a synthetic document. The models are asked to exactly reproduce the passkey after reading the document.
We repeat the experiment 5 times at each context length and report the average performance in Figure[6](#A4.F6 "Figure 6 ‣ Appendix D Combining Activation Beacon with Retrieval. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").
The following observations are derived.

Firstly, Activation Beacon cannot accurately remember the passkey hidden in the long context.
Though Activation Beacon is accurate on the relatively short context (8K), it becomes incapable given 16K context and beyond. This is as expected since longer context requires larger condensing ratio, which impairs the memory accuracy.
However, this doen’t mean the aggressively condensed activations are meaningless.
In Figure[6](#A4.F6 "Figure 6 ‣ Appendix D Combining Activation Beacon with Retrieval. ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon") (B), we evaluate the fuzzy score between the model’s prediction and the ground-truth passkey, which measures how many digits are overlapped between the prediction and the groud-truth. It can be noticed that Activation Beacon always yields positive fuzzy score. This indicates that Activation Beacon can restore several digits in the passkey, while fail to precisely remember it due to the vague memory.

Secondly, Activation Beacon combined with BM25 significantly improves the memory accuracy, achieving 100% accuracy at all context lengths.
The reasons behind such a superior performance is two fold.
On the one hand, the BM25 retriever successfully locates the interval where passkey resides.
On the other hand, the model can effectively utilize the accurate memory of the retrieved intervals.

In conclusion, we demonstrate that Activation Beacon’s memory accuracy can significantly benefit from even the simplest form of retrieval.
This motivates us to investigate more retrieval implementations and explore the potential of Activation Beacon on more real-world long-context tasks.

Appendix E Impact of Different Condensing Ratios
------------------------------------------------

<img src='x7.png' alt='Refer to caption' title='' width='346' height='225' />

*Figure 7: The perplexity of Activation Beacon given different condensing ratios. The theoretical maximum context length is denoted in the brackets ($\alpha\times 3072+1024$ where $\alpha$ is the condensing ratio).*

The condensing ratio of Activation Beacon can be flexibly configured at inference time.
One interesting question is how different condensing ratios affect the performance of Activation Beacon.
We investigate this question on long-context generation and report the results in Figure[7](#A5.F7 "Figure 7 ‣ Appendix E Impact of Different Condensing Ratios ‣ Soaring from 4K to 400K: Extending LLM’s Context with Activation Beacon").

We can observe that lower condensing ratio leads to lower perplexity, which translates to higher generation long-context generation quality.
However, since the window size of Activation Beacon is limited by that of the LLM, lower condensing ratio results in more activations to save, and hence shorter context it can process.

Appendix F Limitations
----------------------

Currently, the sliding window of Activation Beacon strides one interval at a time and only preserves the condensed activations in the previous interval.
This means the tokens in the beginning of the next interval do not have any raw context: it can only attend to the condensed activations in previous intervals.
This lack of raw context may cause the degradation of the generation performance especially when answering user instructions.
In the future, we may adjust the stride of the sliding window to make it shorter than the interval length. Therefore, tokens in the beginning of any interval always attend to some raw activations as local context.
