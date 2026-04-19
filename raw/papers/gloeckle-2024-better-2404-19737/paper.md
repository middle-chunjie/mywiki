# Better & Faster Large Language Models via Multi-token Prediction

Fabian Gloeckle  $^{*12}$  Badr Youbi Idrissi  $^{*13}$  Baptiste Rozière  $^{1}$  David Lopez-Paz  $^{+1}$  Gabriel Synnaeve  $^{+1}$

# Abstract

Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following  $n$  tokens using  $n$  independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves  $12\%$  more problems on HumanEval and  $17\%$  more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to  $3\times$  faster at inference, even with large batch sizes.

# 1. Introduction

Humanity has condensed its most ingenious undertakings, surprising findings and beautiful productions into text. Large Language Models (LLMs) trained on all of these corpora are able to extract impressive amounts of world knowledge, as well as basic reasoning capabilities by implementing a simple—yet powerful—unsupervised learning task: next-token prediction. Despite the recent wave of impressive achievements (OpenAI, 2023), next-token pre

*Equal contribution +Last authors 1FAIR at Meta 2CERMICS Ecole des Ponts ParisTech 3LISN Université Paris-Saclay. Correspondence to: Fabian Gloeckle <fgloeckle@meta.com>, Badr Youbi Idrissi <byoubi@meta.com>.

diction remains an inefficient way of acquiring language, world knowledge and reasoning capabilities. More precisely, teacher forcing with next-token prediction latches on local patterns and overlooks "hard" decisions. Consequently, it remains a fact that state-of-the-art next-token predictors call for orders of magnitude more data than human children to arrive at the same level of fluency (Frank, 2023).


Figure 1: Overview of multi-token prediction. (Top) During training, the model predicts 4 future tokens at once, by means of a shared trunk and 4 dedicated output heads. During inference, we employ only the next-token output head. Optionally, the other three heads may be used to speed-up inference time. (Bottom) Multi-token prediction improves pass@1 on the MBPP code task, significantly so as model size increases. Error bars are confidence intervals of  $90\%$  computed with bootstrapping over dataset samples.

In this study, we argue that training LLMs to predict multiple tokens at once will drive these models toward better sample efficiency. As anticipated in Figure 1, multi-token prediction instructs the LLM to predict the  $n$  future tokens from each position in the training corpora, all at once and in parallel (Qi et al., 2020).

Contributions While multi-token prediction has been studied in previous literature (Qi et al., 2020), the present work offers the following contributions:

1. We propose a simple multi-token prediction architecture with no train time or memory overhead (Section 2).  
2. We provide experimental evidence that this training paradigm is beneficial at scale, with models up to 13B parameters solving around  $15\%$  more code problems on average (Section 3).  
3. Multi-token prediction enables self-speculative decoding, making models up to 3 times faster at inference time across a wide range of batch-sizes (Section 3.2).

While cost-free and simple, multi-token prediction is an effective modification to train stronger and faster transformer models. We hope that our work spurs interest in novel auxiliary losses for LLMs well beyond next-token prediction, as to improve the performance, coherence, and reasoning abilities of these fascinating models.

# 2. Method

Standard language modeling learns about a large text corpus  $x_{1}, \ldots, x_{T}$  by implementing a next-token prediction task. Formally, the learning objective is to minimize the cross-entropy loss

$$
L _ {1} = - \sum_ {t} \log P _ {\theta} \left(x _ {t + 1} \mid x _ {t: 1}\right), \tag {1}
$$

where  $P_{\theta}$  is our large language model under training, as to maximize the probability of  $x_{t + 1}$  as the next future token, given the history of past tokens  $x_{t:1} = x_t,\ldots ,x_1$

In this work, we generalize the above by implementing a multi-token prediction task, where at each position of the training corpus, the model is instructed to predict  $n$  future tokens at once. This translates into the cross-entropy loss

$$
L _ {n} = - \sum_ {t} \log P _ {\theta} \left(x _ {t + n: t + 1} \mid x _ {t: 1}\right). \tag {2}
$$

To make matters tractable, we assume that our large language model  $P_{\theta}$  employs a shared trunk to produce a latent representation  $z_{t:1}$  of the observed context  $x_{t:1}$ , then fed into  $n$  independent heads to predict in parallel each of the

$n$  future tokens (see Figure 1). This leads to the following factorization of the multi-token prediction cross-entropy loss:

$$
\begin{array}{l} L _ {n} = - \sum_ {t} \log P _ {\theta} \left(x _ {t + n: t + 1} \mid z _ {t: 1}\right) \cdot P _ {\theta} \left(z _ {t: 1} \mid x _ {t: 1}\right) \\ = - \sum_ {t} \sum_ {i = 1} ^ {n} \log P _ {\theta} \left(x _ {t + i} \mid z _ {t: 1}\right) \cdot P _ {\theta} \left(z _ {t: 1} \mid x _ {t: 1}\right). \\ \end{array}
$$

In practice, our architecture consists of a shared transformer trunk  $f_{s}$  producing the hidden representation  $z_{t:1}$  from the observed context  $x_{t:1}$ ,  $n$  independent output heads implemented in terms of transformer layers  $f_{h_i}$ , and a shared unembedding matrix  $f_{u}$ . Therefore, to predict  $n$  future tokens, we compute:

$$
P _ {\theta} \left(x _ {t + i} \mid x _ {t: 1}\right) = \operatorname {s o f t m a x} \left(f _ {u} \left(f _ {h _ {i}} \left(f _ {s} \left(x _ {t: 1}\right)\right)\right)\right),
$$

for  $i = 1, \dots, n$ , where, in particular,  $P_{\theta}(x_{t+1} \mid x_{t:1})$  is our next-token prediction head. See Appendix B for other variations of multi-token prediction architectures.

Memory-efficient implementation One big challenge in training multi-token predictors is reducing their GPU memory utilization. To see why this is the case, recall that in current LLMs the vocabulary size  $V$  is much larger than the dimension  $d$  of the latent representation—therefore, logit vectors become the GPU memory usage bottleneck. Naive implementations of multi-token predictors that materialize all logits and their gradients, both of shape  $(n, V)$ , severely limit the allowable batch-size and average GPU memory utilization. Because of these reasons, in our architecture we propose to carefully adapt the sequence of forward and backward operations, as illustrated in Figure 2. In particular, after the forward pass through the shared trunk  $f_{s}$ , we sequentially compute the forward and backward pass of each independent output head  $f_{i}$ , accumulating gradients at the trunk. While this creates logits (and their gradients) for the output head  $f_{i}$ , these are freed before continuing to the next output head  $f_{i+1}$ , requiring the long-term storage only of the  $d$ -dimensional trunk gradient  $\partial L_{n} / \partial f_{s}$ . In sum, we have reduced the peak GPU memory utilization from  $O(nV + d)$  to  $O(V + d)$ , at no expense in runtime (Table S5).

Inference During inference time, the most basic use of the proposed architecture is vanilla next-token autoregressive prediction using the next-token prediction head  $P_{\theta}(x_{t + 1} \mid x_{t:1})$ , while discarding all others. However, the additional output heads can be leveraged to speed up decoding from the next-token prediction head with self-speculative decoding methods such as blockwise parallel decoding (Stern et al., 2018)—a variant of speculative decoding (Leviathan et al., 2023) without the need for an additional draft model—and speculative decoding with Medusa-like tree attention (Cai et al., 2024).

Figure 2: Order of the forward/backward in an  $n$ -token prediction model with  $n = 2$  heads. By performing the forward/backward on the heads in sequential order, we avoid materializing all unembedding layer gradients in memory simultaneously and reduce peak GPU memory usage.


# 3. Experiments on real data

We demonstrate the efficacy of multi-token prediction losses by seven large-scale experiments. Section 3.1 shows how multi-token prediction is increasingly useful when growing the model size. Section 3.2 shows how the additional prediction heads can speed up inference by a factor of  $3 \times$  using speculative decoding. Section 3.3 demonstrates how multi-token prediction promotes learning longer-term patterns, a fact most apparent in the extreme case of byte-level tokenization. Section 3.4 shows that 4-token predictor leads to strong gains with a tokenizer of size  $32k$ . Section 3.5 illustrates that the benefits of multi-token prediction remain for training runs with multiple epochs. Section 3.6 showcases the rich representations promoted by pretraining with multi-token prediction losses by finetuning on the Code-Contests dataset (Li et al., 2022). Section 3.7 shows that the benefits of multi-token prediction carry to natural language models, improving generative evaluations such as summarization, while not regressing significantly on standard benchmarks based on multiple choice questions and negative log-likelihoods.

To allow fair comparisons between next-token predictors and  $n$ -token predictors, the experiments that follow always compare models with an equal amount of parameters. That is, when we add  $n - 1$  layers in future prediction heads, we remove  $n - 1$  layers from the shared model trunk. Please refer to Table S14 for the model architectures and to Table S13 for an overview of the hyperparameters we use in our experiments.

# 3.1. Benefits scale with model size

To study this phenomenon, we train models of six sizes in the range 300M to 13B parameters from scratch on at least 91B tokens of code. The evaluation results in Fig-

Figure 3: Results of  $n$ -token prediction models on MBPP by model size. We train models of six sizes in the range or 300M to 13B total parameters on code, and evaluate pass@1,10,100 on the MBPP (Austin et al., 2021) and HumanEval (Chen et al., 2021) benchmark with 1000 samples. Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale. Error bars are confidence intervals of  $90\%$  computed with bootstrapping over dataset samples.

ure 3 for MBPP (Austin et al., 2021) and HumanEval (Chen et al., 2021) show that it is possible, with the exact same computational budget, to squeeze much more performance out of large language models given a fixed dataset using multi-token prediction.

We believe this usefulness only at scale to be a likely reason why multi-token prediction has so far been largely overlooked as a promising training loss for large language model training.

# 3.2. Faster inference

We implement greedy self-speculative decoding (Stern et al., 2018) with heterogeneous batch sizes using xFormers (Lefaudeauux et al., 2022) and measure decoding speeds of our best 4-token prediction model with 7B parameters on completing prompts taken from a test dataset of code and natural language (Table S2) not seen during training. We observe a speedup of  $3.0 \times$  on code with an average of 2.5 accepted tokens out of 3 suggestions on code, and of

Table 1: Multi-token prediction improves performance and unlocks efficient byte level training. We compare models with 7B parameters trained from scratch on 200B and on 314B bytes of code on the MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021) and APPS (Hendrycks et al., 2021) benchmarks. Multi-token prediction largely outperforms next token prediction on these settings. All numbers were calculated using the estimator from Chen et al. (2021) based on 200 samples per problem. The temperatures were chosen optimally (based on test scores; i.e. these are oracle temperatures) for each model, dataset and pass@k and are reported in Table S12.  

<table><tr><td rowspan="2">Training data</td><td rowspan="2">Vocabulary</td><td rowspan="2">n</td><td colspan="3">MBPP</td><td colspan="3">HumanEval</td><td colspan="3">APPS/Intro</td></tr><tr><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td></tr><tr><td rowspan="4">313B bytes (0.5 epochs)</td><td rowspan="4">bytes</td><td>1</td><td>19.3</td><td>42.4</td><td>64.7</td><td>18.1</td><td>28.2</td><td>47.8</td><td>0.1</td><td>0.5</td><td>2.4</td></tr><tr><td>8</td><td>32.3</td><td>50.0</td><td>69.6</td><td>21.8</td><td>34.1</td><td>57.9</td><td>1.2</td><td>5.7</td><td>14.0</td></tr><tr><td>16</td><td>28.6</td><td>47.1</td><td>68.0</td><td>20.4</td><td>32.7</td><td>54.3</td><td>1.0</td><td>5.0</td><td>12.9</td></tr><tr><td>32</td><td>23.0</td><td>40.7</td><td>60.3</td><td>17.2</td><td>30.2</td><td>49.7</td><td>0.6</td><td>2.8</td><td>8.8</td></tr><tr><td rowspan="5">200B tokens (0.8 epochs)</td><td rowspan="5">32k tokens</td><td>1</td><td>30.0</td><td>53.8</td><td>73.7</td><td>22.8</td><td>36.4</td><td>62.0</td><td>2.8</td><td>7.8</td><td>17.4</td></tr><tr><td>2</td><td>30.3</td><td>55.1</td><td>76.2</td><td>22.2</td><td>38.5</td><td>62.6</td><td>2.1</td><td>9.0</td><td>21.7</td></tr><tr><td>4</td><td>33.8</td><td>55.9</td><td>76.9</td><td>24.0</td><td>40.1</td><td>66.1</td><td>1.6</td><td>7.1</td><td>19.9</td></tr><tr><td>6</td><td>31.9</td><td>53.9</td><td>73.1</td><td>20.6</td><td>38.4</td><td>63.9</td><td>3.5</td><td>10.8</td><td>22.7</td></tr><tr><td>8</td><td>30.7</td><td>52.2</td><td>73.4</td><td>20.0</td><td>36.6</td><td>59.6</td><td>3.5</td><td>10.4</td><td>22.1</td></tr><tr><td rowspan="2">1T tokens (4 epochs)</td><td rowspan="2">32k tokens</td><td>1</td><td>40.7</td><td>65.4</td><td>83.4</td><td>31.7</td><td>57.6</td><td>83.0</td><td>5.4</td><td>17.8</td><td>34.1</td></tr><tr><td>4</td><td>43.1</td><td>65.9</td><td>83.7</td><td>31.6</td><td>57.3</td><td>86.2</td><td>4.3</td><td>15.6</td><td>33.7</td></tr></table>

$2.7 \times$  on text. On an 8-byte prediction model, the inference speedup is  $6.4 \times$  (Table S3). Pretraining with multi-token prediction allows the additional heads to be much more accurate than a simple finetuning of a next-token prediction model, thus allowing our models to unlock self-speculative decoding's full potential.

# 3.3. Learning global patterns with multi-byte prediction

To show that the next-token prediction task latches to local patterns, we went to the extreme case of byte-level tokenization by training a 7B parameter byte-level transformer on 314B bytes, which is equivalent to around 116B tokens. The 8-byte prediction model achieves astounding improvements compared to next-byte prediction, solving  $67\%$  more problems on MBPP pass@1 and  $20\%$  more problems on HumanEval pass@1.

Multi-byte prediction is therefore a very promising avenue to unlock efficient training of byte-level models. Self-speculative decoding can achieve speedups of 6 times for the 8-byte prediction model, which would allow to fully compensate the cost of longer byte-level sequences at inference time and even be faster than a next-token prediction model by nearly two times. The 8-byte prediction model is a strong byte-based model, approaching the performance of token-based models despite having been trained on  $1.7 \times$  less data.

# 3.4. Searching for the optimal  $n$

To better understand the effect of the number of predicted tokens, we did comprehensive ablations on models of scale 7B trained on 200B tokens of code. We try  $n = 1, 2, 4, 6$  and 8 in this setting. Results in table 1 show that training with 4-future tokens outperforms all the other models consistently throughout HumanEval and MBPP for pass at 1, 10 and 100 metrics:  $+3.8\%$ ,  $+2.1\%$  and  $+3.2\%$  for MBPP and  $+1.2\%$ ,  $+3.7\%$  and  $+4.1\%$  for HumanEval. Interestingly, for APPS/Intro,  $n = 6$  takes the lead with  $+0.7\%$ ,  $+3.0\%$  and  $+5.3\%$ . It is very likely that the optimal window size depends on input data distribution. As for the byte level models the optimal window size is more consistent (8 bytes) across these benchmarks.

# 3.5. Training for multiple epochs

Multi-token training still maintains an edge on next-token prediction when trained on multiple epochs of the same data. The improvements diminish but we still have a  $+2.4\%$  increase on pass@1 on MBPP and  $+3.2\%$  increase on pass@100 on HumanEval, while having similar performance for the rest. As for APPS/Intro, a window size of 4 was already not optimal with 200B tokens of training.

# 3.6. Finetuning multi-token predictors

Pretrained models with multi-token prediction loss also outperform next-token models for use in finetunings. We evaluate this by finetuning 7B parameter models from Section 3.3

on the CodeContests dataset (Li et al., 2022). We compare the 4-token prediction model with the next-token prediction baseline, and include a setting where the 4-token prediction model is stripped off its additional prediction heads and finetuned using the classical next-token prediction target. According to the results in Figure 4, both ways of finetuning the 4-token prediction model outperform the next-token prediction model on pass@k across  $k$ . This means the models are both better at understanding and solving the task and at generating diverse answers. Note that CodeContests is the most challenging coding benchmark we evaluate in this study. Next-token prediction finetuning on top of 4-token prediction pretraining appears to be the best method overall, in line with the classical paradigm of pretraining with auxiliary tasks followed by task-specific finetuning. Please refer to Appendix F for details.

Figure 4: Comparison of finetuning performance on CodeContests. We finetune a 4-token prediction model on CodeContests (Li et al., 2022) (train split) using  $n'$ -token prediction as training loss with  $n' = 4$  or  $n' = 1$ , and compare to a finetuning of the next-token prediction baseline model ( $n = n' = 1$ ). For evaluation, we generate 1000 samples per test problem for each temperature  $T \in \{0.5, 0.6, 0.7, 0.8, 0.9\}$ , and compute pass@k for each value of  $k$  and  $T$ . Shown is  $k \mapsto \max_{T} \text{pass\_at}(k, T)$ , i.e. we grant access to a temperature oracle. We observe that both ways of finetuning the 4-token prediction model outperform the next-token prediction baseline. Intriguingly, using next-token prediction finetuning on top of the 4-token prediction model appears to be the best method overall.

# 3.7. Multi-token prediction on natural language

To evaluate multi-token prediction training on natural language, we train models of size 7B parameters on 200B tokens of natural language with a 4-token, 2-token and next-token prediction loss, respectively. In Figure 5, we evaluate the resulting checkpoints on 6 standard NLP benchmarks. On these benchmarks, the 2-future token prediction model performs on par with the next-token prediction baseline

Figure 5: Multi-token training with 7B models doesn't improve performance on choice tasks. This figure shows the evolution of average accuracy of 6 standard NLP benchmarks. Detailed results in Appendix G for 7B models trained on 200B tokens of language data. The 2 future token model has the same performance as the baseline and the 4 future token model regresses a bit. Larger model sizes might be necessary to see improvements on these tasks.

throughout training. The 4-future token prediction model suffers a performance degradation. Detailed numbers are reported in Appendix G.

However, we do not believe that multiple-choice and likelihood-based benchmarks are suited to effectively discern generative capabilities of language models. In order to avoid the need for human annotations of generation quality or language model judges—which comes with its own pitfalls, as pointed out by Koo et al. (2023)—we conduct evaluations on summarization and natural language mathematics benchmarks and compare pretrained models with training sets sizes of 200B and 500B tokens and with next-token and multi-token prediction losses, respectively.

For summarization, we use eight benchmarks where ROUGE metrics (Lin, 2004) with respect to a ground-truth summary allow automatic evaluation of generated texts. We finetune each pretrained model on each benchmark's training dataset for three epochs and select the checkpoint with the highest ROUGE-L  $F_{1}$  score on the validation dataset. Figure 6 shows that multi-token prediction models with both  $n = 2$  and  $n = 4$  improve over the next-token baseline in ROUGE-L  $F_{1}$  scores for both training dataset sizes, with the performance gap shrinking with larger dataset size. All metrics can be found in Appendix H.

For natural language mathematics, we evaluate the pretrained models in 8-shot mode on the GSM8K benchmark (Cobbe et al., 2021) and measure accuracy of the final answer produced after a chain-of-thought elicited by the few-shot examples. We evaluate pass@k metrics to quantify diversity and correctness of answers like in code evaluations

Figure 6: Performance on abstractive text summarization. Average ROUGE-L (longest common subsequence overlap)  $F_{1}$  score for 7B models trained on 200B and 500B tokens of natural language on eight summarization benchmarks. We finetune the respective models on each task's training data separately for three epochs and select the checkpoints with highest ROUGE-L  $F_{1}$  validation score. Both  $n = 2$  and  $n = 4$  multi-token prediction models have an advantage over next-token prediction models. Individual scores per dataset and more details can be found in Appendix H.

and use sampling temperatures between 0.2 and 1.4. The results are depicted in Figure S13 in Appendix I. For 200B training tokens, the  $n = 2$  model clearly outperforms the next-token prediction baseline, while the pattern reverses after 500B tokens and  $n = 4$  is worse throughout.

# 4. Ablations on synthetic data

What drives the improvements in downstream performance of multi-token prediction models on all of the tasks we have considered? By conducting toy experiments on controlled training datasets and evaluation tasks, we demonstrate that multi-token prediction leads to qualitative changes in model capabilities and generalization behaviors. In particular, Section 4.1 shows that for small model sizes, induction capability—as discussed by Olsson et al. (2022)—either only forms when using multi-token prediction as training loss, or it is vastly improved by it. Moreover, Section 4.2 shows that multi-token prediction improves generalization on an arithmetic task, even more so than tripling model size.

# 4.1. Induction capability

Induction describes a simple pattern of reasoning that completes partial patterns by their most recent continuation (Olsson et al., 2022). In other words, if a sentence contains "AB" and later mentions "A", induction is the prediction that the continuation is "B". We design a setup to measure induction

Figure 7: Induction capability of  $n$ -token prediction models. Shown is accuracy on the second token of two token names that have already been mentioned previously. Shown are numbers for models trained with a next-token and a 2-token prediction loss, respectively, with two independent runs each. The lines denote per-loss averages. For small model sizes, next-token prediction models learn practically no or significantly worse induction capability than 2-token prediction models, with their disadvantage disappearing at the size of 100M nonembedding parameters.

capability in a controlled way. Training small models of sizes 1M to 1B nonembedding parameters on a dataset of children stories, we measure induction capability by means of an adapted test set: in 100 stories from the original test split, we replace the character names by randomly generated names that consist of two tokens with the tokenizer we employ. Predicting the first of these two tokens is linked to the semantics of the preceding text, while predicting the second token of each name's occurrence after it has been mentioned at least once can be seen as a pure induction task. In our experiments, we train for up to 90 epochs and perform early stopping with respect to the test metric (i.e. we allow an epoch oracle). Figure 7 reports induction capability as measured by accuracy on the names' second tokens in relation to model size for two runs with different seeds.

We find that 2-token prediction loss leads to a vastly improved formation of induction capability for models of size 30M nonembedding parameters and below, with their advantage disappearing for sizes of 100M nonembedding parameters and above. We interpret this finding as follows: multi-token prediction losses help models to learn transferring information across sequence positions, which lends itself to the formation of induction heads and other in-context learning mechanisms. However, once induction capability has been formed, these learned features transform induction

Figure 8: Accuracy on a polynomial arithmetic task with varying number of operations per expression. Training with multi-token prediction losses increases accuracy across task difficulties. In particular, it also significantly improves out-of-domain generalization performance, albeit at a low absolute level. Tripling the model size, on the other hand, has a considerably smaller effect than replacing next-token prediction with multi-token prediction loss (Figure S16). Shown are two independent runs per configuration with 100M parameter models.

into a task that can be solved locally at the current token and learned with next-token prediction alone. From this point on, multi-token prediction actually hurts on this restricted benchmark—but we surmise that there are higher forms of in-context reasoning to which it further contributes, as evidenced by the results in Section 3.1. In Figure S14, we provide evidence for this explanation: replacing the children stories dataset by a higher-quality 9:1 mix of a books dataset with the children stories, we enforce the formation of induction capability early in training by means of the dataset alone. By consequence, except for the two smallest model sizes, the advantage of multi-token prediction on the task disappears: feature learning of induction features has converted the task into a pure next-token prediction task.

# 4.2. Algorithmic reasoning

Algorithmic reasoning tasks allow to measure more involved forms of in-context reasoning than induction alone. We train and evaluate models on a task on polynomial arithmetic in the ring  $\mathbb{F}_7[X] / (X^5)$  with unary negation, addition, multiplication and composition of polynomials as operations. The coefficients of the operands and the operators are sampled uniformly. The task is to return the coefficients of the polynomials corresponding to the resulting expressions. The number  $m$  of operations contained in the expressions is selected uniformly from the range from 1 to 5 at training time,

and can be used to adjust the difficulty of both in-domain  $(m \leq 5)$  and out-of-domain  $(m > 5)$  generalization evaluations. The evaluations are conducted with greedy sampling on a fixed test set of 2000 samples per number of operations. We train models of two small sizes with 30M and 100M nonembedding parameters, respectively. This simulates the conditions of large language models trained on massive text corpora which are likewise under-parameterized and unable to memorize their entire training datasets.

Multi-token prediction improves algorithmic reasoning capabilities as measured by this task across task difficulties (Figure 8). In particular, it leads to impressive gains in out-of-distribution generalization, despite the low absolute numbers. Increasing the model size from 30M to 100M parameters, on the other hand, does not improve evaluation accuracy as much as replacing next-token prediction by multi-token prediction does (Figure S16). In Appendix K, we furthermore show that multi-token prediction models retain their advantage over next-token prediction models on this task when trained and evaluated with pause tokens (Goyal et al., 2023).

# 5. Why does it work? Some speculation

Why does multi-token prediction afford superior performance on coding evaluation benchmarks, and on small algorithmic reasoning tasks? Our intuition, developed in this section, is that multi-token prediction mitigates the distributional discrepancy between training-time teacher forcing and inference-time autoregressive generation. We support this view with an illustrative argument on the implicit weights multi-token prediction assigns to tokens depending on their relevance for the continuation of the text, as well as with an information-theoretic decomposition of multi-token prediction loss.

# 5.1. Lookahead reinforces choice points

Not all token decisions are equally important for generating useful texts from language models (Bachmann and Nagarajan, 2024; Lin et al., 2024). While some tokens allow stylistic variations that do not constrain the remainder of the text, others represent choice points that are linked with higher-level semantic properties of the text and may decide whether an answer is perceived as useful or derailing.

Multi-token prediction implicitly assigns weights to training tokens depending on how closely they are correlated with their successors. As an illustrative example, consider the sequence depicted in Figure 9 where one transition is a hard-to-predict choice point while the other transitions are considered "inconsequential". Inconsequential transitions following a choice point are likewise hard to predict in advance. By marking and counting loss terms, we find that

Figure 9: Multi-token prediction loss assigns higher implicit weights to consequential tokens. Shown is a sequence in which all transitions except “ $5 \rightarrow A$ ” are easy to predict, alongside the corresponding prediction targets in 3-token prediction. Since the consequences of the difficult transition “ $5 \rightarrow A$ ” are likewise hard to predict, this transition receives a higher implicit weight in the overall loss via its correlates “ $3 \rightarrow A$ ”, ..., “ $5 \rightarrow C$ ”.

$n$ -token prediction associates a weight of  $\frac{n(n + 1)}{2}$  to choice points via their correlates, and a smaller weight of  $n$  to inconsequential points. Please refer to Appendix L.3 for more details. Generally, we believe that the quality of text generations depends on picking the right decisions at choice points, and that  $n$ -token prediction losses promote those.

# 5.2. Information-theoretic argument

Language models are typically trained by teacher-forcing, where the model receives the ground truth for each future token during training. However, during test time generation is unguided and autoregressive, whereby errors accumulate. Teacher-forcing, we argue, encourages models to focus on predicting well in the very short term, at the potential expense of ignoring longer-term dependencies in the overall structure of the generated sequence.

To illustrate the impact of multi-token prediction, consider the following information-theoretic argument. Here,  $X$  denotes the next future token, and  $Y$  the second-next future token. The production of both of these tokens is conditioned on some observed, input context  $C$ , that we omit from our equations for simplicity. When placed before token  $X$ , vanilla next-token prediction concerns the quantity  $H(X)$ , while multi-token prediction with  $n = 2$  aims at  $H(X) + H(Y)$ . We decompose these two quantities as:

$$
\begin{array}{l} H (X) = H (X \mid Y) + I (X; Y), \\ H (X) + H (Y) = H (X \mid Y) + 2 I (X; Y) + H (Y \mid X). \\ \end{array}
$$

By discarding the term  $H(Y \mid X)$  which appears again when predicting at the following position—we observe that 2-token prediction increases the importance of  $I(X;Y)$  by a factor of 2. So, multi-token predictors are more accurate at predicting tokens  $X$  that are of relevance for the remainder

of the text to come. In Appendix L.2, we give a relative version of the above equations that shows the increased weight of relative mutual information in a loss decomposition of 2-token prediction loss.

# 6. Related work

Language modeling losses Dong et al. (2019) and Tay et al. (2022) train on a mixture of denoising tasks with different attention masks (full, causal and prefix attention) to bridge the performance gap with next token pretraining on generative tasks. Tay et al. (2022) uses the span corruption objective, which replaces spans of tokens with special tokens for the encoder and the decoder then predicts the contents of those spans. Unlike UniLM, this allows full causal training with teacher forcing. Similarly, Yang et al. (2019) train on permuted sequences, while conserving the original positional embeddings, effectively training the model to predict various parts of the sequence given a mix of past and future information. This permuted language modeling is the closest task to ours since it allows predicting beyond the next token. However all of these language modeling tasks train on a small percentage of the input text: on average only  $15\%$  of the tokens are backwarded through. For Dong et al. (2019), where the masking is done in BERT style, it is hard to mask more than  $15\%$  since it destroys too much information. For Tay et al. (2022), it is technically possible to have a larger proportion but in practice, the settings used have between  $15\%$  and  $25\%$  of masked tokens. (Yang et al., 2019) also makes it possible to train on the whole sequence since it is only permuted, and no information is lost. Yet, in practice, since the completely random permutation is very hard to reconstruct, only  $15\%$  are predicted for training stability reasons.

Multi-token prediction in language modelling Qi et al. (2020) argue that multi-token prediction encourages planning, improves representations and prevents the overfitting on local patterns that can result from teacher-forced training. However, their technical approach replicates the residual stream  $n$ -fold while ours allows for compute-matched comparisons and makes the residual representations participate more directly in the auxiliary loss terms. Stern et al. (2018) and Cai et al. (2024) propose model finetunings with multi-token prediction for faster inference but do not study the effects of such a loss during pretraining. Pal et al. (2023) use probing methods to show that next-token prediction models are able to predict additional consecutive tokens to a certain extent, but less so than our models which are specifically trained for this task. Jianyu Zhang (2024) observe improvements in language modelling tasks with multi-label binary classification over the occurrence of vocabulary words in the future as an auxiliary learning task.

Self-speculative decoding Stern et al. (2018) are, to the best of our knowledge, the first to suggest a speculative decoding scheme for faster inference. Our architecture replaces their linear prediction heads by transformer layers, but is otherwise similar. By reorganizing the order of the forward/backward, we can use all loss terms instead of stochastically picking one head for loss computation. Cai et al. (2024) present a more elaborate self-speculative decoding scheme that uses the top- $k$  predictions of each head instead of the best one only. It can be used with the multi-token prediction models we train.

Multi-target prediction Multi-task learning is the paradigm of training neural networks jointly on several tasks to improve performance on the tasks of interest (Caruana, 1997). Learning with such auxiliary tasks allows models to exploit dependencies between target variables and can even be preferable in the case of independent targets (Waegeman et al., 2019). While more specifically tailored architectures for multi-target prediction are conceivable (Spyromitros-Xioufis et al., 2016; Read et al., 2021), modern deep learning approaches usually rely on large shared model trunks with separate prediction heads for the respective tasks (Caruana, 1997; Silver et al., 2016; Lample et al., 2022) like we do. Multi-target prediction has been shown to be a successful strategy in various domains, e.g. for learning time series prediction with more distant time steps in the future as auxiliary targets (Vapnik and Vashist, 2009) or for learning from videos with several future frames (Mathieu et al., 2016; Srivastava et al., 2016) or representations of future frames (Vondrick et al., 2016) as auxiliary targets.

# 7. Conclusion

We have proposed multi-token prediction as an improvement over next-token prediction in training language models for generative or reasoning tasks. Our experiments (up to 7B parameters and 1T tokens) show that this is increasingly useful for larger models and in particular show strong improvements for code tasks. We posit that our method reduces distribution mismatch between teacher-forced training and autoregressive generation. When used with speculative decoding, exact inference gets 3 times faster.

In future work we would like to better understand how to automatically choose  $n$  in multi-token prediction losses. One possibility to do so is to use loss scales and loss balancing (Défossez et al., 2022). Also, optimal vocabulary sizes for multi-token prediction are likely different from those for next-token prediction, and tuning them could lead to better results, as well as improved trade-offs between compressed sequence length and compute-per-byte expenses. Finally, we would like to develop improved auxiliary prediction losses that operate in embedding spaces (LeCun, 2022).

# Impact statement

The goal of this paper is to make language models more compute and data efficient. While this may in principle reduce the ecological impact of training LLMs, we shall be careful about rebound effects. All societal advantages, as well as risks, of LLMs should be considered while using this work.

# Environmental impact

In aggregate, training all models reported in the paper required around 500K GPU hours of computation on hardware of type A100-80GB and H100. Estimated total emissions were around 50 tCO2eq,  $100\%$  of which were offset by Meta's sustainability program.

# Acknowledgements

We thank Jianyu Zhang, Léon Bottou, Emmanuel Dupoux, Pierre-Emmanuel Mazaré, Yann LeCun, Quentin Garrido, Megi Dervishi, Mathurin Videau and Timothée Darcet and other FAIR PhD students and CodeGen team members for helpful discussions. We thank Jonas Gehring for his technical expertise and the original Llama team and xFormers team for enabling this kind of research.

# References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.  
Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction, 2024.  
Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks, 2015.  
Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019.  
Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024.  
Rich Caruana. Multitask learning. Machine learning, 28: 41-75, 1997.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde, Jared Kaplan, Harri Edwards, Yura Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  
Nakhun Chumpolsathien. Using knowledge distillation from keyword extraction to improve the informativeness of neural cross-lingual summarization. Master's thesis, Beijing Institute of Technology, 2020.  
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.  
Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pages 13063-13075, 2019.  
Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.  
Moussa Kamal Eddine, Antoine J. P. Tixier, and Michalis Vazirgiannis. Barthez: a skilled pretrained french sequence-to-sequence model, 2021.

Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. Multi-news: a large-scale multi-document summarization dataset and abstractive hierarchical model, 2019.  
Mehrdad Farahani. Summarization using bert2bert model on wikisummary dataset. https://github.com/m3hrdadfi/wiki-summary, 2020.  
Mehrdad Farahani, Mohammad Gharachorloo, and Mohammad Manthouri. Leveraging parsbert and pretrained mt5 for persian abstractive text summarization. In 2021 26th International Computer Conference, Computer Society of Iran (CSICC). IEEE, March 2021. doi: 10.1109/csicc52343.2021.9420563. URL http://dx.doi.org/10.1109/csicc52343.2021.9420563.  
Michael C Frank. Bridging the data gap between children and large language models. Trends in Cognitive Sciences, 2023.  
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. Samsum corpus: A human-annotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization. Association for Computational Linguistics, 2019. doi: 10.18653/v1/d19-5409. URL http://dx.doi.org/10.18653/v1/D19-5409.  
Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens, 2023.  
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.  
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration, 2020.  
Jianyu Zhang Leon Bottou. Multi-label classification as an auxiliary loss for language modelling. personal communication, 2024.  
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *ICLR*, 2015.  
Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. Benchmarking cognitive biases in large language models as evaluators. arXiv preprint arXiv:2309.17012, 2023.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019.  
Guillaume Lample, Marie-Anne Lachaux, Thibaut Lavril, Xavier Martinet, Amaury Hayat, Gabriel Ebner, Aurélien Rodriguez, and Timothée Lacroix. Hypertree proof search for neural theorem proving, 2022.  
Yann LeCun. A path towards autonomous machine intelligence version 0.9.2, 2022-06-27. Open Review, 62(1), 2022.  
Benjamin Lefaudeaux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022.  
Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding, 2023.  
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphabet. Science, 378(6624):1092-1097, 2022.  
Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74-81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.  
Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. Rho-1: Not all tokens are what you need, 2024.  
Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts, 2017.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.  
Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error, 2016.  
Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, and Bing Xiang. Abstractive text

summmarization using sequence-to-sequence rnns and beyond, 2016.  
Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization, 2018.  
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Ohl. In-context learning and induction heads. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.  
OpenAI. Gpt-4 technical report, 2023.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.  
Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, and David Bau. Future lens: Anticipating subsequent tokens from a single hidden state, 2023.  
Weizhen Qi, Yu Yan, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, and Ming Zhou. Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training, 2020.  
Jesse Read, Bernhard Pfahringer, Geoffrey Holmes, and Eibe Frank. Classifier chains: A review and perspectives. Journal of Artificial Intelligence Research, 70:683-718, 2021.  
Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series, 2011.  
Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiaq: Commonsense reasoning about social interactions, 2019.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587): 484-489, 2016.

Aaditya K Singh, Stephanie CY Chan, Ted Moskovitz, Erin Grant, Andrew M Saxe, and Felix Hill. The transient nature of emergent in-context learning in transformers. arXiv preprint arXiv:2311.08360, 2023.  
Eleftherios Spyromitros-Xioufis, Grigorios Tsoumakas, William Groves, and Ioannis Vlahavas. Multi-target regression via input space expansion: treating targets as inputs. Machine Learning, 104:55-98, 2016.  
Nitish Srivastava, Elman Mansimov, and Ruslan Salakhutdinov. Unsupervised learning of video representations using lstms, 2016.  
Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models, 2018.  
Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, et al. Ul2: Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2022.  
Vladimir Vapnik and Akshay Vashist. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544-557, 2009.  
Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Anticipating visual representations from unlabeled video, 2016.  
Willem Waegeman, Krzysztof Dembczyński, and Eyke Hüllermeier. Multi-target prediction: a unifying view on problems and methods. Data Mining and Knowledge Discovery, 33:293-324, 2019.  
Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. arXiv preprint arXiv:1911.07176, 2019.  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pages 5753-5763, 2019.  
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019.

# A. Additional results on self-speculative decoding

Figure S10: Decoding speeds and latencies with self-speculative decoding relative to standard autoregressive decoding. We use  $k$  heads of a 4-token prediction model and evaluate decoding speeds of a code model as explained in Table S2. All numbers are relative to the autoregressive ( $k = 1$ ) baseline with the same batch size.


Table S2: Relative speedups with self-speculative decoding. For wikipedia and books we prompt a 7B parameter model trained on 500B tokens, and for code we prompt a 7B parameter model trained on 1T tokens of code on 4200 sequences of 512 tokens from a test dataset not seen during training, and generate completions consisting of 512 tokens using greedy self-speculative decoding (Stern et al., 2018) using the indicated number of heads from a 4-token prediction model. Note that the maximal speedup that can be obtained with self-speculative decoding using  $k$  heads is  $k$ . The last column shows the average number of tokens retrieved from a forward containing this sequence (both verification and prediction). The speedup was evaluated at the maximal batch size of 42, but is constant across batch sizes (Figure S10).

<table><tr><td rowspan="2"># Heads used</td><td colspan="2">Wikipedia</td><td colspan="2">Books</td><td colspan="2">Code</td></tr><tr><td>Rel. speedup</td><td>Tokens / forward</td><td>Rel. speedup</td><td>Tokens / forward</td><td>Rel. speedup</td><td>Tokens / forward</td></tr><tr><td>1</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>2</td><td>1.79</td><td>1.88</td><td>1.77</td><td>1.87</td><td>1.85</td><td>1.94</td></tr><tr><td>3</td><td>2.35</td><td>2.57</td><td>2.32</td><td>2.56</td><td>2.54</td><td>2.78</td></tr><tr><td>4</td><td>2.74</td><td>3.12</td><td>2.67</td><td>3.09</td><td>3.05</td><td>3.50</td></tr></table>

Table S3: Relative speedups with self-speculative decoding with byte-level models on code. We prompt the 7B parameter models from Section 3.3 on 4096 sequences of 1024 bytes of code not seen during training, and generate completions consisting of 1024 bytes using greedy self-speculative decoding (Stern et al., 2018) as in Table S2. The speedup was evaluated at a batch size of 16.  

<table><tr><td rowspan="2"># Heads used</td><td colspan="2">n = 8</td><td colspan="2">n = 16</td><td colspan="2">n = 32</td></tr><tr><td>Rel. speedup</td><td>Tokens / forward</td><td>Rel. speedup</td><td>Tokens / forward</td><td>Rel. speedup</td><td>Tokens / forward</td></tr><tr><td>1</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>2</td><td>1.94</td><td>1.98</td><td>1.94</td><td>1.98</td><td>1.93</td><td>1.97</td></tr><tr><td>4</td><td>3.67</td><td>3.84</td><td>3.63</td><td>3.81</td><td>3.62</td><td>3.80</td></tr><tr><td>8</td><td>6.39</td><td>7.04</td><td>6.25</td><td>6.92</td><td>6.22</td><td>6.89</td></tr><tr><td>12</td><td>-</td><td>-</td><td>8.07</td><td>9.36</td><td>8.01</td><td>9.30</td></tr><tr><td>16</td><td>-</td><td>-</td><td>9.24</td><td>11.20</td><td>9.15</td><td>11.15</td></tr><tr><td>20</td><td>-</td><td>-</td><td>-</td><td>-</td><td>9.83</td><td>12.61</td></tr><tr><td>24</td><td>-</td><td>-</td><td>-</td><td>-</td><td>10.34</td><td>13.67</td></tr><tr><td>28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>10.55</td><td>14.58</td></tr><tr><td>32</td><td>-</td><td>-</td><td>-</td><td>-</td><td>10.84</td><td>15.35</td></tr></table>

# B. Alternative architectures

Table S4: Alternative architectures improve on baseline but not as consistently. Alternative architectures for multi-token prediction are worth exploring to improve efficiency. Here we tried Anticausal, causal and linear and showed no significant improvement with respect to Parallel architecture.  

<table><tr><td rowspan="2">n</td><td rowspan="2">Head type</td><td rowspan="2">Architecture</td><td rowspan="2">+Layers</td><td colspan="3">MBPP</td><td colspan="3">HumanEval</td><td colspan="3">APPS/Intro</td></tr><tr><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td></tr><tr><td>1</td><td>transformer</td><td>parallel</td><td>0</td><td>30.0</td><td>53.8</td><td>73.7</td><td>22.8</td><td>36.4</td><td>62.0</td><td>2.8</td><td>7.8</td><td>17.4</td></tr><tr><td rowspan="5">4</td><td>linear</td><td>parallel</td><td>0</td><td>33.6</td><td>55.0</td><td>76.2</td><td>21.9</td><td>38.5</td><td>63.7</td><td>3.1</td><td>10.1</td><td>23.0</td></tr><tr><td rowspan="4">transformer</td><td>anticausal</td><td>0</td><td>30.8</td><td>54.8</td><td>75.3</td><td>20.9</td><td>38.4</td><td>64.5</td><td>2.0</td><td>8.7</td><td>21.6</td></tr><tr><td>causal</td><td>0</td><td>31.9</td><td>54.9</td><td>74.9</td><td>20.9</td><td>38.1</td><td>67.3</td><td>4.0</td><td>11.6</td><td>22.8</td></tr><tr><td rowspan="2">parallel</td><td>0</td><td>33.8</td><td>55.9</td><td>76.9</td><td>24.0</td><td>40.1</td><td>66.1</td><td>1.6</td><td>7.1</td><td>19.9</td></tr><tr><td>3</td><td>33.3</td><td>55.7</td><td>77.3</td><td>22.4</td><td>39.4</td><td>66.7</td><td>2.6</td><td>9.5</td><td>22.1</td></tr></table>

The architecture described in Section 2 is not the only sensible option, but proved technically viable and well-performing in our experiments. We describe and compare alternative architectures in this section.

Replicated unembeddings Replicating the unembedding matrix  $n$  times is a simple method for implementing multi-token prediction architectures. However, it requires matrices with shapes  $(d,nV)$  in the notation of Section 2, which is prohibitive for large-scale trainings.

Linear heads Apart from using a single transformer layer for the heads  $H_{i}$ , other architectures are conceivable. We experimented with a single linear layer without any nonlinearity as heads, amounting to linear probing of the model's residual representation  $z$ . Architectures with more than one layer per head are also possible, but we did not pursue this direction further.

Causal and anticausal variant Instead of making the prediction heads  $P_{i}(x_{t + i} \mid z_{t:1})$  architecturally independent of each other, we can also allow them to rely on other heads' (pre-unembedding) outputs. In a causal variant, later prediction heads are applied on top of the previous ones, i.e. the  $i$ -th prediction head  $P_{i}$  is given by

$$
P _ {\theta} (x _ {t + i} | \cdot) = \operatorname {s o f t m a x} \circ f _ {u} \circ f _ {h _ {i}} \circ f _ {h _ {i - 1}} \dots \circ f _ {h _ {1}} \circ f _ {s}.
$$

In another anticausal variant, the network starts by predicting the most distant tokens before gradually refining up to the following token:

$$
P _ {\theta} (x _ {t + i} | \cdot) = \operatorname {s o f t m a x} \circ f _ {u} \circ f _ {h _ {i}} \circ f _ {h _ {i + 1}} \dots \circ f _ {h _ {n}} \circ f _ {s}.
$$

These architectures likewise allow a sequential forward/backward order as the parallel architecture from Section 2. This is described in Figure S11.

Figure S11: Order of the forward/backward in a causal  $n$ -token prediction model with  $n = 2$  heads. Like in the forward/backward depicted for parallel prediction heads in Figure 2, we avoid materializing all unembedding layer gradients in memory simultaneously and reduce peak GPU memory usage significantly. The iteration over the heads starts with the one furthest to the trunk. At each head, a gradient from the succeeding prediction heads and from the head's own loss are accumulated for both the head's output and its weights.

# C. Training speeds

Table S5: Training time relative to next-token prediction training. The slight overhead when using multi-token prediction here is explained by a suboptimal use of Fully Sharded Data Parallel. In our implementation, when doing separate backward passes for each head, we lose the overlap of layer weight communication and computation, therefore it incurs a very slight overhead that can be removed if reimplemented correctly.  

<table><tr><td>Model</td><td>n=1</td><td>n=2</td><td>n=4</td></tr><tr><td>0.3B</td><td>1.00</td><td>1.07</td><td>1.22</td></tr><tr><td>0.6B</td><td>1.00</td><td>1.05</td><td>1.13</td></tr><tr><td>1.3B</td><td>1.00</td><td>1.04</td><td>1.12</td></tr><tr><td>3B</td><td>1.00</td><td>1.02</td><td>1.07</td></tr><tr><td>6.7B</td><td>1.00</td><td>1.02</td><td>1.07</td></tr><tr><td>13B</td><td>1.00</td><td>1.04</td><td>1.09</td></tr></table>

# D. Finetuning

Table S6: Finetuning LLa 2 with multi-token prediction does not significantly improve performance. We tried to finetune LLa 2 with 4-token prediction but this did not yield significant improvements compared to the baseline. We suppose that this new loss changes the initialization too brutally and never really recovers. We still some improvements for example on MBPP Pass@1. All runs use 200B tokens of code.  

<table><tr><td rowspan="2">n</td><td rowspan="2">Head type</td><td rowspan="2">+Layers</td><td colspan="3">MBPP</td><td colspan="3">HumanEval</td><td colspan="3">APPS/Intro</td></tr><tr><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td></tr><tr><td>1</td><td>transformer</td><td>0</td><td>39.6</td><td>65.1</td><td>82.4</td><td>31.4</td><td>57.7</td><td>84.7</td><td>10.0</td><td>21.6</td><td>36.7</td></tr><tr><td rowspan="3">4</td><td>linear</td><td>0</td><td>39.3</td><td>63.7</td><td>81.3</td><td>29.0</td><td>53.4</td><td>82.2</td><td>6.9</td><td>20.0</td><td>34.0</td></tr><tr><td rowspan="2">transformer</td><td>0</td><td>38.3</td><td>62.2</td><td>80.1</td><td>27.9</td><td>53.6</td><td>82.4</td><td>5.8</td><td>18.2</td><td>34.3</td></tr><tr><td>3</td><td>42.5</td><td>64.4</td><td>81.3</td><td>28.7</td><td>56.9</td><td>82.4</td><td>7.8</td><td>21.2</td><td>37.3</td></tr></table>

# E. Additional results on model scaling behavior

Table S7: Scaling model size Full results of scaling model size with  $n = 1,2$  and 4.  

<table><tr><td rowspan="2">Model Size</td><td colspan="4">MBPP</td><td colspan="3">HumanEval</td></tr><tr><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td><td></td></tr><tr><td rowspan="3">0.3B</td><td>1</td><td>1.8</td><td>10.4</td><td>29.9</td><td>1.9</td><td>5.0</td><td>10.9</td></tr><tr><td>2</td><td>1.7</td><td>10.1</td><td>27.2</td><td>1.5</td><td>4.4</td><td>10.3</td></tr><tr><td>4</td><td>1.0</td><td>6.3</td><td>20.1</td><td>1.2</td><td>4.0</td><td>8.6</td></tr><tr><td rowspan="3">0.6B</td><td>1</td><td>4.7</td><td>21.0</td><td>45.2</td><td>2.9</td><td>8.5</td><td>16.7</td></tr><tr><td>2</td><td>4.6</td><td>21.0</td><td>44.7</td><td>3.2</td><td>8.9</td><td>16.2</td></tr><tr><td>4</td><td>3.0</td><td>15.6</td><td>38.0</td><td>2.7</td><td>7.7</td><td>15.5</td></tr><tr><td rowspan="3">1.3B</td><td>1</td><td>6.8</td><td>27.0</td><td>51.0</td><td>4.6</td><td>13.1</td><td>24.3</td></tr><tr><td>2</td><td>7.3</td><td>27.5</td><td>51.7</td><td>5.4</td><td>13.6</td><td>23.3</td></tr><tr><td>4</td><td>7.4</td><td>27.6</td><td>50.1</td><td>4.8</td><td>12.3</td><td>22.5</td></tr><tr><td rowspan="3">3B</td><td>1</td><td>11.1</td><td>36.4</td><td>60.4</td><td>7.2</td><td>17.2</td><td>29.8</td></tr><tr><td>2</td><td>11.8</td><td>37.2</td><td>60.5</td><td>8.0</td><td>18.2</td><td>31.2</td></tr><tr><td>4</td><td>12.7</td><td>37.6</td><td>61.1</td><td>7.2</td><td>18.5</td><td>33.3</td></tr><tr><td rowspan="3">6.7B</td><td>1</td><td>23.9</td><td>54.2</td><td>74.7</td><td>12.8</td><td>29.3</td><td>51.7</td></tr><tr><td>2</td><td>24.7</td><td>54.8</td><td>76.4</td><td>13.2</td><td>32.2</td><td>53.9</td></tr><tr><td>4</td><td>26.0</td><td>55.8</td><td>76.0</td><td>13.8</td><td>33.2</td><td>58.5</td></tr><tr><td rowspan="3">13B</td><td>1</td><td>26.0</td><td>57.1</td><td>77.0</td><td>14.1</td><td>33.6</td><td>56.0</td></tr><tr><td>2</td><td>30.5</td><td>60.5</td><td>79.4</td><td>15.2</td><td>36.9</td><td>60.0</td></tr><tr><td>4</td><td>30.5</td><td>61.0</td><td>79.2</td><td>15.8</td><td>38.6</td><td>63.5</td></tr></table>

# F. Details on CodeContests finetuning

We use the Python subset of the CodeContests (Li et al., 2022) train split with reward annotations ("correct" / "incorrect") and condition on correct solutions at evaluation time. For evaluation, we generate 1000 samples per problem from the test split for each temperature  $T \in \{0.5, 0.6, 0.7, 0.8, 0.9\}$ , and compute the unbiased estimator for pass@k from Chen et al. (2021) for each value of  $k$  and  $T$ . It is possible that models that were pretrained with different losses have different respective optimal temperatures for pass@k, so we compute and show  $k \mapsto \max_{T} \text{pass\_at}(k, T)$  in Figure 4. In other words, we grant pass@k access to a temperature oracle. For small values of  $k$ , pass@k measures the capability of understanding and solving tasks while for large  $k$ , it additionally favors diversity in outputs. According to the results in Figure 4, multi-token prediction pretraining leads to finetuned models that are better on both axes.

# G. Additional results on natural language benchmarks

We evaluate the models from Section 3.7 on standard natural language processing benchmarks: ARC Challenge (Yadav et al., 2019), COPA (Roemmele et al., 2011), Hellaswag (Zellers et al., 2019), Natural Questions (Kwiatkowski et al., 2019), PIQA (Bisk et al., 2019), SIQA (Sap et al., 2019) and TriviaQA (Joshi et al., 2017).







Figure S12: Multiple token training with 7B models doesn't improve performance on choice tasks. This figure shows the evolution of average accuracy of some standard NLP benchmarks (ARC Challenge COPA Hellaswag MMLU Natural Questions PIQA SIQA and TriviaQA. For the 7B models trained on 200B tokens of language data, the 2 future token model has the same performance as the baseline and the 4 future token model regresses a bit. Larger model sizes might be necessary to see improvements on these tasks.

# H. Additional results on abstractive text summarization

In this section, we report comprehensive evaluation results on summarization tasks for the 7B parameter models trained on 200B and 500B tokens of natural language from Section 3.7.

Table S8: Comprehensive evaluation on abstractive text summarization. ROUGE-n (n-gram overlap) and ROUGE-L (longest common subsequence overlap)  $F_{1}$  scores for 7B models trained on 200B and 500B tokens of natural language, respectively. The last three columns correspond to models trained on 500B tokens, the previous three to models trained on 200B tokens. Shown are numbers of the  $n = 1$  baseline and the absolute difference of  $n = 2$  and  $n = 4$  models trained on the same number of tokens. Summary-level ROUGE-L ("ROUGE-Lsum") is reported where it differs from ROUGE-L. Model checkpoints with maximal validation ROUGE-L  $F_{1}$  are selected separately for each model dataset and model type and reported in the first row corresponding to each dataset. Boldface for numbers within 0.05 difference to the best one for each dataset size separately.

<table><tr><td>Task</td><td>Metric</td><td>Baseline 200B</td><td>Δn=2</td><td>Δn=4</td><td>Baseline 500B</td><td>Δn=2</td><td>Δn=4</td></tr><tr><td rowspan="6">CNN/Dailymail (Nallapati et al., 2016)</td><td>evaluation epoch</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>ROUGE-1</td><td>42.88</td><td>+0.74</td><td>+0.74</td><td>43.77</td><td>+0.55</td><td>+0.50</td></tr><tr><td>ROUGE-2</td><td>19.56</td><td>+0.52</td><td>+0.53</td><td>20.34</td><td>+0.52</td><td>+0.34</td></tr><tr><td>ROUGE-3</td><td>11.11</td><td>+0.39</td><td>+0.35</td><td>11.69</td><td>+0.36</td><td>+0.19</td></tr><tr><td>ROUGE-L</td><td>29.72</td><td>+0.66</td><td>+0.49</td><td>30.51</td><td>+0.48</td><td>+0.37</td></tr><tr><td>ROUGE-Lsum</td><td>40.18</td><td>+0.72</td><td>+0.68</td><td>41.02</td><td>+0.56</td><td>+0.52</td></tr><tr><td rowspan="5">Multi-News (Fabbri et al., 2019)</td><td>evaluation epoch</td><td>1</td><td>3</td><td>3</td><td>2</td><td>3</td><td>2</td></tr><tr><td>ROUGE-1</td><td>44.48</td><td>+1.70</td><td>+1.72</td><td>45.87</td><td>+1.05</td><td>+0.69</td></tr><tr><td>ROUGE-2</td><td>16.88</td><td>+0.44</td><td>+0.70</td><td>17.56</td><td>+0.42</td><td>+0.40</td></tr><tr><td>ROUGE-3</td><td>9.63</td><td>-0.06</td><td>+0.17</td><td>9.91</td><td>+0.22</td><td>+0.18</td></tr><tr><td>ROUGE-L</td><td>23.82</td><td>+0.17</td><td>+0.40</td><td>24.22</td><td>+0.20</td><td>+0.26</td></tr><tr><td rowspan="5">OrangeSum (Eddine et al., 2021)</td><td>evaluation epoch</td><td>2</td><td>2</td><td>3</td><td>2</td><td>1</td><td>3</td></tr><tr><td>ROUGE-1</td><td>32.95</td><td>+0.41</td><td>+0.35</td><td>33.37</td><td>+0.32</td><td>+0.78</td></tr><tr><td>ROUGE-2</td><td>13.90</td><td>+0.31</td><td>+0.36</td><td>14.22</td><td>+0.25</td><td>+0.53</td></tr><tr><td>ROUGE-3</td><td>8.01</td><td>+0.19</td><td>+0.21</td><td>8.12</td><td>+0.22</td><td>+0.48</td></tr><tr><td>ROUGE-L</td><td>23.62</td><td>+0.36</td><td>+0.51</td><td>23.91</td><td>+0.23</td><td>+0.66</td></tr><tr><td rowspan="5">pn-summary (Farahani et al., 2021)</td><td>evaluation epoch</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td><td>3</td></tr><tr><td>ROUGE-1</td><td>1.03</td><td>+0.02</td><td>0.00</td><td>0.92</td><td>+0.09</td><td>+0.05</td></tr><tr><td>ROUGE-2</td><td>0.13</td><td>+0.02</td><td>+0.03</td><td>0.15</td><td>0.00</td><td>0.00</td></tr><tr><td>ROUGE-3</td><td>0.02</td><td>0.00</td><td>+0.02</td><td>0.02</td><td>0.00</td><td>+0.02</td></tr><tr><td>ROUGE-L</td><td>1.02</td><td>+0.03</td><td>+0.01</td><td>0.91</td><td>+0.09</td><td>+0.05</td></tr><tr><td rowspan="5">SAMSum (Gliwa et al., 2019)</td><td>evaluation epoch</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>ROUGE-1</td><td>51.39</td><td>+0.70</td><td>+0.63</td><td>52.54</td><td>-0.24</td><td>+0.69</td></tr><tr><td>ROUGE-2</td><td>26.46</td><td>+0.76</td><td>+0.30</td><td>27.74</td><td>-0.20</td><td>+0.82</td></tr><tr><td>ROUGE-3</td><td>16.40</td><td>+0.91</td><td>+0.28</td><td>17.56</td><td>-0.30</td><td>+0.71</td></tr><tr><td>ROUGE-L</td><td>42.59</td><td>+0.90</td><td>+0.51</td><td>43.92</td><td>-0.10</td><td>+0.63</td></tr><tr><td rowspan="5">ThaiSum (Chumpolsathien, 2020)</td><td>evaluation epoch</td><td>2</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>ROUGE-1</td><td>45.08</td><td>+0.63</td><td>+1.12</td><td>45.48</td><td>+0.77</td><td>+0.91</td></tr><tr><td>ROUGE-2</td><td>27.85</td><td>+0.30</td><td>+0.73</td><td>28.07</td><td>+0.74</td><td>+0.64</td></tr><tr><td>ROUGE-3</td><td>15.73</td><td>+0.04</td><td>+0.43</td><td>15.82</td><td>+0.50</td><td>+0.30</td></tr><tr><td>ROUGE-L</td><td>44.92</td><td>+0.64</td><td>+1.12</td><td>45.31</td><td>+0.76</td><td>+0.89</td></tr><tr><td rowspan="5">WikiSummary (Farahani, 2020)</td><td>evaluation epoch</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>ROUGE-1</td><td>10.16</td><td>+0.67</td><td>-0.23</td><td>12.80</td><td>-0.17</td><td>-0.99</td></tr><tr><td>ROUGE-2</td><td>4.46</td><td>-0.03</td><td>-0.09</td><td>6.17</td><td>-0.11</td><td>-0.69</td></tr><tr><td>ROUGE-3</td><td>1.31</td><td>+0.21</td><td>+0.13</td><td>1.98</td><td>-0.08</td><td>-0.33</td></tr><tr><td>ROUGE-L</td><td>10.11</td><td>+0.65</td><td>-0.28</td><td>12.69</td><td>-0.17</td><td>-0.99</td></tr><tr><td rowspan="5">XSum (Narayan et al., 2018)</td><td>evaluation epoch</td><td>2</td><td>2</td><td>3</td><td>2</td><td>2</td><td>3</td></tr><tr><td>ROUGE-1</td><td>42.16</td><td>+0.71</td><td>+1.07</td><td>43.42</td><td>+0.78</td><td>+0.67</td></tr><tr><td>ROUGE-2</td><td>19.19</td><td>+0.54</td><td>+0.55</td><td>20.32</td><td>+0.68</td><td>+0.34</td></tr><tr><td>ROUGE-3</td><td>10.43</td><td>+0.38</td><td>+0.28</td><td>11.23</td><td>+0.48</td><td>+0.20</td></tr><tr><td>ROUGE-L</td><td>34.03</td><td>+0.67</td><td>+0.92</td><td>35.18</td><td>+0.79</td><td>+0.63</td></tr></table>

Table S9: Performance on abstractive text summarization. ROUGE-L (longest common subsequence overlap)  $F_{1}$  score for 7B models trained on 200B and 500B tokens of natural language. We finetune the respective models on each task's training data separately for a given number of epochs and select the checkpoints with maximal ROUGE-L  $F_{1}$  on the validation dataset. The second and fifth column report the numbers for a next-token prediction model, while the third, fourth, sixth and seventh one report the absolute improvements for 2-token and 4-token prediction models trained on the same amount of data, respectively. Boldface for numbers within 0.05 difference to the best one for each dataset size separately.

<table><tr><td>Dataset</td><td>Baseline 200B</td><td>Δn=2</td><td>Δn=4</td><td>Baseline 500B</td><td>Δn=2</td><td>Δn=4</td></tr><tr><td>CNN/Dailymail</td><td>29.72</td><td>+0.66</td><td>+0.49</td><td>30.51</td><td>+0.48</td><td>+0.37</td></tr><tr><td>Multi-News</td><td>23.82</td><td>+0.17</td><td>+0.40</td><td>24.22</td><td>+0.20</td><td>+0.26</td></tr><tr><td>OrangeSum</td><td>23.62</td><td>+0.36</td><td>+0.51</td><td>23.91</td><td>+0.23</td><td>+0.66</td></tr><tr><td>pn-summary</td><td>1.02</td><td>+0.03</td><td>+0.01</td><td>0.91</td><td>+0.09</td><td>+0.05</td></tr><tr><td>SAMSum</td><td>42.59</td><td>+0.90</td><td>+0.51</td><td>43.92</td><td>-0.10</td><td>+0.63</td></tr><tr><td>ThaiSum</td><td>44.92</td><td>+0.64</td><td>+1.12</td><td>45.31</td><td>+0.76</td><td>+0.89</td></tr><tr><td>WikiSummary</td><td>10.11</td><td>+0.65</td><td>-0.28</td><td>12.69</td><td>-0.17</td><td>-0.99</td></tr><tr><td>XSum</td><td>34.03</td><td>+0.67</td><td>+0.92</td><td>35.18</td><td>+0.79</td><td>+0.63</td></tr><tr><td>Average</td><td>26.23</td><td>+0.51</td><td>+0.46</td><td>27.08</td><td>+0.28</td><td>+0.31</td></tr></table>

Table S10: Summary statistics for abstractive text summarization evaluations. Reported are averages for ROUGE-n and ROUGE-L metrics across all datasets from Table S8, separately for precision, recall and  $F_{1}$  score. Both 2-token and 4-token prediction models outperform the next-token prediction baseline. Trained on 500B tokens, 4-token prediction models appear better at recall metrics while 2-token prediction models appear better at precision metrics. Model checkpoints are selected as described in Table S8. Boldface for numbers within 0.05 difference to the best one for each dataset size separately.

<table><tr><td>Metric</td><td>Aspect</td><td>Baseline 200B</td><td>Δn=2</td><td>Δn=4</td><td>Baseline 500B</td><td>Δn=2</td><td>Δn=4</td></tr><tr><td rowspan="3">ROUGE-1</td><td>F1</td><td>33.77</td><td>+0.70</td><td>+0.68</td><td>34.77</td><td>+0.39</td><td>+0.41</td></tr><tr><td>precision</td><td>35.76</td><td>+0.88</td><td>+0.83</td><td>37.03</td><td>+0.42</td><td>-0.04</td></tr><tr><td>recall</td><td>34.37</td><td>+0.45</td><td>+0.45</td><td>35.14</td><td>+0.35</td><td>+0.68</td></tr><tr><td rowspan="3">ROUGE-2</td><td>F1</td><td>16.06</td><td>+0.36</td><td>+0.39</td><td>16.82</td><td>+0.29</td><td>+0.30</td></tr><tr><td>precision</td><td>16.97</td><td>+0.40</td><td>+0.43</td><td>17.91</td><td>+0.29</td><td>+0.03</td></tr><tr><td>recall</td><td>16.34</td><td>+0.28</td><td>+0.35</td><td>16.99</td><td>+0.32</td><td>+0.48</td></tr><tr><td rowspan="3">ROUGE-3</td><td>F1</td><td>9.08</td><td>+0.26</td><td>+0.23</td><td>9.54</td><td>+0.18</td><td>+0.22</td></tr><tr><td>precision</td><td>9.59</td><td>+0.29</td><td>+0.28</td><td>10.17</td><td>+0.18</td><td>+0.05</td></tr><tr><td>recall</td><td>9.26</td><td>+0.21</td><td>+0.20</td><td>9.65</td><td>+0.21</td><td>+0.35</td></tr><tr><td rowspan="3">ROUGE-L</td><td>F1</td><td>26.23</td><td>+0.51</td><td>+0.46</td><td>27.08</td><td>+0.28</td><td>+0.31</td></tr><tr><td>precision</td><td>27.79</td><td>+0.62</td><td>+0.55</td><td>28.85</td><td>+0.28</td><td>-0.09</td></tr><tr><td>recall</td><td>26.71</td><td>+0.37</td><td>+0.32</td><td>27.40</td><td>+0.28</td><td>+0.57</td></tr><tr><td rowspan="3">ROUGE-Lsum</td><td>F1</td><td>27.53</td><td>+0.52</td><td>+0.48</td><td>28.40</td><td>+0.29</td><td>+0.33</td></tr><tr><td>precision</td><td>29.07</td><td>+0.64</td><td>+0.58</td><td>30.15</td><td>+0.29</td><td>-0.08</td></tr><tr><td>recall</td><td>28.13</td><td>+0.35</td><td>+0.33</td><td>28.81</td><td>+0.29</td><td>+0.60</td></tr></table>

# I. Additional results on mathematical reasoning in natural language





Figure S13: Performance on the mathematical reasoning benchmark GSM8K (Cobbe et al., 2021). We evaluate pretrained next-token and multi-token prediction models trained on 200B and 500B tokens of natural language in 8-shot mode using nucleus sampling (Holtzman et al., 2020) with probability mass 0.95 and various sampling temperatures. Reported are the frequencies of the correct final answer to appear among  $k$  samples, for  $k = 1, 10, 100$ , estimated from 200 samples like in code generation benchmarks (Chen et al., 2021). After 200B tokens, the 2-token prediction model has a clear advantage over the next-token baseline but the order reverses after 500B tokens. The 4-token prediction model is worse throughout. We interpret this similarly to the findings in Section 4.1: the follow-your-nose chains-of-thought required for GSM8K may be difficult to learn from a limited amount of data, attesting to the data efficiency of multi-token prediction training. Once the correct circuits for correct autoregressive chains-of-thought in this domain have formed, however, multi-token prediction comes at a cost.


# J. Additional results on induction learning

Figure S14: Induction capability of  $n$ -token prediction models trained on higher-quality data. Shown is accuracy on the second token of two token names that have already been mentioned previously. Training on a 9:1 mix of a books dataset and the children story dataset, we observe that induction capability forms significantly earlier in training (not shown here) and to a higher degree. We believe that this is explained both because our evaluation dataset no longer contains out-of-distribution tokens (Section 4.1) and because the higher-quality data contained in the books dataset makes induction necessary earlier on (especially for small models, cf. Singh et al. (2023)). In particular, by enforcing the formation of induction capability in the model by means of the dataset – instead of the loss – the advantage of 2-token prediction models on this task disappears except for the smallest models: feature learning converts the task into a pure next-token prediction task.

# K. Additional results on algorithmic reasoning

We investigate the following computation-sharing hypothesis for explaining the efficacy of multi-token prediction as training loss.

The prediction difficulty of different tokens in natural text varies greatly. Some tokens may be the continuations of partial words that are uniquely determined from their preceding context without any effort, while others may require to predict theorem names in difficult mathematical proofs or the correct answer to an exam question. Language models with residual connections have been shown to refine their output token distribution with each successive layer, and can be trained with early exit strategies that spend variable amounts of computational resources per token position. Multi-token prediction losses explicitly encourage information-sharing between adjacent token positions and can thus be viewed as a method to learn allocating computational resources in language models more efficiently to the tokens that benefit most of it.

To check the truth of this hypothesis, we augment the polynomial arithmetic task from Section 4.2 with a varying number of pause tokens (Goyal et al., 2023) inserted between the question and a token that denotes the beginning of the answer. Pause tokens introduce additional computational resources that can be expended for computations that are expected to be useful later on in the sequence, in other words: to start thinking about the answer. According to the computation-sharing hypothesis, multi-token prediction models learn information-sharing and thus computation-sharing between token positions more easily, and may be better at making use of these additional computational resources than next-token prediction models are. In Figure S15, we show the evaluation results on the polynomial arithmetic task with a fixed number of pause tokens inserted both at training and evaluation time. Multi-token prediction models likewise outperform next-token prediction models on these task variants across task difficulties and model sizes. However, we do not see strong evidence of a widening or shrinking of this gap i.e. we cannot conclude from these experiments on the veracity of the computation-sharing hypothesis.

In Table S11, we report results from another experiment in the same spirit: by adding spaces and newlines to HumanEval and MBPP prompts, we add "pause tokens" in a somewhat natural way. According to these results, multi-token prediction models have a slight advantage at using this additionally provided compute, but the effect is marginal.

(a) 5 pause tokens

(b) 10 pause tokens  
Figure S15: Accuracy on a polynomial arithmetic task with varying number of operations per expression and pause tokens. We train and evaluate models on the polynomial arithmetic task described in Section 4.2, modified by the addition of pause tokens (Goyal et al., 2023): between the question and the equality sign that indicates the beginning of the answer, we add a constant number of pause tokens both in training and evaluation. For both a variant with five and with ten pause tokens, respectively, we observe comparable improvements from using multi-token prediction to the ones obtained in the case without pause tokens (Figure 8).

Table S11: Utilization of additional whitespace tokens in code benchmarks.  

<table><tr><td>Task</td><td>Whitespace</td><td>n = 1</td><td>n = 4</td></tr><tr><td>APPS/Intro</td><td>spaces + newline</td><td>+0.21</td><td>+0.34</td></tr><tr><td>APPS/Intro</td><td>newline</td><td>+0.79</td><td>+0.69</td></tr><tr><td>HumanEval</td><td>spaces + newline</td><td>-0.72</td><td>-0.16</td></tr><tr><td>HumanEval</td><td>newline</td><td>-0.26</td><td>+0.10</td></tr><tr><td>MBPP</td><td>spaces + newline</td><td>-0.10</td><td>-0.06</td></tr><tr><td>MBPP</td><td>newline</td><td>+0.03</td><td>-0.08</td></tr><tr><td>Average</td><td></td><td>-0.01</td><td>+0.14</td></tr></table>

Figure S16: Accuracy on a polynomial arithmetic task for two model sizes. We train and evaluate models with 30M and 100M parameters on the polynomial arithmetic task described in Section 4.2. Tripling the model size has a smaller effect on performance than replacing next-token prediction loss by multi-token prediction. Shown are two independent runs per configuration and their means, the 100M parameter models being identical to the ones in Figure 8.

Table S12: Optimal temperatures for all numbers in table 1  

<table><tr><td rowspan="2">Training data</td><td rowspan="2">Vocabulary</td><td rowspan="2">n</td><td colspan="3">MBPP</td><td colspan="3">HumanEval</td><td colspan="3">APPS/Intro</td></tr><tr><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td><td>@1</td><td>@10</td><td>@100</td></tr><tr><td rowspan="4">313B bytes (0.5 epochs)</td><td rowspan="4">bytes</td><td>1</td><td>0.2</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>16</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>32</td><td>0.1</td><td>0.4</td><td>0.8</td><td>0.1</td><td>0.4</td><td>0.8</td><td>0.1</td><td>0.4</td><td>0.4</td></tr><tr><td rowspan="4">200B tokens (0.8 epochs)</td><td rowspan="4">32k tokens</td><td>1</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.4</td><td>0.8</td></tr><tr><td>2</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.2</td><td>0.8</td><td>0.8</td><td>0.4</td><td>0.4</td><td>0.8</td></tr><tr><td>4</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.2</td><td>0.8</td><td>0.8</td></tr><tr><td>6</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.2</td><td>0.8</td><td>0.8</td><td>0.4</td><td>0.4</td><td>0.8</td></tr><tr><td rowspan="3">1T tokens (4 epochs)</td><td rowspan="3">32k tokens</td><td>8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.2</td><td>0.4</td><td>0.8</td></tr><tr><td>1</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.1</td><td>0.4</td><td>0.8</td></tr><tr><td>4</td><td>0.1</td><td>0.8</td><td>0.8</td><td>0.2</td><td>0.8</td><td>0.8</td><td>0.4</td><td>0.8</td><td>0.8</td></tr></table>

# L. Additional intuitions on multi-token prediction

# L.1. Comparison to scheduled sampling

In Section 5.2, we argued that multi-token prediction reduces the distribution mismatch between teacher-forced training and autoregressive evaluation of language models. Scheduled sampling (Bengio et al., 2015) is a curriculum learning method that likewise aims to bridge this gap in sequence prediction tasks by gradually replacing more and more input tokens with model-generated ones.

While effective in areas such as time series forecasting, scheduled sampling is, in our opinion, inapplicable to language modelling due to the discrete nature of text. Replacing ground truth input sequences by interleavings of ground truth and model-generated tokens frequently results in ungrammatical, factually wrong or otherwise incoherent text, which should be avoided at all cost. Moreover, unlike multi-token prediction, the technique originally developed for recurrent neural networks cannot easily be adapted for parallel training setups like the ones of transformer models.

# L.2. Information-theoretic argument

We give details on the information-theoretic terms appearing in the decomposition in Section 5.2 and derive a relative version that similarly allows to decompose multi-token prediction losses. As in Section 5.2, denote by  $X$  the next token and by  $Y$  the second-  $n$  one, and omit conditioning on the preceding context  $C$  for ease of notation. In Section 5.2, we decomposed  $H(X) + H(Y)$  —the quantity of interest for 2-token prediction models—as follows:

$$
H (X) + H (Y) = H (X \mid Y) + 2 I (X; Y) + H (Y \mid X). \tag {3}
$$

Let us explain each of the terms. The entropy terms denote the uncertainty contained in the ground-truth random variables  $X$  and  $Y$ . The term  $H(Y \mid X)$  is a classical next-token entropy for the prefix  $(C, X)$ . The conditional entropy  $H(X \mid Y)$  is a more theoretical entity not modelled by causal models. It describes the uncertainty about  $X$  given the prefix  $C$  and suffix  $Y$ , and therefore captures the local variations of  $X$  that do not affect the continuation of the text  $Y$ . The mutual information  $I(X; Y)$  on the other hand describes the information about  $Y$  contained in  $X$  (and vice versa) and therefore captures the variations of  $X$  which constrain the continuation of the text.

However, the argument given in Section 5.2 relies on the assumption that multi-token prediction losses obey a similar decomposition as the sum of the ground-truth entropies themselves. Let us make this rigorous. Denote by  $p(x,y)$  the joint distribution of  $X$  and  $Y$ , by  $p(x)$  (short for  $p_X(x)$ ) the marginal distribution of  $X$  and by  $p(y)$  the one of  $Y$ . Denote the densities of the model's predictions by  $q(x,y)$ ,  $q(x)$  and  $q(y)$ , respectively, conditional distributions by  $p(x|y)$  and Kullback-Leibler divergence from  $q$  to  $p$  by  $D(p\parallel q)$  and cross-entropy from  $q$  to  $p$  by  $H(p,q)$ .

Definition L.1. The conditional cross-entropy  $H(p_{X|Y}, q_{X|Y})$  of  $X$  conditioned on  $Y$  from  $q$  to  $p$  is defined as the

expectation under  $y$  of the cross-entropy between the distributions  $p_X$  and  $q_{X}$  conditioned on  $y$ , in formulas:

$$
H (p _ {X | Y}, q _ {X | Y}) = \underset {y \sim p _ {Y}} {\mathbb {E}} H (p _ {X | Y = y}, q _ {X | Y = y}) = \underset {y \sim p _ {Y}} {\mathbb {E}} H (p (\cdot \mid y), q (\cdot \mid y)).
$$

Definition L.2. The relative mutual information  $I_{p\parallel q}(X;Y)$  of  $X$  and  $Y$  from  $q$  relative to  $p$  is defined by

$$
I _ {p \| q} (X; Y) = D (p \parallel q _ {X} \otimes q _ {Y}) - D (p \parallel q).
$$

We have  $I_{p\|q}(X;Y) = H(p_X, q_X) + H(p_Y, q_Y) - H(p, q), I_{p\|p}(X;Y) = I_p(X;Y)$  reduces to standard mutual information under the distribution  $p$  and  $I_{p\|q}(X;Y)$  is symmetric in  $X$  and  $Y$  but can be negative.

We have the following relative version of the decomposition  $H(X) = H(X \mid Y) + I(X;Y)$ .

Lemma L.3.  $H(p_{X},q_{X}) = H(p_{X|Y},q_{X|Y}) + I_{p\parallel q}(X;Y).$

Proof. We calculate

$$
\begin{array}{l} H \left(p _ {X}, q _ {X}\right) = - \sum_ {x} p (x) \log q (x) \\ = - \sum_ {x, y} p (x, y) \log q (x) \\ = - \sum_ {x, y} p (x, y) \log \frac {q (x) q (y)}{p (x , y)} \frac {p (x , y)}{q (x , y)} \frac {q (x , y)}{q (y)} \\ = D (p \parallel q _ {X} \otimes q _ {Y}) - D (p \parallel q) - \sum_ {x, y} p (y) p (x \mid y) \log q (x \mid y) \\ = I _ {p \parallel q} (X; Y) + \sum_ {y} p (y) H \left(p _ {X \mid y}, q _ {Y \mid y}\right) \\ = I _ {p \parallel q} (X; Y) + H \left(p _ {X \mid Y}, q _ {X \mid Y}\right). \\ \end{array}
$$


Symmetrizing, we get the desired relative version of  $H(X) + H(Y) = H(X \mid Y) + 2I(X;Y) + H(Y \mid X)$ :

$$
H \left(p _ {X}, q _ {X}\right) + H \left(p _ {Y}, q _ {Y}\right) = H \left(p _ {X \mid Y}, q _ {X \mid Y}\right) + 2 I _ {p \parallel q} (X; Y) + H \left(p _ {Y \mid X}, q _ {Y \mid X}\right).
$$

Setting  $p$  to be the empirical distribution of the training data, the left-hand side describes the cross-entropy loss used to train 2-token prediction models. The right-hand side gives the decomposition into a local cross-entropy term, a mutual information term with weight two and a shifted next-token cross-entropy term. We interpret this as follows: by adding the term  $H(p_{Y}, q_{Y})$  to the loss, 2-token prediction incentivizes models to precompute features which will become useful for predicting  $Y$  in the next step and increases the weight of the relative mutual information term in the loss. What does relative mutual information actually mean? By interpreting Kullback-Leibler divergence  $D(p \parallel q)$  as the average number of bits needed in addition to send data from  $p$  with a code optimized for  $q$  instead of  $p$ , we see that minimizing

$$
I _ {p \parallel q} (X; Y) = D (p \parallel q _ {X} \otimes q _ {Y}) - D (p \parallel q)
$$

means minimizing the average number of additional bits needed to send data from  $p$  with a code optimized for  $q$  that treats  $X$  and  $Y$  as independent compared to one that does not. If this number is small,  $q$  managed to exploit the mutual information of  $X$  and  $Y$  under  $p$ .

# L.3. Lookahead reinforces choice points

Training with multi-head prediction increases the importance of choice points in the loss in comparison to inconsequential decisions. To make this argument, we present a simplified model of language modelling. Consider a sequential decision task and a model  $M$  that is trained in a teacher-forced way on optimal trajectories. We distinguish choice points –transitions that lead to different outcomes – and inconsequential decisions which do not (Figure S17 (a) and (b)).

Figure S17: Example of a sequential prediction task with derailing. The goal is to go from the arrow to the trophy. Turning around is not allowed. Most transitions are unique, but there are two turns to be taken correctly, the consequential decisions (a) and (c). Turn (b) is an inconsequential decision: the paths join right after it. Next to transitions (a) and (b), we sketch how a 4-step prediction loss can place more emphasis on consequential transitions than inconsequential ones during teacher-forced training. Next to transition (c), we sketch how a 4-step lookahead can prevent models from taking irreversible suboptimal decisions during autoregressive decoding.

More formally, assume that the language model is deployed in a reinforcement learning setting like in reinforcement learning from human feedback (Ouyang et al., 2022) (states are prompts followed by the partial sequence of tokens  $x_{t:1}$  generated so far, actions are single tokens  $x_{t+1}$  to generate, rewards are external  $R(x_{t:1})$ ). The quantity

$$
V _ {\pi} \left(x _ {t: 1}\right) = \mathbb {E} _ {x _ {t + i} \sim \pi \left(x _ {t + i - 1: 1}\right), i \geq 1} \left[ \sum_ {i \geq 0} R \left(x _ {t + i: 1}\right) \right]
$$

is the value of the state  $x_{t:1}$  following the policy  $\pi$ , while

$$
\sigma_ {\pi} \left(x _ {t: 1}\right) = \sqrt {\operatorname {V a r} _ {x _ {t + 1} \sim \pi \left(x _ {t : 1}\right)} \left[ V _ {\pi} \left(x _ {t + 1 : 1}\right) \right]}
$$

quantifies the importance of the decision  $x_{t+1}$  on the value thereafter. Choice points can formally be viewed as steps  $t$  for which  $\sigma_{\pi}(x_{t:1})$  is large, while inconsequential points are steps where it is low. Note that for completion models, there is no explicit reward, and our argument is merely meant to illustrate what we mean by choice points.

Derailing denotes a situation where autoregressive generation of trajectories from  $M$  at inference time results in bad outcomes after  $M$  made a mistake on a choice point. Even if subsequently,  $M$  acts optimally given this choice, the final outcome can be significantly worse than the outcome of the optimal trajectory.

Staying in the teacher-forced setting, we ask: What is the impact of training  $M$  with  $n$ -step prediction instead of next-step prediction on this task? Say  $x_{t} \rightarrow x_{t+1}$  is a choice point in an optimal trajectory with the suboptimal choice being  $x_{t} \rightarrow \tilde{x}_{t+1}$  (Figure S17 (a)). Assume that the trajectories preceding  $x_{t}$  and succeeding  $x_{t+1}$  and  $\tilde{x}_{t+1}$  consist of inconsequential transitions, the latter denoted by  $\tilde{x}_{t+j} \rightarrow \tilde{x}_{t+j+1}$ . We will compare the losses of a teacher-forced next-step prediction model and a teacher-forced  $n$ -step prediction model on the partial trajectory  $(x_{t-n+1}, \ldots, x_{t})$ . For the next-step prediction model, the predictions are  $(x_{t-n+2}, \ldots, x_{t}, \tilde{x}_{t+1})$  with a single wrong prediction. The predictions of an  $n$ -step prediction model at time  $t-n+i, i=1,\ldots,n$  are  $(x_{t-n+i+1}, \ldots, x_{t}, \tilde{x}_{t+1}, \ldots, \tilde{x}_{t+i})$  with  $i$  wrong predictions. In other words, an  $n$ -step prediction model receives  $1 + \ldots + n = \frac{n(n+1)}{2}$  loss terms pertaining to such a choice point and its consequences, while each inconsequential transition (Figure S17 (b)) is only reinforced  $n$  times as often as in a next-step prediction model. In other words, choice points receive on average  $\frac{n+1}{2}$  times more importance in the loss of  $n$ -step prediction models than in next-step prediction models.

As argued in Section 5.1, we believe that this model captures important features of training and inference with language models: choice points are semantically important turning points in the generated texts, such as the final answer to a question or a specific line of code, while inconsequential decisions can be a choice among synonyms or of variable names in code.

Apart from this training dynamics point of view, we hypothesize that  $n$ -step prediction also allows the formation of circuits that specifically spot inconsistencies between predictions for earlier and later steps. For instance, if in an early layer of the model, it can be predicted that a decision  $x_{t} \rightarrow \tilde{x}_{t+1}$  leads to suboptimal outcomes  $\tilde{x}_{t+n}$  (Figure S17 (c)), subsequent layers can reduce the probability of  $x_{t} \rightarrow \tilde{x}_{t+1}$  in the model's next-step prediction. Such behaviors also happen in next-step prediction models given enough capacity, but our experiments in Section 4.2 point to the fact that circuits of this kind are formed more easily in multi-step architectures that enforce the required information  $\tilde{x}_{t+n}$  to be available to the model when predicting  $\tilde{x}_{t+1}$ . We believe that this situation appears frequently in natural language and code modelling, for instance where an initial answer to a question contradicts the results of the chain of thought brought forward with the intention to justify it

In more general terms, this situation arises whenever predicting first  $\tilde{x}_{n + i}$  for some  $1 < i\leq n$  and then  $\tilde{x}_{n + 1}$  based on  $\tilde{x}_{n + i}$  is easier than predicting  $\tilde{x}_{n + 1}$  directly. We discuss this phenomenon of factorization orders in the next section and present a specific instance of it that frequently appears in modelling natural language.

# L.4. Factorization orders

Causal language modelling factorizes probabilities over text sequences  $x_{t} \cdots x_{1}$  classically as

$$
P \left(x _ {t} \dots x _ {1}\right) = \prod_ {i = 1} ^ {t} P \left(x _ {i} \mid x _ {i - 1} \dots x _ {1}\right).
$$

While moving forward in time is certainly the most natural choice of factorization order, there exist cases where it is suboptimal. In inflectional languages, for instance, agreement between related sentence parts is a frequent pattern with one word directing the grammatical forms of others. Consider the German sentence

Wie konnten auch Worte了我的er durstenden Seele genugen?3

Friedrich Hölderlin, Fragment von Hyperion (1793)

where "genügen" requires a dative case object and then "Seele" requires the possessive pronoun "mein" to be in female singular dative form "meiner" and the participle "durstend" to be in female singular dative form in weak declination "durstenden" because it follows "meiner". In other words, the factorization order

Wie konnten auch Worte  $\rightarrow$  genügen  $\rightarrow$  Seele  $\rightarrow$ /meiner  $\rightarrow$  durstenden?

is arguably an easier one for constructing the above sentence. Humans as well as language models therefore have to perform this factorization (which deviates from the causal order in which predictions take place!) within their latent activations, and a 4-token prediction loss makes this easier as it explicitly encourages models to have all information about the successive 4 tokens in its latent representations.

# M. Training hyperparameters

Table S13: Overview of all training hyperparameters used. We schedule all learning rates with a linear warmup and cosine decay (Loshchilov and Hutter, 2017) to a fraction of the peak learning rate which is depicted in the last column ("decay ratio"). All experiments use the Adam (Kingma and Ba, 2015) optimizer with  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.95$  and decoupled  $L_{2}$  weight decay (Loshchilov and Hutter, 2019) coefficient 0.1. We clip gradients to a maximal Euclidean norm of 1.0 in all experiments except CodeContests finetunings, where we use 0.1 instead. Summarization finetunings correspond to three epochs on all datasets except BigPatent (1 epoch). Byte-level models use the architecture with replicated unembeddings from Appendix B.

<table><tr><td>Model</td><td>Batch size (220)</td><td>Steps</td><td>Tokens (B)</td><td>Warmup steps</td><td>Peak LR</td><td>Context length</td><td>Decay ratio</td></tr><tr><td colspan="8">Model scaling (Section 3.1)</td></tr><tr><td>0.3B</td><td>8</td><td>10,850</td><td>91.0</td><td>1000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>0.6B</td><td>8</td><td>10,850</td><td>91.0</td><td>1000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>1.3B</td><td>8</td><td>10,850</td><td>91.0</td><td>1000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>3B</td><td>8</td><td>10,850</td><td>91.0</td><td>1000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>7B</td><td>8</td><td>25,000</td><td>209.7</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>13B</td><td>8</td><td>25,000</td><td>209.7</td><td>1000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td colspan="8">Code models (Section 3)</td></tr><tr><td>7B 200B</td><td>8</td><td>25,000</td><td>209.7</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>7B 500B</td><td>7</td><td>68,570</td><td>503.3</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td>7B 1T</td><td>7</td><td>136,240</td><td>1000.0</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.03</td></tr><tr><td colspan="8">Byte-level models (Section 3.3)</td></tr><tr><td>7B 314GB</td><td>12</td><td>25,000</td><td>314.6</td><td>2000</td><td>3 × 10-4</td><td>8192</td><td>0.03</td></tr><tr><td colspan="8">Language models (Section 3.7)</td></tr><tr><td>7B 200B</td><td>8</td><td>25,000</td><td>209.7</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.10</td></tr><tr><td>7B 500B</td><td>8</td><td>60,000</td><td>503.3</td><td>2000</td><td>3 × 10-4</td><td>4096</td><td>0.10</td></tr><tr><td colspan="8">Induction task (Section 4.1)</td></tr><tr><td>1M - 1B</td><td>0.25</td><td>100,000</td><td>26.2</td><td>2000</td><td>10-4</td><td>2048</td><td>0.03</td></tr><tr><td>1M - 1B (Appendix J)</td><td>0.5</td><td>50000</td><td>26.2</td><td>2000</td><td>10-4</td><td>2048</td><td>0.03</td></tr><tr><td colspan="8">Arithmetic task (Section 4.2)</td></tr><tr><td>30M</td><td>0.25</td><td>100,000</td><td>26.2</td><td>2000</td><td>10-4</td><td>1024</td><td>0.03</td></tr><tr><td>100M</td><td>0.25</td><td>100,000</td><td>26.2</td><td>2000</td><td>10-4</td><td>2048</td><td>0.03</td></tr><tr><td colspan="8">Summarization (Section 3.7)</td></tr><tr><td>BigPatent</td><td>0.125</td><td>76,680</td><td>10.1</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>CNN/Dailymail</td><td>0.125</td><td>7,140</td><td>0.9</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>Multi-News</td><td>0.125</td><td>3,330</td><td>0.4</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>OrangeSum</td><td>0.125</td><td>360</td><td>0.0</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>pn-summary</td><td>0.125</td><td>3,450</td><td>0.5</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>SAMSum</td><td>0.125</td><td>60</td><td>0.0</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>ThaiSum</td><td>0.125</td><td>23,640</td><td>3.1</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>WikiSummary</td><td>0.125</td><td>2,550</td><td>0.3</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td>XSum</td><td>0.125</td><td>2,760</td><td>0.4</td><td>100</td><td>3 × 10-5</td><td>4096</td><td>0.03</td></tr><tr><td colspan="8">CodeContests (Section 3.6)</td></tr><tr><td>7B</td><td>0.25</td><td>13,000</td><td>3.6</td><td>400</td><td>5 × 10-5</td><td>4096</td><td>0.004</td></tr></table>

Table S14: Overview of model architectures used for scaling analyses.  

<table><tr><td>Name</td><td>Dimension</td><td>Layers</td><td>Heads</td></tr><tr><td>1M</td><td>128</td><td>5</td><td>4</td></tr><tr><td>3M</td><td>256</td><td>4</td><td>8</td></tr><tr><td>10M</td><td>384</td><td>6</td><td>8</td></tr><tr><td>30M</td><td>512</td><td>10</td><td>8</td></tr><tr><td>100M</td><td>768</td><td>14</td><td>12</td></tr><tr><td>300M</td><td>1024</td><td>25</td><td>16</td></tr><tr><td>1B</td><td>1536</td><td>36</td><td>24</td></tr><tr><td>0.3B</td><td>1024</td><td>18</td><td>16</td></tr><tr><td>0.6B</td><td>1280</td><td>27</td><td>20</td></tr><tr><td>1.3B</td><td>2048</td><td>24</td><td>16</td></tr><tr><td>3B</td><td>2560</td><td>36</td><td>20</td></tr><tr><td>6.7B (&quot;7B&quot;)</td><td>4096</td><td>32</td><td>32</td></tr><tr><td>13B</td><td>5120</td><td>40</td><td>40</td></tr></table>

# Footnotes:

Page 5: <sup>1</sup>Note that a perfect score is not reachable in this benchmark as some of the tokens in the names in the evaluation dataset never appear in the training data, and in our architecture, embedding and unembedding parameters are not linked. 
Page 23: 2In particular, they do not refer to model predictions. 
Page 26: <sup>3</sup>roughly: How could words be enough for my thirsty soul? 
