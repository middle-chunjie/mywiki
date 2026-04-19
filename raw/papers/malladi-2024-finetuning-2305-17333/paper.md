# Fine-Tuning Language Models with Just Forward Passes

Sadhika Malladi*

Tianyu Gao*

Eshaan Nichani

Alex Damian

Jason D. Lee

Danqi Chen

Sanjeev Arora

Princeton University

{smalladi, tianyug, eshnich, ad27, jasonlee, danqic, arora}@princeton.edu

# Abstract

Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same budget. We conduct comprehensive experiments across model types (masked and autoregressive LMs), model scales (up to 66B), and downstream tasks (classification, multiple-choice, and generation). Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear probing; (2) MeZO achieves comparable performance to fine-tuning with backpropagation across multiple tasks, with up to  $12 \times$  memory reduction and up to  $2 \times$  GPU-hour reduction in our implementation; (3) MeZO is compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix tuning; (4) MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1). We support our empirical findings with theoretical insights, highlighting how adequate pre-training and task prompts enable MeZO to fine-tune huge models, despite classical ZO analyses suggesting otherwise. $^{1}$

# 1 Introduction

Fine-tuning pre-trained language models (LMs) has been the dominant methodology for solving many language tasks [28], adapting to specialized domains [42], or incorporating human instructions and preferences [73]. However, as LMs are scaled up [13, 72], computing gradients for backpropagation requires a prohibitive amount of memory – in our test, up to  $12 \times$  the memory required for inference – because it needs to cache activations during the forward pass, gradients during the backward pass, and, in the case of Adam [52], also store gradient history (see Section 3.4 for a detailed analysis).

As a result, while it is possible to run inference with a 30-billion (30B) parameter LM on a single Nvidia A100 GPU (with 80GB memory), backpropagation with Adam is feasible only for a 2.7B LM. Parameter-efficient fine-tuning methods (PEFT [46, 57, 54]) update just a fraction of the network

Figure 1: OPT-13B results with zero-shot, in-context learning (ICL), MeZO (we report the best among MeZO/MeZO (LoRA)/MeZO (prefix)), and fine-tuning with Adam (FT). MeZO demonstrates superior results over zero-shot and ICL and performs on par with FT (within  $1\%$  ) on 7 out of 11 tasks, despite using only 1/12 memory. See Table 1 for detailed numbers and Figure 3 for memory profiling.

parameters, but still need to cache many activations, because the tuned parameters are scattered throughout the model. In our tests, fine-tuning an OPT-13B model with full parameter tuning or PEFT requires  $12 \times$  and  $6 \times$  more memory than inference respectively.

In-context learning (ICL [13]) has allowed solving many tasks with a single inference pass, during which the model processes labeled examples (demonstrations) in its context and then outputs a prediction on a test example. While this allows for quick adaptation of the model to specific use cases, current models allow a limited context size (and thus, limited demonstrations) and the performance is sensitive to the formatting and choice of demonstrations [60, 66]. ICL can slow with the number of demonstrations, and it often performs worse than fine-tuning of medium-sized models [13].

Backpropagation also cannot optimize non-differentiable criteria, which have gained popularity in fine-tuning LMs according to human preference scores or set safety standards [89, 73]. Typically, these adaptations involve expensive reinforcement learning from human feedback (RLHF [20]).

A classical zeroth-order optimization method, ZO-SGD [88], uses only differences of loss values to estimate the gradients. Thus, in principle, the method can update neural networks with just forward passes, though naive implementation still doubles the memory overhead and classical lower bounds [69, 32] suggest that convergence slows linearly with model size. As such, ZO methods have been applied in deep learning settings to find adversarial examples or tune input embeddings [91, 90] but not to directly optimize large-scale models (see Liu et al. [61] for a survey).

In this work, we propose a memory-efficient zeroth-order optimizer (MeZO), which adapts the classical ZO-SGD algorithm and reduces its memory consumption to the same as inference. We apply MeZO to fine-tune large LMs and show that, both empirically and theoretically, MeZO can successfully optimize LMs with billions of parameters. Specifically, our contributions are:

1. In MeZO, we adapt the ZO-SGD algorithm [88] and a number of variants to operate in-place on arbitrarily large models with almost no memory overhead (see Algorithm 1 and Section 2).  
2. We conduct comprehensive experiments across model types (masked LM and autoregressive LM), model scales (from 350M to 66B), and downstream tasks (classification, multiple-choice, and generation). MeZO consistently outperforms zero-shot, ICL, and linear probing. Moreover, with RoBERTa-large, MeZO achieves performance close to standard fine-tuning within  $5\%$  gap; with OPT-13B, MeZO outperforms or performs comparably to fine-tuning on 7 out of 11 tasks, despite requiring roughly  $12 \times$  less memory (Figure 1 and Section 3). In our implementation, MeZO requires only half as many GPU-hours as Adam fine-tuning for a 30B model (see Appendix F.6).  
3. We demonstrate MeZO's compatibility with full-parameter tuning and PEFT (e.g., LoRA [46] and prefix-tuning [57]) in Section 3.  
4. Further exploration showcases that MeZO can optimize non-differentiable objectives such as accuracy or F1 score, while still requiring only the same memory as inference (Section 3.3).  
5. Our theory suggests that adequate pre-training ensures the per-step optimization rate (Theorem 1) and global convergence rate (Lemma 3) of MeZO depend on a certain condition number of the landscape (i.e., the local effective rank, see Assumption 1) instead of numbers of parameters. This

Algorithm 1: MeZO  
Require: parameters  $\theta \in \mathbb{R}^d$  , loss  $\mathcal{L}:\mathbb{R}^d\to \mathbb{R}$  , step budget  $T$  , perturbation scale  $\epsilon$  , batch size  $B$  , learning rate schedule  $\{\eta_t\}$    
for  $t = 1,\dots,T$  do Sample batch  $\mathcal{B}\subset \mathcal{D}$  and random seed s  $\theta \leftarrow$  PerturbParameters(0,ε,s)  $\ell_{+}\gets \mathcal{L}(\theta ;\mathcal{B})$ $\theta \leftarrow$  PerturbParameters(0,-2ε,s)  $\ell_{-}\leftarrow \mathcal{L}(\theta ;\mathcal{B})$ $\theta \leftarrow$  PerturbParameters(0,ε,s) Reset parameters before descent projected_grad  $\leftarrow (\ell_{+} - \ell_{-}) / (2\epsilon)$  Reset random number generator with seed s For sampling z for  $\theta_{i}\in \theta$  do  $z\sim \mathcal{N}(0,1)$ $\theta_{i}\gets \theta_{i} - \eta_{t}*$  projected_grad  $^* z$    
end   
Subroutine PerturbParameters(0,ε,s) Reset random number generator with seed s For sampling z for  $\theta_{i}\in \theta$  do  $z\sim \mathcal{N}(0,1)$ $\theta_{i}\gets \theta_{i} + \epsilon z$  Modify parameters in place   
end return  $\theta$

result is in sharp contrast to existing ZO lower bounds [69, 32] suggesting that the convergence rate can slow proportionally to the number of parameters (Section 4).

# 2 Zeroth-order optimization

Zeroth-order (ZO) optimizers have long been studied in the context of convex and strongly convex objectives. In the following, we first introduce a classical ZO gradient estimator, SPSA (Definition 1 [88]) and the corresponding SGD algorithm, ZO-SGD (Definition 2). Then we describe MeZO, our in-place implementation that requires the same memory as inference in Section 2.1 and Algorithm 1. We highlight that SPSA can also be used in more complex optimizers, such as Adam, and we provide memory-efficient implementations for those algorithms too (Section 2.2).

Consider a labelled dataset  $\mathcal{D} = \{(x_i, y_i)\}_{i \in [|\mathcal{D}|]}$  and a minibatch  $\mathcal{B} \subset \mathcal{D}$  of size  $B$ , we let  $\mathcal{L}(\boldsymbol{\theta}; \mathcal{B})$  denote the loss on the minibatch. We introduce a classical ZO gradient estimate in this setting.

Definition 1 (Simultaneous Perturbation Stochastic Approximation or SPSA [88]). Given a model with parameters  $\pmb{\theta} \in \mathbb{R}^d$  and a loss function  $\mathcal{L}$ , SPSA estimates the gradient on a minibatch  $\mathcal{B}$  as

$$
\widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) = \frac {\mathcal {L} (\boldsymbol {\theta} + \epsilon z ; \mathcal {B}) - \mathcal {L} (\boldsymbol {\theta} - \epsilon z ; \mathcal {B})}{2 \epsilon} z \approx z z ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \tag {1}
$$

where  $\mathbf{z} \in \mathbb{R}^d$  with  $\mathbf{z} \sim \mathcal{N}(0, I_d)$  and  $\epsilon$  is the perturbation scale. The  $n$ -SPSA gradient estimate averages  $\widehat{\nabla}\mathcal{L}(\boldsymbol{\theta};\mathcal{B})$  over  $n$  randomly sampled  $\mathbf{z}$ .

SPSA requires only two forward passes through the model to compute the gradient estimate (for  $n$ -SPSA, each estimate requires  $2n$  forward passes). As  $\epsilon \to 0$ , the SPSA estimate can be understood as a rank-1 reconstruction of the gradient. During training,  $n$  can be treated as a hyperparameter and follow a schedule [11, 15], though in cursory experiments (Appendix A),  $n = 1$  is the most efficient.

We use  $n = 1$  as the default. It is widely known that the SPSA estimate can be used to replace the backpropagation gradient in any optimizer such as SGD.

Definition 2 (ZO-SGD). ZO-SGD is an optimizer with learning rate  $\eta$  that updates parameters as  $\pmb{\theta}_{t + 1} = \pmb{\theta}_t - \eta \widehat{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B}_t)$  where  $\mathcal{B}_t$  is the minibatch at time  $t$  and  $\widehat{\nabla}\mathcal{L}$  is the SPSA gradient estimate.

# 2.1 Memory-efficient ZO-SGD (MeZO)

The vanilla ZO-SGD algorithm costs twice the memory of inference, as it needs to store  $z \in \mathbb{R}^d$ . We propose a memory-efficient implementation of ZO-SGD called MeZO, as illustrated in Algorithm 1. At each step, we first sample a random seed  $s$ , and then for each of  $z$ 's four uses in Algorithm 1, we reset the random number generator by  $s$  and resample the relevant entry of  $z$ . Using this in-place implementation, MeZO has a memory footprint equivalent to the inference memory cost.

We note that Algorithm 1 describes perturbing each parameter separately, which may be time-consuming for large models. In practice, we can save time by perturbing an entire weight matrix instead of each scalar independently. This incurs an additional memory cost as large as the largest weight matrix; usually, this is the word embedding matrix (e.g., 0.86GB for OPT-66B).

Storage Efficiency of MeZO. Parameter-efficient fine-tuning (PEFT) techniques fine-tune just a fraction of the network parameters and have thus been proposed as a way to reduce the storage costs of fine-tuned model checkpoints. Fine-tuning with MeZO reduces the storage cost of the resulting checkpoint far more than popular PEFT techniques (e.g., LoRA [46] and prefix tuning [57]). We reconstruct the MeZO trajectory using a single seed, which spawns step-wise seeds to sample  $z$ , and the projected_grad at each step. As such, for fine-tuning a 66B model, MeZO requires saving the seed plus 20,000 (steps) × 2 bytes, which is less than 0.1MB. LoRA fine-tunes 19M parameters and requires 38MB storage, and prefix tuning fine-tunes 6M parameters and requires 12MB storage.

# 2.2 MeZO extensions

We note that SPSA is a popular ZO gradient estimator but not the only one. Many one-point gradient estimators have been proposed in past works [34, 87, 95], and using such estimators in place of SPSA would halve the training time. However, cursory experiments with one such promising estimator [113] reveal that these are not as efficient as SPSA when fixing the number of forward passes (Appendix B.5). As such, we implement MeZO with the SPSA estimator.

MeZO can also be combined with other gradient-based optimizers, including SGD with momentum or Adam. Though naive implementation would require additional memory to store the gradient moment estimates, MeZO-momentum and MeZO-Adam alleviate such overhead by recomputing the moving average of the gradients using saved past losses and  $z$  (see Appendix B for a full discussion).

We also note that all of the coordinates of the SPSA gradient estimate have the same scale, but deep Transformers can have gradients of different scales for each layer [59, 61]. As such, we draw inspiration from layerwise adaptive optimizers [109, 110] to design several MeZO variants. Cursory experiments showed that these algorithms are not more efficient (in terms of forward passes), but we nevertheless present them as potential optimizers for more complex objectives. See Appendix B.

Forward Auto-Differentiation Note that  $z^\top \nabla \mathcal{L}(\theta; \mathcal{B})$  is a Jacobian-vector product (JVP), which can be computed in parallel with an inference pass with excess memory consumption equivalent to that of the largest activation in the network [40]. In this case,  $z$  must be stored on the GPU in order to construct the gradient estimate, so this procedure requires slightly more than two times the memory needed for inference. We analyze this algorithm in detail in Appendix D. Note that using a non-zero  $\epsilon$  in SPSA, which is not possible through the JVP method, may boost generalization by promoting a sharpness-minimizing term. Past works (e.g., Baydin et al. [9]) have also studied JVP-based training but achieved limited empirical success.

Figure 2: Experiments on RoBERTa-large. We report zero-shot, linear probing (LP), and MeZO and fine-tuning (FT) with full parameter, LoRA, and prefix-tuning. MeZO outperforms zero-shot and LP and approaches FT (within  $5\%$  for  $k = 512$ ) with much less memory. Detailed numbers in Table 18.

# 3 Experiments

Preliminary experiments (Appendix A) show that MeZO only works when using prompts [13, 84, 35]. Past works [83, 67] have demonstrated how the inclusion of a suitable prompt ensures the fine-tuning objective is closely related to the pre-training one. In Section 4, we extend these ideas to show how using a simple prompt simplifies the fine-tuning optimization procedure, thereby enabling zeroth order methods to work efficiently. All experiments below use prompts detailed in Appendix E.2. All fine-tuning with backpropagation (FT) experiments follow convention and use Adam, though we also report results when performing FT with SGD in Appendix F.

We conduct comprehensive experiments on both medium-sized masked LMs (RoBERTa-large, 350M [65]) and large autoregressive LMs (OPT-13B, 30B, 66B [112]) in few-shot and many-shot settings with prompts. We also explore both full-parameter tuning and PEFT including LoRA [46] and prefix-tuning [57] (see Appendix E.5 for details). We compare MeZO with zero-shot, in-context learning (ICL), linear-probing (LP), and fine-tuning with Adam (FT). MeZO uses substantially less memory than FT but requires significantly more training steps.

We first show that MeZO improves substantially over zero-shot, ICL, and LP across model types, sizes, and task types. Moreover, MeZO performs comparably to FT over a number of tasks, while drastically reducing the memory cost by, for example,  $12 \times$  on OPT-13B. Further experiments demonstrate that MeZO can optimize non-differentiable objectives, such as accuracy and F1 score (Section 3.3). We compare the memory consumption of ICL, FT, LP, and MeZO in Figures 3 and 4.

# 3.1 Medium-sized masked language models

We conduct experiments with RoBERTa-large on sentiment classification, natural language inference, and topic classification tasks. We follow past works [35, 67] in studying the few-shot and many-shot settings, sampling  $k$  examples per class for  $k = 16$  and  $k = 512$  (details in Appendix E). We run MeZO for 100K steps and fine-tuning for 1000 steps, noting that one MeZO step is substantially faster than one fine-tuning step (see Appendix F.6 for a comparison). We summarize the results from Figure 2 and Table 18 below.

MeZO works significantly better than zero-shot, linear probing, and other memory-equivalent methods. On all six diverse tasks, MeZO can optimize the pre-trained model and consistently perform better than zero-shot and linear probing. We also show for several tasks that MeZO can outperform another ZO algorithm, BBTv2 [90], by up to  $11\%$  absolute (Appendix F.4).<sup>4</sup>

With enough data, MeZO achieves comparable performance (up to  $5\%$  gap) to FT. MeZO achieves close-to-fine-tuning performance on  $k = 16$ , with some tasks only having  $2\%$  gaps. When using  $k = 512$  data, the gap between MeZO and FT further reduced to within  $5\%$  across all tasks.

MeZO works well on both full-parameter tuning and PEFT. Full-parameter tuning (MeZO) and PEFT (MeZO with LoRA and prefix-tuning) achieve comparable performance, while MeZO (prefix)

<table><tr><td>Task
Task type</td><td>SST-2</td><td>RTE</td><td>CB</td><td>BoolQ classification</td><td>WSC</td><td>WIC</td><td>MultiRC</td><td>COPA - multiple choice</td><td>ReCoRD</td><td>SQuAD generation</td></tr><tr><td>Zero-shot</td><td>58.8</td><td>59.6</td><td>46.4</td><td>59.0</td><td>38.5</td><td>55.0</td><td>46.9</td><td>80.0</td><td>81.2</td><td>46.2 14.6</td></tr><tr><td>ICL</td><td>87.0</td><td>62.1</td><td>57.1</td><td>66.9</td><td>39.4</td><td>50.5</td><td>53.1</td><td>87.0</td><td>82.5</td><td>75.9 29.6</td></tr><tr><td>LP</td><td>93.4</td><td>68.6</td><td>67.9</td><td>59.3</td><td>63.5</td><td>60.2</td><td>63.5</td><td>55.0</td><td>27.1</td><td>3.7 11.1</td></tr><tr><td>MeZO</td><td>91.4</td><td>66.1</td><td>67.9</td><td>67.6</td><td>63.5</td><td>61.1</td><td>60.1</td><td>88.0</td><td>81.7</td><td>84.7 30.9</td></tr><tr><td>MeZO (LoRA)</td><td>89.6</td><td>67.9</td><td>66.1</td><td>73.8</td><td>64.4</td><td>59.7</td><td>61.5</td><td>84.0</td><td>81.2</td><td>83.8 31.4</td></tr><tr><td>MeZO (prefix)</td><td>90.7</td><td>70.8</td><td>69.6</td><td>73.1</td><td>60.6</td><td>59.9</td><td>63.7</td><td>87.0</td><td>81.4</td><td>84.2 28.9</td></tr><tr><td>FT (12x memory)</td><td>92.0</td><td>70.8</td><td>83.9</td><td>77.1</td><td>63.5</td><td>70.1</td><td>71.1</td><td>79.0</td><td>74.1</td><td>84.9 31.3</td></tr></table>

Table 1: Experiments on OPT-13B (with 1000 examples). ICL: in-context learning; LP: linear probing; FT: full fine-tuning with Adam. MeZO outperforms zero-shot, ICL, and LP across the board, and achieves comparable (within  $1\%$  ) or better performance than FT on 7 out of 11 tasks.  

<table><tr><td>Task</td><td>SST-2</td><td>RTE</td><td>BoolQ</td><td>WSC</td><td>WIC</td><td>SQuAD</td></tr><tr><td>30B zero-shot</td><td>56.7</td><td>52.0</td><td>39.1</td><td>38.5</td><td>50.2</td><td>46.5</td></tr><tr><td>30B ICL</td><td>81.9</td><td>66.8</td><td>66.2</td><td>56.7</td><td>51.3</td><td>78.0</td></tr><tr><td>30B MeZO/MeZO (prefix)</td><td>90.6</td><td>72.6</td><td>73.5</td><td>63.5</td><td>59.1</td><td>85.2</td></tr><tr><td>66B zero-shot</td><td>57.5</td><td>67.2</td><td>66.8</td><td>43.3</td><td>50.6</td><td>48.1</td></tr><tr><td>66B ICL</td><td>89.3</td><td>65.3</td><td>62.8</td><td>52.9</td><td>54.9</td><td>81.3</td></tr><tr><td>66B MeZO/MeZO (prefix)</td><td>93.6</td><td>66.4</td><td>73.7</td><td>63.5</td><td>58.9</td><td>85.0</td></tr></table>

Table 2: Experiments on OPT-30B and OPT-66B (with 1000 examples). We report the best of MeZO and MeZO (prefix). See Appendix F.2 for more results. We see that on most tasks MeZO effectively optimizes up to 66B models and outperforms zero-shot and ICL.

sometimes outperforms MeZO. We also show in Appendix F.3 that the three variants converge at similar rates, agreeing with our theory in Section 4, which shows that MeZO converges at a rate independent of the number of parameters being optimized.

We show additional results with more FT and MeZO variants in Appendix F.1. We see that (1) ZO-Adam sometimes outperforms ZO-SGD but is not consistent across tasks; (2) LP and then MeZO, as suggested for fine-tuning [53], can sometimes improve the performance.

# 3.2 Large autoregressive language models

With the promising results from RoBERTa-large, we extend MeZO to the OPT family [112], on a scale of 13B (Table 1), 30B, and 66B (Table 2). We select both SuperGLUE [98] tasks $^5$  (including classification and multiple-choice) and generation tasks. We randomly sample 1000, 500, and 1000 examples for training, validation, and test, respectively, for each dataset. We run MeZO for 20K steps and fine-tuning for 5 epochs, or 625 steps, noting that each step of MeZO is substantially faster than fine-tuning (see Appendix F.6 for a comparison). Please refer to Appendix E for details. Table 1 yields the following observations.

MeZO outperforms memory-equivalent methods and closely approaches fine-tuning results. We see that on a 13B-parameter scale, MeZO and its PEFT variants outperform zero-shot, ICL, and LP across almost all tasks. When comparing to FT, which costs  $12 \times$  more memory (Section 3.4), MeZO achieves comparable (within  $1\%$  ) or better performance on 7 out of the 11 tasks.

MeZO exhibits strong performance across classification, multiple-choice, and generation tasks. We investigate MeZO on generation tasks, which are regarded as more intricate than classification or multiple-choice tasks. We evaluate on two question answering datasets, SQuAD [80] and DROP [31]. We use teacher forcing for training and greedy decoding for inference (details in Appendix E).

Table 1 shows that, on all generation tasks, MeZO outperforms zero-shot, ICL, and LP, and achieves comparable performance to FT. Considering that many applications of fine-tuning LMs – including instruction tuning or domain adaptation – target generation tasks, our results underscore the potential of MeZO as a memory-efficient technique to optimize large LMs for realistic and exciting applications.

<table><tr><td>Model</td><td colspan="4">RoBERTa-large (350M)</td><td>OPT-13B</td></tr><tr><td>Task</td><td>SST-2</td><td>SST-5</td><td>SNLI</td><td>TREC</td><td>SQuAD</td></tr><tr><td>Zero-shot</td><td>79.0</td><td>35.5</td><td>50.2</td><td>32.0</td><td>46.2</td></tr><tr><td>Cross entropy (FT)</td><td>93.9</td><td>55.9</td><td>88.7</td><td>97.3</td><td>84.2</td></tr><tr><td>Cross entropy (MeZO)</td><td>93.3</td><td>53.2</td><td>83.0</td><td>94.3</td><td>84.7</td></tr><tr><td>Accuracy/F1 (MeZO)</td><td>92.7</td><td>48.9</td><td>82.7</td><td>68.6</td><td>78.5</td></tr></table>

Figure 3: GPU memory consumption with different OPT models and tuning methods on MultiRC (400 tokens per example on average).  
Figure 4: Largest OPT models that one can tune with specific hardwares and algorithms.  $\dagger$  : projected results without actual testing.

Table 3: MeZO with non-differentiable objectives. For classification  $(k = 512)$ , we use MeZO with full-parameter and optimize accuracy; for SQuAD (1,000 examples), we use MeZO (prefix) and F1.  

<table><tr><td rowspan="2">Hardware</td><td colspan="3">Largest OPT that can fit</td></tr><tr><td>FT</td><td>FT-prefix</td><td>Inference</td></tr><tr><td>1×A100 (80GB)</td><td>2.7B</td><td>6.7B</td><td>30B</td></tr><tr><td>2×A100 (160GB)</td><td>6.7B</td><td>13B</td><td>66B</td></tr><tr><td>4×A100 (320GB)</td><td>13B</td><td>30B</td><td>66B</td></tr><tr><td>8×A100 (640GB)</td><td>30B</td><td>66B</td><td>\( 175\mathrm{B}^{\dagger} \)</td></tr></table>

MeZO scales up to 66 billion parameter models. We demonstrate the efficacy of MeZO on even larger models, up to 66B, in Table 2. While directly fine-tuning models at such scales is extremely costly (Section 3.4), MeZO can effectively optimize these models and outperform zero-shot and ICL.

# 3.3 Training with non-differentiable objectives

We demonstrate the efficacy of MeZO for optimizing non-differentiable objectives through initial experiments. Accuracy and F1 are used as the respective objectives (details in Appendix E.6). Table 3 reveals that MeZO with accuracy/F1 successfully optimizes LMs with superior performance to zero-shot. Although minimizing cross entropy results in stronger performance, these preliminary findings highlight the promising potential of applying MeZO to optimize non-differentiable objectives without clear differentiable surrogates, such as human preferences [73].

# 3.4 Memory usage and wall-clock time analysis

In this section we profile the memory usage of zero-shot, ICL, FT, FT (prefix), and MeZO. We test OPT models of various sizes with Nvidia A100 GPUs (80GB memory) on MultiRC (average  $\# \mathrm{tokens} = 400$ ), and report the peak GPU memory consumption (details in Appendix E.7).

As shown in Figure 3 (refer to Appendix F.5 for detailed numbers), MeZO exhibits the same memory consumption as zero-shot while offering memory savings of up to 12 times compared to standard FT and 6 times compared to FT (prefix). This advantage enables training larger models within a fixed hardware budget, as illustrated in Figure 4. Specifically, using a single A100 GPU, MeZO allows for tuning a model that is 11 times larger than what is feasible with FT.

In Appendix F.6, we compare the wall-clock time efficiencies of our implementations of MeZO and Adam fine-tuning. MeZO achieves  $7.74 \times$  per-step speedup and requires  $8 \times$  fewer GPUs with a 30B model, but takes more steps to converge. Overall, MeZO only requires half as many GPU-hours to fine-tune a 30B model compared to full-parameter fine-tuning. The wall-clock benefit of MeZO is not inherent to the algorithm and is highly dependent on the implementation. We primarily provide this information as a demonstration that MeZO does not take a prohibitively long time to run.

The above measurements are dependent on the computing infrastructure. In Appendix C, we compare the theoretical time-memory tradeoff of MeZO and backpropagation and find that MeZO is always more memory-efficient than backpropagation and is often more time-efficient. The above analyses also do not consider recent advances (e.g., gradient checkpointing [18], FlashAttention [23], and quantization [27]). We leave investigating the how MeZO works with these methods to future work.

# 4 Theory

Our theoretical analysis highlights why MeZO can optimize large LMs, although a number of classical results [69, 47, 79, 3, 70] suggest that optimization should be catastrophically slow when training so many parameters. The inclusion of a simple prompt is crucial for MeZO to succeed (Appendix A). Past works [83, 67] have suggested that including such a prompt ensures that the fine-tuning objective is closely related to the pre-training one. As such, here, we make the assumption that the model has already been trained for many steps on the fine-tuning objective, which implies that the loss landscape exhibits favorable conditions (Assumption 1). Then, we derive a convergence rate independent of the number of parameters. We show that the loss decreases per step at a rate independent of the parameter dimension  $d$  (Theorem 1), and that, under stronger conditions, the algorithm converges in time independent of  $d$  (Lemma 3). Together, these results imply that MeZO is not catastrophically slower than SGD when fine-tuning.<sup>6</sup> For ease of illustration, we assume that  $z$  is sampled from a sphere with radius  $\sqrt{d}$ , and in Appendix G.2, we derive the rate for a general Gaussian  $z$ , which was used in the experiments.

We follow classical analyses of SGD and replace the minibatch gradient estimate with SPSA (Definition 1). Consider the minibatch SGD update  $\theta_{t + 1}\gets \theta_t - \eta \nabla \mathcal{L}(\pmb {\theta};\mathcal{B}_t)$  where  $\mathcal{B}_t$  is a minibatch drawn uniformly from  $\mathcal{D}^B$ . Crucially, the SGD minibatch gradient estimate is unbiased.

Definition 3 (Unbiased Gradient Estimate). Any minibatch gradient estimate  $g(\pmb{\theta}, \mathcal{B})$  is said to be unbiased if  $\mathbb{E}[g(\pmb{\theta}, \mathcal{B})] = \nabla \mathcal{L}(\pmb{\theta})$ .

# 4.1 Per-step analysis

The classical descent lemma uses a Taylor expansion to study how SGD reduces the loss at each optimization step. It highlights that when the gradient covariance is large, the maximum possible decrease in loss at each optimization step is small, thereby resulting in slower optimization.

Lemma 1 (Descent Lemma). Let  $\mathcal{L}(\pmb{\theta})$  be  $\ell$ -smooth. For any unbiased gradient estimate  $g(\pmb{\theta},\mathcal{B})$ ,

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \leq - \eta \| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \| ^ {2} + \frac {1}{2} \eta^ {2} \ell \cdot \mathbb {E} \left[ \| \boldsymbol {g} \left(\boldsymbol {\theta}, \mathcal {B} _ {t}\right) \| ^ {2} \right]. \tag {2}
$$

The descent lemma highlights the importance of the gradient norm, which we derive for MeZO below.

Lemma 2. Let  $\mathcal{B}$  be a random minibatch of size  $B$ . Then, the gradient norm of MeZO is

$$
\mathbb {E} _ {x} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| ^ {2} \right] = \frac {d + n - 1}{n} \mathbb {E} \left[ \| \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \| ^ {2} \right].
$$

where  $n$  is the number of  $z$  sampled in  $n$ -SPSA (Definition 1) and  $d$  is the number of parameters.

Thus, in the usual case where  $n \ll d$ , MeZO has a much larger gradient norm than SGD. The descent lemma also shows that to guarantee loss decrease, one needs to choose the learning rate as

$$
\eta \leq \frac {2 \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2}}{\ell \cdot \mathbb {E} [ \| \boldsymbol {g} (\boldsymbol {\theta} , \mathcal {B}) \| ^ {2} ]} \quad \stackrel {\text {L e m m a 2}} {\Longrightarrow} \quad \eta_ {\mathrm {Z O}} = \frac {n}{d + n - 1} \eta_ {\mathrm {S G D}} \tag {3}
$$

where  $\eta_{\mathrm{ZO}}$  and  $\eta_{\mathrm{SGD}}$  are the maximum permissible learning rates for MeZO and SGD respectively. Thus we see that without any further assumptions, MeZO can slow optimization by decreasing the largest permissible learning rate by a factor of  $d$ . Moreover, MeZO reduces the loss decrease that can be obtained at each step and, as a consequence, slows convergence by a factor of  $d$  as well.

Surprisingly, our experiments show that MeZO can quickly optimize pre-trained models with billions of parameters, and reducing the number of tuned parameters via PEFT techniques does not substantially accelerate optimization (Appendix F.3). We attribute these phenomena to the Hessian of the loss exhibiting small local effective rank. It is prohibitively expensive to directly measure the effective rank of the Hessian of a large LM on a reasonably sized dataset. However, many previous works have shown that the Hessian of the loss for deep neural networks trained by SGD has remarkably low

effective rank [74, 75, 36, 107, 105, 82]. In particular, the bulk of the spectrum concentrates around 0 with only a small number of outliers, and the number of these outliers is an upper bound on the effective rank. In addition, prior works [4, 56] have demonstrated that LM fine-tuning can occur in a very low dimensional subspace ( $< 200$  parameters), which further supports the below assumption. We formalize the assumption on the effective rank below. In particular, we require an upper bound on the Hessian in a neighborhood around the current iterate to have effective rank at most  $r$ .

Assumption 1 (Local  $r$ -effective rank). Let  $G(\pmb{\theta}_t) = \max_{(\pmb{x},\pmb{y})\in \mathcal{D}}\|\nabla\mathcal{L}(\pmb{\theta}_t;\{(\pmb{x},\pmb{y})\})\|$ . There exists a matrix  $\pmb{H}(\pmb{\theta}_t)\preceq \ell \cdot \pmb{I}_d$  such that:

1. For all  $\pmb{\theta}$  such that  $\| \pmb {\theta} - \pmb {\theta}_t\| \leq \eta dG(\pmb {\theta}_t)$ , we have  $\nabla^2\mathcal{L}(\pmb {\theta})\preceq \pmb {H}(\pmb {\theta}_t)$ .  
2. The effective rank of  $\mathbf{H}(\pmb{\theta}_t)$ , i.e.  $\mathrm{tr}(\pmb{H}(\pmb{\theta}_t)) / \| \pmb{H}(\pmb{\theta}_t) \|_{op}$ , is at most  $r$ .

Under this assumption, we show that the convergence rate of ZO-SGD does not depend on the number of parameters. Instead, the slowdown factor only depends on the effective rank of the Hessian.

Theorem 1 (Dimension-Free Rate). Assume the loss exhibits local  $r$ -effective rank (Assumption 1). If  $\pmb{\theta}_{t+1} = \pmb{\theta}_t - \eta_{ZO}\widehat{\nabla}\mathcal{L}(\pmb{\theta}_t;\mathcal{B})$  is a single step of ZO-SGD using the  $n$ -SPSA estimate with a minibatch of size  $B$ , then there exists a  $\gamma = \Theta(r/n)$  such that the expected loss decrease can be bounded as

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \leq - \eta_ {Z O} \| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \| ^ {2} + \frac {1}{2} \eta_ {Z O} ^ {2} \ell \cdot \gamma \cdot \mathbb {E} \left[ \| \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \| ^ {2} \right] \tag {4}
$$

By applying Equation (3), we can directly compare to the SGD descent lemma.

Corollary 1. Choosing the learning rate  $\eta_{ZO} = \gamma^{-1} \cdot \eta_{SGD}$ ,  $ZO$ -SGD obtains a loss decrease of

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \leq \frac {1}{\gamma} \cdot \left[ - \eta_ {S G D} \| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \| ^ {2} + \frac {1}{2} \eta_ {S G D} ^ {2} \ell \cdot \mathbb {E} [ \| \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \| ^ {2} ] \right]. \tag {5}
$$

Here we see that comparing to SGD, the slowdown factor of ZO-SGD scales with the local effective rank  $r$ , which we argue is much smaller than the number of parameters  $d$ . The above analysis focuses on how much ZO-SGD and SGD decrease the loss at each step. Below, we show that under stronger assumptions about the loss landscape, we can obtain rates for how quickly the ZO-SGD algorithm converges to an optimal value.

# 4.2 Global convergence analysis

We show that the global convergence rate also slows by a factor proportional to the local effective rank under stronger assumptions about the loss landscape. We assume that the landscape obeys the classical PL inequality: the gradient norm grows quadratically with the suboptimality of the iterate.

Definition 4 (PL Inequality). Let  $\mathcal{L}^* = \min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$ . The loss  $\mathcal{L}$  is  $\mu$ -PL if, for all  $\boldsymbol{\theta}$ ,  $\frac{1}{2} \| \nabla \mathcal{L}(\boldsymbol{\theta}) \|^2 \geq \mu (\mathcal{L}(\boldsymbol{\theta}) - \mathcal{L}^*)$ .

The PL inequality is not as strong as assuming that optimization exhibits kernel-like dynamics, but it ensures that the landscape is amenable to analysis [50]. In addition to the PL inequality, we assume the trace of the gradient covariance is bounded, so noise does not disrupt the trajectory too drastically.

Definition 5 (Gradient Covariance). The SGD gradient estimate on a minibatch of size  $B$  has covariance  $\pmb{\Sigma}(\pmb{\theta}) = B(\mathbb{E}\left[\nabla \mathcal{L}(\pmb{\theta};\mathcal{B})\nabla \mathcal{L}(\pmb{\theta};\mathcal{B})^\top\right] - \nabla \mathcal{L}(\pmb{\theta})\nabla \mathcal{L}(\pmb{\theta})^\top)$ .

As we show in Appendix G.1, this assumption holds for common loss functions such as square loss or binary cross entropy for several settings (e.g., kernel behavior [67]). With these two assumptions, we show that ZO-SGD has a slowdown proportional to the effective rank  $r$ , not the parameter dimension.

Lemma 3 (Global Convergence of ZO-SGD). Let  $\mathcal{L}(\pmb{\theta})$  be  $\mu$ -PL and let there exist  $\alpha$  such that  $\operatorname{tr}(\pmb{\Sigma}(\pmb{\theta})) \leq \alpha (\mathcal{L}(\pmb{\theta}) - \mathcal{L}^{*})$  for all  $\pmb{\theta}$ . Then after

$$
t = \mathcal {O} \left(\left(\frac {r}{n} + 1\right) \cdot \underbrace {\left(\frac {\ell}{\mu} + \frac {\ell \alpha}{\mu^ {2} B}\right) \log \frac {\mathcal {L} \left(\boldsymbol {\theta} _ {0}\right) - \mathcal {L} ^ {*}}{\epsilon}} _ {S G D r a t e (L e m m a 4)}\right)
$$

iterations of ZO-SGD we have  $\mathbb{E}[\mathcal{L}(\pmb{\theta}_t)] \leq \mathcal{L}^* + \epsilon$ .

# 5 Related work

Zeroth-order optimization Many classical lower bounds have been derived for ZO-SGD in the strongly convex and convex settings [47, 3, 79, 32, 85, 69] as well as non-convex [101]. These bounds generally depend on the number of parameters  $d$ . More recently, [100, 6, 15] showed that if the gradient has low-dimensional structure, then the query complexity scales linearly with the intrinsic dimension and logarithmically with the number of parameters, though the estimation has at least  $\Omega (sd\log d)$  memory cost. Additional tricks such as sampling schedules [11] and other variance reduction methods [48, 62] can be added to ZO-SGD. ZO has inspired distributed methods [93, 43] and black-box adversarial example generation [14, 63, 17, 64] in deep learning. Ye et al. [108], Balasubramanian and Ghadimi [7] estimate the Hessian to perform ZO optimization along important directions. There are also ZO methods that optimize without estimating the gradient [38, 68, 44].

Memory-efficient backpropagation Several algorithms have been proposed to efficiently approximate backpropagation by sparsifying gradients [92, 102], approximating Jacobians [1, 19], and subsampling the computational graph [71, 2]. However, these methods may accrue large approximation errors for deep networks. Gradient checkpointing [18] reduces memory cost of backpropagation at the cost of recomputing some activations. FlashAttention [23] also reduces memory cost by recomputing attention matrices. Dettmers et al. [26, 27] explore quantization of large LMs' weights and optimizer states, which leads to memory reduction in both training and inference.

Gradient-free adaptation of large language models BBT and BBTv2 [91, 90] use evolutionary algorithms to achieve gradient-free optimization; however, due to its sensitivity to high dimensionality, BBT is limited to only optimize a low-dimension projection of prefixes and they focus on RoBERTa-large size models and few-shot settings. Other works in "black-box tuning" of LMs focus on optimizing discrete prompts without updating the model, either via reinforcement learning [16, 25, 29], ensemble [45], or iterative search [78]. Concurrent work in [106] uses iterative forward passes to improve in-context learning performance.

# 6 Conclusion

We have shown that MeZO can effectively optimize large LMs across many tasks and scales. Further experiments suggest that MeZO can optimize non-differentiable objectives, which backpropagation usually cannot do. Our theory illustrates why MeZO is not catastrophically slow when tuning billions of parameters. As a limitation, MeZO takes many steps in order to achieve strong performance, though we show that the per-step speedup in MeZO can often make fine-tuning with MeZO run faster than a standard implementation of fine-tuning with backpropagation. We did not explore combining MeZO with other memory-efficient methods, such as FlashAttention [23] and quantization [26], though we hope to investigate this in the future.

We are excited to explore the applicability of MeZO to a number of areas, including but not limited to: pruning, distillation, saliency, interpretability, and dataset selection for fine-tuning. Non-differentiable objectives are a particularly exciting area, given recent advances in tuning large LMs to adapt to human feedback. Conducting theoretical analyses for how these efficient gradient estimates impact the performance of different applications is also of interest.

# Acknowledgements

We thank Xinyi Chen, Yin Tat Lee, Kaifeng Lyu, Tengyu Ma, Abhishek Panigrahi, Nikunj Saunshi, and Mengzhou Xia for their helpful feedback. SA and SM are funded by NSR, ONR, SRC, and Simons Foundation. JDL, AD, and EN acknowledge the support of the ARO under MURI Award W911NF-11-1-0304, the Sloan Research Fellowship, NSF CCF 2002272, NSF IIS 2107304, NSF CIF 2212262, ONR Young Investigator Award, and NSF CAREER Award 2144994. TG is supported by an IBM PhD Fellowship. This work is also partially funded by the National Science Foundation (IIS-2211779).

# References

[1] Hany S Abdel-Khalik, Paul D Hovland, Andrew Lyons, Tracy E Stover, and Jean Utke. A low rank approach to automatic differentiation. In Advances in Automatic Differentiation, pages 55-65, 2008.  
[2] Menachem Adelman, Kfir Levy, Ido Hakimi, and Mark Silberstein. Faster neural network training with approximate tensor operations. Advances in Neural Information Processing Systems, 34:27877-27889, 2021.  
[3] Alekh Agarwal, Peter L. Bartlett, Pradeep Ravikumar, and Martin J. Wainwright. Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization. IEEE Transactions on Information Theory, 58(5):3235-3249, May 2012.  
[4] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319-7328, 2021.  
[5] Stephen H Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, et al. Promptsource: An integrated development environment and repository for natural language prompts. arXiv preprint arXiv:2202.01279, 2022.  
[6] Krishnakumar Balasubramanian and Saeed Ghadimi. Zeroth-order (non)-convex stochastic optimization via conditional gradient and gradient updates. In Advances in Neural Information Processing Systems, volume 31, 2018.  
[7] Krishnakumar Balasubramanian and Saeed Ghadimi. Zeroth-order nonconvex stochastic optimization: Handling constraints, high dimensionality, and saddle points. Foundations of Computational Mathematics, pages 1-42, 2022.  
[8] Roy Bar Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. In Proceedings of the Second PASCAL Challenges Workshop on Recognising Textual Entailment, 2006.  
[9] Atulim Güneş Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, and Philip Torr. Gradients without backpropagation, 2022.  
[10] Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth PASCAL recognizing textual entailment challenge. In TAC, 2009.  
[11] Raghu Bollapragada, Richard Byrd, and Jorge Nocedal. Adaptive sampling strategies for stochastic optimization. SIAM Journal on Optimization, 28(4):3312-3343, 2018.  
[12] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632-642, 2015.  
[13] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in neural information processing systems, volume 33, pages 1877-1901, 2020.  
[14] HanQin Cai, Yuchen Lou, Daniel McKenzie, and Wotao Yin. A zeroth-order block coordinate descent algorithm for huge-scale black-box optimization. In International Conference on Machine Learning, pages 1193-1203, 2021.  
[15] HanQin Cai, Daniel McKenzie, Wotao Yin, and Zhenliang Zhang. Zeroth-order regularized optimization (zoro): Approximately sparse gradients and adaptive sampling. SIAM Journal on Optimization, 32(2):687-714, 2022.  
[16] Yekun Chai, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Cliptuning: Towards derivative-free prompt learning with a mixture of rewards. arXiv preprint arXiv:2210.12050, 2022.

[17] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pages 15-26, 2017.  
[18] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.  
[19] Krzysztof M Choromanski and Vikas Sindhwani. On blackbox backpropagation and jacobian sensing. In Advances in Neural Information Processing Systems, volume 30, 2017.  
[20] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in neural information processing systems, volume 30, 2017.  
[21] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, 2019.  
[22] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In the First International Conference on Machine Learning Challenges: Evaluating Predictive Uncertainty Visual Object Classification, and Recognizing Textual Entailment, 2005.  
[23] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344-16359, 2022.  
[24] Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In Sinn und Bedeutung, volume 23, pages 107-124, 2019.  
[25] Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric Xing, and Zhiting Hu. RLPrompt: Optimizing discrete text prompts with reinforcement learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3369-3391, 2022.  
[26] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems, 2022.  
[27] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via blockwise quantization. In International Conference on Learning Representations, 2022.  
[28] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, 2019.  
[29] Shizhe Diao, Xuechun Li, Yong Lin, Zhichao Huang, and Tong Zhang. Black-box prompt learning for pre-trained language models. arXiv preprint arXiv:2201.08531, 2022.  
[30] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. arXiv preprint arXiv:2203.06904, 2022.  
[31] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368-2378, 2019.

[32] John C. Duchi, Michael I. Jordan, Martin J. Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788-2806, 2015.  
[33] FairScale authors. Fairscale: A general purpose modular pytorch library for high performance and large scale training, 2021.  
[34] Abraham D. Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. Online convex optimization in the bandit setting: Gradient descent without a gradient. In Proceedings of the Sixteenth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA '05, page 385-394, USA, 2005. Society for Industrial and Applied Mathematics. ISBN 0898715857.  
[35] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816-3830, 2021.  
[36] Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. In International Conference on Machine Learning, pages 2232-2241, 2019.  
[37] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, 2007.  
[38] Daniel Golovin, John Karro, Greg Kochanski, Chansoo Lee, Xingyou Song, and Qiuyi Zhang. Gradientless descent: High-dimensional zeroth-order optimization. In International Conference on Learning Representations, 2020.  
[39] Priya Goyal, Piotr Dólár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: TrainingImagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
[40] Andreas Griewank and Andrea Walther. Evaluating derivatives: principles and techniques of algorithmic differentiation. SIAM, 2008.  
[41] José Grimm, Loic Pottier, and Nicole Rostaing-Schmidt. Optimal time and minimum space-time product for reversing a certain class of programs. PhD thesis, INRIA, 1996.  
[42] Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don't stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, 2020.  
[43] Davood Hajinezhad and Michael M Zavlanos. Gradient-free multi-agent nonconvex nonsmooth optimization. In 2018 IEEE Conference on Decision and Control (CDC), pages 4939-4944, 2018.  
[44] Geoffrey Hinton. The forward-forward algorithm: Some preliminary investigations. arXiv preprint arXiv:2212.13345, 2022.  
[45] Bairu Hou, Joe O'Connor, Jacob Andreas, Shiyu Chang, and Yang Zhang. Promptboosting: Black-box text classification with ten forward passes. arXiv preprint arXiv:2212.09257, 2022.  
[46] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.  
[47] Kevin G Jamieson, Robert Nowak, and Ben Recht. Query complexity of derivative-free optimization. In Advances in Neural Information Processing Systems, volume 25, 2012.  
[48] Kaiyi Ji, Zhe Wang, Yi Zhou, and Yingbin Liang. Improved zeroth-order variance reduced algorithms and analysis for nonconvex optimization. In International conference on machine learning, pages 3100-3109, 2019.

[49] Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26, 2013.  
[50] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal-gradient methods under the polyak-fojasiewicz condition, 2020.  
[51] Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 252-262, 2018.  
[52] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
[53] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Finetuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations, 2022.  
[54] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045-3059, 2021.  
[55] Hector Levesque, Ernest Davis, and Leora Morgenstern. The winograd schema challenge. In Thirteenth international conference on the principles of knowledge representation and reasoning, 2012.  
[56] Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. In International Conference on Learning Representations, 2018.  
[57] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582-4597, 2021.  
[58] Zhiyuan Li, Sadhika Malladi, and Sanjeev Arora. On the validity of modeling SGD with stochastic differential equations (SDEs). In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021.  
[59] Zhiyuan Li, Srinadh Bhojanapalli, Manzil Zaheer, Sashank Reddi, and Sanjiv Kumar. Robust training of neural networks using scale invariant architectures. In International Conference on Machine Learning, pages 12656-12684, 2022.  
[60] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, 2022.  
[61] Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5747-5763, 2020.  
[62] Sijia Liu, Bhavya Kailkhura, Pin-Yu Chen, Paishun Ting, Shiyu Chang, and Lisa Amini. Zeroth-order stochastic variance reduction for nonconvex optimization. In Advances in Neural Information Processing Systems, volume 31, 2018.  
[63] Sijia Liu, Pin-Yu Chen, Xiangyi Chen, and Mingyi Hong. signSGD via zeroth-order oracle. In International Conference on Learning Representations, 2019.  
[64] Sijia Liu, Pin-Yu Chen, Bhavya Kailkhura, Gaoyuan Zhang, Alfred O Hero III, and Pramod K Varshney. A primer on zeroth-order optimization in signal processing and machine learning: Principals, recent advances, and applications. IEEE Signal Processing Magazine, 37(5):43-54, 2020.

[65] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.  
[66] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086-8098, 2022.  
[67] Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, and Sanjeev Arora. A kernel-based view of language model fine-tuning. arXiv preprint arXiv:2210.05643, 2022.  
[68] Horia Mania, Aurelia Guy, and Benjamin Recht. Simple random search of static linear policies is competitive for reinforcement learning. In Advances in Neural Information Processing Systems, volume 31, 2018.  
[69] Arkadj Semenovic Nemirovskij and David Borisovich Yudin. Problem complexity and method efficiency in optimization. 1983.  
[70] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17:527-566, 2017.  
[71] Deniz Oktay, Nick McGreivy, Joshua Aduol, Alex Beatson, and Ryan P Adams. Randomized automatic differentiation. arXiv preprint arXiv:2007.10412, 2020.  
[72] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.  
[73] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022.  
[74] Vardan Papyan. The full spectrum of deepnet hessenians at scale: Dynamics with sgd training and sample size. arXiv preprint arXiv:1811.07062, 2018.  
[75] Vardan Papyan. Traces of class/cross-class structure pervade deep learning spectra. Journal of Machine Learning Research, 21(252):1-64, 2020.  
[76] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024-8035. 2019.  
[77] Mohammad Taher Pilehvar and Jose Camacho-Collados. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1267-1273, 2019.  
[78] Archiki Prasad, Peter Hase, Xiang Zhou, and Mohit Bansal. Grips: Gradient-free, edit-based instruction search for prompting large language models. arXiv preprint arXiv:2203.07281, 2022.  
[79] Maxim Raginsky and Alexander Rakhlin. Information-based complexity, feedback and dynamics in convex programming. IEEE Transactions on Information Theory, 57(10):7036-7056, 2011.  
[80] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, 2016.  
[81] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. 2011.

[82] Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.  
[83] Nikunj Saunshi, Sadhika Malladi, and Sanjeev Arora. A mathematical exploration of why language models help solve downstream tasks. In International Conference on Learning Representations, 2021.  
[84] Timo Schick and Hinrich Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 255–269, 2021.  
[85] Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. The Journal of Machine Learning Research, 18(1):1703-1713, 2017.  
[86] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 2013.  
[87] James C Spall. A one-measurement form of simultaneous perturbation stochastic approximation. Automatica, 33(1):109-112, 1997.  
[88] J.C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3):332-341, 1992.  
[89] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008-3021, 2020.  
[90] Tianxiang Sun, Zhengfu He, Hong Qian, Yunhua Zhou, Xuanjing Huang, and Xipeng Qiu. BBTv2: Towards a gradient-free future with large language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3916-3930, 2022.  
[91] Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, and Xipeng Qiu. Black-box tuning for language-model-as-a-service. In International Conference on Machine Learning, pages 20841-20855, 2022.  
[92] Xu Sun, Xuancheng Ren, Shuming Ma, and Houfeng Wang. meProp: Sparsified back propagation for accelerated deep learning with reduced overfitting. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 3299-3308, 2017.  
[93] Yujie Tang and Na Li. Distributed zero-order algorithms for nonconvex multi-agent optimization. In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 781-786, 2019.  
[94] Zhiwei Tang, Dmitry Rybin, and Tsung-Hui Chang. Zeroth-order optimization meets human feedback: Provable learning via ranking oracles, 2023.  
[95] Alexander Timurovich Vakhitov, Oleg Nikolaevich Granichin, and LS Gurevich. Algorithm for stochastic approximation with trial input perturbation in the nonstationary problem of optimization. Automation and Remote Control, 70:1827-1835, 2009.  
[96] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, volume 30, 2017.  
[97] Ellen M Voorhees and Dawn M Tice. Building a question answering test collection. In the 23rd annual international ACM SIGIR conference on Research and development in information retrieval, 2000.

[98] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in neural information processing systems, volume 32, 2019.  
[99] Chong Wang, Xi Chen, Alexander J Smola, and Eric P Xing. Variance reduction for stochastic gradient optimization. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013.  
[100] Yining Wang, Simon Du, Sivaraman Balakrishnan, and Aarti Singh. Stochastic zeroth-order optimization in high dimensions. In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84, pages 1356–1365, 2018.  
[101] Zhongruo Wang, Krishnakumar Balasubramanian, Shiqian Ma, and Meisam Razaviyayn. Zeroth-order algorithms for nonconvex minimax problems with improved complexities. arXiv preprint arXiv:2001.07819, 2020.  
[102] Bingzhen Wei, Xu Sun, Xuancheng Ren, and Jingjing Xu. Minimal effort back propagation for convolutional neural networks. arXiv preprint arXiv:1709.05804, 2017.  
[103] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2018.  
[104] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45, 2020.  
[105] Yikai Wu, Xingyu Zhu, Chenwei Wu, Annie Wang, and Rong Ge. Dissecting hessian: Understanding common structure of hessian in neural networks. arXiv preprint arXiv:2010.04261, 2020.  
[106] Jiaxi Yang, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. Iterative forward tuning boosts in-context learning in language models, 2023.  
[107] Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W Mahoney. Pyhessian: Neural networks through the lens of the hessian. In 2020 IEEE international conference on big data (Big data), pages 581-590, 2020.  
[108] Haishan Ye, Zhichao Huang, Cong Fang, Chris Junchi Li, and Tong Zhang. Hessian-aware zeroth-order optimization for black-box adversarial attack. arXiv preprint arXiv:1812.11377, 2018.  
[109] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.  
[110] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019.  
[111] Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme. Record: Bridging the gap between human and machine commonsense reading comprehension. arXiv preprint arXiv:1810.12885, 2018.  
[112] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuhui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.  
[113] Yan Zhang, Yi Zhou, Kaiyi Ji, and Michael M Zavlanos. A new one-point residual-feedback oracle for black-box learning and control. Automatica, 136:110006, 2022.

# A Algorithmic Ablations

We perform a number of ablations to select the best algorithm. As is standard in ZO literature, we consider the main computational cost to be the number of forward passes. In our case, the number of forward passes can be affected by the number of gradient steps taken, any usage of gradient accumulation, and using more noise samples to reduce the variance of the gradient estimate.

We observed that the performance of MeZO improves monotonically with the number of steps, and there does not appear to be any overfitting. Hence, when performing algorithmic ablations, we can focus on the efficiency of different algorithms without considering implicit bias. This is also reflected in our theoretical analysis. To ease the computational load, we fix the number of forward passes to 10,000 and compare many different algorithms for RoBERTa-large on a smaller set of tasks that span sentiment analysis, entailment, and topic classification: SST-2, SNLI, and TREC. We emphasize that 10,000 is a small budget and is only used as a setting to compare these ZO algorithms to each other. We find that using a linearly decreasing learning rate schedule during training, as was done for fine-tuning with backpropagation in [65], does not help or hurt MeZO. Similarly, using a learning rate warmup leads to identical results on these three tasks. For simplicity, we use a constant learning rate schedule with no warmup for all of the below experiments. We perform few-shot experiments with  $k = 16$  and average the results across 5 seeds.

<table><tr><td>Experiment</td><td>Hyperparameters</td><td>Values</td></tr><tr><td rowspan="4">MeZO</td><td>Batch size</td><td>{16,64} ×</td></tr><tr><td>Learning rate</td><td>{1e-5,1e-6,1e-7} ×</td></tr><tr><td>ε</td><td>{1e-3,1e-5} ×</td></tr><tr><td>Weight Decay</td><td>{0,0.1}</td></tr></table>

# A.1 Prompting

We study if adding a prompt is crucial to the ability of MeZO to optimize the network. We use prompts from Gao et al. [35]. Malladi et al. [67] claimed the prompt makes the optimization trajectory well-behaved, though we note that the current paper considers RoBERTa-large and large autoregressive models while the previous work only studied RoBERTa-base. We note the similarity between kernel behavior and our theoretical setting in Section 4. MeZO succeeds on tasks that are reported to not exhibit kernel behavior in Malladi et al. [67], so we investigate whether or not the prompt is necessary.

Table 4: The hyperparameter grid used in our ablation experiments. For simplicity, we use a constant learning rate schedule.  

<table><tr><td></td><td>SST-2</td><td>SNLI</td><td>TREC</td></tr><tr><td>Prompt</td><td>89.6 (1.2)</td><td>65.1 (6.2)</td><td>66.7 (6.2)</td></tr><tr><td>No Prompt</td><td>51.9 (2.9)</td><td>34.8 (2.1)</td><td>19.5 (9.0)</td></tr></table>

Table 5: Experiments using MeZO to fine-tune models with and without a prompt.

Both experiments followed the grid in Table 4, but we also expanded the grid to include a learning rate of  $1\mathrm{e} - 4$  for the no prompt case. As a result of these experiments, we fix the setting to prompt-based fine-tuning for all of the below experiments.

# A.2 Sample schedules

One can sample  $n_t$  noise vectors at the  $t$ th step and use  $n_t$ -SPSA to compute the gradient estimate. Similar ideas were proposed in Bollapragada et al. [11], Cai et al. [15]. We study the effect of linearly increasing and constant sampling schedules in the ablation setting. The intuition for the linearly increasing schedule is that the optimizer may need a higher fidelity gradient as it approaches the minimum. Increasing the number of  $z$ s can speed up optimization by reducing the gradient variance, but doing so also increases the number of forward passes required for each optimization step, so there is a trade-off to study. We note that increasing the number of  $z$ s should be accompanied by

a proportional scaling of the learning rate, analogous to the linear scaling rule proposed in [39] (theoretical justification can follow the SDE technique [58]). Table 6 shows no consistent benefit in one schedule over the other, and it demonstrates that increasing the  $n$  in  $n$ -SPSA while fixing the number of forward passes allowed results in only marginal gains at best.

<table><tr><td>n</td><td>Schedule</td><td>SST-2</td><td>SNLI</td><td>TREC</td></tr><tr><td>n = 1</td><td>Constant</td><td>89.6 (1.2)</td><td>65.1 (6.2)</td><td>66.7 (6.2)</td></tr><tr><td>n = 4</td><td>Constant</td><td>89.5 (1.1)</td><td>68.6 (3.2)</td><td>62.3 (5.6)</td></tr><tr><td>n = 4</td><td>Linear</td><td>89.6 (1.4)</td><td>65.3 (6.4)</td><td>66.1 (5.5)</td></tr><tr><td>n = 16</td><td>Constant</td><td>90.4 (0.7)</td><td>67.0 (3.4)</td><td>62.8 (6.3)</td></tr><tr><td>n = 16</td><td>Linear</td><td>88.9 (1.2)</td><td>62.8 (5.9)</td><td>64.2 (5.3)</td></tr></table>

# B MeZO Variants

There is a rich history of transferring ideas from first order optimization to enhance ZO algorithms. Below, we highlight several variants of MeZO that did not achieve as high performance as the algorithm presented in Algorithm 1.

# B.1 Memory-efficient n-SPSA

We highlight how MeZO can perform  $n$ -SPSA (Definition 1) efficiently for  $n > 1$  in Algorithm 2. In particular, if sampling  $n$ $z$  vectors and averaging the projected gradients, we require storing  $2n$  additional scalars: the random seeds and the projected gradients. The same caveat about perturbing individual weights versus entire weight matrices still applies here (see Section 2).

# B.2 Augmenting MeZO with Gradient History

The  $n$ -SPSA algorithm merely provides a gradient estimate that can subsequently be used in place of the gradient in any gradient-based optimizer. Many popular optimizers, such as Adam and SGD with momentum, require storing some historical information about gradients (e.g., a moving average). This requirement causes such algorithms to require  $2 \times$  or  $3 \times$  the memory that is needed for SGD.

However, one advantage of MeZO is that the gradient history can be recomputed at each step without requiring much additional memory. In reference to Algorithm 1, note that the gradient only needs projected_grad and the random seed  $s$  used to compute the perturbation  $z$ , so we need to only store 2 scalars per step to reproduce the gradient history (i.e., up to  $2T$  scalars during training). This is a substantial reduction in added memory overhead that is usually needed for using Adam or momentum instead of vanilla SGD.

Table 18 illustrates that MeZO-Adam can sometimes improve the performance of MeZO, though each gradient step requires additional computation (but no additional forward passes). We leave it to future work to investigate when MeZO-Adam may be more useful than MeZO.

Table 6: Experiments using MeZO with different schedules for  $n$ . We scale the learning rate proportionally to the number of  $z$ 's sampled.  

<table><tr><td>Experiment</td><td>Hyperparameters</td><td>Values</td></tr><tr><td rowspan="4">MeZO-Adam</td><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>{1e-6, 1e-5, 1e-4, 5e-4, 1e-3}</td></tr><tr><td>ε</td><td>1e-3</td></tr><tr><td>Weight Decay</td><td>0</td></tr></table>

Table 7: The hyperparameter grid used for MeZO-Adam. For simplicity, we use a constant learning rate schedule.

Require: parameters  $\pmb{\theta} \in \mathbb{R}^d$ , loss  $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ , step budget  $T$ , perturbation scale  $\epsilon$ , batch size  $B$ , learning rate schedule  $\{\eta_t\}$ ,  $n$  for  $n$ -SPSA estimate (Definition 1)

Algorithm 2: MeZO with  $n > 1$  
```txt
for  $t = 1,\dots,T$  do seeds, projected_grads  $\leftarrow []$  Will each contain  $n$  scalars for  $j = 1,\ldots ,n$  do Sample batch  $\mathcal{B}\subset \mathcal{D}^{\mathcal{B}}$  and random seed s  $\theta \gets$  PerturbParameters  $(\theta ,\epsilon ,s)$ $\ell_{+}\gets \mathcal{L}(\theta ;\mathcal{B})$ $\theta \gets$  PerturbParameters  $(\theta , - 2\epsilon ,s)$ $\ell_{-}\gets \mathcal{L}(\theta ;\mathcal{B})$ $\theta \gets$  PerturbParameters  $(\theta ,\epsilon ,s)$  Reset parameters projected_grad  $\leftarrow (\ell_{+} - \ell_{-}) / (2\epsilon)$  projected_grads[j]  $\leftarrow$  projected_grad seeds [j]  $\leftarrow s$    
end   
for  $j = 1,\dots,n$  do Reset random number generator with seed seeds [j] for  $\theta_i\in \theta$  do  $z\sim \mathcal{N}(0,1)$ $\theta_{i}\gets \theta_{i} - (\eta_{t} / n)*$  projected_grads[j]\*z Avg grad for  $z_{1},\ldots ,z_{n}$  end   
end
```

```latex
Subroutine PerturbParameters  $(\theta ,\epsilon ,s)$  Reset random number generator with seed  $s$  For sampling z for  $\theta_{i}\in \theta$  do  $z\sim \mathcal{N}(0,1)$  0  $\theta_i\gets \theta_i + \epsilon z$  Modify parameters in place end return  $\pmb{\theta}$
```

# B.3 Modifying the Variance of MeZO

Our theory in Section 4 sketches the well-known fact that the variance of the stochastic gradient estimate can impact the rate of optimization. ZO methods can be combined with standard variance reduction techniques to possibly improve optimization speed. For example, Liu et al. [62] designed a variance reduced ZO algorithm, analogous to SVRG [49], to improve the speed of convergence. Below, we show that several variance reduction methods (e.g., using the gradient norm) can be implemented in a memory-efficient manner. However, when controlling for the total budget of forward passes (i.e., function queries), these methods are not as performant as MeZO. We nevertheless present them to demonstrate the ease with which MeZO can be adapted, and we suggest these methods may be useful for optimizing more complex objectives.

First, we define a general SPSA estimate that has the same expectation (i.e., the true gradient) but has a scaled variance.

Definition 6 (Variance-Modified SPSA). Given a matrix  $D = \mathrm{diag}(d)$ , the variance modified SPSA computes

$$
\widetilde {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) = \frac {\mathcal {L} (\boldsymbol {\theta} + \epsilon (\boldsymbol {d} ^ {- 1} \odot z) ; \mathcal {B}) - \mathcal {L} (\boldsymbol {\theta} - \epsilon (\boldsymbol {d} ^ {- 1} \odot z) ; \mathcal {B})}{2 \epsilon} (\boldsymbol {d} \odot z)
$$

where  $\pmb{d} \in \mathbb{R}^d$  has nonzero entries and  $\pmb{d}^{-1}$  denotes the coordinatewise reciprocal.

The above SPSA variant is an unbiased estimator of the gradient, because  $\mathbb{E}[\widetilde{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B})] =$ $\mathbb{E}[D^{-1}zz^{\top}D\nabla \mathcal{L}(\pmb {\theta};\mathcal{B})] = \mathbb{E}[\nabla \mathcal{L}(\pmb {\theta};\mathcal{B})]$  . We will draw inspiration from classical methods (i.e., "control variates") and choose  $\pmb{d}$  to be a block vector with gradient norms or parameter norms [99].

To select the parameter groups, we split the model by layer, keeping the embedding and the head separate (i.e., RoBERTa-large has  $24 + 2 = 26$  parameter groups). It is straightforward to measure the parameter norms without consuming additional memory. We can measure the gradient norms without performing backpropagation, as shown below.

Proposition 1 (ZO Estimate of Gradient Norm of  $\ell$ th Layer). Define  $z_{\ell}$  to have  $z \sim \mathcal{N}(0,1)$  in each coordinate corresponding to parameters in the  $\ell$ th layer and 0 everywhere else. Then, we can estimate the norm of the gradient of the loss w.r.t. the  $\ell$ th layer  $\nabla_{\theta_{\ell}}$  as

$$
\left\| \nabla_ {\boldsymbol {\theta} _ {\ell}} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| _ {2} \approx \left| \frac {\mathcal {L} (\boldsymbol {\theta} + \epsilon \boldsymbol {z} _ {\ell} ; \mathcal {B}) - \mathcal {L} (\boldsymbol {\theta} - \epsilon \boldsymbol {z} _ {\ell} ; \mathcal {B})}{2 \epsilon} \right|
$$

As is true for SPSA, increasing the number of  $z_{\ell}$ 's sampled for each value of  $\ell$  and averaging the result reduces the variance of the estimate. The rationale for this estimate is that for any vector  $v$ ,  $\mathbb{E}_z[(\langle v,z\rangle)^2] = \| v\| _2^2$  for Gaussian  $z$ . It is clear that this estimate can be computed in a memory efficient way, although it requires  $2L$  forward passes to compute gradient norms for  $L$  parameter groups.

We show the experimental results for modifying the variance below. We follow the ablation setting and use a fixed budget of 10,000 steps (Appendix A). Generally, using the gradient norm to reduce the variance substantially hurts performance (Table 8). If we "cheat" and allow one backpropagation through the network to estimate the gradient norm, then we see that reducing the variance using the gradient norm does not substantially hurt or help performance. Modifying the variance using the parameter norm, analogous to layerwise adaptive rate methods, does not substantially impact the performance of MeZO (Table 9).

Our observation is that decreasing the variance by setting  $d$  as the gradient norm does not improve optimization. This empirical result agrees with the exposition in Section 4 that the straightforward variance analysis (which yields a dependence on the number of parameters  $d$ ) is not the best lens to study the rate of optimization when fine-tuning with MeZO. Our effective rank view in Theorem 1 and Lemma 3 is likely a better characterization of fine-tuning dynamics. We leave it to future work to explore if these methods can be useful for other more complex objectives.

<table><tr><td colspan="2">Recompute d ZO estimate of d</td><td>SST-2</td><td>SNLI</td><td>TREC</td></tr><tr><td colspan="2">Baseline MeZO (Algorithm 1)</td><td>89.6 (1.2)</td><td>65.1 (6.2)</td><td>66.7 (6.2)</td></tr><tr><td>X</td><td>X</td><td>89.7 (0.8)</td><td>65.2 (5.2)</td><td>64.3 (6.4)</td></tr><tr><td>X</td><td>✓</td><td>87.0 (2.5)</td><td>49.6 (9.2)</td><td>32.6 (7.7)</td></tr><tr><td>✓</td><td>✓</td><td>79.0 (10.3)</td><td>48.9 (2.2)</td><td>38.7 (7.5)</td></tr></table>

Table 8: Experiments modifying the variance of MeZO using  $\mathbf{d}$  as the gradient norm (see Definition 6). We sometimes recompute  $\mathbf{d}$  at the start of each epoch or use Proposition 1 to estimate  $\mathbf{d}$  without requiring backpropagation.  

<table><tr><td>Recompute d</td><td>SST-2</td><td>SNLI</td><td>TREC</td></tr><tr><td>Baseline MeZO (Algorithm 1)</td><td>89.6 (1.2)</td><td>65.1 (6.2)</td><td>66.7 (6.2)</td></tr><tr><td>X</td><td>89.2 (2.1)</td><td>65.4 (4.2)</td><td>64.8 (5.6)</td></tr><tr><td>✓</td><td>88.2 (4.7)</td><td>65.2 (4.0)</td><td>64.7 (5.5)</td></tr></table>

Table 9: Experiments modifying the variance of MeZO using  $\mathbf{d}$  as the parameter norm (see Definition 6). We sometimes recompute  $\mathbf{d}$  at the start of each epoch.

# B.4 Modifying the Expectation of MeZO

The above experiments show that modifying the variance of MeZO cannot consistently accelerate its convergence. However, a simple modification of Definition 6 allows us to change the expectation of MeZO as well. This can be used to efficiently estimate coordinate-wise normalized gradient-based optimizer updates (e.g., Adam).

Definition 7 (Expectation-Modified SPSA). Given a matrix  $D = \mathrm{diag}(d)$ , the variance modified SPSA computes

$$
\tilde {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) = \frac {\mathcal {L} (\boldsymbol {\theta} + \epsilon (\boldsymbol {d} ^ {- 1} \odot z) ; \mathcal {B}) - \mathcal {L} (\boldsymbol {\theta} - \epsilon (\boldsymbol {d} ^ {- 1} \odot z) ; \mathcal {B})}{2 \epsilon} z
$$

where  $\pmb{d} \in \mathbb{R}^d$ .

Now, we see that  $\widetilde{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B}) = \mathbb{E}[D^{-1}zz^{\top}\nabla \mathcal{L}(\pmb {\theta};\mathcal{B})]$  so the SPSA estimate is no longer an unbiased estimator for  $\nabla \mathcal{L}(\pmb {\theta})$ . If we choose  $\pmb{d}$  to be the gradient norm, for example, then SPSA can estimate the normalized gradient. Concurrent work in Tang et al. [94] gives another ZO estimate of the normalized gradient while assuming access to only rankings of inputs (instead of the noisy function evaluations available in our setting). We find that estimating the normalized gradient does not perform as well as directly estimating the gradient (Table 10). Regardless, we present this algorithm as a way to highlight that any coordinate-wise operation to the gradient can be applied in a memory-efficient manner.

<table><tr><td>Method</td><td>SST-2</td><td>SNLI</td><td>TREC</td></tr><tr><td>Baseline MeZO (Algorithm 1)</td><td>89.6 (1.2)</td><td>65.1 (6.2)</td><td>66.7 (6.2)</td></tr><tr><td>Estimate of normalized gradient (Definition 7)</td><td>88.0 (1.2)</td><td>60.0 (2.4)</td><td>44.0 (14.0)</td></tr></table>

# B.5 One-point estimate

Here, we investigate the efficacy of one-point gradient estimators in place of the two-point SPSA method. Using a one-point estimator instead of SPSA can reduce the MeZO running time by half. Many one-point estimators have been proposed in the past [34, 87, 95]. For simplicity, we focus on one estimator [113] that has a form reminiscent of SPSA but requires only one forward pass to estimate the gradient at each step.

Definition 8 (One-Point Gradient Estimate, Zhang et al. [113]). For a loss function  $\mathcal{L}$  evaluated on a batch  $\mathcal{B}_t$  with parameters  $\theta_t$  at step  $t$ , we can draw random noise  $z_t \sim \mathcal{N}(0, I_d)$  and compute the gradient estimate using hyperparameter  $\epsilon$  as written below.

$$
\widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B} _ {t}) = \frac {\mathcal {L} (\boldsymbol {\theta} _ {t} + \epsilon \boldsymbol {z} _ {t} ; \mathcal {B} _ {t}) - \mathcal {L} (\boldsymbol {\theta} _ {t - 1} + \epsilon \boldsymbol {z} _ {t - 1} ; \mathcal {B} _ {t - 1})}{\epsilon}
$$

Notably, this one-point gradient estimate uses the loss at the previous iterate instead of evaluating the loss again at the current iterate. As such, this estimator requires only one forward pass at each iterate to estimate the gradient. For well-behaved loss functions and slow-moving optimization, these two formulas are intuitively similar. However, Table 11 finds this estimator to be much less efficient than SPSA when fixing the number of forward passes.

Table 10: Experiments modifying the expectation of MeZO using  $\mathbf{d}$  as the gradient norm (see Definition 7). We use the ZO estimate of the gradient norm (Proposition 1).  

<table><tr><td rowspan="2"></td><td rowspan="2">Steps</td><td>SST-2</td><td>SST-5</td><td>SNLI</td><td>MNLI</td><td>RTE</td><td>TREC</td></tr><tr><td>— sentiment —</td><td colspan="4">— natural language inference —</td><td>— topic —</td></tr><tr><td>SPSA [88]</td><td>20K</td><td>92.8 (0.5)</td><td>51.3 (0.9)</td><td>82.9 (1.0)</td><td>74.4 (0.8)</td><td>76.7 (1.7)</td><td>92.7 (0.6)</td></tr><tr><td>One-point estimate [113]</td><td>20K</td><td>90.0 (0.4)</td><td>44.6 (2.0)</td><td>70.1 (1.5)</td><td>57.2 (0.9)</td><td>64.1 (1.0)</td><td>57.3 (5.7)</td></tr><tr><td>One-point estimate [113]</td><td>40K</td><td>91.8 (0.5)</td><td>45.9 (1.7)</td><td>74.4 (0.8)</td><td>61.0 (1.0)</td><td>68.7 (1.2)</td><td>73.0 (3.1)</td></tr></table>

Table 11: Comparison between SPSA and a one-point estimate Zhang et al. [113]. One-point estimate only does one forward pass per step, thus is twice as fast as two-point estimate per step. As such, the number of forward passes after 40K steps with the one-point estimate is the same as the number of forward passes with SPSA after 20K steps. The results show that two-point estimate is much more effective than one-point estimate.

# C Memory Analysis

The compute-memory tradeoff of backpropagation is complex to analyze. Griewank and Walther [40] provides a rigorous theoretical treatment of the problem. We empirically measure the memory

consumption of different methods for commonly used large language models, but here we hope to provide a more rigorous comparison of different gradient estimation algorithms, independent of the software used to implement them. Below, we summarize some key points that may help readers to understand how the MeZO compute-memory tradeoff compares to backpropagation.

Given a network, the first step to perform backpropagation is to decompose the model into easily differentiable blocks. We note that this decomposition is not unique. For each block, one can choose to cache the resulting output during the forward pass (thereby consuming memory) or instead recompute the output when it is needed (thereby consuming compute). The below proposition, adapted from Rule 21 in Griewank and Walther [40], captures this tradeoff.

Proposition 2 (Time-Memory Tradeoff for Backpropagation, Griewank and Walther [40]). Consider a network containing  $N$  bits. For any time-memory tradeoff hyperparameter  $c = O(1)$ , there exists a backpropagation algorithm that runs in time  $O(cN)$  and consumes memory proportional to  $O(N^{1 / c})$ .

Grimm et al. [41] also gave sharp bounds for the memory-time product. Note that the popular gradient checkpointing [18] method allows one to tune  $c$  with limited precision (i.e., one cannot always further split a differentiable block and observe savings). Experiments in Chen et al. [18] choose  $c = 2$  to achieve  $O(\sqrt{N})$  memory while consuming  $O(2N)$  computation. In the extreme case, gradient checkpointing allows one to use  $O(N \log N)$  computation and  $O(\log N)$  memory.

MeZO always consumes  $2N$  compute and  $O(1)$  memory, so it is more compute-efficient at the same memory cost as gradient checkpointing. Our exposition in Section 2 discusses that we can perturb groups of parameters together to save time while consuming additional memory. However, we do not consider that variant here because it is somewhere in the middle of the compute-memory pareto curve, where we cannot reason about what backpropagation will do. In particular, MeZO can split groups differently than backpropagation can, since MeZO does not require that each parameter group is easily differentiable, so it is hard to compare the two algorithms along the entire pareto curve.

We also compare backpropagation for the  $c = 1$  case (i.e., storing everything during the forward pass). When storing everything, backpropagation consumes  $O(N)$  time and  $O(N)$  memory. Hence, SPSA consumes slightly more time and substantially less memory than backpropagation at this end of the tradeoff.

Unlike gradient checkpointing, MeZO computes only an approximation of the gradient. This approximation is only useful for fine-tuning with a prompt, making it less broadly useful than gradient checkpointing. There are other methods that approximate the gradient with less memory consumption than gradient checkpointing (see the Related Work section), though it is unclear how the memory consumption of those algorithms compare to MeZO.

# D Forward Auto-Differentiation

We discuss the merits of using forward auto-differentiation instead of two forward passes to construct a gradient estimate for fine-tuning. As  $\epsilon \rightarrow 0$ , the SPSA gradient estimate (Definition 1) can be written as  $zz^{\top}\nabla \mathcal{L}(\pmb {\theta};\mathcal{B})$ . The term  $z^{\top}\nabla \mathcal{L}(\pmb {\theta};\mathcal{B})$  is a Jacobian-vector product (JVP), and it is well-known that this can be computed in parallel with a single forward pass while consuming additional memory equivalent to that of the largest activation in the model. To fully compute the gradient estimate, one must store  $\pmb{z}$  on the GPU while performing inference, so we observe that this algorithm requires more memory than MeZO.

We note that implementation of the forward auto-differentiation algorithm is not well-supported in PyTorch at the time of writing. The autograd JVP function computes the JVP in a memory-inefficient way, as noted in the documentation, and the other available methods to compute a JVP are not sophisticated enough to easily scale to a complex LLM. Computing the JVP is straightforward when using JAX, so we profile the memory consumption of inference and the JVP for RoBERTa-large when using JAX. We use batch size 16 with the MultiRC task. Note that JAX may automatically use rematerialization to avoid out of memory errors so we focus on settings in which the memory utilization remains below  $50\%$ . The resulting memory usage during inference, backpropagation, and forward auto-differentiation are reported in Table 12.

We see that forward auto-differentiation is substantially more memory efficient than backpropagation but less memory efficient than inference. Furthermore, forward auto-differentiation selects  $\epsilon = 0$ , which removes potentially beneficial third-and-higher order Taylor expansion terms from the estimate.

<table><tr><td>Task</td><td>Inference (and MeZO)</td><td>Backpropagation</td><td>Forward Auto-Differentiation</td></tr><tr><td>Excess Memory (MB)</td><td>327.50</td><td>24156.23</td><td>830.66</td></tr></table>

Table 12: Memory consumption of RoBERTa-large when using batch size 16 with the MultiRC task. The reported memory does not include the cost of storing the model on the GPU, which is required for all three cases.

# E Experiment setup

# E.1 Datasets

For RoBERTa-large, we consider classification datasets: SST-2 [86], SST-5 [86], TREC [97], MNLI [103], SNLI [12], and RTE [22, 8, 37, 10]. We follow Malladi et al. [67] in limiting the test set to 1,000 examples for fast iteration. For training and validation, we have two settings:  $k = 16$  and  $k = 512$ , which mean that we have 16 or 512 examples per class for both training and validation.

For OPT experiments, we consider the SuperGLUE dataset collection [98], including: BoolQ [21], CB [24], COPA [81], MultiRC [51], ReCoRD [111], RTE [22, 8, 37, 10], WiC [77], and WSC [55]. We also include SST-2 [86] and two question answering (QA) datasets, SQuAD [80] and DROP [31]. We randomly sample 1,000 examples for training, 500 examples for validation, and 1,000 examples for testing.

# E.2 Prompts

Table 13 shows the set of downstream tasks and prompts with which we fine-tune RoBERTa-large, which are adapted from [35].  

<table><tr><td>Dataset</td><td>C</td><td>Type</td><td>Prompt</td><td>Label words</td></tr><tr><td>SST-2</td><td>2</td><td>sentiment cls.</td><td>&lt;S1&gt; It was [MASK] .</td><td>{great, terrible}</td></tr><tr><td>SST-5</td><td>5</td><td>sentiment cls.</td><td>&lt;S1&gt; It was [MASK] .</td><td>{great, good, okay, bad, terrible}</td></tr><tr><td>TREC</td><td>6</td><td>topic cls.</td><td>[MASK] : &lt;S1&gt;</td><td>{Description, Expression, Entity, Human, Location, Number}</td></tr><tr><td>MNLI</td><td>3</td><td>NLI</td><td>&lt;S1&gt; ? [MASK] , &lt;S2&gt;</td><td>{Yes, Maybe, No}</td></tr><tr><td>SNLI</td><td>3</td><td>NLI</td><td>&lt;S1&gt; ? [MASK] , &lt;S2&gt;</td><td>{Yes, Maybe, No}</td></tr><tr><td>RTE</td><td>2</td><td>NLI</td><td>&lt;S1&gt; ? [MASK] , &lt;S2&gt;</td><td>{Yes, No}</td></tr></table>

Table 13: The prompts of the datasets we used in our RoBERTa-large experiments (Table 18 and Figure 2). The prompts are adapted from [35] and include a template and a set of label words that can fill in the [MASK] token.  $< S_{1}>$  and  $< S_{2}>$  refer to the first and the second (if any) input sentence.

Table 14 demonstrates the prompts we use for OPT. Note that in OPT experiments we have three types of tasks: classification, multiple-choice, and question answering. Prompts are adopted from GPT-3 [13] and PromptSource with minor changes [5].

# E.3 Hyperparameters

We use the hyperparameters in Table 15 for MeZO experiments on RoBERTa-large (Table 18 and Figure 2). Experiments in Appendix A informed the grid; in particular, the choice of  $\epsilon$  seemed to not significantly impact performance, and using a larger batch size consistently yielded faster optimization. We use the hyperparameters in Table 16 for MeZO experiments on OPT.

Regarding learning rate scheduling and early stopping, we use linear learning scheduling for all fine-tuning with backpropagation experiments and constant learning rate for all MeZO experiments. For RoBERTa experiments, we evaluate the model on validation sets every 1/10 of total training steps and save the best validation checkpoint. All FT experiments use 1K steps and MeZO experiments use

<table><tr><td>Dataset</td><td>Type</td><td>Prompt</td></tr><tr><td>SST-2</td><td>cls.</td><td>It was terrible/great</td></tr><tr><td>RTE</td><td>cls.</td><td>premise&gt;Does this mean that &quot;&lt;hypothesis&gt;&quot; is true? Yes or No?Yes/No</td></tr><tr><td>CB</td><td>cls.</td><td>Suppose &lt;premise&gt; Can we infer that &quot;&lt;hypothesis&gt;?&quot;? Yes, No, or Maybe?Yes/No/Maybe</td></tr><tr><td>BoolQ</td><td>cls.</td><td>&lt;passage&gt; ?Yes/No</td></tr><tr><td>WSC</td><td>cls.</td><td>&lt;text&gt;In the previous sentence, does the pronoun &quot;&lt;span2&gt;&quot; refer to &lt;span1&gt;? Yes or No?Yes/No</td></tr><tr><td>WIC</td><td>cls.</td><td>Does the word &quot;&lt;word&gt;&quot; have the same meaning in these two sentences? Yes, No?&lt;sent1&gt;&lt;sent2&gt;Yes/No</td></tr><tr><td>MultiRC</td><td>cls.</td><td>&lt;paragraph&gt;Question: &lt;question&gt;I found this answer &quot;&lt;answer&quot;. Is that correct? Yes or No?Yes/No</td></tr><tr><td>COPA</td><td>mch.</td><td>&lt;premise&gt; so/because &lt;candidate&gt;</td></tr><tr><td>ReCoRD</td><td>mch.</td><td>&lt; passage&gt; &lt;query&gt;.replace &quot;@placeholder&quot;, &lt;candidate&gt;)</td></tr><tr><td>SQuAD</td><td>QA</td><td>Title: &lt;title&gt;Context: &lt;context&gt;Question: &lt;question&gt;Answer:</td></tr><tr><td>DROP</td><td>QA</td><td>Passage: &lt;context&gt;Question: &lt;question&gt;Answer:</td></tr></table>

Table 14: The prompts of the datasets we used in our OPT experiments. There are three types of tasks: classification (cls.), multiple-choice (mch.), and question answering (QA). Prompts are adopted from GPT-3 [13] and PromptSource [5] with minor changes. <text> represents input from the dataset and Yes represents label words. For inference on multiple choice tasks, we put in different candidates in the prompt and calculate the average log-likelihood for each candidate, and choose the candidate with the highest score. For inference on QA tasks, we use greedy decoding to generate the answer.

100K steps. For OPT experiments, we evaluate the model on validation sets every 1/5 of the total training steps and save the best validation checkpoint. All FT experiments train for 5 epochs and all MeZO experiments use 20K steps. Note that FT experiments mostly converge within 5 epochs but we observe that MeZO performance can still improve with more training steps.

# E.4 Modeling and implementation

For RoBERTa experiments, we follow [35] for the prompt-based fine-tuning paradigm for masked language models. Please refer to the original paper for more details.

In OPT experiments, for classification tasks, we train the model similarly to [35], i.e., we take the logits corresponding to the label words and apply cross entropy loss on them; for multiple choice tasks and generation tasks (QA), we only keep the correct candidate and use teacher forcing to train on the correct examples. We only keep the loss on tokens in the candidate part and exclude the prompt part.

For OPT inference on classification and multiple-choice tasks, we use the model to get the average log-likelihood (by tokens) of all the candidates/label words, and predict the one with the highest average log-likelihood. For generation tasks, we use greedy decoding to generate the answer.

For in-context learning, we use 32 examples in the context. We also try filling in as many examples as possible in the context but this does not improve performance and sometimes leads to unstable results. Thus we keep the 32-example results.

<table><tr><td>Experiment</td><td>Hyperparameters</td><td>Values</td></tr><tr><td rowspan="4">MeZO</td><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>{1e-7, 1e-6, 1e-5}</td></tr><tr><td>ε</td><td>1e-3</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td rowspan="5">MeZO (prefix)</td><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>{1e-2, 5e-3, 1e-3}</td></tr><tr><td>ε</td><td>1e-1</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td># prefix tokens</td><td>5</td></tr><tr><td rowspan="5">MeZO (LoRA)</td><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>{1e-5, 5e-5, 1e-4}</td></tr><tr><td>ε</td><td>1e-3</td></tr><tr><td>Weight Decay</td><td>0.1</td></tr><tr><td>(r, α)</td><td>(8, 16)</td></tr><tr><td rowspan="4">FT with Adam</td><td>Batch size (k=16)</td><td>{2, 4, 8}</td></tr><tr><td>Batch size (k=512)</td><td>{8, 16, 32}</td></tr><tr><td>Learning Rates</td><td>{1e-5, 3e-5, 5e-5}</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td rowspan="4">FT with SGD</td><td>Batch size (k=16)</td><td>{2, 4, 8}</td></tr><tr><td>Batch size (k=512)</td><td>{8, 16, 32}</td></tr><tr><td>Learning Rates</td><td>{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td rowspan="4">FT (prefix)</td><td>Batch size</td><td>{8, 16, 32}</td></tr><tr><td>Learning Rates</td><td>{1e-2, 3e-2, 5e-2}</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td># prefix tokens</td><td>5</td></tr><tr><td rowspan="3">FT (LoRA)</td><td>Batch size</td><td>{4, 8, 16}</td></tr><tr><td>Learning Rates</td><td>{1e-4, 3e-4, 5e-4}</td></tr><tr><td>(r, α)</td><td>(8, 16)</td></tr></table>

Table 15: The hyperparameter grids used for RoBERTa-large experiments. MeZO uses a constant learning rate schedule, and FT uses linear scheduling. All FT experiments use 1K steps and MeZO experiments use 100K steps. We check validation performance every 1/10 total training steps.

For linear probing of classification tasks, we take the output feature and use scipy package to train a linear classifier. For multiple-choice tasks and generation tasks, we found that this leads to poor results since the output space is the whole vocabulary; instead, we do head-tuning, where the whole model is fixed except for the LM projection head. We use a batch size of 8 and a learning rate of  $\{1\mathrm{e} - 45\mathrm{e} - 4\}$ , and train the head for 5 epochs.

For experiments on 30B and 66B OPT models, we largely follow the OPT hyperparameters except that we do not evaluate the intermediate validation performance and directly use the last checkpoint for evaluation, due to the high storage cost of intermediate checkpoints of large models.

# E.5 Parameter-efficient fine-tuning

Fine-tuning and storing a copy of the large language model for each downstream task is expensive. Parameter-efficient fine-tuning (PEFT) techniques alleviate this problem: instead of tuning all model parameters, PEFT only tunes a small number of additional parameters (usually less than  $1\%$ ) and can often achieve comparable or better performance [57, 54, 30]. The ZO optimizer is compatible with PEFT methods, since ZO can operate on any subset of the model parameters. We are interested in the following two common PEFT methods, designed for transformers [96].

LoRA [46] adds a tunable low-rank delta to a linear layer during fine-tuning. Suppose a linear layer performed  $Wx + b$  during pre-training with  $W \in \mathbb{R}^{m \times n}$ . When fine-tuning, LoRA introduces two smaller matrices  $A \in \mathbb{R}^{m \times r}$  and  $B \in \mathbb{R}^{r \times n}$  such that  $r \ll \min(m, n)$ . The linear layer is then

<table><tr><td>Experiment</td><td>Hyperparameters</td><td>Values</td></tr><tr><td rowspan="3">MeZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td>{1e-6, 1e-7} or {1e-6, 5e-7, 1e-7} for SQuAD and DROP</td></tr><tr><td>ε</td><td>1e-3</td></tr><tr><td rowspan="4">MeZO (prefix)</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td>{1e-2, 1e-3} or {5e-2, 1e-2, 5e-3} for SQuAD and DROP</td></tr><tr><td>ε</td><td>1e-1</td></tr><tr><td># prefix tokens</td><td>5</td></tr><tr><td rowspan="4">MeZO (LoRA)</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td>{1e-4, 5e-5} or {1e-4, 5e-5, 1e-5} for SQuAD and DROP</td></tr><tr><td>ε</td><td>1e-2</td></tr><tr><td>(r, α)</td><td>(8, 16)</td></tr><tr><td rowspan="2">FT with Adam</td><td>Batch size</td><td>8</td></tr><tr><td>Learning Rates</td><td>{1e-5, 5e-5, 8e-5}</td></tr></table>

computed as

$$
\left(\boldsymbol {W} + \frac {\alpha}{r} \boldsymbol {A B}\right) \boldsymbol {x} + \boldsymbol {b} \tag {6}
$$

where  $r$  and  $\alpha$  are hyperparameters.  $A$  and  $B$  are trained on the downstream task while  $W$  is frozen at its pre-trained value. In transformers, this modification to the linear layer is applied to the query and value operations of each attention layer. Empirically,  $r$  can be very small, so the number of trainable parameters during fine-tuning is small. We choose  $r = 8$  and  $\alpha = 16$ .

Prefix-tuning [57] adds a prefix of  $m$  tunable representations at each layer and freezes the rest of the model. The representations are added as new keys and values and treated as additional context during the attention operation. We initialize these tunable representations by randomly sampling tokens from the vocabulary and passing them through the LLM to get their keys and values at different attention layers. We found this crucial to make prefix tuning stable with MeZO, and this trick additionally boosts the performance of prefix tuning with backpropagation, as shown in Table 17. We also tried the reparameterization trick in [57], which does not help MeZO training. In our experiments, we find  $m = 5$  to be sufficient to achieve good performance on most tasks.

We also show that MeZO is compatible with parameter-efficient fine-tuning methods, such as prefix tuning and LoRA. Surprisingly, the performance of MeZO does not improve substantially when tuning much fewer parameters, as one might expect from classical analyses (see Section 4). Accordingly, our theoretical analysis in Section 4 suggests that the convergence rate of ZO-SGD does not depend on the parameter dimension during fine-tuning.

Table 16: The hyperparameter grids used for OPT experiments. All weight decay is set to 0. FT uses 5 epochs and linear scheduled learning rates and MeZO uses 20K steps and constant learning rates. We check validation performance and save the best checkpoint every 1/5 total training steps.  

<table><tr><td>Task Type</td><td colspan="2">SST-2 ——sentiment ——</td><td colspan="3">SNLI ——natural language inference ——</td><td>TREC ——topic ——</td></tr><tr><td>FT (prefix, random init)</td><td>90.7 (1.7)</td><td>47.2 (2.0)</td><td>70.7 (2.8)</td><td>62.6 (3.3)</td><td>63.5 (4.4)</td><td>83.4 (4.7)</td></tr><tr><td>FT (prefix, real act init)</td><td>91.9 (1.0)</td><td>47.7 (1.1)</td><td>77.2 (1.3)</td><td>66.5 (2.5)</td><td>66.6 (2.0)</td><td>85.7 (1.3)</td></tr></table>

Table 17: Prefix-tuning ablations. We compare randomly-initialized prefixes and real word activation prefixes. Using real word activations significantly outperforms random initialization.

# E.6 Training with non-differentiable objectives

The experiments maximizing the accuracy of a RoBERTa-large model were all conducted using the same grid as MeZO in Table 15.

For OPT experiments on SQuAD with F1 as objective, we use a batch size of 16. For MeZO, we use a learning rate of  $\{1\mathrm{e} - 6,5\mathrm{e} - 6,1\mathrm{e} - 5\}$  and  $\epsilon = 1\mathrm{e} - 3$ . For MeZO (prefix), we use a learning rate of  $\{1\mathrm{e} - 1,5\mathrm{e} - 2,1\mathrm{e} - 2\}$  and  $\epsilon = 1\mathrm{e} - 1$ .

# E.7 Memory profiling

In memory profiling, we use standard implementation with Huggingface's transformers [104] package. We did not turn on any advance memory-saving options, e.g., gradient checkpointing. We set the per-device batch size as 1 to test the minimum hardware requirement to run the model with specific optimization algorithms. For multi-GPU backpropagation, we use fully sharded data parallel (FSDP) [33] provided by PyTorch [76]. For multi-GPU MeZO, we use transformers multi-GPU inference of large models. We use Nvidia's nvidia-smi command to monitor the GPU memory usage. We call a run "successful" if there is no out of memory error from GPUs for at least 100 steps. We also profile fine-tuning with LoRA, but find its memory usage similar to that of fine-tuning with prefix-tuning. Hence here we only show the analysis with prefix-tuning.

# F More experiment results

# F.1 RoBERTa-large experiments

Table 18 contains the detailed numbers corresponding to Figure 2 and also reports the performance of MeZO-Adam.  

<table><tr><td>Task Type</td><td colspan="2">SST-2 —— sentiment ——</td><td colspan="2">SNLI —— natural language inference ——</td><td colspan="2">RTE —— topic ——</td></tr><tr><td>Zero-shot</td><td>79.0</td><td>35.5</td><td>50.2</td><td>48.8</td><td>51.4</td><td>32.0</td></tr><tr><td colspan="7">Gradient-free methods: k = 16</td></tr><tr><td>LP</td><td>76.0 (2.8)</td><td>40.3 (1.9)</td><td>66.0 (2.7)</td><td>56.5 (2.5)</td><td>59.4 (5.3)</td><td>51.3 (5.5)</td></tr><tr><td>MeZO</td><td>90.5 (1.2)</td><td>45.5 (2.0)</td><td>68.5 (3.9)</td><td>58.7 (2.5)</td><td>64.0 (3.3)</td><td>76.9 (2.7)</td></tr><tr><td>MeZO (LoRA)</td><td>91.4 (0.9)</td><td>43.0 (1.6)</td><td>69.7 (6.0)</td><td>64.0 (2.5)</td><td>64.9 (3.6)</td><td>73.1 (6.5)</td></tr><tr><td>MeZO (prefix)</td><td>90.8 (1.7)</td><td>45.8 (2.0)</td><td>71.6 (2.5)</td><td>63.4 (1.8)</td><td>65.4 (3.9)</td><td>80.3 (3.6)</td></tr><tr><td>MeZO-Adam</td><td>90.4 (1.4)</td><td>45.4 (1.5)</td><td>74.1 (2.7)</td><td>64.3 (0.8)†</td><td>59.2 (11.1)†</td><td>78.3 (1.4)</td></tr><tr><td colspan="7">Gradient-based methods: k = 16</td></tr><tr><td>FT</td><td>91.9 (1.8)</td><td>47.5 (1.9)</td><td>77.5 (2.6)</td><td>70.0 (2.3)</td><td>66.4 (7.2)</td><td>85.0 (2.5)</td></tr><tr><td>FT (LoRA)</td><td>91.4 (1.7)</td><td>46.7 (1.1)</td><td>74.9 (4.3)</td><td>67.7 (1.4)</td><td>66.1 (3.5)</td><td>82.7 (4.1)</td></tr><tr><td>FT (prefix)</td><td>91.9 (1.0)</td><td>47.7 (1.1)</td><td>77.2 (1.3)</td><td>66.5 (2.5)</td><td>66.6 (2.0)</td><td>85.7 (1.3)</td></tr><tr><td colspan="7">Gradient-free methods: k = 512</td></tr><tr><td>LP</td><td>91.3 (0.5)</td><td>51.7 (0.5)</td><td>80.9 (1.0)</td><td>71.5 (1.1)</td><td>73.1 (1.5)</td><td>89.4 (0.5)</td></tr><tr><td>MeZO</td><td>93.3 (0.7)</td><td>53.2 (1.4)</td><td>83.0 (1.0)</td><td>78.3 (0.5)</td><td>78.6 (2.0)</td><td>94.3 (1.3)</td></tr><tr><td>MeZO (LoRA)</td><td>93.4 (0.4)</td><td>52.4 (0.8)</td><td>84.0 (0.8)</td><td>77.9 (0.6)</td><td>77.6 (1.3)</td><td>95.0 (0.7)</td></tr><tr><td>MeZO (prefix)</td><td>93.3 (0.1)</td><td>53.6 (0.5)</td><td>84.8 (1.1)</td><td>79.8 (1.2)</td><td>77.2 (0.8)</td><td>94.4 (0.7)</td></tr><tr><td>MeZO-Adam</td><td>93.3 (0.6)</td><td>53.9 (0.8)</td><td>85.3 (0.8)</td><td>79.6 (0.4)</td><td>79.2 (1.2)</td><td>95.1 (0.3)</td></tr><tr><td colspan="7">Gradient-based methods: k = 512</td></tr><tr><td>FT</td><td>93.9 (0.7)</td><td>55.9 (0.9)</td><td>88.7 (0.8)</td><td>84.4 (0.8)</td><td>82.7 (1.4)</td><td>97.3 (0.2)</td></tr><tr><td>FT (LoRA)</td><td>94.2 (0.2)</td><td>55.3 (0.7)</td><td>88.3 (0.5)</td><td>83.9 (0.6)</td><td>83.2 (1.3)</td><td>97.0 (0.3)</td></tr><tr><td>FT (prefix)</td><td>93.7 (0.3)</td><td>54.6 (0.7)</td><td>88.3 (0.7)</td><td>83.3 (0.5)</td><td>82.5 (0.8)</td><td>97.4 (0.2)</td></tr></table>

Table 18: Experiments on RoBERTa-large (350M parameters). LP: Linear probing; ZO, ZO (LoRA), and ZO (prefix): our memory-efficient ZO-SGD (Section 2.1) with full-parameter tuning, LoRA, and prefix-tuning respectively; FT: fine-tuning with Adam. All reported numbers are averaged accuracy (standard deviation). All experiments use prompts (Appendix E.2). ZO outperforms zero-shot and LP by a large margin and approaches FT performance with much less memory cost.

LP-MeZO We also compare MeZO to performing linear probing and then subsequently performing fine-tuning via MeZO, following the analogous suggestion for fine-tuning in Kumar et al. [53]. We use the MeZO grid described in Table 15. Note that the linear probing checkpoints used here have early stopping, unlike the ones reported in Table 18. We heuristically implement early stopping by limiting the number of iterations (from 5000 to 1000) and increasing the convergence tolerance (from  $1\mathrm{e} - 4$  to 0.01) in the scipy solver. Experiments on a few settings show that LP-MeZO can sometimes improve performance without increasing the memory consumption (see Table 19). However, sometimes, linear probing first can severely hurt performance.

<table><tr><td>Task</td><td>SST-2</td><td>SST-5</td><td>SNLI</td><td>TREC</td></tr><tr><td>Zero-shot</td><td>79.0</td><td>35.5</td><td>50.2</td><td>32.0</td></tr><tr><td>FT</td><td>91.9 (1.8)</td><td>47.5 (1.9)</td><td>77.5 (2.6)</td><td>85.0 (2.5)</td></tr><tr><td>MeZO</td><td>90.5 (1.2)</td><td>45.5 (2.0)</td><td>68.5 (3.9)</td><td>76.9 (2.7)</td></tr><tr><td>LP-MeZO</td><td>91.4 (1.4)</td><td>41.9 (3.3)</td><td>70.7 (3.4)</td><td>54.0 (4.5)</td></tr></table>

Table 19: Performing linear probing before fine-tuning with MeZO, as suggested previously [53], can sometimes improve performance without increasing the memory overhead. We use  $k = 16$  for these experiments.

# F.2 OPT experiments

Table 20 present the full results of OPT-30B and OPT-66B, with detailed MeZO numbers.  

<table><tr><td>Task</td><td>SST-2</td><td>RTE</td><td>BoolQ</td><td>WSC</td><td>WIC</td><td>SQuAD</td></tr><tr><td>30B zero-shot</td><td>56.7</td><td>52.0</td><td>39.1</td><td>38.5</td><td>50.2</td><td>46.5</td></tr><tr><td>30B ICL</td><td>81.9</td><td>66.8</td><td>66.2</td><td>56.7</td><td>51.3</td><td>78.0</td></tr><tr><td>30B MeZO</td><td>90.6</td><td>66.4</td><td>67.2</td><td>63.5</td><td>56.3</td><td>85.2</td></tr><tr><td>30B MeZO (prefix)</td><td>87.5</td><td>72.6</td><td>73.5</td><td>55.8</td><td>59.1</td><td>83.9</td></tr><tr><td>66B zero-shot</td><td>57.5</td><td>67.2</td><td>66.8</td><td>43.3</td><td>50.6</td><td>48.1</td></tr><tr><td>66B ICL</td><td>89.3</td><td>65.3</td><td>62.8</td><td>52.9</td><td>54.9</td><td>81.3</td></tr><tr><td>66B MeZO</td><td>91.2</td><td>65.7</td><td>72.7</td><td>63.5</td><td>58.9</td><td>*</td></tr><tr><td>66B MeZO (prefix)</td><td>93.6</td><td>66.4</td><td>73.7</td><td>57.7</td><td>58.6</td><td>85.0</td></tr></table>

Table 20: Experiments on OPT-30B and OPT-66B (with 1,000 examples).*: MeZO requires further tuning to successfully optimize.

# F.3 Convergence of MeZO with full-parameter and PEFT

We demonstrate the convergence rate of MeZO, MeZO (LoRA) and MeZO (prefix) on SST-2 and SNLI for the first 5,000 steps in Figures 5. We see that despite the different number of parameters they optimize, MeZO demonstrates similar training speed on full parameter and PEFT. This agrees with our theory in Section 4, which shows that MeZO's optimization speed is independent of the number of parameters.

Figure 5: MeZO does not optimize significantly faster when tuning fewer parameters, agreeing with our theory in Section 4.


# F.4 ZO vs BBTv2

We compare ZO with BBTv2 [90] on mutually assessed tasks in Table 21. ZO significantly outperforms BBTv2. Furthermore, BBTv2 is limited to optimize in low-dimensional space and requires prefix-tuning and a down-projection to reduce the number of optimized parameters. BBTv2 also employs an iterative scheme which only optimizes one layer at a time. In contrast, ZO works with both full-parameter tuning and PEFT, as shown in our experiments (Section 3) and theory (Section 4).

<table><tr><td>Task</td><td>SST-2</td><td>SNLI</td><td>RTE</td></tr><tr><td>Task type</td><td>sentiment</td><td colspan="2">- natural language inference</td></tr><tr><td>Zero-shot</td><td>79.0</td><td>50.2</td><td>51.4</td></tr><tr><td>BBTv2</td><td>90.3 (1.7)</td><td>57.3 (2.3)</td><td>56.7 (3.3)</td></tr><tr><td>MeZO</td><td>90.5 (1.2)</td><td>68.5 (3.9)</td><td>64.0 (3.3)</td></tr><tr><td>MeZO (LoRA)</td><td>91.4 (0.9)</td><td>69.7 (6.0)</td><td>64.9 (3.6)</td></tr><tr><td>MeZO (prefix)</td><td>90.8 (1.7)</td><td>71.6 (2.5)</td><td>65.4 (3.9)</td></tr></table>

# F.5 Memory profiling

We show the detailed numbers of memory profiling results Table 22, which also corresponds to Figure 3. For how we profile the memory usage, please refer to Appendix E.7.

Table 21: ZO vs BBTv2 with RoBERTa-large. BBTv2 performance is from Sun et al. [90].  

<table><tr><td>Method</td><td>Zero-shot / MeZO</td><td>ICL</td><td>Prefix FT</td><td>Full-parameter FT</td></tr><tr><td>1.3B</td><td>1xA100 (4GB)</td><td>1xA100 (6GB)</td><td>1xA100 (19GB)</td><td>1xA100 (27GB)</td></tr><tr><td>2.7B</td><td>1xA100 (7GB)</td><td>1xA100 (8GB)</td><td>1xA100 (29GB)</td><td>1xA100 (55GB)</td></tr><tr><td>6.7B</td><td>1xA100 (14GB)</td><td>1xA100 (16GB)</td><td>1xA100 (46GB)</td><td>2xA100 (156GB)</td></tr><tr><td>13B</td><td>1xA100 (26GB)</td><td>1xA100 (29GB)</td><td>2xA100 (158GB)</td><td>4xA100 (316GB)</td></tr><tr><td>30B</td><td>1xA100 (58GB)</td><td>1xA100 (62GB)</td><td>4xA100 (315GB)</td><td>8xA100 (633GB)</td></tr><tr><td>66B</td><td>2xA100 (128GB)</td><td>2xA100 (134GB)</td><td>8xA100</td><td>16xA100</td></tr></table>

Table 22: Memory usage on the MultiRC (avg #tokens=400) dataset.

# F.6 Wallclock time efficiency

In this section, we measure the wallclock time efficiency of MeZO compared to full-parameter FT, with respect to different model sizes. We conduct our experiments with 80GB A100s connected by NVLink and InfiniteBand, which are state-of-the-art solutions for distributed training. As shown in Table 23, on the MultiRC datasets, training with MeZO brings  $7.74 \times$  speedup per step compared to full-parameter FT on a 30B model. This is due to (1) MeZO does not require costly backpropagation and (2) MeZO requires fewer GPUs and reduces the multi-GPU communication overhead. We can see that MeZO has a bigger advantage when training larger models—the multi-GPU overhead for fine-tuning is larger.

Note that even though MeZO has better per-step wallclock efficiency, it requires significantly more steps than standard FT. Taking our OPT-30B experiments as an example: MeZO takes  $32 \times$  more steps than standard FT, while FT takes  $8 \times$  more GPUs and  $7.74 \times$  more time per step. Overall, MeZO requires only half as many GPU-hours as FT for a 30B model.

<table><tr><td></td><td>1.3B</td><td>2.7B</td><td>13B</td><td>30B</td><td>66B</td></tr><tr><td>MeZO (bsz=16)</td><td>0.815s (1)</td><td>1.400s (1)</td><td>2.702s (1)</td><td>5.896s (1)</td><td>12.438s (4)</td></tr><tr><td>MeZO (bsz=8)</td><td>0.450s (1)</td><td>0.788s (1)</td><td>1.927s (1)</td><td>4.267s (1)</td><td>7.580s (2)</td></tr><tr><td>FT (bsz=8)</td><td>0.784s (1)</td><td>1.326s (1)</td><td>13.638s (4)</td><td>45.608s (8)</td><td>84.098s (20)</td></tr><tr><td></td><td>bspd=2, ga=4</td><td>bspd=2, ga=4</td><td>bspd=1, ga=2</td><td>bspd=1, ga=1</td><td>bspd=1, ga=1</td></tr></table>

Table 23: Wallclock time per step of different training methods. Numbers in brackets are numbers of GPUs required. It is measured on 80GB A100s with NVLink and InfiniteBand connections. The wallclock time is averaged over 100 steps. It is measured on the MultiRC task with the OPT family. We use a batch size ("bsz") of 8 for FT and 16 for MeZO (consistent with our main experiment setting). For comparison we also add MeZO with a batch size of 8. For FT (FSDP), we show the following additional information. "bspd": batch size per device. "ga": gradient accumulation steps. The effective batch size is  $\text{bspd} \times \text{ga} \times \# \text{GPUs}$ . Note that for FT 66B, the effective batch size 20.

# G Proofs

Proof of Lemma 2. We first note that in the  $\epsilon \to 0$  limit, we have

$$
\widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) = \frac {1}{B n} \sum_ {(\boldsymbol {x}, \boldsymbol {y}) \in \mathcal {B}} \sum_ {i \in [ n ]} \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \{(\boldsymbol {x}, \boldsymbol {y}) \}).
$$

Taking expectation over the batch  $\mathcal{B}$  and the  $z_{i}$ , we have  $\mathbb{E}[\widehat{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B})] = \nabla \mathcal{L}(\pmb {\theta})$ , so  $\widehat{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B})$  is an unbiased estimator of the gradient.

Computing the second moment, we get

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] \\ = \frac {1}{B ^ {2} n ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \sum_ {i, j \in [ n ]} \mathbb {E} \left[ \left(\boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\})\right) \left(\boldsymbol {z} _ {j} \boldsymbol {z} _ {j} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\})\right) ^ {\top} \right] \\ \end{array}
$$

Let  $\mathbf{u}, \mathbf{v}$  be two arbitrary vectors. We have that

$$
\mathbb {E} _ {\boldsymbol {z} _ {i}, \boldsymbol {z} _ {j}} \left[ \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {u v} ^ {\top} \boldsymbol {z} _ {j} \boldsymbol {z} _ {j} ^ {\top} \right] = \boldsymbol {u v} ^ {\top}
$$

when  $i\neq j$  , and

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {u v} ^ {\top} \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \right] = \mathbb {E} _ {\boldsymbol {z}} \left[ \boldsymbol {z} ^ {\otimes 4} \right] (\boldsymbol {u}, \boldsymbol {v}) \\ = \frac {3 d}{d + 2} \operatorname {S y m} (\boldsymbol {I} ^ {\otimes 2}) (\boldsymbol {u}, \boldsymbol {v}) \\ = \frac {d}{d + 2} \cdot \boldsymbol {u} ^ {\top} \boldsymbol {v} \cdot \boldsymbol {I} + \frac {2 d}{d + 2} \cdot \boldsymbol {u v} ^ {\top}. \\ \end{array}
$$

Therefore

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] \\ = \frac {1}{B ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \left(\frac {n - 1}{n} + \frac {2 d}{n (d + 2)}\right) \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] \\ + \frac {d}{n (d + 2)} \cdot \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) ^ {\top} \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) \right] \boldsymbol {I}. \\ \end{array}
$$

Next, note that when  $(\pmb{x}_1,\pmb{y}_1)\neq (\pmb{x}_2,\pmb{y}_2)$ , we have

$$
\mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] = \nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top},
$$

and when  $(\pmb{x}_1,\pmb{y}_1) = (\pmb{x}_2,\pmb{y}_2)$  we have

$$
\mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] = \nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \boldsymbol {\Sigma} _ {M B} (\boldsymbol {\theta}).
$$

Therefore

$$
\frac {1}{B ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] = \nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \frac {1}{B} \boldsymbol {\Sigma} (\boldsymbol {\theta}),
$$

and plugging this yields

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] = \left(1 + \frac {d - 2}{n (d + 2)}\right) \cdot \left(\nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \frac {1}{B} \boldsymbol {\Sigma} (\boldsymbol {\theta})\right) \tag {7} \\ + \frac {d}{n (d + 2)} \boldsymbol {I} \cdot \left(\| \nabla \mathcal {L} (\boldsymbol {\theta}) \| ^ {2} + \frac {1}{B} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta}))\right). \\ \end{array}
$$

Finally, we have

$$
\begin{array}{l} \mathbb {E} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| ^ {2} \right] = \left(1 + \frac {d ^ {2} + d - 2}{n (d + 2)}\right) \cdot \left(\| \nabla \mathcal {L} (\boldsymbol {\theta}) \| ^ {2} + \frac {1}{B} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta}))\right) \\ = \frac {d + n - 1}{n} \cdot \mathbb {E} \left[ \| \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \| ^ {2} \right]. \\ \end{array}
$$

□

Proof of Theorem 1. By Taylor's theorem with remainder, we have that

$$
\begin{array}{l} \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) = \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) + \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) ^ {\top} \left(\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}\right) \\ + \int_ {0} ^ {1} \lambda (\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}) ^ {\top} \nabla^ {2} \mathcal {L} (\lambda \boldsymbol {\theta} _ {t + 1} + (1 - \lambda) \boldsymbol {\theta} _ {t}) (\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}) ^ {\top} d \lambda \\ \end{array}
$$

Next, note that

$$
\left\| \boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t} \right\| = \eta \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| \leq \eta \sqrt {d} \cdot \frac {1}{B n} \sum \left| \boldsymbol {z} _ {i} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \{(\boldsymbol {x}, \boldsymbol {y}) \}) \right| \leq \eta d G (\boldsymbol {\theta} _ {t}).
$$

Therefore  $\| \lambda \pmb{\theta}_{t + 1} + (1 - \lambda) \pmb{\theta}_t - \pmb{\theta}_t \| \leq \eta dG(\pmb{\theta}_t)$ . By the assumption we have the upper bound  $\nabla^2 \mathcal{L}(\lambda \pmb{\theta}_{t + 1} + (1 - \lambda) \pmb{\theta}_t) \preceq H(\pmb{\theta}_t)$ , and thus

$$
\begin{array}{l} \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) + \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) ^ {\top} \left(\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}\right) + \left(\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}\right) ^ {\top} \boldsymbol {H} \left(\boldsymbol {\theta} _ {t}\right) \left(\boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t}\right) \\ = \mathcal {L} (\boldsymbol {\theta} _ {t}) - \eta \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) ^ {\top} \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) + \frac {1}{2} \eta^ {2} \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) ^ {\top} \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}). \\ \end{array}
$$

Taking the conditional expectation with respect to  $\pmb{\theta}_t$  and plugging in (9), the formula for the covariance of our ZO estimate  $\widehat{\nabla}\mathcal{L}(\pmb{\theta}_t;\mathcal{B})$ , yields

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \eta \left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {\eta^ {2}}{2} \left\langle \boldsymbol {H} \left(\boldsymbol {\theta} _ {t}\right), \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} \left(\boldsymbol {\theta}; \mathcal {B}\right) \widehat {\nabla} \mathcal {L} \left(\boldsymbol {\theta}; \mathcal {B}\right) ^ {\top} \right] \right\rangle \\ = \mathcal {L} (\boldsymbol {\theta} _ {t}) - \eta \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {\eta^ {2}}{2} \cdot \frac {d}{n (d + 2)} \left(\| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {1}{B} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta} _ {t}))\right) \operatorname {t r} (\boldsymbol {H} (\boldsymbol {\theta} _ {t})) \\ + \frac {\eta^ {2}}{2} \left(1 + \frac {d - 2}{n (d + 2)}\right) \left(\nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) ^ {\top} \boldsymbol {H} \left(\boldsymbol {\theta} _ {t}\right) \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) + \frac {1}{B} \langle \boldsymbol {\Sigma} \left(\boldsymbol {\theta} _ {t}\right), \boldsymbol {H} \left(\boldsymbol {\theta} _ {t}\right) \rangle\right) \\ \end{array}
$$

By assumption, the Hessian upper bound  $\mathbf{H}(\pmb{\theta}_t)$  satisfies  $\| \mathbf{H}(\pmb{\theta}_t) \|_{op} \leq \ell$  and  $\mathrm{tr}(\mathbf{H}(\pmb{\theta}_t)) \leq \ell r$ . Thus

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \eta \left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {\eta^ {2} \ell}{2} \cdot \left(\frac {d r + d - 2}{n (d + 2)} + 1\right) \cdot \left(\left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {1}{B} \operatorname {t r} \left(\boldsymbol {\Sigma} \left(\boldsymbol {\theta} _ {t}\right)\right)\right) \\ = \mathcal {L} (\boldsymbol {\theta} _ {t}) - \eta \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {\eta^ {2} \ell}{2} \cdot \left(\frac {d r + d - 2}{n (d + 2)} + 1\right) \cdot \mathbb {E} \left[ \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \| ^ {2} \right], \\ \end{array}
$$

as desired.


# G.1 Proofs of Global Convergence

Lemma 4. Let  $\mathcal{L}(\pmb{\theta})$  be  $\mu$ -PL and let there exist  $\alpha$  such that  $\mathrm{tr}(\pmb{\Sigma}(\pmb{\theta})) \leq \alpha (\mathcal{L}(\pmb{\theta}) - \mathcal{L}^{*})$  for all  $\pmb{\theta}$ . Then after

$$
t = O \left(\left(\frac {\ell}{\mu} + \frac {\ell \alpha}{\mu^ {2} B}\right) \log \frac {\mathcal {L} (\boldsymbol {\theta} _ {0}) - \mathcal {L} ^ {*}}{\epsilon}\right)
$$

iterations of SGD we have  $\mathbb{E}[\mathcal{L}(\pmb{\theta}_t)] \leq \mathcal{L}^* + \epsilon$ .

Proof of Lemma 4. The descent lemma for SGD yields

$$
\mathbb {E} [ \mathcal {L} (\pmb {\theta} _ {t + 1}) \mid \pmb {\theta} _ {t} ] - \mathcal {L} (\pmb {\theta} _ {t}) \leq - \eta \left\| \nabla \mathcal {L} (\pmb {\theta} _ {t}) \right\| ^ {2} + \frac {1}{2} \eta^ {2} \ell \cdot \mathbb {E} [ \| \nabla \mathcal {L} (\pmb {\theta} _ {t}; \mathcal {B}) \| ^ {2} ].
$$

Plugging in  $\mathbb{E}[\| \nabla \mathcal{L}(\pmb{\theta}_t; \mathcal{B})\|^2] = \| \nabla \mathcal{L}(\pmb{\theta}_t)\|^2 + \frac{1}{B}\operatorname{tr}(\pmb{\Sigma}(\pmb{\theta}_t))$  and selecting a learning rate  $\eta \leq \frac{1}{\ell}$  yields

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \frac {\eta}{2} \left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {\eta^ {2} \ell}{2 B} \operatorname {t r} \left(\boldsymbol {\Sigma} \left(\boldsymbol {\theta} _ {t}\right)\right)
$$

Since  $\mathcal{L}$  is  $\mu$ -PL, we get

$$
\mathbb {E} [ \mathcal {L} (\pmb {\theta} _ {t + 1}) \mid \pmb {\theta} _ {t} ] \leq \mathcal {L} (\pmb {\theta} _ {t}) - \eta \mu (\mathcal {L} (\pmb {\theta} _ {t}) - \mathcal {L} ^ {*}) + \frac {\eta^ {2} \ell}{2 B} \operatorname {t r} (\pmb {\Sigma} (\pmb {\theta} _ {t})).
$$

Since  $\operatorname{tr}(\pmb{\Sigma}(\pmb{\theta}_t)) \leq \alpha (\mathcal{L}(\pmb{\theta}_t) - \mathcal{L}^*)$ , we have

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \eta \mu \left(\mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \mathcal {L} ^ {*}\right) + \frac {\eta^ {2} \ell \alpha}{2 B} \left(\mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) - \mathcal {L} ^ {*}\right).
$$

Altogether,

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \right] - \mathcal {L} ^ {*} \leq \left(1 - \eta \mu + \frac {\eta^ {2} \ell \alpha}{2 B}\right) \left(\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right] - \mathcal {L} ^ {*}\right)
$$

Choosing  $\eta = \min \left(\frac{1}{\ell}, \frac{\mu B}{\ell \alpha}\right)$ , we obtain

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \right] - \mathcal {L} ^ {*} \leq \left(1 - \min  \left(\frac {\mu}{2 \ell}, \frac {\mu^ {2} B}{2 \ell \alpha}\right)\right) \left(\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right] - \mathcal {L} ^ {*}\right).
$$

Therefore we reach a solution with  $\mathbb{E}[\mathcal{L}(\pmb{\theta}_t)] - \mathcal{L}^* \leq \epsilon$  after

$$
t = \max \left(\frac {2 \ell}{\mu}, \frac {2 \ell \alpha}{\mu^ {2} B}\right) \log \left(\frac {\mathcal {L} (\pmb {\theta} _ {0}) - \mathcal {L} ^ {*}}{\epsilon}\right) = O \left(\left(\frac {\ell}{\mu} + \frac {\ell \alpha}{\mu^ {2} B}\right) \log \frac {\mathcal {L} (\pmb {\theta} _ {0}) - \mathcal {L} ^ {*}}{\epsilon}\right)
$$

iterations.


Proof of Lemma 3. By Corollary 1, ZO-SGD with  $\eta_{\mathrm{ZO}} = \gamma^{-1}\eta_{\mathrm{SGD}}$  yields

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \leq \frac {1}{\gamma} \cdot \left[ - \eta_ {\mathrm {S G D}} \| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \| ^ {2} + \frac {1}{2} \eta_ {\mathrm {S G D}} ^ {2} \ell \cdot \mathbb {E} [ \| \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \| ^ {2} ] \right].
$$

As in the proof for SGD, choosing  $\eta_{\mathrm{SGD}} \leq \frac{1}{\ell}$  yields

$$
\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \leq \gamma^ {- 1} \cdot \left[ - \frac {\eta_ {\mathrm {S G D}}}{2} \left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {\eta_ {\mathrm {S G D}} ^ {2} \ell}{2 B} \operatorname {t r} \left(\boldsymbol {\Sigma} \left(\boldsymbol {\theta} _ {t}\right)\right) \right].
$$

Therefore under  $\mu$ -PL and the  $\mathrm{tr}(\Sigma(\pmb{\theta}_t)) \leq \alpha (\mathcal{L}(\pmb{\theta}_t) - \mathcal{L}^*)$  assumption we obtain

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \right] - \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right] \leq \gamma^ {- 1} \cdot \left[ - \eta_ {\mathrm {S G D}} \mu + \frac {\eta_ {\mathrm {S G D}} ^ {2} \ell \alpha}{2 B} \right] \cdot \left(\mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right] - \mathcal {L} ^ {*}\right) \\ \Longrightarrow \mathbb {E} [ \mathcal {L} (\boldsymbol {\theta} _ {t + 1}) ] - \mathcal {L} ^ {*} \leq \left(1 - \gamma^ {- 1} \left(\eta_ {\mathrm {S G D}} \mu - \frac {\eta_ {\mathrm {S G D}} ^ {2} \ell \alpha}{2 B}\right)\right) (\mathbb {E} [ \mathcal {L} (\boldsymbol {\theta} _ {t}) ] - \mathcal {L} ^ {*}). \\ \end{array}
$$

Choosing  $\eta_{\mathrm{SGD}} = \min \left(\frac{1}{\ell},\frac{\mu B}{\ell\alpha}\right)$  yields

$$
\mathbb {E} [ \mathcal {L} (\boldsymbol {\theta} _ {t + 1}) ] - \mathcal {L} ^ {*} \leq \left(1 - \gamma^ {- 1} \cdot \min  \left(\frac {\mu}{2 \ell}, \frac {\mu^ {2} B}{2 \ell \alpha}\right)\right) (\mathbb {E} [ \mathcal {L} (\boldsymbol {\theta} _ {t}) ] - \mathcal {L} ^ {*}).
$$

Therefore we reach a solution with  $\mathbb{E}[\mathcal{L}(\pmb{\theta}_t)] - \mathcal{L}^* \leq \epsilon$  after

$$
t = \gamma \cdot \max \left(\frac {2 \ell}{\mu}, \frac {2 \ell \alpha}{\mu^ {2} B}\right) \log \left(\frac {\mathcal {L} (\pmb {\theta} _ {0}) - \mathcal {L} ^ {*}}{\epsilon}\right) = \mathcal {O} \left(\left(\frac {r}{n} + 1\right) \cdot \left(\frac {\ell}{\mu} + \frac {\ell \alpha}{\mu^ {2} B}\right) \log \frac {\mathcal {L} (\pmb {\theta} _ {0}) - \mathcal {L} ^ {*}}{\epsilon}\right)
$$

iterations.


# G.1.1 Verification of assumptions

We show that the  $\mathrm{tr}(\pmb{\Sigma}(\pmb{\theta}_t)) \leq \alpha (\mathcal{L}(\pmb{\theta}_t) - \mathcal{L}^*)$  assumption holds for certain losses.

First, consider optimizing the model  $f(\pmb{x}; \pmb{\theta})$  with square loss, so that

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i \in [ N ]} (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) - \boldsymbol {y} _ {i}) ^ {2}.
$$

One then has that

$$
\boldsymbol {\Sigma} (\boldsymbol {\theta}) = \frac {2}{N} \sum_ {i \in [ N ]} (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) - \boldsymbol {y} _ {i}) ^ {2} \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) ^ {\top} - \nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top}.
$$

Therefore

$$
\begin{array}{l} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta})) \leq \frac {2}{N} \sum_ {i \in [ N ]} (f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) - \boldsymbol {y} _ {i}) ^ {2} \| \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \| ^ {2} \\ \leq 2 \mathcal {L} (\boldsymbol {\theta}) \sum_ {i \in [ N ]} \| \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \| ^ {2}. \\ \end{array}
$$

Assume that the data can be interpolated, i.e.,  $\mathcal{L}^* = 0$ . If the function is  $L$ -Lipschitz, i.e.,  $\| \nabla f(\pmb{x};\pmb{\theta}) \| \leq L$ , then the condition holds with  $\alpha = 2NL^2$ . If we are in the kernel regime, i.e.,  $f(\pmb{x}_i; \pmb{\theta}) = \phi(\pmb{x}_i)^\top \pmb{\theta}$  for some feature map  $\phi$ , then

$$
\nabla^ {2} \mathcal {L} (\boldsymbol {\theta}) = \frac {2}{N} \sum_ {i \in [ N ]} f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) ^ {\top}.
$$

Thus

$$
\operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta})) \leq N \operatorname {t r} \left(\nabla^ {2} \mathcal {L} (\boldsymbol {\theta})\right) \cdot \mathcal {L} (\boldsymbol {\theta}) \leq N \ell r \cdot \mathcal {L} (\boldsymbol {\theta}).
$$

So the condition holds for  $\alpha = N\ell r$

Next, consider the cross entropy loss function, i.e

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i \in [ N ]} \exp (- y _ {i} f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})).
$$

One then has that

$$
\boldsymbol {\Sigma} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i \in [ N ]} \exp (- 2 y _ {i} f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})) y _ {i} ^ {2} \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) ^ {\top} - \mathcal {L} (\boldsymbol {\theta}) \mathcal {L} (\boldsymbol {\theta}) ^ {\top},
$$

Assume that the targets  $y_{i}$  are bounded in  $[-1, 1]$  (which is true for binary classification tasks), and that  $\mathcal{L}^{*} = 0$  (which can be achieved if  $|f(\pmb{x};\pmb{\theta})|$  can be sent to  $\infty$ ) we have that

$$
\operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta})) \leq \frac {1}{N} \sum_ {i \in [ N ]} \exp (- 2 y _ {i} f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})) \| \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \| ^ {2}.
$$

In the kernel regime,  $f(\pmb{x}_i;\pmb{\theta}) = \phi(\pmb{x}_i)^\top \pmb{\theta}$ , and thus

$$
\nabla^ {2} \mathcal {L} (\boldsymbol {\theta}) = \frac {1}{N} \sum_ {i \in [ N ]} \exp (- y _ {i} f (\boldsymbol {x} _ {i}; \boldsymbol {\theta})) \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) \nabla f (\boldsymbol {x} _ {i}; \boldsymbol {\theta}) ^ {\top}.
$$

Therefore

$$
\operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta})) \leq N \operatorname {t r} \left(\nabla^ {2} \mathcal {L} (\boldsymbol {\theta})\right) \cdot \mathcal {L} (\boldsymbol {\theta}) \leq N \ell r \cdot \mathcal {L} (\boldsymbol {\theta}).
$$

Therefore the condition holds with  $\alpha = N\ell r$  as well.

# G.2 Proofs for Gaussian perturbations

The first lemma computes the second moment of the covariance estimate  $\widehat{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B})$  when  $z$  is drawn  $\mathcal{N}(0,\pmb {I})$

Lemma 5. Let  $z_{i}\sim \mathcal{N}(0,I)$  i.i.d. Then

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] = \left(1 + \frac {1}{n}\right) \cdot \left(\nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \frac {1}{B} \boldsymbol {\Sigma} _ {M B} (\boldsymbol {\theta})\right) \tag {8} \\ + \frac {1}{n} \boldsymbol {I} \cdot \left(\| \nabla \mathcal {L} (\boldsymbol {\theta}) \| ^ {2} + \frac {1}{B} \operatorname {t r} \left(\boldsymbol {\Sigma} _ {M B} (\boldsymbol {\theta})\right)\right). \\ \end{array}
$$

Proof. As in the proof of Lemma 2, we have that in the  $\epsilon \to 0$  limit

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] \\ = \frac {1}{B ^ {2} n ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \sum_ {i, j \in [ n ]} \mathbb {E} \left[ \right.\left(\boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right)\right\}\right)\left. \right)\left(\boldsymbol {z} _ {j} \boldsymbol {z} _ {j} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right)\right\})\right) ^ {\top} \left. \right] \\ \end{array}
$$

For vectors  $\mathbf{u}, \mathbf{v}$ , we have that

$$
\mathbb {E} _ {\boldsymbol {z} _ {i}, \boldsymbol {z} _ {j}} [ \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {u v} ^ {\top} \boldsymbol {z} _ {j} \boldsymbol {z} _ {j} ^ {\top} ] = \boldsymbol {u v} ^ {\top}
$$

when  $i\neq j$  , and

$$
\mathbb {E} _ {\boldsymbol {z} _ {i}} [ \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {u} \boldsymbol {v} ^ {\top} \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} ] = \mathbb {E} _ {\boldsymbol {z}} [ \boldsymbol {z} ^ {\otimes 4} ] (\boldsymbol {u}, \boldsymbol {v}) = 3 \mathrm {S y m} (\boldsymbol {I} ^ {\otimes 2}) (\boldsymbol {u}, \boldsymbol {v}) = \boldsymbol {u} ^ {\top} \boldsymbol {v} \cdot \boldsymbol {I} + 2 \boldsymbol {u} \boldsymbol {v} ^ {\top}.
$$

Therefore

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] \\ = \frac {1}{B ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \left(\frac {n - 1}{n} + \frac {2}{n}\right) \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] \\ + \frac {1}{n} \cdot \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) ^ {\top} \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) \right] \boldsymbol {I}. \\ \end{array}
$$

In the proof of Lemma 2 we showed that

$$
\frac {1}{B ^ {2}} \sum_ {\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right), \left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \in \mathcal {B}} \mathbb {E} \left[ \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {1}, \boldsymbol {y} _ {1}\right) \right\}) \mathcal {L} (\boldsymbol {\theta}; \left\{\left(\boldsymbol {x} _ {2}, \boldsymbol {y} _ {2}\right) \right\}) ^ {\top} \right] = \nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \frac {1}{B} \boldsymbol {\Sigma} (\boldsymbol {\theta}).
$$

Plugging this yields

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right] = \left(\frac {n + 1}{n}\right) \cdot \left(\nabla \mathcal {L} (\boldsymbol {\theta}) \nabla \mathcal {L} (\boldsymbol {\theta}) ^ {\top} + \frac {1}{B} \boldsymbol {\Sigma} (\boldsymbol {\theta})\right) \tag {9} \\ + \frac {1}{n} \boldsymbol {I} \cdot \left(\| \nabla \mathcal {L} (\boldsymbol {\theta}) \| ^ {2} + \frac {1}{B} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta}))\right). \\ \end{array}
$$


We can prove an analog to Theorem 1 in the case where the  $z_{i}$  are Gaussian. One challenge is that  $\| \pmb{\theta}_{t + 1} - \pmb{\theta}_t\|$  is no longer bounded; instead we the  $r$ -local effective rank assumption only holds with high probability, and thus to bound the expected loss decrease we must control the probability of the  $\| \pmb{\theta}_{t + 1} - \pmb{\theta}_t\|$  being large.

Consider the following modified version of the local  $r$ -effective rank assumption, where the upper bound on the Hessian is measured over a ball of radius twice as large as the one in Assumption 1.

Assumption 2 (Local  $r$ -effective rank, Gaussian). Let  $G(\pmb{\theta}_t) = \max_{(\pmb{x}, \pmb{y}) \in \mathcal{D}} \| \nabla \mathcal{L}(\pmb{\theta}_t; \{(\pmb{x}, \pmb{y})\}) \|$ . There exists a matrix  $\pmb{H}(\pmb{\theta}_t)$  such that:

1. For all  $\pmb{\theta}$  such that  $\| \pmb {\theta} - \pmb {\theta}_t\| \leq 2\eta dG(\pmb {\theta}_t)$ , we have  $\nabla^2\mathcal{L}(\pmb {\theta})\preceq \pmb {H}(\pmb {\theta}_t)$ .  
2. The effective rank of  $\pmb{H}(\pmb{\theta}_t)$ , i.e.,  $\mathrm{tr}(\pmb{H}(\pmb{\theta}_t)) / \| \pmb{H}(\pmb{\theta}_t) \|_{op'}$  is at most  $r$ .

Theorem 2 (Dimension-Free Rate, Gaussian  $z$ ). Assume the loss exhibits local  $r$ -effective rank (Assumption 2). If  $\pmb{\theta}_{t+1} = \pmb{\theta}_t - \eta_{ZO}\widehat{\nabla}\mathcal{L}(\pmb{\theta}_t; \mathcal{B})$  is a single step of ZO-SGD using the  $n$ -SPSA estimate with a minibatch of size  $B$ , then there exists a  $\gamma = \Theta(r/n)$  such that the expected loss decrease can be bounded as

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \\ \leq - \eta_ {Z O} \| \nabla \mathcal {L} (\pmb {\theta} _ {t}) \| ^ {2} + \frac {1}{2} \eta_ {Z O} ^ {2} \ell \cdot \gamma \cdot \mathbb {E} [ \| \nabla \mathcal {L} (\pmb {\theta} _ {t}; \mathcal {B}) \| ^ {2} ] + \eta_ {Z O} ^ {2} \ell G (\pmb {\theta} _ {t}) ^ {2} \exp (- \Omega (n d)). \\ \end{array}
$$

Proof of Theorem 2. Let  $\mathcal{A}$  be the event that  $\| \pmb{\theta}_{t + 1} - \pmb{\theta}_t\| \leq 2\eta dG(\pmb{\theta}_t)$ . On  $\mathcal{A}$ , we have that

$$
\mathcal {L} (\boldsymbol {\theta} _ {t + 1}) \leq \mathcal {L} (\boldsymbol {\theta} _ {t}) - \eta \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) ^ {\top} \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) + \frac {1}{2} \eta^ {2} \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) ^ {\top} \boldsymbol {H} (\boldsymbol {\theta}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}).
$$

Likewise, since  $\mathcal{L}$  is  $\ell$ -smooth, we have that

$$
\mathcal {L} (\boldsymbol {\theta} _ {t + 1}) \leq \mathcal {L} (\boldsymbol {\theta} _ {t}) - \eta \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) ^ {\top} \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) + \frac {1}{2} \eta^ {2} \ell \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \right\| ^ {2}.
$$

Therefore

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] \leq \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) - \eta \left\| \nabla \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \right\| ^ {2} + \frac {1}{2} \eta^ {2} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \cdot \mathbf {1} (\mathcal {A}) \right], \boldsymbol {H} \left(\boldsymbol {\theta} _ {t}\right) \right\rangle \\ + \frac {1}{2} \eta^ {2} \ell \mathbb {E} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \right\| ^ {2} \cdot \mathbf {1} (\neg \mathcal {A}) \right] \\ = \mathcal {L} (\boldsymbol {\theta} _ {t + 1}) - \eta \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {1}{2} \eta^ {2} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right], \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \right\rangle \\ \frac {1}{2} \eta^ {2} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \cdot \mathbf {1} (\neg \mathcal {A}) \right], \ell I - \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \right\rangle . \\ \end{array}
$$

The latter term can be bounded as follows

$$
\begin{array}{l} \frac {1}{2} \eta^ {2} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \cdot \mathbf {1} (\neg \mathcal {A}) \right], \ell I - \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \right\rangle \leq \eta^ {2} \ell \mathbb {E} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| ^ {2} \cdot \mathbf {1} (\neg \mathcal {A}) \right] \\ \leq \eta^ {2} \ell \mathbb {E} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| ^ {4} \right] ^ {\frac {1}{2}} \Pr [ \neg \mathcal {A} ] ^ {1 / 2}. \\ \end{array}
$$

The gradient estimate  $\widehat{\nabla}\mathcal{L}(\pmb {\theta};\mathcal{B})$  satisfies

$$
\left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| \leq \frac {1}{n} \sum_ {i \in [ n ]} \left| \boldsymbol {z} _ {i} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right| \cdot \| \boldsymbol {z} _ {i} \|
$$

The expectation term is upper bounded as

$$
\begin{array}{l} \mathbb {E} \left[ \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right\| ^ {4} \right] \leq \frac {1}{n} \sum_ {i \in [ n ]} \mathbb {E} \left[ \left| \boldsymbol {z} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right| ^ {4} \cdot \| \boldsymbol {z} \| ^ {4} \right] \\ \leq \mathbb {E} \left[ \left| \boldsymbol {z} ^ {\top} \nabla \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \right| ^ {8} \right] ^ {1 / 2} \mathbb {E} \left[ \| \boldsymbol {z} \| ^ {8} \right] ^ {1 / 2} \\ \leq \sqrt {1 0 5} (d + 6) ^ {2} G (\boldsymbol {\theta} _ {t}) ^ {4}, \\ \end{array}
$$

where we have plugged in explicit formulas for moments of Gaussian and  $\chi^2$  random variables.

Next, note that on the event  $\neg A$ , we have

$$
2 \eta d G (\boldsymbol {\theta} _ {t}) \leq \| \boldsymbol {\theta} _ {t + 1} - \boldsymbol {\theta} _ {t} \| = \eta \left\| \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \right\| \leq \eta \cdot \frac {1}{n} \sum_ {i \in [ n ]} \| z _ {i} \| ^ {2} G (\boldsymbol {\theta} _ {t}).
$$

Therefore

$$
\Pr [ \neg \mathcal {A} ] \leq \Pr \left[ \sum_ {i \in [ n ]} \| z _ {i} \| ^ {2} \geq 2 n d \right]
$$

Lemma 6 (Standard  $\chi^2$ -tail bound). Let  $Z$  be a  $\chi^2$  random variable with  $k$  degrees of freedom. Then

$$
\Pr [ Z \geq k + u ] \leq \exp \left(- \min  \left(\frac {u ^ {2}}{1 6 k}, \frac {u}{1 6}\right)\right)
$$

Since  $\sum_{i\in [n]}\| \pmb {z}_i\|^2$  is a  $\chi^2$  random variable with nd degrees of freedom, we thus have that

$$
\Pr [ \neg \mathcal {A} ] \leq \exp \left(- \frac {n d}{1 6}\right).
$$

Altogether,

$$
\begin{array}{l} \frac {1}{2} \eta^ {2} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \cdot \mathbf {1} (\neg \mathcal {A}) \right], \ell I - \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \right\rangle \leq \eta^ {2} \ell 1 0 5 ^ {1 / 4} (d + 6) G (\boldsymbol {\theta} _ {t}) ^ {2} \exp (- \frac {n d}{3 2}) \\ = \eta^ {2} \ell G (\boldsymbol {\theta} _ {t}) ^ {2} \exp (- \Omega (n d)). \\ \end{array}
$$

Finally, plugging in (8), along with the fact that  $\| H(\pmb{\theta}_t)\|_{op} \leq \ell$  and  $\mathrm{tr}(H(\pmb{\theta}_t)) \leq \ell r$ ,

$$
\begin{array}{l} \left\langle \mathbb {E} \left[ \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) \widehat {\nabla} \mathcal {L} (\boldsymbol {\theta}; \mathcal {B}) ^ {\top} \right], \boldsymbol {H} (\boldsymbol {\theta} _ {t}) \right\rangle = \frac {r + n + 1}{n} \cdot \ell \left(\| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {1}{B} \operatorname {t r} (\boldsymbol {\Sigma} (\boldsymbol {\theta} _ {t}))\right) \\ = \frac {r + n + 1}{n} \cdot \mathbb {E} \left[ \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \| ^ {2} \right] \\ \end{array}
$$

Thus letting  $\gamma = \frac{r + n + 1}{n}$  yields

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \mid \boldsymbol {\theta} _ {t} \right] - \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \\ \leq - \eta \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}) \| ^ {2} + \frac {1}{2} \eta^ {2} \ell \cdot \gamma \cdot \mathbb {E} [ \| \nabla \mathcal {L} (\boldsymbol {\theta} _ {t}; \mathcal {B}) \| ^ {2} ] + \eta^ {2} \ell G (\boldsymbol {\theta} _ {t}) ^ {2} \exp (- \Omega (n d)), \\ \end{array}
$$

as desired.

# Footnotes:

Page 0: *Equal contribution and corresponding authors. <sup>1</sup>Our code is available at https://github.com/princeton-nlp/MeZ0. 
Page 2: 2The original SPSA algorithm [88] perturbs the model by  $1 / z$  and thus requires that  $\mathcal{Z}$  has finite inverse moments, precluding the choice of  $\textbf{z}$  as Gaussian.  $1 / z$  is very large with high probability for a zero-mean Gaussian  $\mathcal{Z}$ , so we adopt the standard in many theoretical [70, 32] and empirical [64] works and perturb the parameters by  $\mathcal{Z}$  with  $\mathcal{Z}$  as a Gaussian random variable. 
Page 3: Note that this reconstruction requires no additional forward passes through the model and no access to the data used during fine-tuning, since projected_grad implicitly encodes this information. 
Page 4: $^{4}$ BBTv2 can only train low-dimensional projected prefixes instead of the full model. 
Page 5: We also include SST-2, which is a simple sentiment classification task that we use for development. 
Page 7: <sup>6</sup>Section 3 uses the standard choice of Adam for FT; we provide SGD experiments in Appendix F.1. This is satisfied for the standard cross-entropy objective. All of our experiments use  $n = 1$ . 
