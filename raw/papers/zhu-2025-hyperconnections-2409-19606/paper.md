# HYPER-CONNECTIONS

Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou

Seed-Foundation-Model Team, ByteDance

{zhudefa,huanghongzhi.51,huangzihao.notabot,yutao.zeng, maoyunyao.myy,wubanggu,minqiyang,zhouxun}@bytedance.com

# ABSTRACT

We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections. This approach specifically addresses common drawbacks observed in residual connection variants, such as the seesaw effect between gradient vanishing and representation collapse. Theoretically, hyperconnections allow the network to adjust the strength of connections between features at different depths and dynamically rearrange layers. We conduct experiments focusing on the pre-training of large language models, including dense and sparse models, where hyper-connections show significant performance improvements over residual connections. Additional experiments conducted on vision tasks also demonstrate similar improvements. We anticipate that this method will be broadly applicable and beneficial across a wide range of AI problems.

# 1 INTRODUCTION

Figure 1: The performance of the baseline model OLMoE-1B-7B and the model with hyperconnections, OLMoE-1B-7B-DHC×4. (1) and (2) show the training loss (0.99 EMA smoothed) and the C4-en validation loss, respectively. Our method converges 1.8 times faster compared to the baseline and maintains a significant advantage at the 500B tokens. (3) and (4) show the accuracy curves on HellaSwag and ARC-Challenge, demonstrating the superior performance of the OLMoE-1B-7B-DHC×4 model.




Deep learning has achieved tremendous success across various domains, where residual connections (He et al., 2016) have been instrumental in contemporary neural network architectures, including transformers and CNNs. Residual connections help mitigate the problem of gradient vanishing, enabling the effective training of very deep networks. However, it is important to acknowledge that residual connections are not infallible solutions and still present limitations that remain unresolved.

The two main variants of residual connections, Pre-Norm and Post-Norm, each make distinct trade-offs between gradient vanishing and representation collapse. Pre-Norm applies normalization operations to the input before each residual block, effectively addressing the problem of gradient vanishing (Bengio et al., 1994; Glorot & Bengio, 2010). However, it can also lead to the issue of collapse in deep representations (Liu et al., 2020), where hidden features in deeper layers become highly similar, diminishing the contribution of additional layers as their number increases. In contrast, Post-Norm applies normalization after the output of each residual block, reducing the influence of a hidden state on subsequent layers. This approach can alleviate the issue of representation collapse but

Figure 2: Hyper-connections (HC) with an expansion rate of  $n = 2$ . (a) Residual connections. (b) Hyper-connections:  $\beta_{1}, \beta_{2}, \alpha_{0,0}, \alpha_{0,1}, \alpha_{1,0}, \alpha_{1,1}, \alpha_{2,1}$ , and  $\alpha_{2,2}$  are learnable scalars or scalars predicted by the network, depending on the specific HC version. These connections enable lateral information exchange and vertical integration of features across depths. The Transformer with HC is shown in Fig. 17. They can be decoupled into depth-connections and width-connections. (c) Depth-connections perform a weighted sum between the layer output and the hidden vector  $h_{1}$ . (d) Width-connections allow information exchange between the hidden vectors  $h_{1}$  and  $h_{2}$ .

also reintroduces the problem of vanishing gradients. The vanishing gradient and the representation collapse are like two ends of a seesaw, with these two variants making respective trade-offs between these issues. The key issue is that residual connections, including both Pre-Norm and Post-Norm variants, predefine the strength of connections between the output and input within a layer.

Driven by the limitations of residual connections, an important question arises: Can neural networks autonomously learn the optimal strength of connections to improve performance? To address this, we propose hyper-connections (HC), which lead to significantly improved performance with a negligible increase in computation and parameters. We will show that both Post-Norm and Pre-Norm variants can be expressed as specific nontrainable forms of hyper-connections, as discussed in § 3.1.

The core idea of hyper-connections (HC) is to propose learnable depth-connections and width-connections, as depicted in Fig.2 (b). These connections flexibly integrate features vertically across depths, compared to the residual connections shown in Fig.2 (a). Depth-connections can be considered as a generalized residual connections, assigning weights to the connections between the inputs and outputs of each layer. To enable the network to model different depth-connections simultaneously, we expand the network's input into  $n$  copies, each having its own depth connection, as shown in Fig. 2 (b). This design allows multiple hidden vectors to reserve multiple patterns connecting preceding layers, as shown in § 4.5. Moreover, we establish width connections between the  $n$  hidden vectors, allowing information exchange between hidden vectors within the same layer, as shown in Fig. 2 (b). We argue that  $n (>1)$  hidden states are necessary. As analyzed in Appendix F, the seesaw effect persists when  $n = 1$ , and experiments show that it does not improve performance, as shown in Fig. 5. In contrast, when  $n > 1$ , hyper-connections can not only learn to adjust the

Figure 3: Cosine similarity between the input of the current and the previous layers for the OLMo-1B models (Groeneveld et al., 2024). The curve represents the median of similarity, while the shaded area indicates the range between the 5th and 95th percentiles. The red curve shows the model with Pre-Norm, and the blue curve shows that with hyper-connections.

strength of residuals but also rearrange layers, either sequentially or in parallel, as discussed in § 3.2. To further enhance flexibility, we introduce dynamic hyper-connections (DHC), enabling the network to adjust connection weights according to the input. Notably, although HC seem to increase the network's width by  $n$  times, the additional parameters and computational cost are almost negligible, as analyzed in Appendix B. The Transformer with HC is shown in Fig. 17.

Our research, primarily centered on large language models (LLMs) pre-training, also extends to visual generation and classification tasks. Using Pre-Norm as a baseline, we demonstrate the significant benefits of hyper-connections, including 1B and 7B dense models as well as 7B MoE models, as

detailed in § 4. The benefits are particularly prominent for OLMoE (Muennighoff et al., 2024) as presented in Fig.1. The model utilizing DHC converges 1.8 times faster and shows an improvement of 6 points on ARC-Challenge compared to the baseline trained with 500 B tokens. According to our visualization analysis, as shown in Fig.3, the baseline model tends toward representation collapse, characterized by high similarity between features of adjacent layers. In contrast, models with HC exhibit significantly lower similarity between features across adjacent layers and a wider range of similarities. This suggests that HC enhance the impact of each layer. Further discussion is provided in §4.5 and in Appendix F. These compelling pieces of evidence demonstrate the generality of the hyper-connections principle, and we anticipate their applicability in numerous other AI challenges.

# 2 METHOD

# 2.1 STATIC HYPER-CONNECTIONS

Consider the hidden vector  $\mathbf{h}^{k - 1}\in \mathbb{R}^d$  (or  $\mathbf{h}^{k - 1}\in \mathbb{R}^{d\times 1}$ ) as the input to the  $k$ -th layer, with the initial input  $\mathbf{h}^0$  to the network. Initially,  $\mathbf{h}^0\in \mathbb{R}^d$  is replicated  $n$  times to form the initial hyper hidden matrix  $\mathbf{H}^0 = \left(\mathbf{h}^0\quad \mathbf{h}^0\quad \ldots \quad \mathbf{h}^0\right)^\top \in \mathbb{R}^{n\times d}$ . Here,  $n$  is the expansion rate. For the  $k$ -th layer, the input consists of the hyper hidden matrix from the previous layer  $\mathbf{H}^{k - 1} = \left(\mathbf{h}_1^{k - 1}\quad \mathbf{h}_2^{k - 1}\quad \ldots \quad \mathbf{h}_n^{k - 1}\right)^\top \in \mathbb{R}^{n\times d}$ . Finally, we sum the last hyper hidden matrix row-wise to obtain the required hidden vector, which is then passed through a final projector to produce the final output of the network (i.e., a normalization layer and an unembedding layer in transformers). To simplify the notation in subsequent analysis, we omit the layer index and simply denote the hyper-hidden matrix as  $\mathbf{H} = (\mathbf{h}_1\quad \mathbf{h}_2\quad \ldots \quad \mathbf{h}_n)^\top$ .

The hyper-connections (HC) can be represented by a matrix  $\mathcal{H}\mathcal{C}$ , where each element defines the connection weight. The matrix is structured as follows:

$$
\mathcal {H C} = \left( \begin{array}{c c} \mathbf {0} _ {1 \times 1} & \mathbf {B} \\ \mathbf {A} _ {\mathbf {m}} & \mathbf {A} _ {\mathbf {r}} \end{array} \right) = \left( \begin{array}{c c c c c} 0 & \beta_ {1} & \beta_ {2} & \dots & \beta_ {n} \\ \alpha_ {1, 0} & \alpha_ {1, 1} & \alpha_ {1, 2} & \dots & \alpha_ {1, n} \\ \alpha_ {2, 0} & \alpha_ {2, 1} & \alpha_ {2, 2} & \dots & \alpha_ {2, n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \alpha_ {n, 0} & \alpha_ {n, 1} & \alpha_ {n, 2} & \dots & \alpha_ {n, n} \end{array} \right) \in \mathbb {R} ^ {(n + 1) \times (n + 1)}. \tag {1}
$$

Consider a network layer  $\mathcal{T}$ , it integrates self-attention layers and feed-forward networks within transformers. The output of the HC, denoted by  $\hat{\mathbf{H}}$ , can be simply formulated as follows:

$$
\hat {\mathbf {H}} = \mathcal {H C} (\mathcal {T}, \mathbf {H}) = \mathbf {B} ^ {\intercal} \mathcal {T} \left(\mathbf {H} ^ {\intercal} \mathbf {A} _ {\mathbf {m}}\right) ^ {\intercal} + \mathbf {A} _ {\mathbf {r}} ^ {\intercal} \mathbf {H}. \tag {2}
$$

We use  $\mathbf{A}_{\mathbf{m}}$  as weights to perform a weighted sum on the input  $\mathbf{H} = (\mathbf{h}_1 \quad \mathbf{h}_2 \quad \ldots \quad \mathbf{h}_n)^\top$  to obtain the input  $\mathbf{h}_0$  of the current layer  $\mathcal{T}$ , which is given by:

$$
\mathbf {h} _ {0} ^ {\intercal} = \mathbf {A} _ {\mathbf {m}} ^ {\intercal} \mathbf {H}, \tag {3}
$$

While  $\mathbf{A}_{\mathbf{r}}$  is used to connect  $\mathbf{H}$  and map it to a hyper hidden matrix  $\mathbf{H}'$ , as shown below:

$$
\mathbf {H} ^ {\prime} = \mathbf {A} _ {\mathbf {r}} ^ {\top} \mathbf {H}. \tag {4}
$$

Subsequently, the output is given by:

$$
\hat {\mathbf {H}} = \mathbf {B} ^ {\intercal} \left(\mathcal {T} \mathbf {h} _ {0}\right) ^ {\intercal} + \mathbf {H} ^ {\prime}. \tag {5}
$$

The depth-connections can be decoupled as the following matrix, which is shown at Fig 2 (a):

$$
\mathcal {D C} = \binom {\mathbf {B}} {\operatorname {d i a g} \left(\mathbf {A} _ {\mathbf {r}}\right)} = \left( \begin{array}{c c c c} \beta_ {1} & \beta_ {2} & \dots & \beta_ {n} \\ \alpha_ {1, 1} & \alpha_ {2, 2} & \dots & \alpha_ {n, n} \end{array} \right) \in \mathbb {R} ^ {2 \times n}, \tag {6}
$$

where the first row  $\mathbf{B}$  represents the weights of the output of the current layer  $\mathcal{T}$ , and the last row  $\mathrm{diag}(\mathbf{A_r})$  represents the weights of the input. We use  $\mathrm{diag}(\mathbf{A_r})$  to represent the flatten vector of the diagonal entries of  $\mathbf{A_r}$ .

The width-connections matrix can be defined as follows, which is shown at Fig 2 (b):

$$
\mathcal {W C} = \left( \begin{array}{c c} \mathbf {A _ {m}} & \mathbf {A _ {r}} \end{array} \right) \in \mathbb {R} ^ {n \times (n + 1)}. \tag {7}
$$

The algorithm that employs hyper-connections is presented in Algorithm 1.

# 2.2 DYNAMIC HYPER-CONNECTIONS

The entries of  $\mathcal{HC}$  can dynamically depend on the input  $\mathbf{H}$ , which the matrix representation of dynamic hyper-connections (DHC) is defined as follows:

$$
\mathcal {H C} (\mathbf {H}) = \left( \begin{array}{l l} \mathbf {0} _ {1 \times 1} & \mathcal {B} (\mathbf {H}) \\ \mathcal {A} _ {m} (\mathbf {H}) & \mathcal {A} _ {r} (\mathbf {H}) \end{array} \right) \tag {8}
$$

Similarly, given a layer  $\mathcal{T}$  and input  $\mathbf{H}$ , we obtain the output of the DHC as follows:

$$
\hat {\mathbf {H}} = \mathcal {H C} (\mathbf {H}) (\mathcal {T}, \mathbf {H}). \tag {9}
$$

In practice, we combine the dynamic and static matrices to achieve DHC. The dynamic parameters are obtained through a linear transformation. To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor. The following equations detail how these dynamic parameters are computed:

$$
\overline {{\mathbf {H}}} = \operatorname {n o r m} (\mathbf {H}) \tag {10}
$$

$$
\mathcal {B} (\mathbf {H}) = s _ {\beta} \circ \tanh  \left(\overline {{\mathbf {H}}} \mathbf {W} _ {\beta}\right) ^ {\intercal} + \mathbf {B} \in \mathbb {R} ^ {1 \times n} \tag {11}
$$

$$
\mathcal {A} _ {m} (\mathbf {H}) = s _ {\alpha} \circ \tanh  \left(\overline {{\mathbf {H}}} \mathbf {W} _ {m}\right) + \mathbf {A} _ {m} \in \mathbb {R} ^ {n \times 1} \tag {12}
$$

$$
\mathcal {A} _ {r} (\mathbf {H}) = s _ {\alpha} \circ \tanh  \left(\overline {{\mathbf {H}}} \mathbf {W} _ {r}\right) + \mathbf {A} _ {r} \in \mathbb {R} ^ {n \times n} \tag {13}
$$

Our experiments in § 4 demonstrate that dynamic hyper-connections outperform static hyperconnections in language modeling tasks. The PyTorch implementations for both the static and dynamic variants of hyper-connections are detailed in Algorithm 2 and 3.

# 2.3 INITIALIZATION

In order to make the initialization of the hyper-connections equivalent to the Pre-Norm residual connections, we adopt the following initialization strategy. The dynamic parameters  $\mathbf{W}_{\beta}, \mathbf{W}_m$ , and  $\mathbf{W}_r$  in Eqs. 11, 12, and 13 are initialized to 0, while the static matrices are initialized as follows:

$$
\left( \begin{array}{l l} \mathbf {0} _ {1 \times 1} & \mathbf {B} ^ {k} \\ \mathbf {A} _ {\mathbf {m}} ^ {k} & \mathbf {A} _ {\mathbf {r}} ^ {k} \end{array} \right) = \left( \begin{array}{c c} \mathbf {0} _ {1 \times 1} & \mathbf {1} _ {1 \times n} \\ \mathbf {e} _ {k \bmod n} & \mathbf {e} _ {n \times n} \end{array} \right), \tag {14}
$$

where  $k$  is the index of the layer. mod denotes the modulo operation.

# 3 WHY HYPER-CONNECTIONS

In this section, we elucidate the rationale behind hyper-connections. We explore how variants of residual connections, namely Pre-Norm and Post-Norm, can be viewed as non-trainable hyperconnections, and introduce the concept of sequential-parallel duality, demonstrating how hyperconnections can dynamically optimize layer arrangements to enhance network performance. A visulize analysis of hyper-connections through an unfolded view is discussed in  $\S 4.5$ .

# 3.1 RESIDUAL CONNECTIONS AS NON-TRAINABLE HYPER-CONNECTIONS

The Pre-Norm and Post-Norm residual connections can be represented as the following hyperconnections matrices with an expansion rate  $n = 1$ :

$$
\mathcal {H C} _ {\text {P r e N o r m}} = \left( \begin{array}{l l} 0 & 1 \\ 1 & 1 \end{array} \right), \quad \tag {15} \qquad \mathcal {H C} _ {\text {P o s t N o r m}} = \left( \begin{array}{c c} 0 & \frac {1}{\sqrt {\sigma_ {i} ^ {2} + \sigma_ {o} ^ {2} + 2 \sigma_ {i o}}} \\ 1 & \frac {1}{\sqrt {\sigma_ {i} ^ {2} + \sigma_ {o} ^ {2} + 2 \sigma_ {i o}}} \end{array} \right), \tag{16}
$$

where  $\sigma_{i}$  and  $\sigma_{o}$  denote the standard deviations of the input and output of the neural network layer, respectively, and  $\sigma_{io}$  is the covariance between them.

For Pre-Norm, its hyper-connection matrix is a  $2 \times 2$  matrix where the bottom right triangular part is filled with 1 and the rest is a placeholder 0. For Post-Norm, the weights depend on the variances and covariance of the input and output, forming a  $2 \times 2$  matrix. Therefore, their hyper-connection matrices are non-trainable. In this work, we propose hyper-connections that can be  $(n + 1) \times (n + 1)$  matrices, with weights that are trainable or even predicted based on the input. The complete derivation is provided in Appendix G.

# 3.2 SEQUENTIAL-PARALLEL DUALITY

Given a series of neural network modules, we have the option to arrange them either sequentially or in parallel. However, hyper-connections offer an approach that learns to rearrange these layers in a configuration blending both sequential and parallel arrangements.

(a) Sequential Arrangement

(b) Parallel Arrangement  
Figure 4: Sequential and parallel arrangements of hyper-connections with  $n = 2$ .

Without loss of generality, we set the expansion rate to  $n = 2$ . If the hyper-connections are learned as the following matrix, the neural network will be arranged sequentially:

$$
\mathcal {H C} = \left( \begin{array}{c c c} 0 & 1 & 1 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{array} \right). \tag {17}
$$

In this case, the depth connection degenerates into a residual connection, as shown in Fig. 4 (a).

When the hyper-connections for odd and even layers (with layer numbering starting from 1) are defined by the following matrices, the neural network will be arranged in parallel every two consecutive layers, similar to the arrangement of parallel transformer blocks in transformers (Wang, 2021), as shown in Fig. 4 (b). The general and complete derivation is provided in Appendix H.

$$
\mathcal {H C} _ {o d d} = \left( \begin{array}{l l l} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{array} \right), \tag {18}
$$

Thus, learning the hyper-connection matrix in various forms can create layer arrangements that surpass traditional sequential and parallel configurations, resulting in a soft-mixture or even dynamic

arrangement. For static hyper-connections, the layer arrangement within the network remains fixed after training. In contrast, dynamic hyper-connections allow the arrangement to adapt dynamically for each token.

# 4 RESULTS

Figure 5: Comparison of training loss curves for different expansion rate. The left subfigure includes models with dynamic hyper-connections (DHC) at various expansion rates, while the right subfigure shows the effect of omitting the tanh function. Both subfigures illustrate how increasing the expansion rate leads to improved training loss performance over 500B tokens. Results are smoothed using an exponential moving average with a coefficient of 0.99.


Table 1: Ablation study on expansion rates  $n$  with training on  ${500}\mathrm{\;B}$  tokens.  

<table><tr><td>Methods</td><td>V2 Eval Loss ↓</td><td>V2 Eval PPL ↓</td><td>V3 Eval Loss ↓</td><td>V3 Eval PPL ↓</td><td>Down Stream Avg, Acc. ↑</td></tr><tr><td>OLMo-1B</td><td>2.811</td><td>18.023</td><td>2.544</td><td>14.229</td><td>62.5</td></tr><tr><td>OLMo-1B-DHC×1 W/O tanh</td><td>2.822</td><td>18.270</td><td>2.556</td><td>14.428</td><td>62.3</td></tr><tr><td>OLMo-1B-DHC×2 W/O tanh</td><td>2.792</td><td>17.663</td><td>2.537</td><td>14.033</td><td>63.8</td></tr><tr><td>OLMo-1B-DHC×4 W/O tanh</td><td>2.779</td><td>17.451</td><td>2.516</td><td>13.844</td><td>64.4</td></tr><tr><td>OLMo-1B-DHC×8 W/O tanh</td><td>2.777</td><td>17.425</td><td>2.514</td><td>13.819</td><td>63.8</td></tr><tr><td>OLMo-1B-DHC×1</td><td>2.819</td><td>18.125</td><td>2.556</td><td>14.418</td><td>62.3</td></tr><tr><td>OLMo-1B-DHC×2</td><td>2.802</td><td>17.950</td><td>2.534</td><td>14.114</td><td>63.0</td></tr><tr><td>OLMo-1B-DHC×4</td><td>2.781</td><td>17.509</td><td>2.514</td><td>13.826</td><td>63.8</td></tr><tr><td>OLMo-1B-DHC×8</td><td>2.778</td><td>17.445</td><td>2.516</td><td>13.843</td><td>62.8</td></tr></table>

We primarily conduct experiments on pre-training of large language model, including dense and Mixture-of-Experts (MoE) (Shazeer et al., 2017) models, and extend to visual generation and classification tasks. Due to space constraints, we include the vision experiments in the Appendix E.

Experiment Settings. We employ the experimental setup outlined by OLMo (Groeneveld et al., 2024) for dense models and by OLMoE (Muennighoff et al., 2024) for MoE models. For dense models, we use dolmap-v1.5-sample (Soldaini et al., 2024) as our training dataset. We conduct ablation studies on 1B models and assess the effectiveness of our method at the 7B model scale. For MoE models, we train the OLMoE-1B-7B model, both with and without hyper-connections, on the OLMOE-MIX dataset. These models activate 1.3B out of a total of 7B parameters. All experiments are trained on 500B tokens.

Implementation. We maintain the training configuration of the baseline model, replacing the residual connections with hyper-connections. The static component in Eqs. 1, 11, 12, 13 does not utilize weight decay, whereas the dynamic component does. Since the hyper hidden vectors of the final transformer block are ultimately summed, we ensure that the standard deviation (std) of the output (before the final layernorm and unembedding layers) remains consistent with the original. At initialization, we scale the std of the weights of the output module at all layers, including those of the second linear layer of the feedforward network and the output projector of the attention module, by a factor of  $\sqrt{n}$ ,

where  $n$  represents the expansion rate. The parameters and computational overhead introduced by hyper-connections is negligible, see Table. 7 and 8.

Metrics. In accordance with the methodology of OLMo (Groeneveld et al., 2024), we report the average perplexities (PPL) and losses on both the V2 and V3 validation sets, along with the average metrics for zero-shot evaluation on downstream benchmarks (refer to Table 13). We observe significant volatility in the zero-shot performance indicators for the datasets (highlighted in grey in Table 13), with fluctuations exceeding  $20\%$  across neighboring checkpoints. For more reliable and consistent results, we excludes these volatile datasets from our analysis. For the MoE models, in line with OLMoE, we also present losses on V3 validation sets, and accuracies on downstream benchmarks (refer to Table 14).

# 4.1 ABLATION STUDY

We use the dynamic hyperconnections with an expansion rate of  $n = 4$  and include the tanh function as the default method, marked with the suffix -DHC, while -SHC denotes static hyper-connections.

The evaluation results are presented in Table 1, and the training loss curves are depicted in Fig. 5. We observe that with an expansion rate of  $n = 1$ , the performance of DHC is inferior to the baseline. However, for  $n > 1$ , DHC significantly outperforms the baseline, achieving superior results at  $n = 4$  with the increase to  $n = 8$  providing minimal additional benefits. Notably, OLMo-1B-DHC×8 W/O tanh excels on both V2 and V3 validation sets, with a reduction in V2 Eval Loss by 0.034 and V3 Eval Loss by 0.029 compared to the baseline. Furthermore, the decline rate of training losses for DHC ( $n \geq 2$ ) is steeper than that of the baseline, and DHC demonstrates greater stability, with no spikes observed in any DHC experiments.

Static and dynamic hyper-connections. Table 2 presents an ablation study comparing SHC and DHC. All hyper-connection (HC) variants significantly outperform the baseline. At an expansion rate of 2, the improvements of DHC and SHC are similar. However, at an expansion rate of 4, DHC performs notably better than SHC.

Table 2: Ablation study on static and dynamic hyper-connections with training on  ${500}\mathrm{\;B}$  tokens.  

<table><tr><td>Methods</td><td>V2 Eval Loss ↓</td><td>V2 Eval PPL ↓</td><td>V3 Eval Loss ↓</td><td>V3 Eval PPL ↓</td><td>Down Stream Avg, Acc. ↑</td></tr><tr><td>OLMo-1B</td><td>2.811</td><td>18.023</td><td>2.544</td><td>14.229</td><td>62.5</td></tr><tr><td>OLMo-1B-SHC×2</td><td>2.799</td><td>17.778</td><td>2.538</td><td>14.152</td><td>63.4</td></tr><tr><td>OLMo-1B-DHC×2</td><td>2.802</td><td>17.950</td><td>2.534</td><td>14.114</td><td>63.0</td></tr><tr><td>OLMo-1B-DHC×2 W/O tanh</td><td>2.792</td><td>17.663</td><td>2.529</td><td>14.033</td><td>63.8</td></tr><tr><td>OLMo-1B-SHC×4</td><td>2.791</td><td>17.671</td><td>2.528</td><td>14.025</td><td>63.6</td></tr><tr><td>OLMo-1B-DHC×4</td><td>2.781</td><td>17.509</td><td>2.515</td><td>13.826</td><td>63.8</td></tr><tr><td>OLMo-1B-DHC×4 W/O tanh</td><td>2.779</td><td>17.451</td><td>2.516</td><td>13.844</td><td>64.4</td></tr></table>

The importance of B and  $\mathcal{WC}$ . As shown in Table 3, not training  $\mathcal{WC}$  leads to significant performance declines, with the V2 loss increasing by 0.021 and the V3 loss by 0.017, as seen when comparing the 4th and 6th lines of Table 3. In contrast, the impact is less pronounced when B is not trained. Therefore, ensuring the trainability of both  $\mathcal{WC}$  and B is crucial.

# 4.2 COMPARISON WITH RELATED WORKS

We implemented the Altup (Baykal et al., 2024) and ResiDual (Xie et al., 2023) methods in OLMo. Altup is motivated to widen the hidden dimension while maintaining low computation cost by passing only a part of hidden state to transformer blocks. By contrast, ResiDual is proposed to combine both Pre- and Post-Norm in a two-stream style. Both methods expand the hidden size by  $n$  times with negligible computational overhead, with ResiDual expanding it exactly 2 times. For a fair comparison, we set  $n = 2$  in our experiments. Unfortunately, although these methods show gains in the early stages of training, they are gradually surpassed by the baseline, as demonstrated by the results in Table 4 and the training loss curves in Fig. 15.

Table 3: Ablation study on OLMo-1B-DHC×4. In the B or Wc column, the symbol "x" denotes parameters that are not trainable from initialization.  

<table><tr><td>WC</td><td>B</td><td>Tanh</td><td>V2 Eval Loss ↓</td><td>V2 Eval PPL ↓</td><td>V3 Eval Loss ↓</td><td>V3 Eval PPL ↓</td><td>Down Stream Avg, Acc. ↑</td></tr><tr><td>X</td><td>✓</td><td>X</td><td>2.804</td><td>17.912</td><td>2.537</td><td>14.145</td><td>62.5</td></tr><tr><td>✓</td><td>X</td><td>X</td><td>2.781</td><td>17.493</td><td>2.518</td><td>13.874</td><td>63.6</td></tr><tr><td>✓</td><td>✓</td><td>X</td><td>2.779</td><td>17.773</td><td>2.516</td><td>13.823</td><td>64.4</td></tr><tr><td>X</td><td>✓</td><td>✓</td><td>2.802</td><td>17.914</td><td>2.532</td><td>14.072</td><td>63.4</td></tr><tr><td>✓</td><td>X</td><td>✓</td><td>2.783</td><td>17.504</td><td>2.520</td><td>13.906</td><td>63.4</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>2.781</td><td>17.835</td><td>2.515</td><td>13.807</td><td>63.8</td></tr></table>

Table 4: Performance of related methods on OLMo-1B models.  

<table><tr><td>Methods</td><td>V2 Eval Loss ↓</td><td>V2 Eval PPL ↓</td><td>V3 Eval Loss ↓</td><td>V3 Eval PPL ↓</td><td>Down Stream Avg, Acc. ↑</td></tr><tr><td>OLMo-1B</td><td>2.811</td><td>18.023</td><td>2.544</td><td>14.229</td><td>62.5</td></tr><tr><td>OLMo-1B-ResiDual</td><td>2.825</td><td>18.375</td><td>2.551</td><td>14.346</td><td>62.0</td></tr><tr><td>OLMo-1B-Altup×2</td><td>2.827</td><td>18.268</td><td>2.558</td><td>14.454</td><td>62.4</td></tr><tr><td>OLMo-1B-DHC×2</td><td>2.802</td><td>17.950</td><td>2.534</td><td>14.114</td><td>63.0</td></tr><tr><td>OLMo-1B-DHC×2 W/O tanh</td><td>2.792</td><td>17.663</td><td>2.529</td><td>14.033</td><td>63.8</td></tr></table>

# 4.3 7B MODELS

Figure 6: (1) and (2) Training loss (0.99 EMA smoothed) and C4-en validation loss for OLMo-7B and OLMo-7B-DHC×4 models. (3) and (4) Accuracy curves on hellaswag and sciq, demonstrating the superior performance of the OLMo-7B-DHC×4 model.




We evaluate the effectiveness of hyper-connections on the 7B model, training a model with DHCs with an expansion rate of 4, denoted as OLMo-7B-DHC  $\times 4$ . According to Table 5, OLMo-7B-DHC  $\times 4$  significantly outperforms the baseline OLMo-7B model in all average metrics. In the V2 evaluation, OLMo-7B-DHC  $\times 4$  shows improvements of 0.022 for loss and 0.293 for PPL. Furthermore, the average score of downstream benchmarks 0.710 surpasses the baseline 0.701, with the results of specific tasks shown in Fig. 10.

Based on Fig 6, the OLMo-7B-DHC  $\times 4$  model consistently shows better metrics compared to baseline, including training and validation loss and accuracy in downstream benchmarks. Notably, after  $400\mathrm{~B}$  tokens, the model maintains its improvement without the gains diminishing. This indicates that the OLMo-7B-DHC  $\times 4$  model continues to provide consistent benefits in reducing loss, even at higher token counts. Furthermore, according to Fig. 6, the baseline model exhibits frequent spikes, while our model with DHCs shows no spikes throughout the training. This shows that our approach not only achieves better loss but also ensures more stable training.

# 4.4 MOE MODELS

We evaluate the effectiveness of hyper-connections on the Mixture-of-Experts (MoE) model. We retrain the original OLMoE-1B-7B model as the baseline and train a model that applies Dynamic

Table 5: Performance of 7B models. FLOPs refers to the computation per token in the forward pass.  

<table><tr><td>Methods</td><td>Params (B)</td><td>FLOPs (G)</td><td>V2 Loss ↓</td><td>V2 PPL ↓</td><td>V3 Loss ↓</td><td>V3 PPL ↓</td><td>Tasks Avg. Acc. ↑</td></tr><tr><td>OLMo-7B</td><td>6.9</td><td>13.36</td><td>2.581</td><td>14.316</td><td>2.322</td><td>11.324</td><td>70.1</td></tr><tr><td>OLMo-7B-DHC×4</td><td>6.9</td><td>13.38</td><td>2.559</td><td>14.023</td><td>2.304</td><td>11.120</td><td>71.0</td></tr></table>

Hyper-Connections (DHC) with  $n = 4$ , replacing the residual connections. The full results are shown in Fig. 9, which illustrates that hyper-connections outperform residual connections in almost all metrics. In many metrics, our method requires only half of the training tokens to achieve the same performance as the baseline. Fig. 1 and Table 6 highlight some of the results, such as a reduction in training loss of approximately 0.027, a reduction in loss on the C4-en validation set of 0.028, an improvement of 6 points on the ARC-Challenge and an improvement of 1.2 points on MMLU Var.

Table 6: Downstream evaluations for MoE models training with 500B tokens under the OLMoE evaluation setting. ARC-C stands for ARC-Challenge, and ARC-E for ARC-Easy. MMLU Var is a modified version of MMLU that includes varying few-shot examples, providing stable feedback during early training, as outlined in the OLMoE setting (Muennighoff et al., 2024).

<table><tr><td>Methods</td><td>MMLU Var</td><td>Hella-Swag</td><td>ARC-C</td><td>ARC-E</td><td>PIQA</td><td>Wino-Grande</td><td>BoolQ</td></tr><tr><td>OLMoE-1B-7B</td><td>38.5</td><td>69.5</td><td>41.8</td><td>72.8</td><td>77.6</td><td>64.4</td><td>65.4</td></tr><tr><td>OLMoE-1B-7B-DHC×4</td><td>39.7</td><td>70.2</td><td>47.8</td><td>76.7</td><td>78.2</td><td>64.6</td><td>68.5</td></tr></table>

# 4.5 VISUALIZATION ANALYSIS

Figure 7: Visualization of connection matrices for hyper-connections and various related baseline methods. The attention layers, which have odd ids, are marked with green tick marks.

In this section, we investigate the learned hyper-connection weights and show how the output of the former layer contributes to the latter ones. To this end, we convert hyper-connections to dense connections cross layers. Consider the input hidden vectors  $\mathbf{h}_0^k$  in  $k$ -th layer, it can be unfolded as a weighted summation over previous layer outputs:

$$
\mathbf {h} _ {0} ^ {k} = \sum_ {j = 0} ^ {k - 1} c _ {k j} ^ {(0)} \mathcal {T} ^ {j} \left(\mathbf {h} _ {0} ^ {j}\right), \tag {20}
$$

where  $c_{kj}^{(0)}$  describes how much layer-  $j$  ( $\mathcal{T}^j$ ) contributes to layer-  $k$ 's input  $\mathbf{h}_0^k$ . Then,  $\mathbf{C}^{(0)}$  denotes a dense connection weight matrix. In particular, let layer-0 be the word embedding and  $\mathcal{T}^0$  be an identity mapping, layer-  $L + 1$  be the hidden state before the unembedding layer, which is a summation over the last hidden vectors, i.e.,  $\mathbf{h}_0^{L + 1} = \sum_j\mathbf{h}_j^L$ .

OLMo-1B-DHC  $\times 4$  model is adopted for visualization. We take the checkpoint at 500B tokens and forward random validation text to obtain dynamic hyper-connection weights. In addition, we show connection patterns for some related baseline methods. Finally, the visualization is illustrated in Fig. 13. We present the following findings, with more detailed discussions provided in Appendix F.

Connection patterns for baseline methods. For Pre-Norm baseline, the connection matrix is simply a lower triangular matrix with diagonal elements erased, because each transformer layer joins the residual equally. In the Pre-Norm parallel transformer block (PTB) baseline, the connection matrix appears jagged because the input to the FFN layer does not depend on the output of the previous attention layer. For Post-Norm baseline, the connection only holds for adjacent layers, as the weight for bottom layers decays every time the residual passes a post-norm layer. For the two-hop residual baseline (Ma et al., 2024), the outputs of attention layers are not added to residual and only contributes to the next one FFN layer, resulting in a vertical strip pattern in the connection matrix.

$\Lambda$ -shaped connection pattern. In the connection matrix for hyper-connections, a long-term decay pattern can be observed, where layers are generally preferred to rely on a few adjacent layer outputs. Moreover, the bottom layers (e.g. layer 0,2) are observed frequently used in most of subsequent layers. Therefore, the two patterns together form a  $\Lambda$ -shaped connection pattern. Note that the long-term decay pattern is a Post-Norm style pattern, while the frequently accessed pattern is Pre-Norm style, indicating that the hyper-connection introduces a free mixture of Pre- and Post-Norm architecture.

Input word embedding is eliminated from model output. As per the first column in the connection matrix for layer inputs, the input word embedding contributes to most of the layers except for the final one. This last layer, which products the model's output, is used for next token prediction. In most cases, keeping a component of input embedding in model output is harmful to next token prediction, especially when using a tied word embedding such as that employed by OLMo-1B. Similar results are found in previous works (Ma et al., 2023).

Parallel transformer blocks are observed. As discussed in § 3.2, parallel transformer block, which performs attention and FFN in parallel, is a special case for hyper-connection. In practice, PTB-like patterns, which can be identified by the local jagged pattern, are surprisingly observed to be learned by hyper-connections. For instance, layer 11 has a minimal contribution to the input of layer 12 (refer to row 12 in the hyper-connection connection matrix). This suggests that layers 11 and 12 can operate in parallel, thereby forming a PTB module.

Attention layers tend to have fewer long-term connections. It is observed that attention layers at the bottom barely have long-term contribution, a trend that persists until layer 17. Upon examining the connection matrix for hyper hiddens (refer to Fig. 13 in the appendix), it's evident that the outputs of the FFN layers have significantly greater magnitudes than those of the attention layers. This pattern resembles a two-hop residual connection design, wherein the attention output contributes to the input of the following FFN layer, but doesn't join the main residual path.

# 5 RELATED WORK

Transformers (Vaswani et al., 2017) have revolutionized various fields, particularly natural language processing and computer vision. They rely heavily on residual connections to facilitate the training of deep models. Our hyper-connections approach can replace residual connections, providing stable training and consistent improvements in both natural language processing and computer vision.

The issues of gradient vanishing and representation collapse (Bengio et al., 1994; Glorot & Bengio, 2010; Liu et al., 2020) have been extensively studied. The combinations of normalization techniques (Ioffe & Szegedy, 2015; Ba et al., 2016) and residual connections (He et al., 2016), like Pre-Norm and Post-Norm, actually reflects different emphases in solving these two issues. However, despite these advancements, the fundamental trade-off between gradient vanishing and representation collapse in deep networks remains a critical challenge. Building on these findings, our work introduces a novel approach that enables neural networks to autonomously learn the optimal strength of connections, potentially improving both gradient stability and representation quality.

# 6 CONCLUSION

In conclusion, we have introduced hyper-connections as an effective alternative to residual connections in transformers. Our analysis reveals that hyper-connections not only overcome the limitations of residuals but also enable dynamic adjustments in network architecture. Experimental results confirm their promising benefits across various tasks, including pre-training of large language model, image generation, and image classification.

# ACKNOWLEDGEMENTS

This research was conducted at ByteDance Inc. We are grateful for the suggestions and assistance provided by Yaowei Zheng, Yuyu Zhang, Yunshui Li, Xiang Li, Bairen Yi, Zhenyi Lu and Xintian Han.

# REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. In arXiv preprint arXiv:1607.06450, 2016.  
Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, and Xin Wang. Alternating updates for efficient transformers. Advances in Neural Information Processing Systems, 36, 2024.  
Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2), 1994.  
Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 2020.  
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.  
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.  
Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Machine learning challenges workshop. Springer, 2005.  
Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, 2019.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009.  
Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Third international workshop on paraphrasing (IWP2005), 2005.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.  
Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning. PMLR, 2015.  
Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017.  
Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. arXiv preprint arXiv:2205.05198, 2022.  
Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249, 2020.  
Haoyan Ma, Xiang Li, Xia Yuan, and Chunxia Zhao. Denseformer: A dense transformer framework for person re-identification. IET Computer Vision, 17(5), 2023.  
Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. arXiv preprint arXiv:2404.08801, 2024.  
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.  
Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060.  
William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.  
Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI spring symposium series, 2011.  
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9), 2021.  
Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiaq: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.  
N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, 2013.  
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159, 2024.  
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.

Ben Wang. Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX. https://github.com/kingofflolz/mesh-transformer-jax, May 2021.  
Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023.  
Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, and Rui Yan. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.  
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.  
Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

# A TRANSFORMER WITH HYPER-CONNECTIONS

Figure 8: Comparison between transformers with hyper-connections and that with residual connections.

# B PARAMETERS,COMPUTATION AND MEMORY FOOTPRINT ANALYSIS

Static Hyper-Connections. All learnable parameters are included in the hyper-connection matrix  $\mathcal{H}\mathcal{C}$  in Eq. 1. The number of parameters in one  $\mathcal{H}\mathcal{C}$  is given by:

$$
\left| \theta_ {\mathrm {S H C}} \right| = \left| \theta_ {\mathbf {B}} \right| + \left| \theta_ {\mathbf {A}} \right| = n + n \cdot (n + 1) = n \cdot (n + 2), \tag {21}
$$

where  $n$  is the expansion rate,  $|\theta_{\mathbf{B}}|$  is the number of parameters in  $\mathbf{B}$  in SHC, and  $|\theta_{\mathbf{A}}|$  is the number of parameters in  $\mathbf{A}$ . Each layer contains two hyper-connection modules (one for the self attention and one for the feedforward network). Thus, the number of extra parameters is:

$$
P _ {\text {e x t r a}} = \left| \theta_ {\mathrm {S H C}} \right| \times 2 \times L, \tag {22}
$$

where  $L$  is the number of layers. For example, in OLMo-1B-SHC×4,  $P_{\text{extra}} = 4 \times (4 + 2) \times 2 \times 16 = 768$ .

Dynamic Hyper-Connections. The parameters of DHC are defined in Eqs. 10, 11, 12, and 13, and the number of parameters is given by:

$$
\begin{array}{l} \left| \theta_ {\mathrm {D H C}} \right| = \left| \theta_ {\text {n o r m}} \right| + \left| s _ {\beta} \right| + \left| \theta_ {\mathbf {W} _ {\beta}} \right| + \left| \theta_ {\mathbf {B}} \right| + \left| s _ {\alpha} \right| + \left| \theta_ {\mathbf {W} _ {m}} \right| + \left| \theta_ {\mathbf {A} _ {m}} \right| + \left| \theta_ {\mathbf {W} _ {r}} \right| + \left| \theta_ {\mathbf {A} _ {r}} \right| (23) \\ = \left| \theta_ {\text {n o r m}} \right| + 1 + d _ {\text {m o d e l}} + n + 1 + d _ {\text {m o d e l}} + n + d _ {\text {m o d e l}} \times n + n \times n (24) \\ = \left| \theta_ {\text {n o r m}} \right| + d _ {\text {m o d e l}} \times (n + 2) + n \times (n + 2) + 2, (25) \\ \end{array}
$$

where  $d_{\mathrm{model}}$  is the dimension of the hidden states in the transformer, and  $|\theta_{\mathrm{norm}}|$  depends on the type of normalization module. In OLMo models, there are no parameters for normalization, so  $|\theta_{\mathrm{norm}}| = 0$ . In OLMoE,  $|\theta_{\mathrm{norm}}| = d_{\mathrm{model}}$ . Similar to the static hyper-connections, the number of extra parameters is:

$$
P _ {\text {e x t r a}} = \left| \theta_ {\mathrm {D H C}} \right| \times 2 \times L, \tag {26}
$$

For example, for OLMo-1B-DHC  $\times 4$ ,  $P_{\mathrm{extra}} = (0 + 2048 \times (4 + 2) + 4 \times (4 + 2) + 2) \times 2 \times 16 = 394.048$ .

The number of parameters for DHC and SHC used in the experiments is detailed in Table 7, while their corresponding FLOPs comparisons are provided in Table 8. Regardless of whether SHC or DHC is used, the additional parameters and computational overhead introduced are minimal and can be considered negligible.

Table 7: Comparison of number of parameters.  

<table><tr><td>Method</td><td>HC Params(B)</td><td>Total Params(B)</td><td>Total Params Δ rate (%)</td></tr><tr><td>OLMo-1B</td><td>-</td><td>1.17676442</td><td>-</td></tr><tr><td>OLMo-1B-SHC×2</td><td>0.0000026</td><td>1.17676467</td><td>+0.00002%</td></tr><tr><td>OLMo-1B-SHC×4</td><td>0.0000077</td><td>1.17676518</td><td>+0.00007%</td></tr><tr><td>OLMo-1B-DHC×2</td><td>0.0002625</td><td>1.17702688</td><td>+0.02230%</td></tr><tr><td>OLMo-1B-DHC×4</td><td>0.0003940</td><td>1.17715846</td><td>+0.03349%</td></tr><tr><td>OLMo-7B</td><td>-</td><td>6.88809574</td><td>-</td></tr><tr><td>OLMo-7B-DHC×4</td><td>0.0013124</td><td>6.88967027</td><td>+0.02286%</td></tr><tr><td>OLMoE-1B-7B</td><td>-</td><td>6.91909427</td><td>-</td></tr><tr><td>OLMoE-1B-7B-DHC×4</td><td>0.0003940</td><td>6.91948832</td><td>+0.00570%</td></tr></table>

**Computation Analysis.** The main computational cost of SHC and DHC lies in line 5 of Algorithm 1, where the complexity is  $\mathcal{O}(d_{\mathrm{model}} \times n \times (n + 1))$ . The computational cost of the FFN is  $\mathcal{O}(2 \times d_{\mathrm{model}} \times d_{\mathrm{ffn}})$ , and that of the projection part of attention is  $\mathcal{O}(4 \times d_{\mathrm{model}} \times d_{\mathrm{model}})$ . Since  $\mathcal{O}(d_{\mathrm{model}} \times n \times (n + 1)) \ll \mathcal{O}(4 \times d_{\mathrm{model}} \times d_{\mathrm{model}}) < \mathcal{O}(2 \times d_{\mathrm{model}} \times d_{\mathrm{ffn}})$ , the computational cost of HC is negligible compared to the cost of both FFN and the attention projection part. Here,  $d_{\mathrm{ffn}}$  is the inner dimension of the FFN. The detailed computation cost statistics are presented in Table 8.

Table 8: FLOPs per token in forward pass.  

<table><tr><td>Method</td><td>HC FLOPs (G)</td><td>Total FLOPs (G)</td><td>Total FLOPs Δ rate (%)</td></tr><tr><td>OLMo-1B</td><td>-</td><td>2.3536</td><td>-</td></tr><tr><td>OLMo-1B-SHC×2</td><td>0.0010</td><td>2.3545</td><td>+0.038%</td></tr><tr><td>OLMo-1B-SHC×4</td><td>0.0031</td><td>2.3566</td><td>+0.127%</td></tr><tr><td>OLMo-1B-DHC×2</td><td>0.0020</td><td>2.3554</td><td>+0.076%</td></tr><tr><td>OLMo-1B-DHC×4</td><td>0.0049</td><td>2.3583</td><td>+0.200%</td></tr><tr><td>OLMo-7B</td><td>-</td><td>13.3647</td><td>-</td></tr><tr><td>OLMo-7B-DHC×4</td><td>0.0197</td><td>13.3844</td><td>+0.147%</td></tr><tr><td>OLMoE-1B-7B</td><td>-</td><td>2.3580</td><td>-</td></tr><tr><td>OLMoE-1B-7B-DHC×4</td><td>0.0049</td><td>2.3629</td><td>+0.208%</td></tr></table>

Memory Footprint. The introduction of HC results in a minor increase in activation memory usage during training. For a transformer model with  $L$  layers, a model dimension of  $d_{\mathrm{model}}$ , batch size  $b$ , sequence length  $s$ , and number of attention heads  $a$ , the activation memory is calculated as  $sbd_{\mathrm{model}}L(34 + 5as / d_{\mathrm{model}})$ , as outlined in Korthikanti et al. (2022). Incorporating HC with an expansion rate of  $n$  adds an extra memory overhead of  $2nsbd_{\mathrm{model}}L$ . For  $n = 2$ , this contributes less than  $15\%$  to the total memory usage of a standard transformer. Notably, the memory consumption is mostly driven by the weight parameters, which experience only a slight increase with HC. Additionally, given HC's low computational cost, the hidden states generated by HC can be discarded post forward pass and recomputed during backpropagation to further optimize memory usage. With this approach, the additional memory requirement is reduced to  $nsbd_{\mathrm{model}}$ . During inference, the memory usage for activations is largely determined by the Key-Value cache, which is not impacted by the extra activations brought by HC. Moreover, the hidden states from earlier layers can be released as soon as the next layer's computations start, significantly lowering memory requirements. The actual memory footprint is empirically measured on 8 GPUs, as shown in Table 9.

Table 9: Measured Memory Footprint on 8 GPUs.  

<table><tr><td>Method</td><td>Memory (GB)</td><td>Memory Δ Rate (%)</td><td>Micro Batch Size (tokens per GPU)</td></tr><tr><td>OLMo-1B</td><td>41.11</td><td>-</td><td>16,384</td></tr><tr><td>OLMo-1B-SHC×2</td><td>47.55</td><td>+15.7%</td><td>16,384</td></tr><tr><td>OLMo-1B-SHC×4</td><td>51.85</td><td>+26.0%</td><td>16,384</td></tr><tr><td>OLMo-1B-DHC×2</td><td>47.56</td><td>+15.7%</td><td>16,384</td></tr><tr><td>OLMo-1B-DHC×4</td><td>51.86</td><td>+26.1%</td><td>16,384</td></tr><tr><td>OLMo-7B</td><td>26.27</td><td>-</td><td>2,048</td></tr><tr><td>OLMo-7B-DHC×4</td><td>33.70</td><td>+28.28%</td><td>2,048</td></tr><tr><td>OLMoE-1B-7B</td><td>31.59</td><td>-</td><td>4,096</td></tr><tr><td>OLMoE-1B-7B-DHC×4</td><td>34.65</td><td>+9.7%</td><td>4,096</td></tr></table>

# C MOE 1B/7B MODEL EXPERIMENTS

















Figure 9: Loss curves in V3 validation sets and accuracy curves on downstream tasks for OLMoE-1B7B and OLMoE-1B7B-DHC×4 models.












# D 7B MODEL EXPERIMENTS

c4 en val. loss

dolma books val, loss

dolma cc val, loss

dolma pes2o val.

dolma reddit val. loss

dolma stack val. loss

dolma wiki val. loss

ice val. loss

m2d2-s2orc val. loss

pile val. loss

wiktext 103 val. loss

HellaSwag Acc.  $(\%)$  
Openbook QA Acc.  $(\%)$

SciQ Acc. (%)

COPA Acc. (%)


PIQA Acc.  $(\%)$  
Figure 10: Loss curves in V3 validation set and accuracy curves on downstream tasks for OLMo-7B and OLMo-7B-DHC  $\times 4$  models.

WinoGrande Acc.  $(\%)$

ARC-Easy Acc.  $(\%)$

# E VISION EXPERIMENTS

Datasets. We use the ILSVRC-2012 ImageNet dataset (Deng et al., 2009) with 1k classes and 1.3M images (see ImageNet in the following) for image generation and classification.

# E.1 IMAGE GENERATION

To investigate the generalizability of hyper-connections in image generation, our experiments are conducted using the DiT framework (Peebles & Xie, 2022) training the models for 1400 epochs. In order to save experimental costs, we use FP16 precision, introduce flash-attention to speed up training, and introduce QK-Norm (Wortsman et al., 2023) to stabilize training.

Table 10: Benchmarking class-conditional image generation on ImageNet  $256 \times 256$ , with  $\mathrm{cfg} = 1.50$ . NP, P, and R are short for Numerical Precision, Precision, and Recall, respectively.  

<table><tr><td>Method</td><td>NP</td><td>QK-Norm</td><td>Size (M)</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>P↑</td><td>R↑</td></tr><tr><td>DiT-XL/2</td><td>FP32</td><td>×</td><td>675</td><td>2.27</td><td>4.60</td><td>278.24</td><td>0.83</td><td>0.57</td></tr><tr><td>DiT-XL/2</td><td>FP16</td><td>✓</td><td>675</td><td>2.36</td><td>4.54</td><td>269.46</td><td>0.83</td><td>0.58</td></tr><tr><td>DiT-1B/2</td><td>FP16</td><td>✓</td><td>983</td><td>2.13</td><td>4.50</td><td>288.69</td><td>0.82</td><td>0.59</td></tr><tr><td>DiT-XL/2-SHC×2</td><td>FP16</td><td>✓</td><td>675</td><td>2.18</td><td>4.52</td><td>287.24</td><td>0.82</td><td>0.60</td></tr></table>

Our experimental results demonstrate that DiT models incorporating hyper-connections exhibit comparable performance metrics to DiT models with  $50\%$  more parameters. This finding underscores the efficiency and efficacy of hyper-connections in enhancing model performance without increasing model size.

# E.2 IMAGE CLASSIFICATION

For the image classification experiments, we train ViT/16-Base and ViT/16-Large models with images at a resolution of  $224 \times 224$  for 300 epochs, following the experimental setup used by (Dosovitskiy et al., 2020). To speed up the training process, we use bfloat16 numerical precision. The training configuration is detailed in Table 12. Within this configuration, we replace the residual connections with static and dynamic hyper-connections, referred to as SHC and DHC, respectively, using an expansion rate of  $n = 2$ . The top-1 accuracy results are presented in Table 11, and the training loss curves for ViT/16-Large and ViT/16-Large with  $\mathrm{DHC} \times 2$  are shown in Fig. 11.

For the Base model (85M), our re-implemented ViT/16 achieves  $76.38\%$  accuracy on  $224 \times 224$  images. The SHC and DHC enhance performance to  $77.60\%$  and  $77.26\%$ , respectively. representing relative increases of  $1.22\%$  and  $0.88\%$ . For the Large model (307M parameters), ViT/16 achieves  $77.25\%$  accuracy. The SHC and DHC configurations further enhance accuracy to  $78.38\%$  and  $79.94\%$ , respectively. This corresponds to relative improvements of  $1.13\%$  and  $2.69\%$ , with DHC showing the highest performance. These results demonstrate that hyper-connections (SHC and DHC) significantly improve accuracy, especially in the Large model scale.

Table 11: Accuracy on ImageNet. ViT*/16 refers to the results reported by (Dosovitskiy et al., 2020), whereas ViT/16 denotes our re-implemented baseline. SHC and DHC indicate that residual connections are replaced with static and dynamic hyper-connections, respectively.  

<table><tr><td rowspan="2">Model Scales</td><td rowspan="2">Params (M)</td><td>ViT*/16</td><td>ViT/16</td><td>ViT/16-SHC×2</td><td>ViT/16-DHC×2</td></tr><tr><td>384 × 384</td><td></td><td>224 × 224</td><td></td></tr><tr><td>Base</td><td>85</td><td>77.91</td><td>76.38</td><td>77.60</td><td>77.26</td></tr><tr><td>Large</td><td>307</td><td>76.53</td><td>77.25</td><td>78.38</td><td>79.94</td></tr></table>

Figure 11: Training loss curves of ViT/16-Large and ViT/16-Large-DHC×2, smoothed using an Exponential Moving Average (EMA) with a decay rate of 0.999. The gain from Hyper-Connections decreases as training progresses, likely due to pass over the same dataset across many epochs, resulting in diminishing returns from the additional capacity provided by Hyper-Connections.

# E.3 VISULATION OF DHC

We randomly select three categories from the ImageNet dataset and sample the corresponding examples from the validation set. These samples are fed into the ViT-Base/16-DHC×2 model to compute the dynamic connection weights of the DHC in the final layer. As shown in Fig. 12, we visualize the distribution of these weights. We observe that the intra-class distribution of beta is highly concentrated, indicating that samples within the same category tend to have similar beta values. In contrast, the distribution of alpha is less concentrated, but the differences between the distributions of different categories are more pronounced, as exemplified by  $\alpha_{2,0}$ .

Table 12: Training hyperparameters for ViT.  

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate (lr)</td><td>0.003</td></tr><tr><td>Batch Size</td><td>4096</td></tr><tr><td>Scheduler</td><td>Cosine Annealing with Linear Warmup (10k steps)</td></tr><tr><td>Data Augmentation</td><td>Mixup (α = 0.2)</td></tr><tr><td>Epochs</td><td>300</td></tr><tr><td>Optimizer</td><td>AdamW (β1 = 0.9, β2 = 0.999, ε = 1e-8)</td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td>Weight Decay</td><td>0.3</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Precision</td><td>bf16</td></tr></table>



















Figure 12: Distribution of weights of last DHC in ViT-Base/16-DHC  $\times 2$  model.



# F MORE VISUALIZATION AND ANALYSIS

Unfolding hyper-connections. We first introduce how to determine the connection matrix  $\mathbf{C}^{(0)}$  for hyper-connections. To simplify writing, the layer output  $\mathcal{T}^k (\mathbf{h}_0^k)$  is denoted by  $\mathcal{T}^k$  for short. The







Figure 13: Visualization of unfolded connection matrix. Matrices from left to right are  $\mathbf{C}^{(0)}(\mathrm{Connections~for~}\{\mathbf{h}_0^j\}_{j = 0}^{L + 1})$ ,  $\mathbf{C}^{(i)}$  (Connections for  $\{\mathbf{h}'_i^j\}_{j = 0}^{L + 1}$ ) for  $i\in \{1,2,3,4\}$ . The attention layers, which have odd ids, are marked with green tick marks.

(a) Connection matrix for DHC model.


(b) Connection matrix for SHC model.



recurrent form of hyper connection in Eq. 2 is expanded as follows:

$$
\begin{array}{l} \mathbf {h} _ {\mathbf {0}} ^ {k} = \mathbf {H} ^ {k ^ {\intercal}} \mathbf {A} _ {\mathbf {m}} ^ {k} = \left(\mathcal {T} ^ {k - 1} \mathbf {B} ^ {k - 1} + \mathbf {H} ^ {k - 1 ^ {\intercal}} \mathbf {A} _ {\mathbf {r}} ^ {k - 1}\right) \mathbf {A} _ {\mathbf {m}} ^ {k} \\ = \sum_ {j = 0} ^ {k - 1} \mathcal {T} ^ {j} \mathbf {B} ^ {j} \left(\mathbf {A _ {r}} ^ {j + 1} \mathbf {A _ {r}} ^ {j + 2} \dots \mathbf {A _ {r}} ^ {k - 1}\right) \mathbf {A _ {m}} ^ {k} \\ = \sum_ {j = 0} ^ {k - 1} \mathcal {T} ^ {j} \mathbf {B} ^ {j} \left(\prod_ {t = j + 1} ^ {k - 1} \mathbf {A} _ {\mathbf {r}} ^ {t}\right) \mathbf {A} _ {\mathbf {m}} ^ {k}. \tag {27} \\ \end{array}
$$

Therefore, we obtain connection matrix  $c_{kj}^{(0)} = \mathbf{B}^j (\prod_{t = j + 1}^{k - 1}\mathbf{A_r}^t)\mathbf{A_m}^k$ . Similarly, the connection matrix  $\mathbf{C}^{(i)}$  for the  $i$ -th hyper hidden from  $k$ -th layer can be computed by substituting the last  $\mathbf{A}_{\mathbf{m}}^{k}$  with  $\mathbf{A}_{\mathbf{r}}^{k}$  in Eq. 27, i.e.,

$$
\mathbf {H} ^ {\prime k} = \mathbf {A} _ {\mathbf {r}} ^ {k ^ {\intercal}} \mathbf {H} ^ {k} = \sum_ {j = 0} ^ {k - 1} \left(\prod_ {t = j + 1} ^ {k} \mathbf {A} _ {\mathbf {r}} ^ {t}\right) ^ {\intercal} \mathbf {B} ^ {j ^ {\intercal}} \mathcal {T} ^ {j ^ {\intercal}} \tag {28}
$$

$$
c _ {k j} ^ {(i)} = \left(\left(\prod_ {t = j + 1} ^ {k} \mathbf {A} _ {\mathbf {r}} ^ {t}\right) ^ {\intercal} \mathbf {B} ^ {j \intercal}\right) _ {i}. \tag {29}
$$

Visualization for hyper hidden. We visualize connection matrices for hyper hiddens in Fig. 13 to reveal how hyper-connection maintains intermediate layer outputs. First of all, the four hyper hiddens are dissimilar and show completely different connection patterns. Then, we can see outputs from FFN layers are preserved long-termly in hyper hiddens, while attention layers are reserved less. It is also observed that the long-term connections are usually stored in pairs of hyper hiddens, where the connection is positive in one hyper hidden but negative in the other, for example, column 0 and 2 in  $\mathbf{C}^{(1)},\mathbf{C}^{(3)}$ . With such strategy, these connections can be easily eliminated in the sum-pooling operation before the unembedding layer.

SHC shares similar connection pattern with DHC. We show the connection matrices for OLMo-1B-SHC×4 model in Fig. 13b. Comparing to DHC, as shown in Fig. 13a, SHC shares exactly the same connection patterns. Moreover, we observe many more PTB-like blocks in SHC, e.g., layers from 13 to 18. Note that the connection relation for SHC is token independent, and such PTB-like blocks can be physically reorganized to be parallelly computed.

(a) OLMo-1B-DHC×1

(b) OLMo-1B-DHC×2  
Figure 14: Comparison of unfolded connection matrices for OLMo-1B-DHC×1, OLMo-1B-DHC×2 and OLMo-1B-DHC×4 model.

(c) OLMo-1B-DHC×4

How  $\mathbf{HC} \times 1$  fails. The OLMo-  $1\mathrm{B} \times 1$  model is observed to perform worse than baseline in our experiments. Its connection matrix is visualized in Fig. 14 to show how it fails. Above all, we observe that layer 17 is wasted, who has no connection to subsequent layers at all. Secondly, compared to  $\mathrm{HC} \times 2$  and  $\mathrm{HC} \times 4$  models, the  $\Lambda$  shaped pattern does not appear. Note that  $\mathrm{HC} \times 1$  does not support the pattern of  $\Lambda$  in its mathematical formulation, where the connections to previous layers must be weakened or strengthened simultaneously. Thus, the lack of connection from the early layers to the final layers may suffer from gradient vanishing, like post-norm style transformers, which leads to performance degeneration.

# G DERIVATION OF NON-TRAINABLE HYPER-CONNECTION MATRIX FOR RESIDUAL CONNECTIONS

# G.1 PRE-NORM RESIDUAL CONNECTION

In the Pre-Norm residual connection, the input to a layer is first normalized before being passed through the layer. The output of the layer is then added to the original input. This can be represented as:

$$
\hat {\mathbf {h}} = \mathcal {T} (\operatorname {N o r m} (\mathbf {h})) + \mathbf {h}. \tag {30}
$$

By incorporating the normalization operator into the layer,  $\mathcal{T} \coloneqq \mathcal{T} \circ \mathrm{Norm}$ , we can express the entire process as:

$$
\hat {\mathbf {h}} = \mathcal {T} (\mathbf {h}) + \mathbf {h}. \tag {31}
$$

To express this using hyper-connections, the matrix for Pre-Norm can be structured as follows:

$$
\mathcal {H C} _ {\text {P r e N o r m}} = \left( \begin{array}{l l} 0 & 1 \\ 1 & 1 \end{array} \right) \tag {32}
$$

Given hyper hidden matrix  $\mathbf{H} = \mathbf{h}^{\mathrm{T}}$ , we prove that the output of  $\mathcal{HC}_{\mathrm{PreNorm}}\hat{\mathbf{H}} = \hat{\mathbf{h}}^{\mathrm{T}}$ .

Proof.

$$
\begin{array}{l} \hat {\mathbf {H}} = \mathcal {H C} (\mathcal {T}, \mathbf {H}) \\ = \mathbf {B} ^ {\intercal} \mathcal {T} (\mathbf {H} ^ {\intercal} \mathbf {A} _ {\mathbf {m}}) ^ {\intercal} + \mathbf {A} _ {\mathbf {r}} ^ {\intercal} \mathbf {H} \\ = \mathcal {T} (\mathbf {h}) ^ {\intercal} + \mathbf {h} ^ {\intercal} \tag {33} \\ = \hat {\mathbf {h}} ^ {\intercal}. \\ \end{array}
$$


# G.2 POST-NORM RESIDUAL CONNECTION

In the Post-Norm residual connection, the input to a layer is passed through the layer first, and then the output is normalized after being added to the original input. In matrix form, this can be represented as:

$$
\mathbf {h} ^ {\prime} = \mathcal {T} (\mathbf {h}) \tag {34}
$$

The summation of the input and the normalized output of the layer is:

$$
\hat {\mathbf {h}} = \operatorname {N o r m} \left(\mathbf {h} + \mathbf {h} ^ {\prime}\right) \tag {35}
$$

We consider Norm to be LayerNorm (Zhang & Sennrich, 2019). The analysis process for RMSNorm is almost identical. In fact, the affine transformation can be incorporated into the subsequent layer, while the mean subtraction operation can be integrated into the current layer.

$$
\mathcal {T} = \mathcal {C} \circ \mathcal {T} \circ \mathcal {A}, \tag {36}
$$

where  $\mathcal{A}$  is the affine transformation, and  $\mathcal{C}$  is the re-centering operator. Thus, the mean of the output of  $\mathcal{T}$  is 0.

To express this using hyper-connections with an expansion rate  $n = 1$ , we need a hyper-connection matrix  $\mathcal{HC}$  that encapsulates this operation:

$$
\mathcal {H C} _ {\text {P o s t N o r m}} = \left( \begin{array}{l l} 0 & \frac {1}{\sqrt {\sigma_ {\mathrm {h}} ^ {2} + \sigma_ {\mathrm {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathrm {h h} ^ {\prime}}}} \\ 1 & \frac {1}{\sqrt {\sigma_ {\mathrm {h}} ^ {2} + \sigma_ {\mathrm {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathrm {h h} ^ {\prime}}}} \end{array} \right) = \left( \begin{array}{l l} 0 & \mathbf {B} \\ \mathbf {A} _ {m} & \mathbf {A} _ {r} \end{array} \right). \tag {37}
$$

Similar to the previous proof, we prove that the output of  $\mathcal{HC}_{\mathrm{PostNorm}}$  is equivalent to the transpose of the output of the Post-Norm residual connection:

$$
\hat {\mathbf {H}} = \hat {\mathbf {h}} ^ {\intercal}. \tag {38}
$$

Proof: Note that

$$
\sigma_ {\mathbf {h} + \mathbf {h} ^ {\prime}} = \sqrt {\sigma_ {\mathbf {h}} ^ {2} + \sigma_ {\mathbf {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathbf {h h} ^ {\prime}}}. \tag {39}
$$

Given this fact, we can derive the Post-Norm:

$$
\begin{array}{l} \hat {\mathbf {h}} = \operatorname {N o r m} \left(\mathbf {h} ^ {\prime} + \mathbf {h}\right) \\ = \frac {\mathbf {h} ^ {\prime} + \mathbf {h} - \mu_ {\mathbf {h} ^ {\prime} + \mathbf {h}}}{\sigma_ {\mathbf {h} + \mathbf {h} ^ {\prime}}} \\ = \frac {1}{\sigma_ {\mathbf {h} ^ {\prime} + \mathbf {h}}} \left(\mathbf {h} ^ {\prime} + \mathbf {h}\right) \tag {40} \\ = \frac {1}{\sqrt {\sigma_ {\mathbf {h}} ^ {2} + \sigma_ {\mathbf {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathbf {h h} ^ {\prime}}}} (\mathbf {h} ^ {\prime} + \mathbf {h}) \\ \end{array}
$$

For hyper-connections side, we have:

$$
\begin{array}{l} \hat {\mathbf {H}} = \mathbf {B} ^ {\intercal} \mathbf {h} ^ {\prime \intercal} + \mathbf {H} ^ {\prime} \\ = \mathbf {B} ^ {\mathrm {T}} \mathbf {h} ^ {\prime \mathrm {T}} + \mathbf {A} _ {r} \mathbf {H} \\ = \mathbf {B} ^ {\intercal} \mathbf {h} ^ {\prime \intercal} + \mathbf {A} _ {r} \mathbf {h} ^ {\intercal} \tag {41} \\ {= \frac {1}{\sqrt {\sigma_ {\mathbf {h}} ^ {2} + \sigma_ {\mathbf {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathbf {h h} ^ {\prime}}}} \mathbf {h} ^ {\prime \intercal} + \frac {1}{\sqrt {\sigma_ {\mathbf {h}} ^ {2} + \sigma_ {\mathbf {h} ^ {\prime}} ^ {2} + 2 \sigma_ {\mathbf {h h} ^ {\prime}}}} \mathbf {h} ^ {\intercal}} = \hat {\mathbf {h}} ^ {\intercal}.} \\ \end{array}
$$

# H SEQUENTIAL-PARALLEL DUALITY

# H.1 HYPER-CONNECTION MATRIX OF SEQUENTIAL ARRANGEMENT

In this section, we demonstrate that the following hyper-connection matrix will produce  $n$  identical networks arranged sequentially with residual connections between them:

$$
\mathcal {H C} = \left( \begin{array}{c c} \mathbf {0} _ {1 \times 1} & \mathbf {1} _ {1 \times n} \\ \mathbf {e} _ {1} & \mathbf {e} _ {n \times n} \end{array} \right), \tag {42}
$$

where  $\mathbf{e}_{n\times n}$  denotes an  $n\times n$  identity matrix,  $\mathbf{e}_i\in \mathbb{R}^{n\times 1}$  represents the  $i$ -th column of  $\mathbf{e}_{n\times n}$ , and  $\mathbf{1}_{1\times n}$  signifies a  $1\times n$  matrix of ones.

We will use mathematical induction to prove that  $\mathbf{h}_i^k = \mathbf{h}_j^k$  and  $\mathbf{h}_i^{k + 1} = \mathcal{T}^k (\mathbf{h}_i^k) + \mathbf{h}_i^k$ ,  $\forall i,j\in \{0,1,\ldots ,n\}$ ,  $\forall k\in \{0,1,\dots ,L\}$ , where  $L$  is the number of layers.

# Proof. BASE CASE

For  $k = 0$ , we have the initial condition  $\mathbf{h}_i^0 = \mathbf{h}_j^0, \forall i,j \in \{0,1,\ldots,n\}$ , as we define  $\mathbf{H}^0 = (\mathbf{h}^0, \mathbf{h}^0, \ldots, \mathbf{h}^0)^\top \in \mathbb{R}^{n\times d}$ .

# INDUCTION HYPOTHESIS

Assume that for some  $k \in \{1, \dots, L - 1\}$ , we have  $\mathbf{h}_i^k = \mathbf{h}_j^k$  and  $\mathbf{h}_i^k = \mathcal{T}^k(\mathbf{h}_i^{k-1}) + \mathbf{h}_i^{k-1}$ ,  $\forall i, j \in \{0, 1, \dots, n\}$ .

# INDUCTION STEP

We have

$$
\begin{array}{l} \mathbf {H} ^ {k + 1} = \mathcal {H C} \left(\mathcal {T} ^ {k}, \mathbf {H} ^ {k}\right) (43) \\ = \mathbf {B} ^ {\intercal} \left(\mathbf {h} _ {0} ^ {\prime k}\right) ^ {\intercal} + \mathbf {H} ^ {\prime k} (44) \\ = \mathbf {B} ^ {\intercal} \mathbf {A} _ {\mathbf {m}} ^ {\intercal} \mathbf {H} ^ {k} + \mathbf {A} _ {\mathbf {r}} ^ {\intercal} \mathbf {H} ^ {k} (45) \\ = \mathbf {1} _ {n \times 1} \mathcal {T} ^ {k} \left(\mathbf {e} _ {1} ^ {\intercal} \mathbf {H} ^ {k}\right) + \mathbf {e} _ {n \times n} \mathbf {H} ^ {k} (46) \\ = \left(\mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) \quad \mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) \quad \dots \quad \mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right)\right) ^ {\intercal} + \left(\mathbf {h} _ {1} ^ {k} \quad \mathbf {h} _ {2} ^ {k} \quad \dots \quad \mathbf {h} _ {n} ^ {k}\right) ^ {\intercal} (47) \\ = \left(\mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) + \mathbf {h} _ {1} ^ {k} \quad \mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) + \mathbf {h} _ {2} ^ {k} \quad \dots \quad \mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) + \mathbf {h} _ {n} ^ {k}\right) ^ {\intercal} (48) \\ = \left(\mathbf {h} _ {1} ^ {k + 1} \quad \mathbf {h} _ {2} ^ {k + 1} \quad \dots \quad \mathbf {h} _ {n} ^ {k + 1}\right) ^ {\intercal} (49) \\ \end{array}
$$

Since  $\mathbf{h}_i^k = \mathbf{h}_j^k$ ,  $\forall i, j \in \{0, 1, \dots, n\}$ , it follows that  $\mathcal{T}^k(\mathbf{h}_1^k) + \mathbf{h}_i^k = \mathcal{T}^k(\mathbf{h}_1^k) + \mathbf{h}_j^k$ . Thus, we have

$$
\mathbf {h} _ {i} ^ {k + 1} = \mathbf {h} _ {j} ^ {k + 1} \tag {50}
$$

Since  $\mathbf{h}_i^k = \mathbf{h}_j^k$ ,  $\forall i, j \in \{0, 1, \dots, n\}$ , it follows that  $\mathbf{h}_1^k = \mathbf{h}_i^k$ ,  $\forall i \in \{0, 1, \dots, n\}$ . Thus, we have

$$
\begin{array}{l} \mathbf {h} _ {i} ^ {k + 1} = \mathcal {T} ^ {k} \left(\mathbf {h} _ {1} ^ {k}\right) + \mathbf {h} _ {i} ^ {k} (51) \\ = \mathcal {T} ^ {k} \left(\mathbf {h} _ {i} ^ {k}\right) + \mathbf {h} _ {i} ^ {k} (52) \\ \end{array}
$$


# H.2 HYPER-CONNECTION MATRIX OF PARALLEL ARRANGEMENT

In this section, we demonstrate that the following hyper-connection matrix will produce a network where every  $n$  adjacent layers are arranged in parallel, with each layer incorporating residual connections. We define a parallel-arranged network such that  $n$  adjacent layers form a group, with layers within a group being parallel and groups arranged sequentially. The output of  $k$ -th group is given by:

$$
\mathbf {h} ^ {k + 1} = \sum_ {i = 1} ^ {n} \left(\mathcal {T} ^ {k \times n + i} \left(\mathbf {h} ^ {k}\right) + \mathbf {h} ^ {k}\right). \tag {53}
$$

It can be proved that this arrangement can be described by the following hyper-connection matrices.

First, for  $k$  where  $k - 1 \equiv 0 \pmod{n}$ :

$$
\mathcal {H C} ^ {\{k \mid k - 1 \equiv 0 \pmod {n} \}} = \left( \begin{array}{l l} \mathbf {0} _ {1 \times 1} & \mathbf {e} _ {1} ^ {\intercal} \\ \mathbf {1} _ {n \times 1} & \mathbf {1} _ {n \times n}, \end{array} \right) \tag {54}
$$

where the  $\mathcal{HC}$  matrix can be decomposed into two operations: 1) sum up all the outputs of the previous group and use it as the input of the current layer and as the residual of the subsequent layers; 2) sum up the output and input saving to the first hidden vector slot.

Next, for  $k$  where  $k - 1 \equiv i \pmod{n}$  and  $i \neq 0$ :

$$
\mathcal {H C} ^ {\{k \mid k - 1 \equiv i \pmod {n}, i \neq 0 \}} = \left( \begin{array}{c c} \mathbf {0} _ {1 \times 1} & \mathbf {e} _ {i} ^ {\intercal} \\ \mathbf {e} _ {i} & \mathbf {e} _ {n \times n}, \end{array} \right). \tag {55}
$$

where the  $\mathcal{HC}$  matrix selects the  $i$ -th hidden vector as the input of the current layer, and sums up the output and input, saving to the  $i$ -th hidden vector slot.

This means:

$$
\mathbf {h} ^ {k + 1} = \mathcal {H C} ^ {(k + 1) \times n} \left(\mathcal {T} ^ {(k + 1) \times n}, \right. \tag {56}
$$

$$
\mathcal {H C} ^ {(k + 1) \times n - 1} \left(\mathcal {T} ^ {(k + 1) \times n - 1}, \right. \tag {57}
$$

$$
\dots \tag {58}
$$

$$
\mathcal {H C} ^ {k \times n + 1} \left(\mathcal {T} ^ {k \times n + 1}, \mathbf {h} ^ {k}\right)) \tag {59}
$$

This can also be proved by mathematical induction; however, the conclusion is quite obvious through drawing, and the proof process is very tedious. Therefore, we don't repeat the similar proof here.

# I PSEUDOCODE OF HYPER-CONNECTIONS

<table><tr><td colspan="2">Algorithm 1 Network with Hyper-Connections</td></tr><tr><td colspan="2">Require: Initial hidden vector h0 ∈ Rd</td></tr><tr><td colspan="2">Require: Expansion rate n</td></tr><tr><td colspan="2">Ensure: Final output y</td></tr><tr><td colspan="2">1: Initialize:</td></tr><tr><td colspan="2">2: H0← (h0 h0 ... h0)T ∈ Rn×d</td></tr><tr><td>3: for k = 1 to L do</td><td>▷ For each layer</td></tr><tr><td colspan="2">4: H ← Hk-1</td></tr><tr><td>5: (h0 H&#x27;) ← WCTH</td><td>▷ Width Connections</td></tr><tr><td>6: h0&#x27; ← Tk(h0)</td><td>▷ Layer Computation</td></tr><tr><td>7: H ← BkTH&#x27; + H&#x27;</td><td>▷ Depth Connections</td></tr><tr><td colspan="2">8: Hk ← H</td></tr><tr><td colspan="2">9: end for</td></tr><tr><td colspan="2">10: Final Output:</td></tr><tr><td colspan="2">11: hL ← sum rows of HL</td></tr><tr><td colspan="2">12: hL ← Normalization Layer(hL)</td></tr><tr><td colspan="2">13: y ← Output Layer(hL)</td></tr><tr><td colspan="2">14: return y</td></tr></table>

# J PYTORCH IMPLEMENTATION OF HYPER-CONNECTIONS

Algorithm 2 Pseudocode of hyper-connections in a PyTorch-like style.  
```python
h: hyper hidden matrix (BxLxNxD)   
class HyperConnection(nnModule): def __init__(self, dim, rate, layer_id, dynamic, device=None): super(HyperConnection, self).__init_(   ) self RATE  $=$  rate self(layer_id  $\equiv$  layer_id self.dynamic  $=$  dynamic self(static_beta  $=$  nn.Parameters(torch.ones((rate,) device=device)) init_alpha0  $\equiv$  torch.zeros((rate, 1), device  $\equiv$  device) init_alpha0[layer_id % rate, 0] = 1. self/static_alpha  $\equiv$  nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device= device)], dim=1)) if self.dynamic: self.dynamic_alpha_fn  $\equiv$  nn.Parameter(torch.zeros((dim, rate+1), device  $\equiv$  device)) self.dynamic_alpha_scale  $\equiv$  nn.Parameter(torch.ones(1, device  $\equiv$  device)  $\star 0.01$  self.dynamic_beta_fn  $\equiv$  nn.Parameter(torch.zeros((dim,) device  $\equiv$  device)) self.dynamic_beta_scale  $\equiv$  nn.Parameter(torch.ones(1, device  $\equiv$  device)  $\star 0.01$  self(layer_norm  $=$  LayerNorm(dim) def width_connection(self, h): # get alpha and beta if self.dynamic: norm_h  $=$  self(layer_norm(h) if self.dynamic: wc_weight  $=$  norm_h @ self.dynamic_alpha_fn wc_weight  $=$  F.tanh(wc_weight) dynamic_alpha  $\equiv$  wc_weight  $\star$  self.dynamic_alpha_scale alpha  $=$  dynamic_alpha + self/static_alpha[None, None, ... ] else: alpha  $=$  self/static_alpha[None, None, ... ] if self.dynamic: dc_weight  $=$  norm_h @ self.dynamic_beta_fn dc_weight  $=$  F.tanh(dc_weight) dynamic_beta  $=$  dc_weight  $\star$  self.dynamic_beta_scale beta  $=$  dynamic_beta + self/static_beta[None, None, ... ] else: beta  $=$  self/static_beta[None, None, ... ] # width connection mix_h  $=$  alpha.transpose(-1, -2) @ h return mix_h, beta   
def depth_connection(self, mix_h, h_o, beta): h  $=$  torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[., 1:, :] return h
```

Algorithm 3 Pseudocode of transformer with hyper-connections in a PyTorch-like style.  
```python
h: hyper hidden matrix (BxLxNxD)  
# attenuHyper_connection, ffn_hyper_connection: hyper-connection modules  
# attenu_norm, ffn_norm: normalization modules  
# Attention Block  
mix_h, beta = attenu_hyper_connection.width_connection(h)  
h = attenu_norm(mix_h[.,0,:])  
h = selfattention(h)  
h = attenu_hyper_connection.depth_connection(mix_h, dropout(h), beta)  
# FFN Block  
mix_h, beta = ffn_hyper_connection.width_connection(h)  
h = ffn_norm(mix_h[.,0,:])  
h = ffn(h)  
h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta)
```

# K VALIDATION SETS AND DOWNSTREAM TASKS

Table 13: OLMo's default configuration was evaluated using multiple metrics. Perplexity (PPL) and loss were used for the V2 and V3 Validation Sets, while zero-shot testing was applied to the Downstream Benchmarks. However, the grey benchmarks were excluded from our analysis due to the instability of their performance indicators.

V2 Validation Sets  
V3 Validation Sets  
```txt
v2-small-4chan-validation  
v2-small-c4_100_domains-validation  
v2-small-c4_en-validation  
v2-small-gab-validation  
v2-small-ice-validation  
v2-small-m2d2_s2orc-validation  
v2-small-m2d2_wiki-validation  
v2-small-manosphere-validation  
v2-small-mc4_en-validation  
v2-small-pile-validation  
v2-small-ptb-validation  
v2-small-twitterAEE-validation  
v2-small-wikitext_103-validation
```

Downstream Benchmarks  
```txt
v3-small-c4_en-validation  
v3-small-dolma_books-validation  
v3-small-dolma_common-crawl-validation  
v3-small-dolma_pes2o-validation  
v3-small-dolma_redgit-validation  
v3-small-dolma_stack-validation  
v3-small-dolma_wiki-validation  
v3-small-ice-validation  
v3-small-m2d2_s2orc-validation  
v3-small-pile-validation  
v3-small-wikitext_103-validation
```

```txt
piqa (Bisk et al., 2020)  
hellaswag (Zellers et al., 2019)  
winogrande (Sakaguchi et al., 2021)  
openbook_qa (Mihaylov et al., 2018)  
sciq (Johannes Welbl, 2017)  
arc_easy (Clark et al., 2018)  
copa (Roemmle et al., 2011)  
commitment_bank (De Marneffé et al., 2019)  
mrpc (Dolan & Brockett, 2005)  
rte (Dagan et al., 2005)  
sst2 (Socher et al., 2013)
```

Table 14: Downstream Benchmarks for OLMoE.  

<table><tr><td>Downstream Benchmarks for OLMoE</td></tr><tr><td>piqa (Bisk et al., 2020)</td></tr><tr><td>hellaswag (Zellers et al., 2019)</td></tr><tr><td>winogrande (Sakaguchi et al., 2021)</td></tr><tr><td>openbook_qa (Mihaylov et al., 2018)</td></tr><tr><td>sciq (Johannes Welbl, 2017)</td></tr><tr><td>arc_easy (Clark et al., 2018)</td></tr><tr><td>arc_challenage (Clark et al., 2018)</td></tr><tr><td>copa (Roemmele et al., 2011)</td></tr><tr><td>boolq (Clark et al., 2019)</td></tr><tr><td>commonsense_qa (Talmor et al., 2018)</td></tr><tr><td>social_iqa (Sap et al., 2019)</td></tr><tr><td>mmlu (Hendrycks et al., 2021)</td></tr></table>

# L 1B MODEL EXPERIMENTS

Figure 15: Training loss curves of related works, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

Figure 16: Training loss curves of DHC with tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

Figure 17: Training loss curves of DHC without tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

Figure 18: Training loss curves compared with parallel transformer blocks (PTB), smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99.

Table 15: Results on downstream benchmarks for 1B models.  

<table><tr><td>Method</td><td>arc.easy</td><td>copa</td><td>hellaswag</td><td>openbook_qa</td><td>piqa</td><td>sciq</td><td>winogrande</td><td>avg.</td></tr><tr><td>OLMo-1B</td><td>56.8</td><td>76.0</td><td>56.1</td><td>33.8</td><td>74.4</td><td>85.1</td><td>55.6</td><td>62.5</td></tr><tr><td colspan="9">Scaling n in DHC W/O tanh</td></tr><tr><td>OLMo-1B-DHCx1 W/O tanh</td><td>56.8</td><td>75.0</td><td>55.3</td><td>33.4</td><td>72.9</td><td>85.4</td><td>57.1</td><td>62.3</td></tr><tr><td>OLMo-1B-DHCx2 W/O tanh</td><td>63.0</td><td>74.0</td><td>57.1</td><td>34.6</td><td>73.5</td><td>86.0</td><td>58.2</td><td>63.8</td></tr><tr><td>OLMo-1B-DHCx4 W/O tanh</td><td>61.2</td><td>80.0</td><td>57.5</td><td>33.6</td><td>75.5</td><td>85.8</td><td>56.9</td><td>64.4</td></tr><tr><td>OLMo-1B-DHCx8 W/O tanh</td><td>61.1</td><td>75.0</td><td>57.6</td><td>35.4</td><td>73.8</td><td>85.2</td><td>58.5</td><td>63.8</td></tr><tr><td colspan="9">Scaling n in DHC</td></tr><tr><td>OLMo-1B-DHCx1</td><td>59.7</td><td>74.0</td><td>55.5</td><td>33.6</td><td>73.5</td><td>85.4</td><td>54.5</td><td>62.3</td></tr><tr><td>OLMo-1B-DHCx2</td><td>59.7</td><td>73.0</td><td>56.7</td><td>34.0</td><td>74.7</td><td>85.2</td><td>57.9</td><td>63.0</td></tr><tr><td>OLMo-1B-DHCx4</td><td>59.8</td><td>79.0</td><td>58.1</td><td>32.4</td><td>74.3</td><td>86.1</td><td>57.1</td><td>63.8</td></tr><tr><td>OLMo-1B-DHCx8</td><td>56.8</td><td>75.0</td><td>58.0</td><td>34.4</td><td>73.8</td><td>84.2</td><td>57.3</td><td>62.8</td></tr><tr><td colspan="9">Scaling n in SHC</td></tr><tr><td>OLMo-1B-SHCx2</td><td>59.1</td><td>77.0</td><td>56.6</td><td>35.4</td><td>74.2</td><td>85.3</td><td>56.4</td><td>63.4</td></tr><tr><td>OLMo-1B-SHCx4</td><td>59.3</td><td>77.0</td><td>56.7</td><td>34.0</td><td>74.3</td><td>86.6</td><td>57.1</td><td>63.6</td></tr><tr><td colspan="9">Non-trainable WC</td></tr><tr><td>OLMo-1B-DHCx4</td><td>60.5</td><td>78.0</td><td>56.2</td><td>34.0</td><td>73.5</td><td>86.0</td><td>55.8</td><td>63.4</td></tr><tr><td>OLMo-1B-DHCx4 W/O tanh</td><td>59.1</td><td>72.0</td><td>56.8</td><td>35.0</td><td>73.3</td><td>86.0</td><td>55.5</td><td>62.5</td></tr><tr><td colspan="9">Non-trainable B</td></tr><tr><td>OLMo-1B-DHCx4</td><td>59.5</td><td>77.0</td><td>57.9</td><td>33.8</td><td>73.3</td><td>85.6</td><td>56.6</td><td>63.4</td></tr><tr><td>OLMo-1B-DHCx4 W/O tanh</td><td>60.4</td><td>74.0</td><td>57.6</td><td>34.0</td><td>74.9</td><td>86.7</td><td>57.5</td><td>63.6</td></tr></table>

<table><tr><td>18L&#x27;Z</td><td>9097°</td><td>8E9&#x27;€</td><td>L88&#x27;€</td><td>222&#x27;€</td><td>964&#x27;€</td><td>500&#x27;€</td><td>L95&#x27;€</td><td>990&#x27;€</td><td>689&#x27;€</td><td>L4E&#x27;€</td><td>6E&#x27;L&#x27;€</td><td>265&#x27;€</td><td>562&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td>E8L&#x27;Z</td><td>229&#x27;€</td><td>L29&#x27;€</td><td>L16&#x27;€</td><td>122&#x27;€</td><td>L64&#x27;€</td><td>800&#x27;€</td><td>695&#x27;€</td><td>150&#x27;€</td><td>t89&#x27;€</td><td>84E&#x27;€</td><td>24L&#x27;€</td><td>t65&#x27;€</td><td>962&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td colspan="15">Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-
non</td><td></td></tr><tr><td>t08&#x27;Z</td><td>t49&#x27;Z</td><td>£99&#x27;€</td><td>S66&#x27;€</td><td>017&#x27;€</td><td>015&#x27;€</td><td>520&#x27;€</td><td>585&#x27;€</td><td>001&#x27;€</td><td>01L&#x27;€</td><td>L5E&#x27;€</td><td>5SL&#x27;€</td><td>609&#x27;€</td><td>80E&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td>208&#x27;Z</td><td>9E9&#x27;Z</td><td>8L9&#x27;€</td><td>656&#x27;€</td><td>8E9&#x27;€</td><td>805&#x27;€</td><td>t20&#x27;€</td><td>£85&#x27;€</td><td>LL0&#x27;€</td><td>00L&#x27;€</td><td>L5E&#x27;€</td><td>2SL&#x27;€</td><td>809&#x27;€</td><td>21E&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td colspan="16">CWM Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non- Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-
DHS un i u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uu u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuUUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuucuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuccuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuccuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuddu:</td></tr><tr><td>8L&#x27;Z</td><td>809&#x27;Z</td><td>1E9&#x27;€</td><td>9L8&#x27;Z</td><td>612&#x27;€</td><td>864&#x27;€</td><td>800&#x27;€</td><td>L95&#x27;€</td><td>t50&#x27;€</td><td>t89&#x27;€</td><td>E55&#x27;€</td><td>6E&#x27;L&#x27;€</td><td>165&#x27;€</td><td>562&#x27;€</td><td>862&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td></tr><tr><td>18L&#x27;Z</td><td>119&#x27;Z</td><td>1E9&#x27;€</td><td>068&#x27;Z</td><td>812&#x27;€</td><td>764&#x27;€</td><td>500&#x27;€</td><td>t90&#x27;€</td><td>t90&#x27;€</td><td>t89&#x27;€</td><td>t55&#x27;€</td><td>8E&#x27;L&#x27;€</td><td>165&#x27;€</td><td>062&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td>208&#x27;Z</td><td>9E9&#x27;Z</td><td>t0L&#x27;€</td><td>056&#x27;Z</td><td>L22&#x27;€</td><td>605&#x27;€</td><td>720&#x27;€</td><td>L85&#x27;€</td><td>190&#x27;€</td><td>E0L&#x27;€</td><td>L95&#x27;€</td><td>809&#x27;€</td><td>605&#x27;€</td><td>605&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td>618&#x27;Z</td><td>8L9&#x27;Z</td><td>2E9&#x27;€</td><td>196&#x27;Z</td><td>292&#x27;€</td><td>335&#x27;€</td><td>L30&#x27;€</td><td>909&#x27;€</td><td>060&#x27;€</td><td>82L&#x27;€</td><td>9E&#x27;L&#x27;€</td><td>5LL&#x27;€</td><td>529&#x27;€</td><td>2E2&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td colspan="15">DHS un i u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uu u u uu u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uu u u u uu:</td><td></td></tr><tr><td>LLL&#x27;Z</td><td>609&#x27;Z</td><td>8E9&#x27;€</td><td>8L8&#x27;Z</td><td>812&#x27;€</td><td>764&#x27;€</td><td>900&#x27;€</td><td>295&#x27;€</td><td>090&#x27;€</td><td>589&#x27;€</td><td>055&#x27;€</td><td>685&#x27;€</td><td>262&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td><td></td></tr><tr><td>6L&#x27;Z</td><td>019&#x27;Z</td><td>2E9&#x27;€</td><td>868&#x27;Z</td><td>122&#x27;€</td><td>764&#x27;€</td><td>500&#x27;€</td><td>295&#x27;€</td><td>950&#x27;€</td><td>989&#x27;€</td><td>5E&#x27;L&#x27;€</td><td>165&#x27;€</td><td>562&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td><td></td></tr><tr><td>26L&#x27;Z</td><td>529&#x27;Z</td><td>3E9&#x27;€</td><td>806&#x27;Z</td><td>122&#x27;€</td><td>605&#x27;€</td><td>510&#x27;€</td><td>685&#x27;€</td><td>690&#x27;€</td><td>00L&#x27;€</td><td>795&#x27;€</td><td>64L&#x27;€</td><td>009&#x27;€</td><td>11E&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td></tr><tr><td>228&#x27;Z</td><td>2L9&#x27;Z</td><td>3E9&#x27;€</td><td>846&#x27;Z</td><td>122&#x27;€</td><td>960&#x27;€</td><td>609&#x27;€</td><td>201&#x27;€</td><td>52L&#x27;€</td><td>6E&#x27;L&#x27;€</td><td>5LL&#x27;€</td><td>929&#x27;€</td><td>02E&#x27;€</td><td>quen O/A+HCH-BH-IR-WTO</td><td></td><td></td></tr><tr><td colspan="15">quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-IR-WTO quen O/A+HCH-BH-BH-IR-WTO quen O/A+HCH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BH-BB:</td><td></td></tr></table>

IaONnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnaeNnnae

<table><tr><td>t0S&#x27;LI</td><td>99L&#x27;EL</td><td>019&#x27;LE</td><td>8L&#x27;8I</td><td>02C&#x27;6</td><td>7I&#x27;2I</td><td>85&#x27;2O</td><td>150&#x27;EL</td><td>0E&#x27;1&#x27;2</td><td>149&#x27;I</td><td>9E&#x27;8Z</td><td>01S&#x27;5I</td><td>98E&#x27;EL</td><td>266&#x27;6</td><td>quen O/A</td><td>t+XHCH-BI-WTO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>86&#x27;&#x27;L&#x27;1</td><td>86&#x27;&#x27;EL</td><td>000&#x27;8E</td><td>7E&#x27;6LI</td><td>8E&#x27;6I</td><td>9E&#x27;2I</td><td>88&#x27;1O</td><td>120&#x27;EL</td><td>7E&#x27;5&#x27;2</td><td>7E&#x27;2&#x27;2</td><td>7E&#x27;1&#x27;8Z</td><td>5E&#x27;5&#x27;5</td><td>7E&#x27;5&#x27;5</td><td>7E&#x27;5&#x27;5</td><td>7E&#x27;5&#x27;5</td><td>7E&#x27;5&#x27;5</td><td>7E&#x27;5&#x27;5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>t+16&#x27;LI</td><td>896&#x27;EL</td><td>0LS&#x27;6E</td><td>7L&#x27;6I</td><td>8E&#x27;6I</td><td>9E&#x27;2I</td><td>85&#x27;2O</td><td>150&#x27;EL</td><td>969&#x27;12</td><td>8L&#x27;8&#x27;2</td><td>999&#x27;EL</td><td>995&#x27;EL</td><td>995&#x27;EL</td><td>760&#x27;01</td><td>quen O/A</td><td>t+XHCH-BI-WTO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>716&#x27;LI</td><td>020&#x27;AI</td><td>696&#x27;8E</td><td>910&#x27;6I</td><td>06E&#x27;6</td><td>01E&#x27;2I</td><td>765&#x27;02</td><td>892&#x27;EL</td><td>981&#x27;22</td><td>8E&#x27;20&#x27;5</td><td>689&#x27;8Z</td><td>172&#x27;5&#x27;1</td><td>785&#x27;EL</td><td>785&#x27;EL</td><td>785&#x27;EL</td><td>785&#x27;EL</td><td>785&#x27;EL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-ON-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1L&#x27;9&#x27;LI</td><td>988&#x27;EL</td><td>695&#x27;8E</td><td>6H&#x27;8I</td><td>8E&#x27;6I</td><td>9E&#x27;2I</td><td>85&#x27;2O</td><td>150&#x27;EL</td><td>961&#x27;1</td><td>7E&#x27;2&#x27;2</td><td>991&#x27;EL</td><td>169&#x27;8Z</td><td>859&#x27;51</td><td>766&#x27;6</td><td>quen O/A</td><td>t+XHCH-BI-WTO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8L&#x27;LL&#x27;LI</td><td>090&#x27;AI</td><td>712&#x27;8E</td><td>16L&#x27;8I</td><td>7E&#x27;6I</td><td>61E&#x27;2I</td><td>795&#x27;02</td><td>762&#x27;EL</td><td>163&#x27;12</td><td>16E&#x27;6&#x27;1</td><td>788&#x27;8Z</td><td>172&#x27;5&#x27;1</td><td>109&#x27;EL</td><td>940&#x27;01</td><td>quen O/A</td><td>t+XHCH-BI-WTO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="101">ONS u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uu u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuUUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuucuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuccuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuccuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuddu:</td></tr></table>

sopwRnrsnsnnprrnA ZJ Jo sənXxDd: Ll 1qL

<table><tr><td>02S&#x27;2</td><td>829&#x27;2</td><td>+02&#x27;2</td><td>£81&#x27;2</td><td>089&#x27;2</td><td>79&#x27;2</td><td>+20&#x27;1</td><td>996&#x27;2</td><td>90&#x27;2</td><td>20L&#x27;2</td><td>988&#x27;2</td><td>189&#x27;2</td><td>quen O/A+XCHH-BI-WTO</td><td></td></tr><tr><td>81&#x27;S&#x27;2</td><td>219&#x27;2</td><td>+02&#x27;2</td><td>881&#x27;2</td><td>+89&#x27;2</td><td>85&#x27;2</td><td>+20&#x27;1</td><td>196&#x27;2</td><td>90&#x27;2</td><td>L69&#x27;2</td><td>088&#x27;2</td><td>6L&#x27;2</td><td>+XCHH-BI-WTO</td><td></td></tr><tr><td colspan="13">non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non- Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-ND</td><td></td></tr><tr><td colspan="4">non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-Non-ND</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>82S&#x27;2</td><td>£89&#x27;2</td><td>+12&#x27;2</td><td>$61&#x27;2</td><td>889&#x27;2</td><td>79&#x27;2</td><td>820&#x27;1</td><td>£L6&#x27;2</td><td>£L&#x27;2</td><td>11L&#x27;2</td><td>268&#x27;2</td><td>689&#x27;2</td><td>+XCHS-BI-WTO</td><td></td></tr><tr><td>82S&#x27;2</td><td>099&#x27;2</td><td>12&#x27;2</td><td>861&#x27;2</td><td>00L&#x27;2</td><td>6L&#x27;2</td><td>£L&#x27;2</td><td>086&#x27;2</td><td>£L&#x27;2</td><td>81L&#x27;2</td><td>L06&#x27;2</td><td>869&#x27;2</td><td>+XCHS-BI-WTO</td><td></td></tr><tr><td colspan="14">CHS u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uu u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuddu</td></tr><tr><td>91S&#x27;2</td><td>+19&#x27;2</td><td>10&#x27;2</td><td>LLI&#x27;2</td><td>089&#x27;2</td><td>95&#x27;2</td><td>220&#x27;1</td><td>+96&#x27;2</td><td>+08&#x27;2</td><td>10L&#x27;2</td><td>088&#x27;2</td><td>LLL&#x27;2</td><td>88&#x27;2</td><td>88&#x27;2</td></tr><tr><td>91S&#x27;2</td><td>L19&#x27;2</td><td>00&#x27;2</td><td>9L&#x27;2</td><td>6L&#x27;2</td><td>55&#x27;2</td><td>120&#x27;1</td><td>296&#x27;2</td><td>10L&#x27;2</td><td>L69&#x27;2</td><td>9L&#x27;2</td><td>5L&#x27;2</td><td>+XCHD-BI-WTO</td><td></td></tr><tr><td>+19&#x27;S&#x27;2</td><td>+19&#x27;2</td><td>81&#x27;2</td><td>20L&#x27;2</td><td>669&#x27;2</td><td>84&#x27;2</td><td>220&#x27;1</td><td>9L&#x27;2</td><td>120&#x27;2</td><td>21L&#x27;2</td><td>106&#x27;2</td><td>+69&#x27;2</td><td>+XCHD-BI-WTO</td><td></td></tr><tr><td>95&#x27;S&#x27;2</td><td>£89&#x27;2</td><td>54&#x27;2</td><td>112&#x27;2</td><td>35&#x27;2</td><td>66&#x27;2</td><td>+08&#x27;1</td><td>166&#x27;2</td><td>94&#x27;2</td><td>22L&#x27;2</td><td>L26&#x27;2</td><td>+1L&#x27;2</td><td>+XCHD-BI-WTO</td><td></td></tr><tr><td colspan="14">CHD u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUAUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUGUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUucUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCAUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUUUUCUUuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu ku uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uu uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu uu</td></tr></table>

[OpoWg]I Jos OssnepaA EJossoSOT:8I OqL

<table><tr><td>906&#x27;EI</td><td>6E8&#x27;EI</td><td>090&#x27;6</td><td>801&#x27;2</td><td>885&#x27;1</td><td>7L&#x27;11</td><td>S8L&#x27;2</td><td>S0F&#x27;61</td><td>ZE0&#x27;01</td><td>t06&#x27;1</td><td>9Z6&#x27;LI</td><td>E6S&#x27;1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td>t4L&#x27;8&#x27;EI</td><td>1Z9&#x27;EI</td><td>6S0&#x27;6</td><td>E2C&#x27;2</td><td>L&#x27;9&#x27;1</td><td>LL&#x27;9&#x27;1</td><td>L8&#x27;2</td><td>0ZE&#x27;61</td><td>8E0&#x27;01</td><td>0t8&#x27;1</td><td>0Z8&#x27;LI</td><td>tL&#x27;S&#x27;1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td colspan="14">non-hydrogenation</td></tr><tr><td>ZD&#x27;0+I</td><td>1Z0&#x27;1</td><td>E0Z&#x27;6</td><td>E2Z&#x27;2</td><td>L&#x27;08&#x27;1</td><td>898&#x27;1</td><td>908&#x27;2</td><td>E19&#x27;61</td><td>161&#x27;01</td><td>S60&#x27;S1</td><td>091&#x27;81</td><td>9SL&#x27;1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td>S1&#x27;1+I</td><td>SE1&#x27;1</td><td>0Z2&#x27;6</td><td>ZS&#x27;2</td><td>t&#x27;6&#x27;1</td><td>Z06&#x27;1</td><td>918&#x27;2</td><td>0S9&#x27;61</td><td>S1&#x27;201</td><td>0Z1&#x27;S1</td><td>tZ2&#x27;81</td><td>018&#x27;1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td colspan="14">non-hydrogenation</td></tr><tr><td>SZ&#x27;0+I</td><td>Z16&#x27;1</td><td>S1&#x27;6</td><td>L&#x27;0+2</td><td>669&#x27;1</td><td>9t8&#x27;11</td><td>96L&#x27;2</td><td>0SS&#x27;61</td><td>1Z1&#x27;01</td><td>6t0&#x27;S1</td><td>8Z0&#x27;81</td><td>L1L&#x27;1</td><td>HCH-SB-1WTO</td><td></td></tr><tr><td>ZS&#x27;1+I</td><td>OS1&#x27;1</td><td>t&#x27;12&#x27;6</td><td>8L&#x27;4&#x27;2</td><td>9L&#x27;8&#x27;1</td><td>t&#x27;6&#x27;11</td><td>L&#x27;08&#x27;2</td><td>689&#x27;61</td><td>0E2&#x27;01</td><td>0S1&#x27;S1</td><td>£6Z&#x27;81</td><td>tS8&#x27;1</td><td>HCH-SB-1WTO</td><td></td></tr><tr><td colspan="14">CHS u n 133</td></tr><tr><td>E8&#x27;8&#x27;EI</td><td>E89&#x27;EI</td><td>0E0&#x27;6</td><td>t&#x27;96&#x27;2</td><td>6LS&#x27;1</td><td>E99&#x27;11</td><td>6LL&#x27;2</td><td>99E&#x27;61</td><td>110&#x27;01</td><td>688&#x27;1</td><td>L08&#x27;LI</td><td>9tS&#x27;1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td>9Z8&#x27;EI</td><td>689&#x27;EI</td><td>8Z0&#x27;6</td><td>8t6&#x27;2</td><td>ELS&#x27;1</td><td>099&#x27;11</td><td>9LL&#x27;2</td><td>E4t&#x27;61</td><td>686&#x27;6</td><td>6Z8&#x27;1</td><td>E4L&#x27;LI</td><td>t1S&#x27;1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td>t11&#x27;1</td><td>E0&#x27;1</td><td>L81&#x27;6</td><td>685&#x27;2</td><td>0L8&#x27;1</td><td>S16&#x27;11</td><td>908&#x27;2</td><td>Z19&#x27;61</td><td>161&#x27;01</td><td>190&#x27;S1</td><td>061&#x27;81</td><td>t6L&#x27;1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td>81&#x27;1+I</td><td>Z9&#x27;1</td><td>9E9&#x27;6</td><td>018&#x27;2</td><td>S22&#x27;S1</td><td>tL&#x27;121</td><td>tL&#x27;8&#x27;2</td><td>606&#x27;61</td><td>Zt&#x27;01</td><td>09E&#x27;S1</td><td>£9&#x27;81</td><td>£60&#x27;S1</td><td>HCH-BI-1WTO</td><td></td></tr><tr><td colspan="14">CHD u n 133</td></tr><tr><td>618&#x27;EI</td><td>t89&#x27;EI</td><td>1Z0&#x27;6</td><td>8t6&#x27;2</td><td>L&#x27;85&#x27;1</td><td>0E9&#x27;11</td><td>6LL&#x27;2</td><td>90E&#x27;61</td><td>000&#x27;01</td><td>E18&#x27;1</td><td>6tL&#x27;LI</td><td>t6F&#x27;1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td>E0&#x27;0+I</td><td>L&#x27;16&#x27;EI</td><td>9t1&#x27;6</td><td>8t2&#x27;2</td><td>6E8&#x27;1</td><td>0E8&#x27;11</td><td>008&#x27;2</td><td>6L&#x27;61</td><td>9t&#x27;01</td><td>S6&#x27;1</td><td>966&#x27;LI</td><td>11L&#x27;1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td>t8&#x27;8&#x27;EI</td><td>t69&#x27;EI</td><td>190&#x27;6</td><td>906&#x27;2</td><td>809&#x27;1</td><td>Z99&#x27;11</td><td>£8L&#x27;2</td><td>E2E&#x27;61</td><td>8E0&#x27;01</td><td>L8&#x27;8&#x27;1</td><td>L18&#x27;LI</td><td>E2E&#x27;1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td>8Z&#x27;1+I</td><td>0s&#x27;S&#x27;1</td><td>1S&#x27;6</td><td>£10&#x27;S2</td><td>161&#x27;S1</td><td>L&#x27;91&#x27;21</td><td>t&#x27;8&#x27;2</td><td>606&#x27;61</td><td>£L&#x27;01</td><td>9S&#x27;S1</td><td>669&#x27;81</td><td>t90&#x27;S1</td><td>quenO/A</td><td>HCH-BI-1WTO</td></tr><tr><td colspan="14">non-hydrogenation</td></tr><tr><td colspan="14">O/Λ CHD u n 133</td></tr><tr><td>6Z&#x27;2+I</td><td>tE&#x27;E&#x27;1</td><td>61E&#x27;6</td><td>E0S&#x27;2</td><td>860&#x27;S1</td><td>9Z0&#x27;21</td><td>ZE8&#x27;2</td><td>989&#x27;61</td><td>S0E&#x27;01</td><td>912&#x27;S1</td><td>682&#x27;81</td><td>806&#x27;1</td><td>BI-NTO</td><td></td></tr><tr><td>S8&#x27;2</td><td>£0&#x27;-1&#x27;</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td><td>An d</td></tr></table>

sopwI rssneppIeA eA Jo sənnxod:61 1qL
