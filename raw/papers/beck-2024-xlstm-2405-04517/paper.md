Maximilian Beck* 1,2,3

Andreas Auer  $^{1,2}$

Günter Klambauer  $^{1,2,3}$

Korbinian Poppel* 1,2,3

Oleksandra Prudnikova

Johannes Brandstetter  $^{1,2,3}$

*Equal contribution

Markus Spanring

Michael Kopp

Sepp Hochreiter 1,2,3

<sup>1</sup>ELLIS Unit, LIT AI Lab, Institute for Machine Learning, JKU Linz, Austria

$^{2}$ NXAI Lab, Linz, Austria,  $^{3}$ NXAI GmbH, Linz, Austria

# Abstract

In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood the test of time and contributed to numerous deep learning success stories, in particular they constituted the first Large Language Models (LLMs). However, the advent of the Transformer technology with parallelizable self-attention at its core marked the dawn of a new era, outpacing LSTMs at scale. We now raise a simple question: How far do we get in language modeling when scaling LSTMs to billions of parameters, leveraging the latest techniques from modern LLMs, but mitigating known limitations of LSTMs? Firstly, we introduce exponential gating with appropriate normalization and stabilization techniques. Secondly, we modify the LSTM memory structure, obtaining: (i) sLSTM with a scalar memory, a scalar update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a matrix memory and a covariance update rule. Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures. Exponential gating and modified memory structures boost xLSTM capabilities to perform favorably when compared to state-of-the-art Transformers and State Space Models, both in performance and scaling.

Code available at: https://github.com/NX-AI/xlstm

Figure 1: The extended LSTM (xLSTM) family. From left to right: 1. The original LSTM memory cell with constant error carousel and gating. 2. New sLSTM and mLSTM memory cells that introduce exponential gating. sLSTM offers a new memory mixing technique. mLSTM is fully parallelizable with a novel matrix memory cell state and new covariance update rule. 3. mLSTM and sLSTM in residual blocks yield xLSTM blocks. 4. Stacked xLSTM blocks give an xLSTM architecture.

# 1 Introduction

The Long Short-Term Memory (LSTM) ideas (Hochreiter, 1991; Hochreiter & Schmidhuber, 1997b,a), i.e., the constant error carousel and gating, were introduced to overcome the vanishing gradient problem of recurrent neural networks (Hochreiter, 1991; Hochreiter et al., 2000):

$$
\left. c _ {t} \right. = \left. \mathrm {f} _ {t} \right. \left. c _ {t - 1} \right. + \left. \mathrm {i} _ {t} \right. \left. z _ {t} \right., \quad h _ {t} = \left. \mathrm {o} _ {t} \right. \psi \left(\left. c _ {t} \right.\right). \tag {1}
$$

The constant error carousel is the additive update of the cell state  $c_{t-1}$  (green) by cell inputs  $z_t$  and moderated by sigmoid gates (blue). The input gate  $\mathrm{i}_t$  and the forget gate  $\mathrm{f}_t$  control this update, while the output gate  $\mathrm{o}_t$  controls the output of the memory cell, i.e. the hidden state  $h_t$ . The cell state is normalized or squashed by  $\psi$  and then output gating gives the hidden state.

LSTMs have been successfully applied to various domains (Hochreiter et al., 2001, 2007; Schmidhuber, 2015), and prevailed over text generation until the dawn of Transformers in 2017 (Vaswani et al., 2017). The effectiveness of LSTMs has been demonstrated at numerous sequence-related tasks such as generating text (Graves, 2013; Karpathy, 2015), generating handwritings (Graves, 2013), sequence-to-sequence translation (Sutskever et al., 2014), evaluating computer programs (Zaremba & Sutskever, 2014), generating image captions (Karpathy & Fei-Fei, 2015; Hossain et al., 2019), generating source code (Karpathy, 2015), rainfall-runoff modeling (Kratzert et al., 2018, 2019), or hydrological models for flooding warnings (Nearing et al., 2024). In reinforcement learning, LSTMs are the best performing sequence models, e.g., the AlphaStar model for StarCraft II (Vinyals et al., 2017), the OpenAI Five model for Dota 2 (Karpathy, 2019), and models of the magnetic controller for nuclear fusion (Degrave et al., 2022). LSTMs excel at learning abstractions, i.e., adeptly extracting semantic information and storing it in their memory cells (Karpathy, 2015), which for example became evident by number and syntax neurons (Lakretz et al., 2019), linguistic neurons (Bau et al., 2019), and sentiment neurons (Radford et al., 2017). LSTMs are still used in highly relevant applications (Degrave et al., 2022; Nearing et al., 2024) and have stood the test of time.

Despite their tremendous successes, LSTMs have three main limitations: (i) Inability to revise storage decisions. We exemplify this limitation via the Nearest Neighbor Search problem (see also Appendix B): With a reference vector given, a sequence must be scanned sequentially for the most similar vector in order to provide its attached value at sequence end. The left panel of Figure 2 shows the mean squared error at this task. LSTM struggles to revise a stored value when a more similar vector is found, while our new xLSTM remediates this limitation by exponential gating. (ii) Limited storage capacities, i.e., information must be compressed into scalar cell states. We exemplify this limitation via Rare Token Prediction. In the right panel of Figure 2, the perplexity of token prediction on Wikitext-103 (Merit et al., 2017) is given for partitions of different token frequency.

Figure 2: LSTM limitations. Left: Nearest Neighbor Search problem in terms of mean squared error (MSE). Given a reference vector, a sequence is scanned sequentially for the most similar vector with the objective to return its attached value at sequence end. LSTM struggles to revise a stored value when a more similar vector is found. Our new xLSTM overcomes this limitation by exponential gating. Right: Rare Token Prediction. The perplexity (PPL) of token prediction on Wikitext-103, in partitions of token frequency. LSTM performs worse on predicting rare tokens because of its limited storage capacities, whereas our new xLSTM solves this problem via a matrix memory.

LSTM performs worse on rare tokens because of its limited storage capacities. Our new xLSTM solves this problem by a matrix memory. (iii) Lack of parallelizability due to memory mixing, i.e., the hidden-hidden connections between hidden states from one time step to the next, which enforce sequential processing.

These limitations of LSTM have paved the way for the emergence of Transformers (Vaswani et al., 2017) in language modeling. What performances can we achieve in language modeling when overcoming these limitations and scaling LSTMs to the size of current Large Language Models?

# 2 Extended Long Short-Term Memory

To overcome the LSTM limitations, Extended Long Short-Term Memory (xLSTM) introduces two main modifications to the LSTM idea of Equation (1). Those modifications — exponential gating and novel memory structures — enrich the LSTM family by two members: (i) the new sLSTM (see Section 2.2) with a scalar memory, a scalar update, and memory mixing, and (ii) the new mLSTM (see Section 2.3) with a matrix memory and a covariance (outer product) update rule, which is fully parallelizable. Both sLSTM and mLSTM enhance the LSTM through exponential gating. To enable parallelization, the mLSTM abandons memory mixing, i.e., the hidden-hidden recurrent connections. Both mLSTM and sLSTM can be extended to multiple memory cells, where sLSTM features memory mixing across cells. Further, the sLSTM can have multiple heads without memory mixing across the heads, but only memory mixing across cells within each head. This introduction of heads for sLSTM together with exponential gating establishes a new way of memory mixing. For mLSTM multiple heads and multiple cells are equivalent.

Integrating these new LSTM variants into residual block modules results in xLSTM blocks (see Section 2.4). Residually stacking those xLSTM blocks in architectures provides xLSTM architectures (see Section 2.4). See Figure 1 for the xLSTM architecture with its components.

# 2.1 Review of the Long Short-Term Memory

The original LSTM idea (Hochreiter, 1991; Hochreiter & Schmidhuber, 1997b,a) introduced the scalar memory cell as a central processing and storage unit that avoids vanishing gradients (Hochreiter, 1991; Hochreiter et al., 2000) through the constant error carousel (the cell state update). The memory cell contains three gates: input, output, and forget gate. The forget gate has been introduced by Gers et al. (2000). The update rules of the LSTM memory cell at time step  $t$  are:

$$
c _ {t} = \mathrm {f} _ {t} \quad c _ {t - 1} + \mathrm {i} _ {t} z _ {t} \quad \text {c e l l s t a t e} \tag {2}
$$

$$
h _ {t} = \left[ \begin{array}{l l} \mathrm {o} _ {t} & \tilde {h} _ {t} \end{array} \right], \quad \tilde {h} _ {t} = \psi \left(\boxed {c _ {t}}\right) \quad \text {h i d d e n s t a t e} \tag {3}
$$

$$
\left. z _ {t} \right. = \varphi \left(\tilde {z} _ {t}\right), \quad \tilde {z} _ {t} = \boldsymbol {w} _ {z} ^ {\top} \boldsymbol {x} _ {t} + r _ {z} h _ {t - 1} + b _ {z} \quad \text {c e l l i n p u t} \tag {4}
$$

$$
\mathrm {i} _ {t} = \sigma (\tilde {\mathrm {i}} _ {t}), \quad \tilde {\mathrm {i}} _ {t} = \boldsymbol {w} _ {\mathrm {i}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {i}} h _ {t - 1} + b _ {\mathrm {i}} \quad \text {i n p u t g a t e} \tag {5}
$$

$$
\mathrm {f} _ {t} = \sigma \left(\tilde {\mathrm {f}} _ {t}\right), \quad \tilde {\mathrm {f}} _ {t} = \boldsymbol {w} _ {\mathrm {f}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {f}} h _ {t - 1} + b _ {\mathrm {f}} \quad \text {f o r g e t g a t e} \tag {6}
$$

$$
\mathrm {o} _ {t} = \sigma (\tilde {\mathrm {o}} _ {t}), \quad \tilde {\mathrm {o}} _ {t} = \boldsymbol {w} _ {\mathrm {o}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {o}} h _ {t - 1} + b _ {\mathrm {o}} \quad \text {o u t p u t g a t e} \tag {7}
$$

The weight vectors  $\boldsymbol{w}_z, \boldsymbol{w}_i, \boldsymbol{w}_f$ , and  $\boldsymbol{w}_o$  correspond to the input weight vectors between inputs  $\boldsymbol{x}_t$  and cell input, input gate, forget gate, and output gate, respectively. The weights  $r_z, r_i, r_f,$  and  $r_o$  correspond to the recurrent weights between hidden state  $h_{t-1}$  and cell input, input gate, forget gate, and output gate, respectively.  $b_z, b_i, b_f,$  and  $b_o$  are the corresponding bias terms.  $\varphi$  and  $\psi$  are the cell input and hidden state activation functions (typically tanh).  $\psi$  is used to normalize or squash the cell state, which would be unbounded otherwise. All gate activation functions are sigmoid, i.e.,  $\sigma(x) = 1/(1 + \exp(-x))$ . In later formulations, multiple scalar memory cells  $\boldsymbol{c}_t \in \mathbb{R}^d$  were combined in a vector, which allows the usage of recurrent weight matrices  $\boldsymbol{R} \in \mathbb{R}^{d \times d}$  to mix the cell outputs of memory cells (Greff et al., 2015), for more details see Appendix A.1. Ablation studies showed that all components of the memory cell are crucial (Greff et al., 2015).

# 2.2 sLSTM

To empower LSTMs with the ability to revise storage decisions, we introduce exponential gates (red) together with normalization and stabilization. In particular, input and forget gates can have exponential activation functions. For normalization, we introduce a normalizer state that sums up the product of the input gate times all future forget gates.

The scalar sLSTM forward pass is:

$$
\boxed {c _ {t}} = \mathrm {f} _ {t} \boxed {c _ {t - 1}} + \mathrm {i} _ {t} \boxed {z _ {t}} \quad \text {c e l l s t a t e} \tag {8}
$$

$$
n _ {t} = \mathrm {f} _ {t} \quad n _ {t - 1} + \mathrm {i} _ {t} \quad \text {n o r m a l i z e r s t a t e} \tag {9}
$$

$$
h _ {t} = \mathrm {o} _ {t} \tilde {h} _ {t}, \quad \tilde {h} _ {t} = \left. c _ {t} / n _ {t} \right. \quad \text {h i d d e n s t a t e} \tag {10}
$$

$$
z _ {t} = \varphi (\tilde {z} _ {t}), \quad \tilde {z} _ {t} = \boldsymbol {w} _ {z} ^ {\top} \boldsymbol {x} _ {t} + r _ {z} h _ {t - 1} + b _ {z} \quad \text {c e l l i n p u t} \tag {11}
$$

$$
\mathrm {i} _ {t} = \exp (\tilde {\mathrm {i}} _ {t}), \quad \tilde {\mathrm {i}} _ {t} = \boldsymbol {w} _ {\mathrm {i}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {i}} h _ {t - 1} + b _ {\mathrm {i}} \quad \text {i n p u t g a t e} \tag {12}
$$

$$
\mathrm {f} _ {t} = \sigma \left(\tilde {\mathrm {f}} _ {t}\right) \text {O R} \exp \left(\tilde {\mathrm {f}} _ {t}\right), \quad \tilde {\mathrm {f}} _ {t} = \boldsymbol {w} _ {\mathrm {f}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {f}} h _ {t - 1} + b _ {\mathrm {f}} \quad \text {f o r g e t g a t e} \tag {13}
$$

$$
\mathrm {o} _ {t} = \sigma (\tilde {\mathrm {o}} _ {t}), \quad \tilde {\mathrm {o}} _ {t} = \boldsymbol {w} _ {\mathrm {o}} ^ {\top} \boldsymbol {x} _ {t} + r _ {\mathrm {o}} h _ {t - 1} + b _ {\mathrm {o}} \quad \text {o u t p u t g a t e} \tag {14}
$$

We transfer the original LSTM gating techniques, i.e., input- and/or hidden-dependent gating plus bias term, to the new architectures. Exponential activation functions can lead to large values that cause overflows. Therefore, we stabilize gates with an additional state  $m_{t}$  (Milakov & Gimelshein, 2018):

$$
\left. \overline {{m _ {t}}} = \max  \left(\log \left(\mathrm {f} _ {t}\right) + \overline {{m _ {t - 1}}}, \log \left(\mathrm {i} _ {t}\right)\right) \right. \quad \text {s t a b i l i z e r s t a t e} \tag {15}
$$

$$
\dot {\mathrm {i}} _ {t} ^ {\prime} = \exp (\log (\dot {\mathrm {i}} _ {t}) - m _ {t}) = \exp (\tilde {\mathrm {i}} _ {t} - m _ {t}) \quad \text {s t a b i l . i n p u t g a t e} \tag {16}
$$

$$
\left. \mathrm {f} _ {t} ^ {\prime} \right. = \exp \left(\log \left(\mathrm {f} _ {t}\right) + \left. m _ {t - 1} \right. - \left. m _ {t} \right.\right) \quad \text {s t a b i l . f o r g e t g a t e} \tag {17}
$$

We show in Appendix A.2, that replacing  $\mathrm{f}_t$  by  $\mathrm{f}_t'$  and  $\mathrm{i}_t$  by  $\mathrm{i}_t'$  in the forward pass does neither change the output of the whole network nor the derivatives of the loss with respect to the parameters.

New Memory Mixing. sLSTM can have multiple memory cells like the original LSTM (see Appendix A.2). Multiple memory cells enable memory mixing via recurrent connections  $R_{z}$ ,  $R_{i}$ ,  $R_{f}$ ,  $R_{o}$  from hidden state vector  $h$  to memory cell input  $z$  and the gates i, f, o, respectively. A new aspect in memory mixing is the effect of exponential gating. The new sLSTM can have multiple heads with memory mixing within each head but not across heads. The introduction of heads for sLSTM together with exponential gating establishes a new way of memory mixing.

# 2.3 mLSTM

To enhance storage capacities of LSTMs, we increase the LSTM memory cell from a scalar  $c \in \mathbb{R}$  to a matrix  $C \in \mathbb{R}^{d \times d}$ . Hence, retrieval is performed via a matrix multiplication. At time  $t$ , we want to store a pair of vectors, the key  $\pmb{k}_t \in \mathbb{R}^d$  and the value  $\pmb{v}_t \in \mathbb{R}^d$  (we use the Transformer terminology). Later at time  $t + \tau$ , the value  $\pmb{v}_t$  should be retrieved by a query vector  $\pmb{q}_{t + \tau} \in \mathbb{R}^d$ . This is the setting of Bidirectional Associative Memories (BAMs) (Kohonen, 1972; Anderson, 1972; Nakano, 1972; Anderson et al., 1977). The covariance update rule (Sejnowski, 1977; Dayan & Willshaw, 1991) for storing a key-value pair is

$$
\boldsymbol {C} _ {t} = \boldsymbol {C} _ {t - 1} + \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top}. \tag {18}
$$

We assume a layer-norm before projecting inputs to keys and values, therefore they have zero mean. The covariance update rule is optimal (Dayan & Willshaw, 1991) for a maximal separability of retrieved binary vectors, which is equivalent to a maximal signal/noise ratio. Higher separability is possible when limiting retrieval to pairwise interactions and conceding quadratic complexity like attention (Krotov & Hopfield, 2016, 2017; Ramsauer et al., 2021). The covariance update rule is equivalent to Fast Weight Programmers (Schmidhuber, 1992; Schlag et al., 2021), which have later been equipped with a constant decay rate multiplied to  $C_{t-1}$  and a constant learning rate multiplied to  $v_t k_t^\top$  (Ba et al., 2016a). In this spirit, we integrate the covariance update rule into the LSTM framework, where the forget gate corresponds to decay rate and the input gate to the learning rate, while the output gate scales the retrieved vector.

For this matrix memory, the normalizer state is the weighted sum of key vectors, where each key vector is weighted by the input gate and all future forget gates. Again, the normalizer state keeps

record of the strength of the gates. Since the dot product between query and normalizer state can be close to zero, we use the absolute value of this dot product and lower bound it by a threshold (typically 1.0) as done previously (Sun et al., 2023). The mLSTM forward pass is:

$$
\boldsymbol {C} _ {t} = \mathrm {f} _ {t} \quad \boldsymbol {C} _ {t - 1} + \mathrm {i} _ {t} \quad \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top}
$$

cell state (19)

$$
\boldsymbol {n} _ {t} = \mathrm {f} _ {t} \quad \boldsymbol {n} _ {t - 1} + \mathrm {i} _ {t} \quad \boldsymbol {k} _ {t}
$$

normalizer state (20)

$$
\boldsymbol {h} _ {t} = \left. \boldsymbol {0} _ {t} \odot \tilde {\boldsymbol {h}} _ {t}, \right. \quad \tilde {\boldsymbol {h}} _ {t} = \left. \boldsymbol {C} _ {t} \boldsymbol {q} _ {t} / \max  \left\{\left| \boldsymbol {n} _ {t} ^ {\top} \boldsymbol {q} _ {t} \right|, 1 \right\} \right.
$$

hidden state (21)

$$
\boldsymbol {q} _ {t} = \boldsymbol {W} _ {q} \boldsymbol {x} _ {t} + \boldsymbol {b} _ {q}
$$

query input (22)

$$
\boldsymbol {k} _ {t} = \frac {1}{\sqrt {d}} \boldsymbol {W} _ {k} \boldsymbol {x} _ {t} + \boldsymbol {b} _ {k}
$$

key input (23)

$$
\boldsymbol {v} _ {t} = \boldsymbol {W} _ {v} \boldsymbol {x} _ {t} + \boldsymbol {b} _ {v}
$$

value input (24)

$$
\mathrm {i} _ {t} = \exp (\tilde {\mathrm {i}} _ {t}),
$$

$$
\tilde {\mathrm {i}} _ {t} = \boldsymbol {w} _ {\mathrm {i}} ^ {\top} \boldsymbol {x} _ {t} + b _ {\mathrm {i}}
$$

input gate (25)

$$
\mathrm {f} _ {t} = \sigma (\tilde {\mathrm {f}} _ {t}) \operatorname {O R} \exp (\tilde {\mathrm {f}} _ {t}), \tilde {\mathrm {f}} _ {t} = \boldsymbol {w} _ {\mathrm {f}} ^ {\top} \boldsymbol {x} _ {t} + b _ {\mathrm {f}}
$$

forget gate (26)

$$
\mathbf {o} _ {t} = \sigma (\tilde {\mathbf {o}} _ {t}),
$$

$$
\tilde {\mathbf {o}} _ {t} = \boldsymbol {W} _ {\mathrm {o}} \boldsymbol {x} _ {t} + \boldsymbol {b} _ {\mathrm {o}}
$$

output gate (27)

mLSTM can have multiple memory cells like the original LSTM. For mLSTM, multiple heads and multiple cells are equivalent as there is no memory mixing. In order to stabilize the exponential gates of mLSTM, we use the same stabilization techniques as for sLSTM (see Equation 15). Since the mLSTM has no memory mixing, this recurrence can be reformulated in a parallel version. For more details we refer to Appendix A.3.

# 2.4 xLSTM Architecture

xLSTM Blocks. An xLSTM block should non-linearly summarize the past in a high-dimensional space to better separate different histories or contexts. Separating histories is the prerequisite to correctly predict the next sequence element such as the next token. We resort to Cover's Theorem (Cover, 1965), which states that in a higher dimensional space non-linearly embedded patterns can more likely be linearly separated than in the original space. We consider two residual block architectures: (i) A residual block with post upprojection (like Transformers), which non-linearly summarizes the past in the original space, then linearly maps into a high-dimensional space, applies a non-linear activation function, and linearly maps back to the original

space; see left panel of Figure 3 and third column in Figure 1. A more detailed version is depicted in Figure 10 in the appendix. (ii) A residual block with pre up-projection (like State Space Models), which linearly maps to a high-dimensional space, non-linearly summarizes the past in the high-dimensional space and then linearly maps back to the original space. For an xLSTM block containing an sLSTM, we use the post up-projection block. For an xLSTM block containing an mLSTM, we use the pre up-projection block since the memory capacity becomes larger in the high-dimensional space. We refer to the left panel of Figure 3 and third column in Figure 1, or Figure 10 in the appendix for more details.

Figure 3: xLSTM blocks. Left: A residual sLSTM block with post up-projection (like Transformers): The input is fed into an sLSTM — with an optional convolution — followed by a gated MLP. Right: A residual mLSTM block with pre up-projection (like State Space models): mLSTM is wrapped inside two MLPs, via a convolution, a learnable skip connection, and an output gate that acts component-wise. See Figure 10 and Figure 11 in the appendix for details.


xLSTM Architecture. An xLSTM architecture is constructed by residually stacking building blocks (Srivastava et al., 2015; He et al., 2016). We rely on the most commonly used preLayerNorm (Ba et al., 2016b) residual backbones as used in contemporary Large Language Models. See last column in Figure 1.

# 2.5 Memory and Speed Considerations

Contrary to Transformers, xLSTM networks have a linear computation and a constant memory complexity with respect to the sequence length. Since the xLSTM memory is compressive, it is well suited for industrial applications and implementations on the edge.

The memory of mLSTM does not require parameters but is computationally expensive through its  $d \times d$  matrix memory and  $d \times d$  update. We trade off memory capacity against computational complexity. Nevertheless, the computations can be done in parallel on GPUs, therefore these computations have only a minor effect on the wall clock time.

While mLSTM is parallelizable analog to FlashAttention (Dao et al., 2022; Dao, 2024) or GLA (Yang et al., 2023), sLSTM is not parallelizable due to the memory mixing (hidden-hidden connections). However, we developed a fast CUDA implementation with GPU memory optimizations to the register level which is typically less than two times slower than mLSTM.

# 3 Related Work

Linear Attention. Several methods have been suggested to overcome the quadratic complexity in terms of context length of the Transformer and make attention linear in the context length. The Synthesizer learns synthetic attention weights without token-token interactions (Tay et al., 2020). Linformer realizes self-attention by a low-rank matrix and even linearly approximates it (Wang et al., 2020). Linear Transformer linearizes the attention mechanism (Katharopoulos et al., 2020). Performer linearly approximates the attention softmax by positive orthogonal random features approach (Choromanski et al., 2021). Attention has been replaced by fast long convolutions in the Structured Global Convolution (SGConv) (Li et al., 2022) and the Hyena Hierarchy (Poli et al., 2023).

State Space Models. Recently, State Space Models (SSMs) became very popular since they are linear in the context length and show promising performance compared to Transformers. One of the first proposed models was Structured State Space sequence model (S4) (Gu et al., 2021), followed by Diagonal State Space (DSS) model (Gupta et al., 2022), Gated State Space (GSS) models (Mehta et al., 2022), S5 model (Smith et al., 2022), Bidirectional Gated SSM (BiGS) (Wang et al., 2022), H3 model (Fu et al., 2023), and Mamba (Gu & Dao, 2023).

Recurrent Neural Networks. Recurrent Neural Networks (RNNs) have been suggested to replace Transformer and attention due to their linearity in the context length. RNNs with Deep Linear Recurrent Units (LRUs) showed promising results for language modeling (Orvieto et al., 2023; De et al., 2024), as did hierarchically Gated Linear RNN (HGRN) (Qin et al., 2023) and HGRN2 (Qin et al., 2024). A well-known RNN approach to large language modeling is RWKV (Peng et al., 2023, 2024), showcasing competitive performance to Transformers.

Gating. One of the key ideas of LSTM is gating, which was rediscovered and reinterpreted in many recent approaches. Gating was used in HGRN (Qin et al., 2023), HGRN2 (Qin et al., 2024), Gated Linear Attention (GLA) (Yang et al., 2023), Gated State Space (GSS) models (Mehta et al., 2022), Bidirectional Gated SSM (BiGS) (Wang et al., 2022), Moving Average Equipped Gated Attention (MEGA) (Ma et al., 2022), RWKV (Peng et al., 2023), and Mamba (Gu & Dao, 2023).

Covariance Update Rule. To enhance storage capacities, we equipped the mLSTM cell with a matrix memory with a covariance update rule. Other methods which build on such an update mechanism are Fast Weight Programmers (Schmidhuber, 1992; Schlag et al., 2021), RWKV-5 and RWKV-6 (Peng et al., 2024), Retention (Sun et al., 2023), Linear Transformer (Katharopoulos et al., 2020), and HGRN2 (Qin et al., 2024).

Most Related. Conceptually the closest models to xLSTM are Retention (Sun et al., 2023), RWKV (Peng et al., 2023, 2024), and HGRN2 (Qin et al., 2024). These models share the concepts matrix memory and/or gating. However, in contrast to the new sLSTM, these approaches do not allow memory mixing. Memory mixing enables to solve state tracking problems, and therefore LSTMs are more expressive than State Space Models (SSMs) and Transformers (Merrill et al., 2024; Delétang et al., 2023). State tracking is required to evaluate code or to track entities in a long narrative.

Residually Stacking Architectures. Like almost all contemporary large deep learning models, xLSTM architectures are constructed by residually stacking building blocks (Srivastava et al., 2015; He et al., 2016). This construction enabled deep convolutional networks (He et al., 2016) and Transformers (Vaswani et al., 2017). Transformers are the ultimate force behind Large Language Models (LLMs) like GPT-3 (Brown et al., 2020), ChatGPT (Schulman et al., 2022), GPT-4 (Achiam et al., 2023), Megatron-LM (Shoeybi et al., 2019), Gopher (Rae et al., 2021), ERNIE 3.0 Titan (Wang et al., 2021), GLaM (Du et al., 2021), Chinese M6 (Lin et al., 2021), multilingual AlexaTM 20B (Soltan et al., 2022), OPT (Zhang et al., 2022), Chinchilla (Hoffmann et al., 2022), BLOOM (Scao et al., 2022), GLM-130B (Zeng et al., 2022), LaMDA (Thoppilan et al., 2022), PaLM (Chowdhery et al., 2022), Llama (Touvron et al., 2023), Gemini (Google, 2023; Reid et al., 2024).

# 4 Experiments

In this section, we experimentally evaluate xLSTM and compare it to existing methods with a focus on language modeling. We investigate xLSTM's specific capabilities on synthetic tasks in Section 4.1. In Section 4.2, we compare the validation set perplexity of various current language modeling methods that were trained on 15B tokens from SlimPajama (Soboleva et al., 2023). On the same dataset, we perform ablation studies for xLSTM. Then, we assess the scaling behavior of the different methods analogous to Kaplan et al. (2020) and Brown et al. (2020). In Section 4.3, we conduct a more thorough language modeling experiment. We compare xLSTM and the best performing methods from Section 4.2 after being trained on 300B tokens from SlimPajama (Soboleva et al., 2023). First, we assess how well the methods perform in extrapolating to longer contexts, secondly we test the methods via validation perplexity and performance on downstream tasks (Sutawika et al., 2024), thirdly we evaluate the methods on 571 text domains of the PALOMA language benchmark dataset (Magnusson et al., 2023), fourthly we again assess the scaling behavior of the different methods, but now with 20 times more training data.

For all experiments, we use the notation xLSTM[a:b] for the ratio  $a / b$  of mLSTM-based versus sLSTM-based xLSTM blocks. For example, xLSTM[7:1] means that out of eight blocks, seven are mLSTM-based blocks and one is an sLSTM-based block. For a common total block number of 48, this translates to 6 sLSTM-based blocks and 42 mLSTM-based blocks. Further, for all experiments, we use pre and post up-projection blocks for mLSTM and sLSTM, respectively.

# 4.1 Synthetic Tasks and Long Range Arena

First, we test the effectiveness of xLSTM's new exponential gating with memory mixing on formal languages (Delétang et al., 2023). Then, we assess the effectiveness of xLSTM's new matrix memory on the Multi-Query Associative Recall task (Arora et al., 2023). Finally, xLSTM's performance at processing long sequences in the Long Range Arena is evaluated (Tay et al., 2021).

Test of xLSTM's Exponential Gating with Memory Mixing. We test xLSTM's new exponential gating with memory mixing, which should enable it to solve state tracking problems (Merrill et al., 2024; Merrill & Sabharwal, 2023). We implement and extend the formal language tasks from Deletang et al. (2023) to enable multi-length training for length extrapolation. For a detailed description of all tasks and extended results see Appendix B.1.1. We compare xLSTM to other methods including Transformers, State Space Models, and Recurrent Neural Networks. The accuracy of the tested methods is evaluated on those tokens relevant to the task. The accuracy is scaled between 0 (random) and 1 (perfect). We compare 2-block architectures of the following methods on these tasks: xLSTM[0:1] (i.e., only sLSTM), xLSTM[1:0] (i.e., only mLSTM), xLSTM[1:1], Llama, Mamba, RWKV, Retention, Hyena, LSTM, and LSTM in Transformer blocks (LSTM (Block)). The results of this experiment are shown in Figure 4. Models such as Transformers or State Space Models without memory mixing (no state tracking) cannot solve, e.g. regular grammars like the parity task.

Figure 4: Test of xLSTM's exponential gating with memory mixing. Results are given by the scaled accuracy of different models at solving formal language tasks, of which some require state tracking. The different tasks are grouped by the Chomsky hierarchy.

This result is in agreement with findings that Transformers and State Space models are fundamentally less powerful than RNNs (Merrill et al., 2024; Merrill & Sabharwal, 2023; Delétang et al., 2023).

Test of xLSTM's Memory Capacities on Associative Recall Tasks. In this experiment, we test xLSTM's new matrix memory in terms of the memory capacity on the Multi-Query Associative Recall task (Arora et al., 2023): For each sequence, key-value pairs are randomly chosen from a large vocabulary, which must be memorized for later retrieval. To enhance the difficulty of the original task, we increase the number of key-value pairs up to 256 and extend the context length up to 2048. Thus, we have broader tests for the memory capacities of different models. We compare 2-block architectures of Llama, Mamba, RWKV-5, RWKV-6, xLSTM[1:1] and xLSTM[1:0]. The models are evaluated by the accuracy at recalling the pairs. Since Transformers (e.g. Llama) have a memory that is exponential in the coding dimension (Ramsauer et al., 2021), they constitute the gold standard at this task. Results are shown in Figure 5. xLSTM[1:1] performs best among all non-Transformer models, also for small models. Interestingly, the sLSTM block does not diminish the memory capacity but rather leverages it, which becomes evident at the most difficult task with 256 key-value pairs. Additional results are presented in Appendix B.1.2, where extrapolation analyses indicate that xLSTM's enhanced memory capacities also allow for extrapolating to contexts that are longer than those seen during training.

Figure 5: Test of memory capacities of different models at the Multi-Query Associative Recall task with context length 2048. Each panel is dedicated to a different number of key-value pairs. The  $x$ -axis displays the model size and the  $y$ -axis the validation accuracy.

Test of xLSTM's Long Context Capabilities on Long Range Arena. To assess xLSTM's performance on long sequences and large contexts, we compare different methods on the Long Range Arena (Tay et al., 2021). xLSTM demonstrates consistent strong performance on all of the tasks, suggesting that the xLSTM architecture is remarkably efficient in handling different aspects of long context problems. For more details, see Appendix B.1.3.

# 4.2 Method Comparison and Ablation Study

To address the main question of our paper, i.e. what can our new LSTM variants achieve when scaled up in language modelling, we train xLSTMs, Transformers, State Space Models, and other methods on 15B tokens from SlimPajama in the same auto-regressive setting. We compare the trained models on the validation set and perform ablation studies for the xLSTMs.

Comparing xLSTM to Other Methods. For comparison, we train models on 15B tokens from SlimPajama (Soboleva et al., 2023). The trained models are evaluated by their perplexity on the validation set. We compare the following methods: xLSTM (our new method), GPT-3 (Transformer) (Brown et al., 2020), Llama (Transformer) (Touvron et al., 2023), H3 (SSM) (Fu et al., 2023), Mamba (SSM) (Gu & Dao, 2023), RWKV-4 (RNN) (Peng et al., 2023), RWKV-5 (RNN) (Peng et al., 2024), RWKV-6 (RNN) (Peng et al., 2024), GLA (linear Transformer) (Yang et al., 2023), HGRN (RNN) (Qin et al., 2023), HGRN2 (RNN) (Qin et al., 2024). RetNet (linear Transformer) (Sun et al., 2023), Hyena (linear Transformer) (Poli et al., 2023), xLSTM[1:0], and xLSTM[7:1] (see Section 4). The models were trained with mixed precision, for RWKV-5, RWKV-6, GLA, HGRN2, the mixed-precision training did not utilize the PyTorch automated mixed precision (see also Appendix Section B.2). We categorize the methods into (a) Transformers, (b) State Space Models (SSMs), and (c) Recurrent Neural Networks (RNNs) together with linear Transformers. Linear Transformers are linear methods that substitute the Transformer attention mechanism. The models match a GPT-3 model with 350M parameters in size, i.e. embedding dim 1024 and 24 residual blocks. Only GPT-3 uses shared weights for token and output embeddings, therefore has fewer parameters. The results in Table 1 show that xLSTM outperforms all existing methods in validation perplexity. For details see Appendix B.2. Figure 6 shows the scaling behaviour for this experiment, indicating that xLSTM will also perform favorably for larger models.

<table><tr><td>Model</td><td>#Params M</td><td>SlimPajama (15B) ppl ↓</td></tr><tr><td>GPT-3</td><td>356</td><td>14.26</td></tr><tr><td>Llama</td><td>407</td><td>14.25</td></tr><tr><td>H3</td><td>420</td><td>18.23</td></tr><tr><td>Mamba</td><td>423</td><td>13.70</td></tr><tr><td>Hyena</td><td>435</td><td>17.59</td></tr><tr><td>RWKV-4</td><td>430</td><td>15.62</td></tr><tr><td>RWKV-5</td><td>456</td><td>14.25</td></tr><tr><td>RWKV-6</td><td>442</td><td>15.03</td></tr><tr><td>RetNet</td><td>431</td><td>16.23</td></tr><tr><td>HGRN</td><td>411</td><td>17.59</td></tr><tr><td>GLA</td><td>412</td><td>16.15</td></tr><tr><td>HGRN2</td><td>411</td><td>14.32</td></tr><tr><td>xLSTM[1:0]</td><td>409</td><td>13.43</td></tr><tr><td>xLSTM[7:1]</td><td>408</td><td>13.48</td></tr></table>

Table 1: Method comparison on next token prediction when trained on 15B tokens from SlimPajama. Best validation perplexities within model classes, i.e., Transformers, LSTMs, SSMs, RNNs, and linear Transformers are underlined and overall best is in bold. For each model class, the best performing methods are later used in Section 4.3 for LLM training. xLSTMs with new memory (xLSTM[1:0] and xLSTM[7:1]) perform best.

Ablation Studies. Table 1 and Figure 6 demonstrate that xLSTM achieves excellent results at language modeling when being trained on 15B tokens from SlimPajama. To ablate the changes from LSTM to xLSTM, we morph a vanilla LSTM architecture step-by-step into an xLSTM architecture. Firstly, we integrate LSTM layers into pre-LayerNorm residual backbones. Secondly, we extend this to a post up-projection block. Finally, we add exponential gating and matrix memory. The results are shown in Table 2 (top). The ablation studies attribute the strong performance improvement to both the exponential gating and the matrix memory. Additionally, due to the importance of gating in RNNs and State Space Models, we ablate different gating mechanisms. In Table 2 (bottom), we conclude that having each gate learnable and influenced by the input has an incremental positive effect. Additional studies on the individual backbone components are discussed in Appendix B.2.

Figure 6: Method comparison on next token prediction when trained on 15B tokens from SlimPajama. Performance measure in validation perplexity for the best methods of each model class (see Table 1) are reported. The performance degradation of xLSTM[7:1] at 2.7B is due to initially slower training convergence that leads to an especially undertrained model. xLSTM is the best method at all sizes.

Ablation studies on the new xLSTM components.  

<table><tr><td>Model</td><td>Modification</td><td>Exponential Gating</td><td>Matrix Memory</td><td>#Params M</td><td>SlimPajama (15B) ppl ↓</td></tr><tr><td rowspan="3">LSTM</td><td>Vanilla Multi-Layer LSTM</td><td>X</td><td>X</td><td>607.8</td><td>2417.86</td></tr><tr><td>Adding Resnet Backbone</td><td>X</td><td>X</td><td>506.1</td><td>35.46</td></tr><tr><td>Adding Up-Projection Backbone</td><td>X</td><td>X</td><td>505.9</td><td>26.01</td></tr><tr><td>xLSTM[0:1]</td><td>Adding Exponential Gating</td><td>✓</td><td>X</td><td>427.3</td><td>17.70</td></tr><tr><td>xLSTM[7:1]</td><td>Adding Matrix Memory</td><td>✓</td><td>✓</td><td>408.4</td><td>13.48</td></tr></table>

Ablation studies on different gating techniques.  

<table><tr><td rowspan="2">Learnable Gates</td><td colspan="3">Forget Gate</td><td colspan="3">Input Gate</td><td rowspan="2">SlimPajama (15B) ppl ↓</td></tr><tr><td>Input Dependent</td><td>Learnable Bias</td><td>Bias Init</td><td>Input Dependent</td><td>Learnable Bias</td><td>Bias Init</td></tr><tr><td>No Gates</td><td>X</td><td>X</td><td>+∞</td><td>X</td><td>X</td><td>0</td><td>NaN</td></tr><tr><td>No Gates</td><td>X</td><td>X</td><td>[3, 6]</td><td>X</td><td>X</td><td>0</td><td>13.95</td></tr><tr><td>Forget Gate</td><td>✓</td><td>✓</td><td>[3, 6]</td><td>X</td><td>X</td><td>0</td><td>13.58</td></tr><tr><td>Input Gate</td><td>X</td><td>X</td><td>[3, 6]</td><td>✓</td><td>✓</td><td>N(0, 0.1)</td><td>13.69</td></tr><tr><td>Forget Gate Bias</td><td>X</td><td>✓</td><td>[3, 6]</td><td>X</td><td>X</td><td>0</td><td>13.76</td></tr><tr><td>Forget + Input Gate Bias</td><td>X</td><td>✓</td><td>[3, 6]</td><td>X</td><td>✓</td><td>N(0, 0.1)</td><td>13.73</td></tr><tr><td>Forget Gate + Input Gate Bias</td><td>✓</td><td>✓</td><td>[3, 6]</td><td>X</td><td>✓</td><td>N(0, 0.1)</td><td>13.55</td></tr><tr><td>Forget Gate + Input Gate</td><td>✓</td><td>✓</td><td>[3, 6]</td><td>✓</td><td>✓</td><td>N(0, 0.1)</td><td>13.43</td></tr></table>

Table 2: Ablation studies. Top: Ablation studies on the new xLSTM components, contributing the strong performance improvement of xLSTM over vanilla LSTM to both the exponential gating and the matrix memory. Bottom: Ablation studies on different gating techniques. We consider an xLSTM[1:0] with sigmoid forget gate and exponential input gate. Bias initialization  $\infty$  means that the forget gate is set to one, [3, 6] indicates that values are taken equidistant in the respective interval, and  $\mathcal{N}(0,0.1)$  that values are randomly chosen from a Gaussian with mean 0 and std 0.1. PPL denotes validation perplexity. The first two lines correspond to models similar to linearized attention, line four to Retention, line five to RWKV-5, and line six to RWKV-6. Dependencies of the gates on the input lead to better performance.

# 4.3 xLSTM as Large Language Model

We culminate this study in large-scale language modeling experiments, testing the potential of xLSTM as an LLM. We therefore increase the amount of training data and train on 300B tokens from SlimPajama. The same number of tokens is used in, e.g., Mamba (Gu & Dao, 2023) and Griffin (De et al., 2024). We compare xLSTM to RWKV-4, Llama, and Mamba - one method from each respective method class in Section 4.2. We select RWKV-4 as RNN representative since for RWKV-5, RWKV-6 and HGRN2 a reasonable training precision setting has been found only after the training start of the 300B token experiments (see Appendix B.2). We train different model sizes (125M, 350M, 760M, 1.3B), test all models for length extrapolation capabilities and evaluate their performance on the validation set. We assess their performance on downstream tasks, test their performance in language modeling on 571 text domains of the PALOMA benchmark, and, finally, investigate their scaling law behavior.

Sequence Length Extrapolation. Firstly, we test the sequence length extrapolation for 1.3B-sized, large models of xLSTM, RWKV-4, Llama, and Mamba. All models are trained on context length 2048, and then tested for context lengths up to 16384. See Figure 7 for the results. In contrast to other methods, xLSTM models maintain low perplexities for longer contexts.

Figure 7: Sequence extrapolation in language modeling. This is a comparison of 1.3B-sized, large models of xLSTM, RWKV-4, Llama, and Mamba at next token prediction on the SlimPajama validation set after training on 300B tokens from SlimPajama. Models are trained with context length 2048 and then tested for context lengths up to 16384. Left: Token perplexities evaluated at different context lengths. In contrast to other methods, xLSTM models remain at low perplexities for longer contexts. Right: Prediction quality when extrapolating to long context sizes in terms of validation perplexity (PPL). xLSTM yields the best PPL values (best in bold, second best underlined).

Validation Perplexity and Downstream Tasks. Secondly, for all model sizes, we evaluate the performance of xLSTM, RWKV-4, Llama, and Mamba models on the SlimPajama validation set for next token prediction and on downstream tasks that measure common sense reasoning. The third column of Table 3 lists the validation set perplexities of different methods. Both xLSTM[1:0] and xLSTM[7:1] are the best models for all model sizes with respect to the validation set perplexity. The other columns of Table 3 provide the performance on downstream tasks. In the vast majority of tasks and across all model sizes xLSTM is the best method — only on the ARC task Mamba is in some cases the best method. For details see Appendix B.3.

Performance on PALOMA Language Tasks. Thirdly, for all model sizes, we test the next token prediction performance of xLSTM, RWKV-4, Llama, and Mamba models on PALOMA language tasks (Magnusson et al., 2023). We measure the performance by the perplexity for next token prediction on 571 text domains, which range from nytimes.com to r/depression on Reddit. Table 4 shows token prediction perplexity grouped into language modeling (first seven columns) and fine-grained domain benchmarks (last 5 columns). xLSTM[1:0] performs better than xLSTM[7:1] on these language tasks. xLSTM[1:0] has in 568 out of 571 (99.5%) text domains a lower perplexity

<table><tr><td></td><td>Model</td><td>#Params M</td><td>SlimPajama (300B) ppl ↓</td><td>LAMBADA ppl ↓</td><td>LAMBADA acc ↑</td><td>HellaSwag acc ↑</td><td>PIQA acc ↑</td><td>ARC-E acc ↑</td><td>ARC-C acc ↑</td><td>WinoGrande acc ↑</td><td>Average acc ↑</td></tr><tr><td rowspan="5">125M</td><td>RWKV-4</td><td>169.4</td><td>16.66</td><td>54.72</td><td>23.77</td><td>34.03</td><td>66.00</td><td>47.94</td><td>24.06</td><td>50.91</td><td>41.12</td></tr><tr><td>Llama</td><td>162.2</td><td>15.89</td><td>39.21</td><td>31.54</td><td>34.09</td><td>65.45</td><td>45.33</td><td>23.63</td><td>50.67</td><td>41.78</td></tr><tr><td>Mamba</td><td>167.8</td><td>15.08</td><td>27.76</td><td>34.14</td><td>36.47</td><td>66.76</td><td>48.86</td><td>24.40</td><td>51.14</td><td>43.63</td></tr><tr><td>xLSTM[1:0]</td><td>163.8</td><td>14.63</td><td>25.98</td><td>36.52</td><td>36.74</td><td>65.61</td><td>47.81</td><td>24.83</td><td>51.85</td><td>43.89</td></tr><tr><td>xLSTM[7:1]</td><td>163.7</td><td>14.60</td><td>26.59</td><td>36.08</td><td>36.75</td><td>66.87</td><td>48.32</td><td>25.26</td><td>51.70</td><td>44.16</td></tr><tr><td rowspan="5">350M</td><td>RWKV-4</td><td>430.5</td><td>12.62</td><td>21.57</td><td>36.62</td><td>42.47</td><td>69.42</td><td>54.46</td><td>25.43</td><td>51.22</td><td>46.60</td></tr><tr><td>Llama</td><td>406.6</td><td>12.19</td><td>15.73</td><td>44.19</td><td>44.45</td><td>69.15</td><td>52.23</td><td>26.28</td><td>53.59</td><td>48.32</td></tr><tr><td>Mamba</td><td>423.1</td><td>11.64</td><td>12.83</td><td>46.24</td><td>47.55</td><td>69.70</td><td>55.47</td><td>27.56</td><td>54.30</td><td>50.14</td></tr><tr><td>xLSTM[1:0]</td><td>409.3</td><td>11.31</td><td>11.49</td><td>49.33</td><td>48.06</td><td>69.59</td><td>55.72</td><td>26.62</td><td>54.38</td><td>50.62</td></tr><tr><td>xLSTM[7:1]</td><td>408.4</td><td>11.37</td><td>12.11</td><td>47.74</td><td>47.89</td><td>71.16</td><td>56.61</td><td>27.82</td><td>53.28</td><td>50.75</td></tr><tr><td rowspan="5">760M</td><td>RWKV-4</td><td>891.0</td><td>10.55</td><td>10.98</td><td>47.43</td><td>52.29</td><td>72.69</td><td>58.84</td><td>28.84</td><td>55.41</td><td>52.58</td></tr><tr><td>Llama</td><td>834.1</td><td>10.60</td><td>9.90</td><td>51.41</td><td>52.16</td><td>70.95</td><td>56.48</td><td>28.75</td><td>56.67</td><td>52.74</td></tr><tr><td>Mamba</td><td>870.5</td><td>10.24</td><td>9.24</td><td>50.84</td><td>53.97</td><td>71.16</td><td>60.44</td><td>29.78</td><td>56.99</td><td>53.86</td></tr><tr><td>xLSTM[1:0]</td><td>840.4</td><td>9.86</td><td>8.09</td><td>54.78</td><td>55.72</td><td>72.69</td><td>62.75</td><td>32.59</td><td>58.17</td><td>56.12</td></tr><tr><td>xLSTM[7:1]</td><td>839.7</td><td>9.91</td><td>8.07</td><td>55.27</td><td>56.12</td><td>72.74</td><td>61.36</td><td>29.61</td><td>56.43</td><td>55.26</td></tr><tr><td rowspan="5">1.3B</td><td>RWKV-4</td><td>1515.2</td><td>9.83</td><td>9.84</td><td>49.78</td><td>56.20</td><td>74.70</td><td>61.83</td><td>30.63</td><td>55.56</td><td>54.78</td></tr><tr><td>Llama</td><td>1420.4</td><td>9.44</td><td>7.23</td><td>57.44</td><td>57.81</td><td>73.12</td><td>62.79</td><td>31.74</td><td>59.04</td><td>56.99</td></tr><tr><td>Mamba</td><td>1475.3</td><td>9.14</td><td>7.41</td><td>55.64</td><td>60.45</td><td>74.43</td><td>66.12</td><td>33.70</td><td>60.14</td><td>58.41</td></tr><tr><td>xLSTM[1:0]</td><td>1422.6</td><td>8.89</td><td>6.86</td><td>57.83</td><td>60.91</td><td>74.59</td><td>64.31</td><td>32.59</td><td>60.62</td><td>58.48</td></tr><tr><td>xLSTM[7:1]</td><td>1420.1</td><td>9.00</td><td>7.04</td><td>56.69</td><td>60.26</td><td>74.92</td><td>65.11</td><td>32.34</td><td>59.27</td><td>58.10</td></tr></table>

Table 3: Validation set perplexity and downstream tasks. Comparison of xLSTM, RWKV-4, Llama, and Mamba on the validation set at next token prediction and on downstream tasks after training on 300B tokens from SlimPajama. Model sizes are 125M, 250M, 760M, and 1.3B. The first column shows the methods and the second the actual number of parameters. The third column lists the validation set perplexities, while the remaining columns show the performance on downstream tasks. Best model per model size is depicted bold and the second best is underlined. In the vast majority of tasks and across all model sizes xLSTM is the best method — only on the ARC task Mamba is in some cases the best method. xLSTM[1:0] and xLSTM[7:1] are the two best models with respect to validation set perplexity.

<table><tr><td></td><td>Model</td><td>#Params M</td><td>C4</td><td>MC4 EN</td><td>Wikitext 103</td><td>Penn Treebank</td><td>Red Pajama</td><td>Refined Web</td><td>Dolma</td><td>M2D2 S2ORC</td><td>M2D2 Wikipedia</td><td>C4 Domains</td><td>Dolma Subreddits</td><td>Dolma Coding</td><td>Average</td></tr><tr><td rowspan="5">125M</td><td>RWKV-4</td><td>169.4</td><td>26.25</td><td>22.33</td><td>29.18</td><td>38.45</td><td>8.99</td><td>32.47</td><td>17.04</td><td>23.86</td><td>21.42</td><td>22.68</td><td>37.08</td><td>5.12</td><td>23.74</td></tr><tr><td>Llama</td><td>162.2</td><td>24.64</td><td>17.23</td><td>23.16</td><td>31.56</td><td>8.26</td><td>29.15</td><td>15.10</td><td>19.71</td><td>20.41</td><td>21.45</td><td>36.73</td><td>3.61</td><td>20.92</td></tr><tr><td>Mamba</td><td>167.8</td><td>23.12</td><td>17.04</td><td>22.49</td><td>30.63</td><td>7.96</td><td>27.73</td><td>14.60</td><td>19.38</td><td>19.36</td><td>20.14</td><td>34.32</td><td>3.77</td><td>20.05</td></tr><tr><td>xLSTM[1:0]</td><td>163.8</td><td>22.54</td><td>16.32</td><td>21.98</td><td>30.47</td><td>7.80</td><td>27.21</td><td>14.35</td><td>19.02</td><td>19.04</td><td>19.65</td><td>34.15</td><td>3.64</td><td>19.68</td></tr><tr><td>xLSTM[7:1]</td><td>163.7</td><td>22.39</td><td>16.13</td><td>21.47</td><td>30.01</td><td>7.75</td><td>26.91</td><td>14.13</td><td>18.6</td><td>18.84</td><td>19.52</td><td>33.9</td><td>3.59</td><td>19.44</td></tr><tr><td rowspan="5">350M</td><td>RWKV-4</td><td>430.5</td><td>19.55</td><td>15.82</td><td>19.64</td><td>27.58</td><td>6.97</td><td>24.28</td><td>12.94</td><td>17.59</td><td>15.96</td><td>16.98</td><td>29.40</td><td>3.90</td><td>17.55</td></tr><tr><td>Llama</td><td>406.6</td><td>18.38</td><td>13.28</td><td>16.41</td><td>21.82</td><td>6.56</td><td>22.09</td><td>11.76</td><td>15.05</td><td>15.25</td><td>15.99</td><td>28.30</td><td>3.12</td><td>15.67</td></tr><tr><td>Mamba</td><td>423.1</td><td>17.33</td><td>13.05</td><td>16.11</td><td>22.24</td><td>6.34</td><td>21.04</td><td>11.42</td><td>14.83</td><td>14.53</td><td>15.16</td><td>27.02</td><td>3.20</td><td>15.19</td></tr><tr><td>xLSTM[1:0]</td><td>409.3</td><td>17.01</td><td>12.55</td><td>15.17</td><td>22.51</td><td>6.20</td><td>20.66</td><td>11.16</td><td>14.44</td><td>14.27</td><td>14.85</td><td>26.70</td><td>3.08</td><td>14.88</td></tr><tr><td>xLSTM[7:1]</td><td>408.4</td><td>16.98</td><td>12.68</td><td>15.43</td><td>21.86</td><td>6.23</td><td>20.70</td><td>11.22</td><td>14.62</td><td>14.30</td><td>14.85</td><td>26.61</td><td>3.11</td><td>14.88</td></tr><tr><td rowspan="5">760M</td><td>RWKV-4</td><td>891.0</td><td>15.51</td><td>12.76</td><td>14.84</td><td>21.39</td><td>5.91</td><td>19.28</td><td>10.70</td><td>14.27</td><td>13.04</td><td>13.68</td><td>24.22</td><td>3.32</td><td>14.08</td></tr><tr><td>Llama</td><td>834.1</td><td>15.75</td><td>11.59</td><td>13.47</td><td>18.33</td><td>5.82</td><td>19.04</td><td>10.33</td><td>13.00</td><td>13.05</td><td>13.76</td><td>24.80</td><td>2.90</td><td>13.49</td></tr><tr><td>Mamba</td><td>870.5</td><td>15.08</td><td>11.54</td><td>13.47</td><td>19.34</td><td>5.69</td><td>18.43</td><td>10.15</td><td>13.05</td><td>12.62</td><td>13.25</td><td>23.94</td><td>2.99</td><td>13.30</td></tr><tr><td>xLSTM[1:0]</td><td>840.4</td><td>14.60</td><td>11.03</td><td>12.61</td><td>17.74</td><td>5.52</td><td>17.87</td><td>9.85</td><td>12.50</td><td>12.20</td><td>12.81</td><td>23.46</td><td>2.87</td><td>12.76</td></tr><tr><td>xLSTM[7:1]</td><td>839.7</td><td>14.72</td><td>11.11</td><td>12.68</td><td>17.61</td><td>5.55</td><td>18.01</td><td>9.87</td><td>12.59</td><td>12.25</td><td>12.89</td><td>23.43</td><td>2.88</td><td>12.80</td></tr><tr><td rowspan="5">1.3B</td><td>RWKV-4</td><td>1515.2</td><td>14.51</td><td>12.04</td><td>13.73</td><td>19.37</td><td>5.62</td><td>18.25</td><td>10.11</td><td>13.46</td><td>12.10</td><td>12.87</td><td>22.85</td><td>3.25</td><td>13.18</td></tr><tr><td>Llama</td><td>1420.4</td><td>13.93</td><td>10.44</td><td>11.74</td><td>15.92</td><td>5.29</td><td>17.03</td><td>9.35</td><td>11.61</td><td>11.53</td><td>12.24</td><td>22.63</td><td>2.74</td><td>12.04</td></tr><tr><td>Mamba</td><td>1475.3</td><td>13.35</td><td>10.40</td><td>11.76</td><td>16.65</td><td>5.21</td><td>16.50</td><td>9.17</td><td>11.73</td><td>11.18</td><td>11.83</td><td>21.43</td><td>2.83</td><td>11.84</td></tr><tr><td>xLSTM[1:0]</td><td>1422.6</td><td>13.13</td><td>10.09</td><td>11.41</td><td>15.92</td><td>5.10</td><td>16.25</td><td>9.01</td><td>11.43</td><td>10.95</td><td>11.60</td><td>21.29</td><td>2.73</td><td>11.58</td></tr><tr><td>xLSTM[7:1]</td><td>1420.1</td><td>13.31</td><td>10.21</td><td>11.32</td><td>16.00</td><td>5.16</td><td>16.48</td><td>9.11</td><td>11.61</td><td>11.10</td><td>11.76</td><td>21.50</td><td>2.75</td><td>11.69</td></tr></table>

Table 4: Performance on PALOMA Language Modeling Tasks. Comparison of xLSTM, RWKV-4, Llama, and Mamba by the perplexity of next token prediction on the PALOMA language benchmark after training on 300B tokens from SlimPajama. Model sizes are 125M, 250M, 760M, and 1.3B. The second column shows the actual number of parameters. The 571 text domains are grouped into language modeling (next seven columns) and fine-grained domain benchmarks (further 5 columns). The last column shows the average perplexity across all of these tasks. Best model per model size is given in bold and the second best is underlined. xLSTM yields the best performance.

than Mamba, in 486 out of 571 (85.1%) a lower perplexity than Llama, in 570 out of 571 (99.8%) a lower perplexity than RWKV-4. For details see Appendix B.3.

Scaling Laws. Fourthly, we assess the power-law scaling behavior, which allows to extrapolate the performance to larger model sizes (Kaplan et al., 2020; Brown et al., 2020). Figure 8 presents the scaling behavior. All models share a similar scaling behavior but with different offsets. RWKV-4 performs worst, followed by Llama and Mamba. xLSTM is better than Mamba with a similar margin to Mamba as Mamba has to Llama. The scaling behavior indicates that for larger models xLSTM will continue to perform favourable compared to Transformers and State-Space models.

Figure 8: Scaling laws. Next token prediction perplexity of xLSTM, RWKV-4, Llama, and Mamba on the SlimPajama validation set when trained on 300B tokens from SlimPajama. Model sizes are 125M, 350M, 760M, and 1.3B. Best models for each model class, see Table 1, were selected. The scaling laws indicate that for larger models xLSTM will perform well too.

Generation Times and Maximal Throughput. Finally, we measure the text generation time in Figure 9 and the maximal throughput in Figure 9 (left) for our xLSTM variants at 1.3B scale. We compare against similar sized Mamba, Llama and RWKV implementations from HuggingFace, including a static key-value cache for the Llama model. At the time of the experiments, both full cache compilation of the Transformer Model and compilation of the Mamba model with torch.com did not work. For the text generation experiments all of the models are tested at batch size 1 and pre-fill 16. This pre-fill should be maximally favorable for the Transformer. Figure 9 shows the linear scaling of the xLSTM and the other recurrent models Mamba and RWKV-4 compared to the quadratic scaling of Llama. For the decoding throughput we measure different batch sizes and prefetch for the Llama model. Figure 9 (right) shows that xLSTM can use much higher batch sizes than Llama due to its constant memory and thus achieves the highest throughput.

Figure 9: Inference Generative Speed. Left: Generation times of different 1.3B models for a pre-fill context of 16 tokens (to mitigate cache initialization). The recurrent models (xLSTM[1:0], xLSTM[7:1], Mamba and RWKV-4) show linear behavior, whereas the Transformer (Llama) inference/decoding time is quadratic in sequence length. Right: Token throughput for different batch sizes on a A100-80GB GPU for 1.3B sized models. Note that the Transformer / Llama model goes out of memory (OOM) already for small batch sizes, whereas xLSTM and Mamba can sustain very large batch sizes. xLSTM[1:0] consistently outperforms Mamba in throughput.


# 5 Limitations

(i) In contrast to mLSTM, memory mixing of the sLSTM prohibits parallelizable operations, and therefore does not allow a fast parallel implementation. Nevertheless, we developed a fast CUDA kernel for sLSTM, which is currently less than two times slower than our parallel mLSTM implementation. (ii) The CUDA kernels for mLSTM are not optimized, and therefore the current implementation is about 4 times slower than FlashAttention or the scan used in Mamba. Faster CUDA kernels could be obtained in the vein of FlashAttention. (iii) The matrix memory of mLSTM has high computation complexity since  $d \times d$  matrices must be processed. Still, the memory update and retrieval does not use parameters and can be parallelized using standard matrix operations, therefore the wall clock time overhead due to the complex memory is minor. (iv) The initialization of the forget gates must be chosen carefully. (v) Since the matrix memory is independent of the sequence length, increasing the sequence length might overload the memory for longer context sizes. Still, this does not appear to be a limitation for contexts up to 16k, see Section 4.3. (vi) Due to the expensive computational load for large language experiments, we did neither fully optimize the architecture nor the hyperparameters, especially for larger xLSTM architectures. We anticipate that an extensive optimization process is needed for xLSTM to reach its full potential.

# 6 Conclusion

We have partly answered our simple question: How far do we get in language modeling when scaling LSTM to billions of parameters? So far, we can answer: "At least as far as current technologies like Transformers or State Space Models". We have enhanced LSTM to xLSTM by exponential gating with memory mixing and a new memory structure. xLSTM models perform favorably on language modeling when compared to state-of-the-art methods like Transformers and State Space Models. The scaling laws indicate that larger xLSTM models will be serious competitors to current Large Language Models that are built with the Transformer technology. xLSTM has the potential to considerably impact other fields like Reinforcement Learning, Time Series Prediction, or the modeling of physical systems.

# Acknowledgements

We thank Sebastian Lehner, Daniel Klotz, Thomas Adler, Matthias Dellago, Gerald Gutenbrunner, Fabian Paischer, Vihang Patil, Niklas Schmidinger, Benedikt Alkin, Kajetan Schweighofer, Anna Zimmel, Lukas Aichberger, Lukas Hauzenberger, Bernhard Schäfl and Johannes Lehner for helpful discussions and feedback.

# References

J. Achiam, S. Adler, S. Agarwal, et al. GPT-4 technical report. ArXiv, 2303.08774, 2023.  
J. Anderson, J. Silverstein, S. Ritz, and R. Jones. Distinctive features, categorical perception, and probability learning: Some applications of a neural model. *Psychological Review*, 84:413-451, 1977. doi: 10.1037/0033-295X.84.5.413.  
J. A. Anderson. A simple neural network generating an interactive memory. *Mathematical Biosciences*, 14, 1972. doi: 10.1016/0025-5564(72)90075-2.  
S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, and C. Ré. Zoology: Measuring and improving recall in efficient language models. *ArXiv*, 2312.04927, 2023.  
J. Ba, G. E. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu. Using fast weights to attend to the recent past. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 4331-4339. Curran Associates, Inc., 2016a.  
J. Ba, J. R. Kiros, and G. Hinton. Layer normalization. ArXiv, 1607.06450, 2016b.  
A. Bau, Y. Belinkov, H. Sajjad, N. Durrani, F. Dalvi, and J. Glass. Identifying and controlling important neurons in neural machine translation. In International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id=H1z-PsR5KX.  
Y. Bisk, R. Zellers, R. LeBras, J. Gao, and Y. Choi. Piqa: Reasoning about physical commonsense in natural language. In AAAI Conference on Artificial Intelligence, volume 34, pp. 7432-7439, 2020.  
S. L. Blodgett, L. Green, and B. O'Connor. Demographic dialectal variation in social media: A case study of African-American English. In Conference on Empirical Methods in Natural Language Processing, pp. 1119-1130, 2016. doi: 10.18653/v1/D16-1120.  
T. Brown, B. Mann, N. Ryder, et al. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877-1901. Curran Associates, Inc., 2020.  
K. M. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlós, P. Hawkins, J. Q. Davis, A. Mohiuddin, L. Kaiser, D. B. Belanger, L. J. Colwell, and A. Weller. Rethinking attention with performers. In 9th International Conference on Learning Representations (ICLR). OpenReview.net, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH.  
A. Chowdhery, S. Narang, J. Devlin, et al. PaLM: scaling language modeling with pathways. ArXiv, 2204.02311, 2022.  
A. Chronopoulou, M. Peters, and J. Dodge. Efficient hierarchical domain adaptation for pretrained language models. In Conference of the North American Chapter of the Association for Computational Linguistics, pp. 1336-1351, 2022. doi: 10.18653/v1/2022.naacl-main.96.  
P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. ArXiv, 1803.05457, 2018.  
T. M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. Electronic Computers, IEEE Transactions on, EC-14(3):326-334, 1965.  
T. Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), volume 12, 2024. URL https://openreview.net/forum?id=mZn2Xyh9Ec.  
T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (eds.), Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://openreview.net/forum?id=H4DqfPSibmx.  
P. Dayan and D. J. Willshaw. Optimising synaptic learning rules in linear associative memories. Biological Cybernetics, 65, 1991. doi: 10.1007/bf00206223.

S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen, S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. DeFreitas, and C. Gulcehre. Griffin: Mixing gated linear recurrences with local attention for efficient language models. ArXiv, 2402.19427, 2024.  
J. Degrave, F. Felici, J. Buchli, et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602:414-419, 2022. doi: 10.1038/s41586-021-04301-9.  
G. Deletang, A. Ruoss, J. Grau-Moya, T. Genewein, L. K. Wenliang, E. Catt, C. Cundy, M. Hutter, S. Legg, J. Veness, and P. A. Ortega. Neural networks and the Chomsky hierarchy. In International Conference on Learning Representations (ICLR), volume 11, 2023. URL https://openreview.net/forum?id=WbxHAzkeQcn.  
N. Du, Y. Huang, A. M. Dai, et al. GLaM: efficient scaling of language models with mixture-of-experts. ArXiv, 2012.06905, 2021.  
D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, and C. Re. Hungry hungry hippos: Towards language modeling with state space models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=COZDyOWYGg.  
L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy. The Pile: An 800gb dataset of diverse text for language modeling. ArXiv, 2101.00027, 2021.  
F. A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: Continual prediction with LSTM. Neural Compututatlon, 12(10):2451-2471, 2000.  
Gemini Team Google. Gemini: A family of highly capable multimodal models. ArXiv, 2312.11805, 2023.  
A. Graves. Generating sequences with recurrent neural networks. ArXiv, 1308.0850, 2013.  
S. Greenbaum and G. Nelson. The international corpus of English (ICE) project. World Englishes, 15(1):3-15, 1996.  
K. Greff, R. K. Srivastava, J. Koutnik, B. R. Steunebrink, and J. Schmidhuber. LSTM: A search space odyssey. ArXiv, 1503.04069, 2015.  
A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. *ArXiv*, 2312.00752, 2023.  
A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with structured state spaces. *ArXiv*, 2111.00396, 2021.  
A. Gupta, A. Gu, and J. Berant. Diagonal state spaces are as effective as structured state spaces. ArXiv, 2203.14343, 2022.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.  
S. Hochreiter. Untersuchungen zu dynamischen neuronalen Netzen. Master's thesis, Technische Universität München, 1991.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735-1780, 1997a.  
S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. In M. C. Mozer, M. I. Jordan, and T. Petsche (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 9, pp. 473-479. MIT Press, Cambridge MA, 1997b.  
S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. In J. Kolen and S. Kremer (eds.), A Field Guide to Dynamical Recurrent Networks. IEEE, 2000.

S. Hochreiter, A. Steven Younger, and Peter R. Conwell. Learning to learn using gradient descent. In G. Dorffner, H. Bischof, and K. Hornik (eds.), Proc. Int. Conf. on Artificial Neural Networks (ICANN 2001), pp. 87-94. Springer, 2001.  
S. Hochreiter, M. Heusel, and K. Obermayer. Fast model-based protein homology detection without alignment. Bioinformatics, 23(14):1728-1736, 2007.  
J. Hoffmann, S. Borgeaud, A. Mensch, et al. Training compute-optimal large language models. ArXiv, 2203.15556, 2022.  
M. D. Hossain, F. Sohel, M. F. Shiratuddin, and H. Laga. A comprehensive survey of deep learning for image captioning. ACM Computing Surveys (CSUR), 51(6):118, 2019.  
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. *ArXiv*, 2001.08361, 2020.  
A. Karpathy. The unreasonable effectiveness of recurrent neural networks. http://karpathy.github.io/2015/05/21/rnn-effectiveness/, 2015.  
A. Karpathy. OpenAI Five defeats Dota 2 world champions. https://openai.com/research/openai-five-defeats-dota-2-world-champions, 2019.  
A. Karpathy and L. Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3128-3137, 2015.  
A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In E. H. Daumé III and A. Singh (eds.), International Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, pp. 5156-5165. PMLR, 2020.  
T. Katsch. GateLoop: Fully data-controlled linear recurrence for sequence modeling. *ArXiv*, 2311.01927, 2023.  
D. Kocetkov, R. Li, L. BenAllal, J. Li, C. Mou, C. Mu nozFerrandis, Y. Jernite, M. Mitchell, S. Hughes, T. Wolf, D. Bahdanau, L. vonWerra, and H. deVries. The Stack: 3 TB of permissively licensed source code. ArXiv, 2211.15533, 2022.  
T. Kohonen. Correlation matrix memories. IEEE Transactions on Computers, C-21(4), 1972. doi: 10.1109/tc.1972.5008975.  
F. Kratzert, D. Klotz, C. Brenner, K. Schulz, and M. Herrnegger. Rainfall-runoff modelling using long short-term memory (LSTM) networks. Hydrology and Earth System Sciences, 22(11):6005-6022, 2018.  
F. Kratzert, D. Klotz, G. Shalev, G. Klambauer, S. Hochreiter, and G. Nearing. Benchmarking a catchment-aware long short-term memory network (LSTM) for large-scale hydrological modeling. ArXiv, 1907.08456, 2019.  
A. Krizhevsky. Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto, 2009.  
D. Krotov and J. J. Hopfield. Dense associative memory for pattern recognition. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems, pp. 1172-1180. Curran Associates, Inc., 2016.  
D. Krotov and J. J. Hopfield. Dense associative memory is robust to adversarial inputs. *ArXiv*, 1701.00939, 2017.  
Y. Lakretz, G. Kruszewski, T. Desbordes, D. Hupkes, S. Dehaene, and M. Baroni. The emergence of number and syntax units in LSTM language models. In J. Burstein, C. Doran, and T. Solorio (eds.), Conference of the North American Chapter of the Association for Computational Linguistics, pp. 11-20. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1002.

Y. Li, T. Cai, Y. Zhang, D. Chen, and D. Dey. What makes convolutional models great on long sequence modeling? ArXiv, 2210.09298, 2022.  
P. Liang, R. Bommasani, T. Lee, et al. Holistic evaluation of language models. Annals of the New York Academy of Sciences, 1525:140-146, 2023.  
J. Lin, R. Men, A. Yang, C. Zhou, M. Ding, Y. Zhang, P. Wang, A. Wang, L. Jiang, X. Jia, J. Zhang, J. Zhang, X. Zou, Z. Li, X. Deng, J. Liu, J. Xue, H. Zhou, J. Ma, j. Yu, Y. Li, W. Lin, J. Zhou, J. Tang, and H. Yang. M6: A Chinese multimodal pretrainer. ArXiv, 2103.00823, 2021.  
D. Linsley, J. Kim, V. Veerabadran, C. Windolf, and T. Serre. Learning long-range spatial dependencies with horizontal gated recurrent units. Advances in Neural Information Processing Systems (NeurIPS), 31, 2018.  
I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.  
X. Ma, C. Zhou, X. Kong, J. He, L. Gui, G. Neubig, J. May, and L. Zettlemoyer. Mega: Moving average equipped gated attention. ArXiv, 2209.10655, 2022.  
A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Annual Meeting of the Association for Computational Linguistics, volume 49, pp. 142-150, 2011.  
I. Magnusson, A. Bhagia, V. Hofmann, et al. Paloma: A benchmark for evaluating language model fit. ArXiv, 2312.10523, 2023.  
H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur. Long range language modeling via gated state spaces. ArXiv, 2206.13947, 2022.  
S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. In International Conference on Learning Representations (ICRL), 2017. URL https://openreview.net/forum?id=Byj72udxe.  
W. Merrill and A. Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531-545, 2023. doi: 10.1162/tacl_a_00562.  
W. Merrill, J. Petty, and A. Sabharwal. The illusion of state in state-space models. ArXiv, 2404.08819, 2024.  
M. Milakov and N. Gimelshein. Online normalizer calculation for softmax. *ArXiv*, 1805.02867, 2018.  
K. Nakano. Associatron - a model of associative memory. IEEE Transactions on Systems, Man, and Cybernetics, SMC-2(3):380-388, 1972. doi: 10.1109/TSMC.1972.4309133.  
G. Nearing, D. Cohen, V. Dube, M. Gauch, O. Gilon, S. Harrigan, A. Hassidim, D. Klotz, F. Kratzert, A. Metzger, S. Nevo, F. Pappenberger, C. Prudhomme, G. Shalev, S. Shenzis, T. Y. Tekalign, D. Weitzner, and Y. M. B. Kosko. Global prediction of extreme floods in ungauged watersheds. Nature, 627:559-563, 2024. doi: 10.1038/s41586-024-07145-1.  
C. Olsson, N. Elhage, N. Nanda, et al. In-context learning and induction heads. ArXiv, 2209.11895, 2022.  
A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De. Resurrecting recurrent neural networks for long sequences. In Proceedings of the 40th International Conference on Machine Learning (ICML). JMLR.org, 2023. doi: 10.5555/3618408.3619518.  
A. Papasavva, S. Zannettou, E. DeCristofaro, G. Stringhini, and J. Blackburn. Raiders of the lost KeK: 3.5 years of augmented 4chan posts from the politically incorrect board. In International AAAI Conference on Web and Social Media (ICWSM), volume 14, pp. 885-894, 2020.

D. Paperno, G. Kruszewski, A. Lazaridou, N.-Q. Pham, R. Bernardi, S. Pezzelle, M. Baroni, Gemma G. Boleda, and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Annual Meeting of the Association for Computational Linguistics, volume 1, pp. 1525-1534, 2016.  
G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, A. Cappelli, H. Alobeidli, B. Pannier, E. Al-mazrouei, and J. Launay. The RefinedWeb dataset for Falcon LLM: Outperforming curated corpora with web data, and web data only. ArXiv, 2306.01116, 2023.  
B. Peng, E. Alcaide, Q. Anthony, et al. RWKV: Reinventing RNNs for the transformer era. ArXiv, 2305.13048, 2023.  
B. Peng, D. Goldstein, Q. Anthony, et al. Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence. ArXiv, 2404.05892, 2024.  
M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, and C. Ré. Hyena hierarchy: Towards larger convolutional language models. In Proceedings of the 40th International Conference on Machine Learning (ICML). JMLR.org, 2023. doi: 10.5555/3618408.3619572.  
M. Poli, A. W. Thomas, E. Nguyen, P. Ponnusamy, B. Deiseroth, K. Kersting, T. Suzuki, B. Hie, S. Ermon, C. Ré, C. Zhang, and S. Massaroli. Mechanistic design and scaling of hybrid architectures. ArXiv, 2403.17844, 2024.  
Z. Qin, S. Yang, and Y. Zhong. Hierarchically gated recurrent neural network for sequence modeling. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2023. URL https://openreview.net/forum?id=P1TCHxJwLB.  
Z. Qin, S. Yang, W. Sun, X. Shen, D. Li, W. Sun, and Y. Zhong. HGRN2: Gated linear RNNs with state expansion. ArXiv, 2404.07904, 2024.  
D. R. Radev, P. Muthukrishnan, and V. Qazvinian. The ACL anthology network corpus. In Workshop on Text and Citation Analysis for Scholarly Digital Libraries (NLPIR4DL), pp. 54-61. Association for Computational Linguistics, 2009.  
A. Radford, R. Jozefowicz, and I. Sutskever. Learning to generate reviews and discovering sentiment. ArXiv, 1704.01444, 2017.  
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. https://openai.com/index/better-language-models, 2019.  
J. W. Rae, S. Borgeaud, T. Cai, et al. Scaling language models: Methods, analysis & insights from training Gopher. ArXiv, 2112.11446, 2021.  
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv, 1910.10683, 2019.  
H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, M. Pavlovic, G. K. Sandve, V. Greiff, D. Kreil, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. Hopfield networks is all you need. In International Conference on Learning Representations (ICLR). OpenReview, 2021.  
M. Reid, V. Zhong, S. Gururangan, and L. Zettlemoyer. M2D2: A massively multi-domain language modeling dataset. In Conference on Empirical Methods in Natural Language Processing, pp. 964-975, 2022.  
M. Reid, N. Savinov, D. Teptyashin, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv, 2403.05530, 2024.  
M. H. Ribeiro, J. Blackburn, B. Bradlyn, E. DeCristofaro, G. Stringhini, S. Long, S. Greenberg, and S. Zannettou. The evolution of the manosphere across the web. In Proceedings of the international AAAI conference on web and social media, volume 15, pp. 196-207, 2021.  
K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99-106, 2021.

T. L. Scao, A. Fan, C. Akiki, et al. BLOOM: A 176B-parameter open-access multilingual language model. ArXiv, 2211.05100, 2022.  
I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In M. Meila and T. Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pp. 9355-9366. PMLR, 2021.  
J. Schmidhuber. Learning to control fast-weight memories: An alternative to recurrent nets. Neural Computation, 4(1):131-139, 1992.  
J. Schmidhuber. Deep learning in neural networks: An overview. *Neural Networks*, 61:85–117, 2015. doi: 10.1016/j.neunet.2014.09.003.  
J. Schulman, B. Zoph, C. Kim, J. Hilton, et al. ChatGPT: Optimizing language models for dialogue. https://openai.com/blog/chatgpt/, 2022. OpenAI Research.  
T. J. Sejnowski. Storing covariance with nonlinearly interacting neurons. Journal of Mathematical Biology, 4, 1977. doi: 10.1007/BF00275079.  
M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. ArXiv, 1909.08053, 2019.  
J. T. H. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling. *ArXiv*, 2208.04933, 2022.  
D. Soboleva, F. Al-Khateeb, R. Myers, J. R. Steeves, J. Hestness, and N. Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023. URL https://huggingface.co/datasets/cerebras/SlimPajama-627B.  
L. Soldaini, R. Kinney, A. Bhagia, et al. Dolma: an open corpus of three trillion tokens for language model pretraining research. ArXiv, 2306.01116, 2023.  
S. Soltan, S. Ananthakrishnan, J. FitzGerald, R. Gupta, W. Hamza, H. Khan, C. Peris, S. Rawls, A. Rosenbaum, A. Rumshisky, C. S. Prakash, M. Sridhar, F. Triefenbach, A. Verma, G. Tur, and P. Natarajan. AlexaTM 20B: Few-shot learning using a large-scale multilingual Seq2Seq model. ArXiv, 2208.01448, 2022.  
R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 28. Curran Associates, Inc., 2015.  
Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. ArXiv, 2307.08621, 2023.  
L. Sutawika, L. Gao, H. Schoelkopf, et al. EleutherAI/lm-evaluation-harness: Major refactor, 2023.  
L. Sutawika, H. Schoelkopf, L. Gao, B. Abbasi, S. Biderman, J. Tow, B. fattori, C. Lovering, farzanehnakhaee70, J. Phang, A. Thite, Fazz, T. Wang, N. Muennighoff, Aflah, sdtblck, nopperl, gakada, tttyuntian, researcher2, Chris, J. Etxaniz, H. A. Lee, Z. Kasner, Khalid, J. Hsu, A. Kanekar, P. S. Ammanamanchi, V. Boykis, and AndyZwei. EleutherAI/lm-evaluation-harness, 2024.  
I. Sutskever, O. Vinyals, and Q. V. V. Le. Sequence to sequence learning with neural networks. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 27 (NIPS'13), pp. 3104-3112. Curran Associates, Inc., 2014.  
Y. Tay, D. Bahri, D. Metzler, D.-C. Juan, Z. Zhao, and C. Zheng. Synthesizer: Rethinking self-attention in transformer models. ArXiv, 2005.00743, 2020.  
Y. Tay, M. Dehghani, S. Abnar, Y. Shen, D. Bahri, P. Pham, J. Rao, L. Yang, S. Ruder, and D. Metzler. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations (ICRL), 2021. URL https://openreview.net/forum?id=qVyeW-grC2k.

R. Thoppilan, D. deFreitas, J. Hall, et al. LaMDA: Language models for dialog applications. ArXiv, 2201.08239, 2022.  
TogetherComputer. Redpajama: an open dataset for training large language models, 2023. URL https://github.com/togethercomputer/RedPajama-Data.  
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models. ArXiv, 2302.1397, 2023.  
D. Vadas and J. R. Curran. Parsing noun phrases in the Penn Treebank. Computational Linguistics, 37(4):753-809, 2011.  
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, pp. 5998-6008. Curran Associates, Inc., 2017.  
O. Vinyals, T. Ewalds, S. Bartunov, et al. Starcraft II: A new challenge for reinforcement learning. ArXiv, 1708.04782, 2017.  
J. Wang, J. N. Yan, A. Gu, and A. M. Rush. Pretraining without attention. ArXiv, 2212.10544, 2022.  
S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma. Linformer: Self-attention with linear complexity. ArXiv, 2006.04768, 2020.  
S. Wang, Y. Sun, Y. Xiang, et al. ERNIE 3.0 Titan: Exploring larger-scale knowledge enhanced pre-training for language understanding and generation. ArXiv, 2112.12731, 2021.  
Y. Wu and K. He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pp. 3-19, 2018.  
L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In Conference of the North American Chapter of the Association for Computational Linguistics, pp. 483-498, 2021. doi: 10.18653/v1/2021.naacl-main.41.  
S. Yang and Y. Zhang. FLA: A Triton-based library for hardware-efficient implementations of linear attention mechanism, 2024. URL https://github.com/sustcsonglin/flash-linear-attention.  
S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. ArXiv, 2312.06635, 2023.  
S. Zannettou, B. Bradlyn, E. DeCristofaro, H. Kwak, M. Sirivianos, G. Stringini, and J. Blackburn. What is Gab: A bastion of free speech or an alt-right echo chamber. In The Web Conference, pp. 1007-1014, 2018. doi: 10.1145/3184558.3191531.  
W. Zaremba and I. Sutskever. Learning to execute. ArXiv, 1410.4615, 2014.  
R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish your sentence? In Annual Meeting of the Association for Computational Linguistics, pp. 4791-4800, 2019.  
A. Zeng, X. Liu, Z. Du, et al. GLM-130B: An open bilingual pre-trained model. ArXiv, 2210.02414, 2022.  
S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer. OPT: Open pre-trained transformer language models. ArXiv, 2205.01068, 2022.

# Contents

# A Extended Long Short-Term Memory 23

A.1 Vanilla Long Short-Term Memory Formulation: Vector Notation 23  
A.2 sLSTM 23  
A.3 mLSTM 25  
A.4 Detailed Block Structure 29

# B Experiments 31

B.1 Synthetic Tasks and Long Range Arena 31

B.1.1 Test of xLSTM's Exponential Gating with Memory Mixing. 31  
B.1.2 Test of xLSTM's Memory Capacities on Associative Recall Tasks. 34  
B.1.3 Test of xLSTM's Long Range Capabilities on the Long Range Arena. 36

B.2 Method Comparison and Ablation Study on SlimPajama (15B) 40  
B.3 xLSTM Large Language Models-SlimPajama300B 42

# C Detailed Results on PALOMA Language Model Evaluation 44

# A Extended Long Short-Term Memory

# A.1 Vanilla Long Short-Term Memory Formulation: Vector Notation

The vanilla LSTM memory cell update rules (Greff et al., 2015) at time step  $t$  extend the scalar cell state formulation to a vector of cell states:

$$
\boldsymbol {c} _ {t} = \left. \boldsymbol {f} _ {t} \odot \boldsymbol {c} _ {t - 1} + \boldsymbol {i} _ {t} \odot \boldsymbol {z} _ {t} \right. \quad \text {c e l l s t a t e} \tag {28}
$$

$$
\boldsymbol {h} _ {t} = \left. \boldsymbol {0} _ {t} \odot \tilde {\boldsymbol {h}} _ {t}, \right. \quad \tilde {\boldsymbol {h}} _ {t} = \psi \left(\left. \boldsymbol {c} _ {t}\right) \right. \quad \text {h i d d e n s t a t e} \tag {29}
$$

$$
\left. \boldsymbol {z} _ {t} \right. = \varphi \left(\tilde {\boldsymbol {z}} _ {t}\right), \quad \tilde {\boldsymbol {z}} _ {t} = W _ {z} \boldsymbol {x} _ {t} + R _ {z} h _ {t - 1} + \boldsymbol {b} _ {z} \quad \text {c e l l i n p u t} \tag {30}
$$

$$
\mathbf {i} _ {t} = \sigma (\tilde {\mathbf {i}} _ {t}), \quad \tilde {\mathbf {i}} _ {t} = W _ {\mathrm {i}} x _ {t} + R _ {\mathrm {i}} h _ {t - 1} + b _ {\mathrm {i}} \quad \text {i n p u t g a t e} \tag {31}
$$

$$
\left. \mathbf {f} _ {t} = \sigma \left(\tilde {\mathbf {f}} _ {t}\right), \quad \tilde {\mathbf {f}} _ {t} = W _ {\mathbf {f}} x _ {t} + R _ {\mathbf {f}} h _ {t - 1} + b _ {\mathbf {f}} \quad \text {f o r g e t g a t e} \right. \tag {32}
$$

$$
\mathbf {o} _ {t} = \sigma (\tilde {\mathbf {o}} _ {t}), \quad \tilde {\mathbf {o}} _ {t} = W _ {\mathbf {o}} x _ {t} + R _ {\mathbf {o}} h _ {t - 1} + b _ {\mathbf {o}} \quad \text {o u t p u t g a t e} \tag {33}
$$

The matrices  $W_{z}$ ,  $W_{\mathrm{i}}$ ,  $W_{\mathrm{f}}$ , and  $W_{\mathrm{o}}$  correspond to the input weights between inputs  $x_{t}$  and cell input, input gate, forget gate, and output gate, respectively. The matrices  $R_{z}$ ,  $R_{\mathrm{i}}$ ,  $R_{\mathrm{f}}$ , and  $R_{\mathrm{o}}$  correspond to the recurrent weights between hidden state  $h_{t-1}$  and cell input, input gate, forget gate, and output gate, respectively.  $b_{z}$ ,  $b_{\mathrm{i}}$ ,  $b_{\mathrm{f}}$ , and  $b_{\mathrm{o}}$  are the corresponding bias vectors.  $\varphi$  and  $\psi$  are the cell input and hidden state activation functions (typically tanh).  $\psi$  is used to normalize or squash the cell state, which would be unbounded otherwise.

# A.2 sLSTM

Similar to the LSTM in Section A.1, also the sLSTM can be vectorized to multiple cells:

$$
\boldsymbol {c} _ {t} = \left. \mathbf {f} _ {t} \odot \boldsymbol {c} _ {t - 1} + \mathbf {i} _ {t} \odot \boldsymbol {z} _ {t} \right. \tag {34}
$$

$$
\boldsymbol {n} _ {t} = \boldsymbol {f} _ {t} \odot \boldsymbol {n} _ {t - 1} + \boldsymbol {i} _ {t} \quad \text {n o r m a l i z e r s t a t e} \tag {35}
$$

$$
\boldsymbol {h} _ {t} = \left. \boldsymbol {0} _ {t} \odot \tilde {\boldsymbol {h}} _ {t}, \right. \quad \tilde {\boldsymbol {h}} _ {t} = \left. \boldsymbol {c} _ {t} \odot \boldsymbol {n} _ {t} ^ {- 1} \right. \quad \text {h i d d e n s t a t e} \tag {36}
$$

$$
\left. \boldsymbol {z} _ {t} \right. = \varphi (\tilde {\boldsymbol {z}} _ {t}), \quad \tilde {\boldsymbol {z}} _ {t} = W _ {z} \boldsymbol {x} _ {t} + R _ {z} h _ {t - 1} + \boldsymbol {b} _ {z} \quad \text {c e l l i n p u t (3 7)}
$$

$$
\mathbf {i} _ {t} = \exp (\tilde {\mathbf {i}} _ {t}), \quad \tilde {\mathbf {i}} _ {t} = W _ {\mathbf {i}} x _ {t} + R _ {\mathbf {i}} h _ {t - 1} + b _ {\mathbf {i}} \quad \text {i n p u t g a t e (3 8)}
$$

$$
\mathbf {f} _ {t} = \exp (\tilde {\mathbf {f}} _ {t}) \operatorname {O R} \sigma (\tilde {\mathbf {f}} _ {t}), \quad \tilde {\mathbf {f}} _ {t} = \boldsymbol {W} _ {\mathbf {f}} \boldsymbol {x} _ {t} + \boldsymbol {R} _ {\mathbf {f}} \boldsymbol {h} _ {t - 1} + \boldsymbol {b} _ {\mathbf {f}} \quad \text {f o r g e t g a t e (3 9)}
$$

$$
\mathbf {o} _ {t} = \sigma (\tilde {\mathbf {o}} _ {t}), \quad \tilde {\mathbf {o}} _ {t} = W _ {\mathrm {o}} x _ {t} + R _ {\mathrm {o}} h _ {t - 1} + b _ {\mathrm {o}} \quad \text {o u t p u t g a t e (4 0)}
$$

Here, the cell input activation function  $\varphi$  is tanh, the hidden state activation function is the identity.  $\varphi$  helps stabilizing the recurrence.

Considering external gradient contribution  $\delta_{\pmb{h}_t}^{\mathrm{ext}}$  from subsequent layers and recurrent gradient contribution  $\delta_{\pmb{h}_t}^R$  from gradients from future states flowing over the cell interaction matrix  $\pmb{R}$ , we obtain the recursive backward pass of sLSTM, where  $\delta_{a}$  indicates gradients with respect to parameter / internal variable  $a$ :

$$
\delta_ {\boldsymbol {h} _ {t}} = \delta_ {\boldsymbol {h} _ {t}} ^ {e x t} + \delta_ {\boldsymbol {h} _ {t}} ^ {\boldsymbol {R}} \tag {41}
$$

$$
\delta_ {\boldsymbol {c} _ {t - 1}} = \mathbf {f} _ {t} \odot \delta_ {\boldsymbol {c} _ {t}} + \mathbf {o} _ {t - 1} \odot \boldsymbol {n} _ {t - 1} ^ {- 1} \odot \delta_ {\boldsymbol {h} _ {t - 1}} \tag {42}
$$

$$
\delta_ {\boldsymbol {n} _ {t - 1}} = \mathbf {f} _ {t} \odot \delta_ {\boldsymbol {n} _ {t}} - \mathbf {o} _ {t - 1} \odot \boldsymbol {c} _ {t - 1} \odot \boldsymbol {n} _ {t - 1} ^ {- 2} \odot \delta_ {\boldsymbol {h} _ {t - 1}} \tag {43}
$$

$$
\delta_ {\tilde {\mathbf {f}} _ {t}} = \mathbf {f} _ {t} ^ {\prime} \odot \boldsymbol {c} _ {t - 1} \odot \delta_ {\boldsymbol {c} _ {t}} + \mathbf {f} _ {t} ^ {\prime} \odot \boldsymbol {n} _ {t - 1} \odot \delta_ {\boldsymbol {n} _ {t}} \tag {44}
$$

$$
\delta_ {\tilde {\mathbf {i}} _ {t}} = \mathbf {i} _ {t} ^ {\prime} \odot \boldsymbol {z} _ {t} \odot \delta_ {\boldsymbol {c} _ {t}} + \mathbf {i} _ {t} ^ {\prime} \odot \delta_ {\boldsymbol {n} _ {t}} \tag {45}
$$

$$
\delta_ {\tilde {\mathbf {z}} _ {t}} = \mathbf {i} _ {t} \odot \varphi^ {\prime} (\tilde {\mathbf {z}} _ {t}) \odot \delta_ {\mathbf {c} _ {t}} \tag {46}
$$

$$
\delta_ {\bar {\mathbf {o}} _ {t}} = \mathbf {o} _ {t} ^ {\prime} \odot \boldsymbol {c} _ {t} \odot \boldsymbol {n} _ {t} ^ {- 1} \odot \delta_ {\boldsymbol {h} _ {t}} \tag {47}
$$

$$
\delta_ {\mathbf {x} _ {t}} = \sum_ {\mathbf {g} \in \{\mathbf {f}, \mathbf {i}, \mathbf {z}, \mathbf {o} \}} W _ {\mathbf {g}} ^ {\top} \delta_ {\tilde {\mathbf {g}} _ {t}} \tag {48}
$$

$$
\delta_ {\boldsymbol {h} _ {t - 1}} ^ {\boldsymbol {R}} = \sum_ {\mathbf {g} \in \{\mathbf {f}, \mathrm {i}, \mathbf {z}, \mathbf {o} \}} \boldsymbol {R} _ {\mathbf {g}} ^ {\top} \delta_ {\tilde {\mathbf {g}} _ {t}} \tag {49}
$$

$$
\delta_ {\boldsymbol {R} _ {\mathbf {g}}} ^ {\top} = \sum_ {t} \boldsymbol {h} _ {t - 1} \delta_ {\tilde {\mathbf {g}} _ {t}} ^ {\top}, \quad \mathbf {g} \in \{\mathbf {i}, \mathbf {f}, \mathbf {z}, \mathbf {o} \} \tag {50}
$$

$$
\delta_ {\boldsymbol {W} _ {\mathbf {g}}} ^ {\top} = \sum_ {t} \boldsymbol {x} _ {t} \delta_ {\tilde {\mathbf {g}} _ {t}} ^ {\top}, \quad \mathbf {g} \in \{\mathbf {i}, \mathbf {f}, \mathbf {z}, \mathbf {o} \} \tag {51}
$$

with the derivatives of the respective gate activation function  $\mathbf{i}_t' = \exp'(\tilde{\mathbf{i}}_t) = \exp(\tilde{\mathbf{i}}_t) = \mathbf{i}_t$ ,  $\mathbf{o}_t' = \sigma'(\tilde{\mathbf{o}}_t)$ , and  $\mathbf{f}_t' = \sigma'(\tilde{\mathbf{f}}_t)$  or  $\mathbf{f}_t' = \mathbf{f}_t$  depending on the forget gate activation.  $\varphi'(z)$  is the derivative of the cell input activation function  $\varphi(z)$ .

The matrices  $R_{z}, R_{\mathrm{i}}, R_{\mathrm{f}}, R_{\mathrm{o}}$  are block-diagonal which is analogous to multiple heads in the mLSTM. This way, the parameters reduce to  $d^{2} / (N_{h})$ , where  $N_{h}$  is the number of heads, limiting the cell interactions to individual heads. This parameter efficient formulation of cell interactions together with the exponential gating is called the new memory mixing. Finally, to stabilize the backward pass, we clip the magnitude of  $\delta_{h_t}^R$  to 10, as a means to prohibit exploding gradients for long context lengths.

Proof of Equivalence for sLSTM Stabilized Version. The stabilization state  $m$ , see Equation (15) in the main paper, has no gradient, and hence does not influence the other gradients. We go back to the scalar version (Equation 8) here for simplicity. We re-define  $c_t^{(s)}$  and  $n_t^{(s)}$  as stabilized cell and normalizer states:

$$
c _ {t} = c _ {t} ^ {(s)} \exp \left(\boxed {m _ {t}}\right) \tag {52}
$$

$$
n _ {t} = n _ {t} ^ {(s)} \exp \left(\boxed {m _ {t}}\right) \tag {53}
$$

Inserting Equation 15 into Equation 8 yields:

$$
\begin{array}{l} \tilde {h} _ {t} ^ {(s)} = c _ {t} ^ {(s)} / n _ {t} ^ {(s)} = (54) \\ = \frac {\exp \left(\log \left(\mathrm {f} _ {t}\right) + \boxed {m _ {t - 1}} - \boxed {m _ {t}}\right) c _ {t - 1} ^ {(s)} + \exp \left(\log \left(\mathrm {i} _ {t}\right) - \boxed {m _ {t}}\right) z _ {t}}{\exp \left(\log \left(\mathrm {f} _ {t}\right) + \boxed {m _ {t - 1}} - \boxed {m _ {t}}\right) n _ {t - 1} ^ {(s)} + \exp \left(\log \left(\mathrm {i} _ {t}\right) - \boxed {m _ {t}}\right)} (55) \\ = \frac {\exp \left(\log \left(\mathrm {f} _ {t}\right) + \boxed {m _ {t - 1}}\right) c _ {t - 1} ^ {(s)} + \exp \left(\log \left(\mathrm {i} _ {t}\right)\right) z _ {t}}{\exp \left(\log \left(\mathrm {f} _ {t}\right) + \boxed {m _ {t - 1}}\right) n _ {t - 1} ^ {(s)} + \exp \left(\log \left(\mathrm {i} _ {t}\right)\right)} (56) \\ = \frac {\exp \left(\log \left(\mathrm {f} _ {t}\right)\right) c _ {t - 1} + \exp \left(\log \left(\mathrm {i} _ {t}\right)\right) z _ {t}}{\exp \left(\log \left(\mathrm {f} _ {t}\right)\right) n _ {t - 1} + \exp \left(\log \left(\mathrm {i} _ {t}\right)\right)} (57) \\ = \frac {\mathrm {f} _ {t} c _ {t - 1} + \mathrm {i} _ {t} z _ {t}}{\mathrm {f} _ {t} n _ {t - 1} + \mathrm {i} _ {t}} = c _ {t} / n _ {t} = \tilde {h} _ {t} (58) \\ \end{array}
$$

Therefore, since the loss solely depends on  $h_t$ , there's no dependency on  $m_t$ , and consequently, no gradient exists for this stabilization state. Note that  $m_t$  can be chosen arbitrarily. We choose  $m_t = \max(\log(\mathbf{f}_t) + m_{t-1}, \log(\mathbf{i}_t))$ , which stabilizes the exponential function. One can even find  $m_t$ , such that the normalizer state  $n_t$  can be eliminated, but this version was experimentally found to be numerically unstable in the backward pass.

# A.3 mLSTM

Throughout this section,  $\mathbf{1} \in \mathbb{R}^T$  denotes a column vector of ones and  $\mathbf{1}^\top \in \mathbb{R}^{1 \times T}$  a row vector of ones, where  $T$  is the dimension of this vector.

Recurrent mLSTM Backward Pass. The recurrent formulation of the mLSTM cell in Equation 19 yields the following backward pass recurrence, where  $\delta_{a}$  indicates gradients with respect to parameter or internal variable  $a$  and  $\delta_{h_t}^{\mathrm{ext}}$  denotes gradients from subsequent layers:

$$
\delta_ {\tilde {h} _ {t}} = \mathbf {o} _ {t} \odot \delta_ {h _ {t}} ^ {\text {e x t}} \tag {59}
$$

$$
\delta_ {\boldsymbol {C} _ {t - 1}} ^ {\top} = \mathrm {f} _ {t} \delta_ {\boldsymbol {C} _ {t}} ^ {\top} + \frac {\boldsymbol {q} _ {t - 1} \delta_ {\boldsymbol {h} _ {t - 1}} ^ {\top}}{\max  \left\{\left| \boldsymbol {n} _ {t - 1} ^ {\top} \boldsymbol {q} _ {t - 1} \right| , 1 \right\}} \tag {60}
$$

$$
\delta_ {\boldsymbol {n} _ {t - 1}} = \mathrm {f} _ {t} \delta_ {\boldsymbol {n} _ {t}} - \frac {\boldsymbol {q} _ {t - 1} ^ {\top} \boldsymbol {C} _ {t - 1} ^ {\top} \delta_ {\tilde {\mathbf {h}} _ {t - 1}}}{\max  \left\{\left| \boldsymbol {n} _ {t - 1} ^ {\top} \boldsymbol {q} _ {t - 1} \right| , 1 \right\} ^ {2}} \Omega (\boldsymbol {n} _ {t - 1} ^ {\top} \boldsymbol {q} _ {t - 1}) \boldsymbol {q} _ {t - 1} \tag {61}
$$

$$
\delta_ {\boldsymbol {v} _ {t}} ^ {\top} = \mathrm {i} _ {t} \boldsymbol {k} _ {t} ^ {\top} \delta_ {\boldsymbol {C} _ {t}} ^ {\top} \tag {62}
$$

$$
\delta_ {\boldsymbol {k} _ {t}} ^ {\top} = \mathrm {i} _ {t} \left(\boldsymbol {v} _ {t} ^ {\top} \delta_ {\boldsymbol {C} _ {t}} + \delta_ {\boldsymbol {n} _ {t}} ^ {\top}\right) \tag {63}
$$

$$
\delta_ {\boldsymbol {q} _ {t}} = \frac {\boldsymbol {C} _ {t} ^ {\top} \delta_ {\tilde {\boldsymbol {h}} _ {t}}}{\max  \left\{\left| \boldsymbol {n} _ {t} ^ {\top} \boldsymbol {q} _ {t} \right| , 1 \right\}} - \frac {\boldsymbol {q} _ {t} ^ {\top} \boldsymbol {C} _ {t} ^ {\top} \delta_ {\tilde {\boldsymbol {h}} _ {t}}}{\max  \left\{\left| \boldsymbol {n} _ {t} ^ {\top} \boldsymbol {q} _ {t} \right| , 1 \right\} ^ {2}} \Omega \left(\boldsymbol {n} _ {t} ^ {\top} \boldsymbol {q} _ {t}\right) \boldsymbol {n} _ {t} \tag {64}
$$

$$
\delta_ {\boldsymbol {x} _ {t}} = \sum_ {g \in \{q, k, v \}} \boldsymbol {W} _ {g} ^ {\top} \delta_ {\boldsymbol {g} _ {t}} \tag {65}
$$

$$
\delta_ {\boldsymbol {W} _ {g}} ^ {\top} = \sum_ {t} \boldsymbol {x} _ {t} \delta_ {\boldsymbol {g} _ {t}} ^ {\top}, \quad g \in \{q, k, v \} \tag {66}
$$

$$
\delta_ {\boldsymbol {b} _ {g}} = \sum_ {t} \delta_ {\boldsymbol {g} _ {t}}, \quad g \in \{q, k, v \} \tag {67}
$$

$$
\delta_ {\tilde {\mathbf {f}} _ {t}} = \left(\mathbf {1} ^ {\top} \left(\boldsymbol {C} _ {t - 1} \odot \delta_ {\boldsymbol {C} _ {t}}\right) \mathbf {1} + \mathbf {1} ^ {\top} \left(\boldsymbol {n} _ {t - 1} \odot \delta_ {\boldsymbol {n} _ {t}}\right)\right) \gamma (\tilde {\mathbf {f}} _ {t}) \tag {68}
$$

$$
\delta_ {\tilde {\mathbf {i}} _ {t}} = \left(\mathbf {1} ^ {\top} \left(\left(\boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top}\right) \odot \delta_ {\boldsymbol {C} _ {t}}\right) \mathbf {1} + \mathbf {1} ^ {\top} \left(\boldsymbol {k} _ {t} \odot \delta_ {\boldsymbol {n} _ {t}}\right)\right) \exp (\tilde {\mathrm {i}} _ {t}) \tag {69}
$$

$$
\delta_ {\tilde {\mathbf {o}} _ {t}} = \tilde {\mathbf {h}} _ {t} \odot \sigma^ {\prime} (\tilde {\mathbf {o}} _ {t}) \odot \delta_ {\mathbf {h} _ {t}} \tag {70}
$$

and  $\Omega (z) = \Theta (z - 1) - \Theta (-z - 1),\Theta (z)$  being the Heaviside step function.  $\gamma (z)$  is either  $\sigma^{\prime}(z)$  or  $\exp (z)$ , depending on the forget gate activation.

Parallel mLSTM Forward Pass. The mLSTM recurrence in Equations (19-27) can be reformulated in a parallel form, which is used to speed up training. After training we can still use the recurrent formulation for fast text generation.

Instead of processing each input  $\pmb{x}_t\in \mathbb{R}^d$  at time step  $t$  sequentially, the parallel version processes all timesteps of a full sequence  $\pmb {X}\in \mathbb{R}^{T\times d}$  at once, where  $T$  is the sequence length and  $d$  is the head dimension. We present the forward pass of the mLSTM for a single head and drop the head dimension for simplicity.

Let  $\tilde{\mathbf{f}}\in \mathbb{R}^T$  be the forget gate pre-activations and  $\tilde{\mathbf{i}}\in \mathbb{R}^T$  be the input gate pre-activations for a full sequence. We construct the forget gate activation matrix  $\mathbf{F}\in \mathbb{R}^{T\times T}$  by

$$
\mathbf {F} _ {i j} = \left\{ \begin{array}{l l} 0 & \text {f o r} i <   j \\ 1 & \text {f o r} i = j \\ \prod_ {k = j + 1} ^ {i} \sigma \left(\tilde {\mathrm {f}} _ {k}\right) & \text {f o r} i > j \end{array} , \right. \tag {71}
$$

and the input gate pre-activation matrix  $\tilde{\mathbf{I}}\in \mathbb{R}^{T\times T}$  by

$$
\tilde {\mathbf {I}} _ {i j} = \left\{ \begin{array}{l l} 0 & \text {f o r} i <   j \\ \tilde {\mathrm {i}} _ {j} & \text {f o r} i \geqslant j \end{array} . \right. \tag {72}
$$

By applying the elementwise exponential input gate activation function naively, we obtain the unstabilized gate activation matrix  $\mathbf{D} \in \mathbb{R}^{T \times T}$  as

$$
\mathbf {D} = \mathbf {F} \odot \exp (\tilde {\mathbf {I}}). \tag {73}
$$

In order to avoid overflow due to the exponential function we apply the same stabilization as in the recurrent sLSTM, see Equation 15. In the parallel formulation of the mLSTM we get a numerically stable gate activation matrix  $\mathbf{D}' \in \mathbb{R}^{T \times T}$  by taking the logarithm of  $\mathbf{D}$  element-wise and subtracting the row-wise maximum value of  $\mathbf{D}$  from each element:

$$
\widetilde {\mathbf {D}} = \log \mathbf {D} = \log (\mathbf {F} \odot \exp (\widetilde {\mathbf {I}})) = \log \mathbf {F} + \widetilde {\mathbf {I}} \tag {74}
$$

$$
\mathbf {D} ^ {\prime} = \exp (\widetilde {\mathbf {D}} - \max  \widetilde {\mathbf {D}}) \tag {75}
$$

Given the queries, keys and values  $Q, K, V \in \mathbb{R}^{T \times d}$ , for a full sequence we can compute all hidden pre-activation states  $\widetilde{\mathbf{H}} \in \mathbb{R}^{T \times d}$  in parallel for the un-stabilized version by

$$
\tilde {\mathbf {H}} = \boldsymbol {C} \boldsymbol {V}, \quad \text {w i t h} \quad \boldsymbol {C} = \frac {\tilde {\boldsymbol {C}}}{\max  \left(\left| \sum_ {j = 1} ^ {T} \tilde {\boldsymbol {C}} _ {i j} \right| , 1\right)}, \quad \text {a n d} \quad \tilde {\boldsymbol {C}} = \frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\sqrt {d}} \odot \mathbf {D}. \tag {76}
$$

Note that we extract the  $\frac{1}{\sqrt{d}}$  factor for  $\pmb{K}$  explicitly here and further on. For the stabilized version this yields

$$
\tilde {\mathbf {H}} = \boldsymbol {C} \boldsymbol {V}, \quad \text {w i t h} \quad \boldsymbol {C} = \frac {\tilde {\boldsymbol {C}} ^ {\prime}}{\max  \left(\left| \sum_ {j = 1} ^ {T} \tilde {\boldsymbol {C}} _ {i j} ^ {\prime} \right| , \exp (- \max  \tilde {\mathbf {D}})\right)}, \quad \text {a n d} \quad \tilde {\boldsymbol {C}} ^ {\prime} = \frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\sqrt {d}} \odot \mathbf {D} ^ {\prime}, \tag {77}
$$

where for both versions the hidden pre-activation states  $\widetilde{\mathbf{H}}$  are identical.

With the output gate pre-activations  $\widetilde{\mathbf{O}}\in \mathbb{R}^{T\times d}$  we can compute the hidden states  $\pmb {H}\in \mathbb{R}^{T\times d}$  for all timesteps by applying the output gate in parallel for each timestep element-wise:

$$
\mathbf {H} = \sigma (\widetilde {\mathbf {O}}) \odot \widetilde {\mathbf {H}}. \tag {78}
$$

This gives the parallel forward pass of the mLSTM for a full input sequence  $\mathbf{X} \in \mathbb{R}^{T \times d}$ .

Parallel mLSTM Backward Pass. We present the backward pass of the mLSTM for the stabilized version only. For completeness we summarize the forward pass in the stabilized version before we present the backward pass.

Given the forget gate matrix  $\mathbf{F} \in \mathbb{R}^{T \times T}$ , the logarithm of the forget gate matrix  $\overline{\mathbf{F}} = \log \mathbf{F} \in \mathbb{R}^{T \times T}$  and the input gate matrix  $\mathbf{I} \in \mathbb{R}^{T \times T}$  as introduced above, together with the queries, keys and values

$\pmb{Q}, \pmb{K}, \pmb{V} \in \mathbb{R}^{T \times d}$ , we can write the forward pass of the mLSTM in the stabilized version as:

$$
\widetilde {\mathbf {D}} = \overline {{\mathbf {F}}} + \widetilde {\mathbf {I}} \tag {79}
$$

$$
\boldsymbol {m} = \max  _ {j} \widetilde {\mathbf {D}} _ {i j}, \quad \text {r o w - w i s e m a x i m u m} \tag {80}
$$

$$
\mathbf {D} ^ {\prime} = \exp (\widetilde {\mathbf {D}} - m \mathbf {1} ^ {\top}) \tag {81}
$$

$$
\widetilde {\boldsymbol {C}} ^ {\prime} = \frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\sqrt {d}} \odot \mathbf {D} ^ {\prime} \tag {82}
$$

$$
\boldsymbol {b} = \sum_ {j = 1} ^ {T} \widetilde {\boldsymbol {C}} _ {i j} ^ {\prime} = \widetilde {\boldsymbol {C}} ^ {\prime} \mathbf {1}, \quad \text {r o w - w i s e s u m} \tag {83}
$$

$$
\boldsymbol {n} = \max  (| \boldsymbol {b} |, \exp (- \boldsymbol {m})) \tag {84}
$$

$$
\boldsymbol {C} = \tilde {\boldsymbol {C}} ^ {\prime} \odot \left(\boldsymbol {n} ^ {- 1} \mathbf {1} ^ {\top}\right) \tag {85}
$$

$$
\widetilde {\mathbf {H}} = \boldsymbol {C} \boldsymbol {V} \tag {86}
$$

With this forward pass we can compute the gradients  $\delta_{a}$  for all intermediate and input variables to the mLSTM forward pass in the backward pass. We denote the gradient with respect to variable  $a$  as  $\delta_{a}$ . Given the output gradient  $\delta_{\widetilde{\mathbf{H}}} \in \mathbb{R}^{T \times d}$  we can compute the backward pass for the intermediate gradients as:

$$
\delta_ {\boldsymbol {C}} ^ {\top} = \boldsymbol {V} \delta_ {\tilde {\mathbf {H}}} ^ {\top} \tag {87}
$$

$$
\begin{array}{l} \delta_ {\boldsymbol {n}} = - \left(\widetilde {C} ^ {\prime} \odot \left(\boldsymbol {n} ^ {- 2} \mathbf {1} ^ {\top}\right) \odot \delta_ {\boldsymbol {C}}\right) \mathbf {1} (88) \\ = - \left(\left(\tilde {\boldsymbol {C}} ^ {\prime} \odot \delta_ {\boldsymbol {C}}\right) \mathbf {1}\right) \odot \boldsymbol {n} ^ {- 2} (89) \\ \end{array}
$$

$$
\delta_ {\boldsymbol {b}} = \operatorname {s i g n} (\boldsymbol {n}) \odot \delta_ {\boldsymbol {n}} \odot \left\{ \begin{array}{l l} 1 & \text {i f} | \boldsymbol {b} | > \exp (- \mathbf {m}) \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {90}
$$

$$
\delta_ {\tilde {C} ^ {\prime}, C} = \left(\boldsymbol {n} ^ {- 1} \boldsymbol {1} ^ {\top}\right) \odot \delta_ {C}, \quad \text {c o l u m n - w i s e b r o a d c a s t} \tag {91}
$$

$$
\delta_ {\tilde {C} ^ {\prime}, \boldsymbol {b}} ^ {\top} = 1 \delta_ {\boldsymbol {b}} ^ {\top}, \quad \text {c o l u m n - w i s e b r o a d c a s t} \tag {92}
$$

$$
\delta_ {\tilde {C} ^ {\prime}} = \delta_ {\tilde {C} ^ {\prime}, C} + \delta_ {\tilde {C} ^ {\prime}, B} \tag {93}
$$

$$
\delta_ {\mathbf {D} ^ {\prime}} = \frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\sqrt {d}} \odot \delta_ {\tilde {C} ^ {\prime}} \tag {94}
$$

$$
\delta_ {\widetilde {\mathbf {D}}} = \exp (\widetilde {\mathbf {D}} - \boldsymbol {m}) \odot \delta_ {\mathbf {D} ^ {\prime}} = \mathbf {D} ^ {\prime} \odot \delta_ {\mathbf {D} ^ {\prime}} \tag {95}
$$

We do not compute the gradients for  $m$  as they cancel out (see the proof in the recurrent sLSTM).

With these intermediate gradients the gradients for the logarithmic forget gate matrix  $\delta_{\overline{\mathbf{F}}}\in \mathbb{R}^{T\times T}$ , the input gate matrix  $\delta_{\mathbf{I}}\in \mathbb{R}^{T\times T}$ , and the queries, keys and values  $\delta_{Q},\delta_{K},\delta_{V}\in \mathbb{R}^{T\times d}$  are given by

$$
\delta_ {\overline {{\mathbf {F}}}} = \delta_ {\widetilde {\mathbf {D}}} \tag {96}
$$

$$
\delta_ {\mathbf {I}} = \delta_ {\widetilde {\mathbf {D}}} \tag {97}
$$

$$
\delta_ {\boldsymbol {Q}} = \left(\mathbf {D} ^ {\prime} \odot \delta_ {\tilde {\mathcal {C}} ^ {\prime}}\right) \frac {\boldsymbol {K}}{\sqrt {d}} \tag {98}
$$

$$
\delta_ {\boldsymbol {K}} = \left(\mathbf {D} ^ {\prime} \odot \delta_ {\tilde {C} ^ {\prime}}\right) ^ {\top} \frac {\boldsymbol {Q}}{\sqrt {d}} \tag {99}
$$

$$
\delta_ {\boldsymbol {V}} = \boldsymbol {C} ^ {\top} \delta_ {\tilde {\mathbf {H}}} \tag {100}
$$

Having computed the gradients for the logarithmic forget gate matrix  $\delta_{\overline{\mathbf{F}}}$ , we can compute the gradients for the forget gate pre-activations  $\delta_{\tilde{\mathbf{f}}} = \left[\delta_{\tilde{\mathbf{f}}_1},\delta_{\tilde{\mathbf{f}}_2},\dots,\delta_{\tilde{\mathbf{f}}_T}\right]^\top \in \mathbb{R}^T$ .

Recall the logarithmic forget gate matrix  $\overline{\mathbf{F}} = \log \mathbf{F}$  is computed by

$$
\overline {{\mathbf {F}}} _ {i j} = \log \mathbf {F} _ {i j} = \left\{ \begin{array}{l l} - \infty & \text {f o r} i <   j \\ 0 & \text {f o r} i = j \\ \sum_ {k = j + 1} ^ {i} \underbrace {\log \sigma (\tilde {\mathrm {f}} _ {k})} _ {=: \bar {\mathrm {f}} _ {k}} = \sum_ {k = j + 1} ^ {i} \bar {\mathrm {f}} _ {k} & \text {f o r} i > j \end{array} . \right. \tag {101}
$$

With the substitution  $\overline{\mathbf{f}} = \log \sigma (\widetilde{\mathbf{f}})$  we compute the gradients for the logarithmic forget gate activations  $\delta_{\overline{\mathbf{f}}} = \left[\delta_{\overline{\mathbf{f}}_1},\delta_{\overline{\mathbf{f}}_2},\dots,\delta_{\overline{\mathbf{f}}_T}\right]^\top \in \mathbb{R}^T$  as

$$
\delta_ {\overline {{\mathbf {f}}} _ {k}} = \sum_ {j = 1} ^ {k - 1} \sum_ {i = k} ^ {T} \left(\delta_ {\overline {{\mathbf {F}}}}\right) _ {i j}, \tag {102}
$$

$$
\delta_ {\tilde {\mathrm {f}} _ {k}} = \sigma \left(- \tilde {\mathrm {f}} _ {k}\right) \cdot \delta_ {\bar {\mathrm {f}} _ {k}}, \tag {103}
$$

where the last equation makes use of the following:

$$
\begin{array}{l} \frac {\mathrm {d}}{\mathrm {d} x} (\log \sigma (x)) = - (1 + \exp (- x)) ^ {- 1} \cdot \exp (- x) \cdot (- 1) \\ = \frac {\exp (- x)}{1 + \exp (- x)} = \frac {1}{1 + \exp (x)} \tag {104} \\ = \sigma (- x) \\ \end{array}
$$

Finally, we compute the input gate pre-activations' gradients  $\delta_{\tilde{\mathbf{i}}} = \left[\delta_{\tilde{\mathbf{i}}_1},\delta_{\tilde{\mathbf{i}}_2},\dots,\delta_{\tilde{\mathbf{i}}_S}\right]^\top \in \mathbb{R}^T$  as the column-wise sum over the rows of the input gate matrix  $\delta_{\mathbf{I}}$ :

$$
\delta_ {\bar {1} _ {k}} = \sum_ {i = k} ^ {T} \left(\delta_ {\mathbf {I}}\right) _ {i k} \tag {105}
$$

This completes the backward pass of the parallel mLSTM for a full input sequence  $\mathbf{X} \in \mathbb{R}^{T \times d}$ .

# A.4 Detailed Block Structure

Figure 10: Schematic representation of an sLSTM Block – post up-projection: Embedded in a pre-LayerNorm residual structure, the input is optionally passed through a causal convolution of window size 4 that includes a Swish activation for input and forget gates. Then, for all input, forget and output gates i, f, o, and the cell update z the input is fed through a block-diagonal linear layer with four diagonal blocks or “heads”. These diagonal blocks coincide with the recurrent gate pre-activations from the last hidden state, which corresponds to an sLSTM with four heads depicted with the circular arrows. The resulting hidden state goes through a GroupNorm layer (Wu & He, 2018) – a head-wise LayerNorm for each of the four heads. Finally, the output is up- and down-projected using a gated MLP, with GeLU activation function and projection factor  $\frac{4}{3}$  to match parameters.

Figure 11: Schematic representation of an mLSTM block - pre up-projection: Embedded in a pre-LayerNorm residual structure, the input is up-projected first with projection factor 2, once for an externalized output gate and once as input for the mLSTM cells. The mLSTM cell input is dimension-wise causally convoluted (kernel size 4), before entering a learnable skip connection. We obtain input  $q$  and  $k$  via block-diagonal projection matrices of block size 4. The values  $v$  are fed directly, skipping the convolution part. After the mLSTM sequence mixing, outputs are normalized via GroupNorm (Wu & He, 2018) - a head-wise layer norm for each of the four heads. Finally, the learnable skip input is added and the result is gated component-wise with the external output gate. The output is down-projected.

# B Experiments

Training Setup. For all experiments, we use Python  $^{1}$  3.11 with PyTorch  $2.2.0^{2}$ , and CUDA  $12.1^{3}$  on NVIDIA A100 GPUs. We developed and trained all our models and baselines over the course of three months on a cluster with 128 nodes of eight NVIDIA A100 GPUs each. More than  $95\%$  of this compute were used for the Language Modeling experiments in Sections 4.2 and 4.3.

Nearest Neighbor Search Task. For this auxiliary task, we use randomly sampled feature vectors of dimension 2 and unit norm. The attached value is a uniformly distributed random number from [0, 1], leading to inputs vectors of dimension 3. The first feature vector serves as search key, with the first value being ignored. Then the model has to predict the value of the nearest neighbor so far in the sequence. We train on 8192 sequences of context length up to 64 (uniformly sampled) and validate on 8192 different samples. All models have two blocks and embedding dimension 128. We use a dropout of 0.1,  $10\%$  linear warm-up steps and cosine decay to 1e-7 for 100k total training steps. We sweep over learning rates 1e-4, 1e-3, 1e-2, 1e-1 and 5 seeds each. The reported values in Figure 2 are mean values for the best learning rate and  $99\%$  confidence intervals. Note that LSTM requires very high learning rates, whereas Transformers (Llama) perform best at the smallest learning rate. The xLSTM[0:1] reaches similar performance across all learning rates.

Wikitext-103 Rare Token Prediction. For this exemplary experiment on rare token prediction, we trained 125M-sized models on Wikitext-103 (Merit et al., 2017). All models have an embedding dimension of 768 in a post up-projection structure of 12 residual blocks. The Transformer model (Llama) uses Multi-Head Attention, for what is called LSTM the Multi-Head Attention is replaced by an LSTM and the xLSTM[1:0] contains mLSTM layers with matrix memory. Models were trained with maximum learning rate 1e-3, 4k steps linear warm-up and cosine decay for in total 50k steps, using a batch size of 256 and context length of 512. We use the validation perplexity as a stopping criterion and evaluate on the test set.

# B.1 Synthetic Tasks and Long Range Arena

# B.1.1 Test of xLSTM's Exponential Gating with Memory Mixing.

We evaluate xLSTM on a suite of formal language tasks to test its exponential gating and memory mixing mechanism.

Formal languages provide a framework to probe the generalization capabilities of models. They allow to specifically test different expressivity levels, e.g. along the Chomsky hierarchy. Typical language model architectures do not necessarily fit perfectly in these hierarchies (Delétang et al., 2023) — nevertheless these languages allow to illustrate differences in generalization expressivity between different architectures. Our evaluation tasks are heavily based on the work of Delétang et al. (2023).

Experiment Setup. The different formal language tasks in the experiment (see individual tasks description below) encompass different levels of the Chomsky hierarchy as well as additional counting and memory-focused tasks. We use different lengths per sample, which allows us to validate in a length extrapolation setting. We train on a varying task length up to 40. The evaluation is done for task lengths between 40 and 256 as we are only interested in the "task generalization capabilities" of the models.

In all experiments, we use two blocks (or layers for the pure LSTM) for all models. We compare Llama, Mamba, Retention, Hyena, RWKV-4, RWKV-5, RWKV-6, LSTM, xLSTM[0:1], xLSTM[1:0] and xLSTM[1:1]. The sLSTM block is used without a convolution and with normal weight initialization. LSTM (Block) refers to an architecture where a vanilla LSTM is used instead of self-attention inside a Transformer block.

All models are trained with 3 different learning rates (1e-2, 1e-3, 1e-4), each with two seeds. Batch size is 256 — cosine annealing (min lr: 1e-5) with  $10\%$  warm-up steps is applied. We use AdamW (Loshchilov & Hutter, 2019)  $(\beta_{1} = 0.9, \beta_{2} = 0.99)$  and a weight decay of 0.1 for training.

<table><tr><td rowspan="2"></td><td rowspan="2">Context Sensitive Odds First</td><td colspan="2">Deterministic Context Free</td><td rowspan="2">Repetition</td><td rowspan="2">Set</td></tr><tr><td>Reverse String</td><td>Stack Manipulation</td></tr><tr><td>Llama-</td><td>0.07 ± 0.0</td><td>0.06 ± 0.0</td><td>0.11 ± 0.01</td><td>0.08 ± 0.0</td><td>0.04 ± 0.0</td></tr><tr><td>Retention-</td><td>0.03 ± 0.0</td><td>0.11 ± 0.0</td><td>0.03 ± 0.0</td><td>0.02 ± 0.0</td><td>0.02 ± 0.0</td></tr><tr><td>RWKV-4-</td><td>0.08 ± 0.0</td><td>0.12 ± 0.01</td><td>0.2 ± 0.0</td><td>0.1 ± 0.0</td><td>0.1 ± 0.02</td></tr><tr><td>Hyena-</td><td>0.04 ± 0.0</td><td>0.15 ± 0.0</td><td>0.07 ± 0.0</td><td>0.07 ± 0.0</td><td>0.03 ± 0.0</td></tr><tr><td>RWKV-5-</td><td>0.08 ± 0.01</td><td>0.09 ± 0.01</td><td>0.16 ± 0.0</td><td>0.16 ± 0.0</td><td>0.13 ± 0.01</td></tr><tr><td>RWKV-6-</td><td>0.13 ± 0.01</td><td>0.11 ± 0.0</td><td>0.23 ± 0.01</td><td>0.15 ± 0.01</td><td>0.19 ± 0.01</td></tr><tr><td>xLSTM[0:1]</td><td>0.09 ± 0.01</td><td>0.14 ± 0.03</td><td>0.13 ± 0.01</td><td>0.09 ± 0.01</td><td>0.17 ± 0.01</td></tr><tr><td>Mamba-</td><td>0.08 ± 0.01</td><td>0.13 ± 0.02</td><td>0.21 ± 0.0</td><td>0.15 ± 0.01</td><td>0.12 ± 0.0</td></tr><tr><td>LSTM (Block)</td><td>0.08 ± 0.01</td><td>0.17 ± 0.02</td><td>0.25 ± 0.02</td><td>0.15 ± 0.01</td><td>0.18 ± 0.01</td></tr><tr><td>xLSTM[0:1]</td><td>0.09 ± 0.01</td><td>0.14 ± 0.03</td><td>0.13 ± 0.01</td><td>0.09 ± 0.01</td><td>0.17 ± 0.01</td></tr><tr><td>xLSTM[1:0]</td><td>0.15 ± 0.03</td><td>0.22 ± 0.02</td><td>0.25 ± 0.03</td><td>0.28 ± 0.0</td><td>0.17 ± 0.01</td></tr><tr><td>xLSTM[1:1]</td><td>0.08 ± 0.0</td><td>0.2 ± 0.01</td><td>0.17 ± 0.0</td><td>0.09 ± 0.0</td><td>0.15 ± 0.03</td></tr></table>

Figure 12: Supplementary results given by scaled accuracy of different models at solving formal language tasks. Tasks are grouped by the Chomsky hierarchy.

In each experiment we train for 100k steps — the samples are generated randomly, however, all experiments are trained and evaluated on the same samples.

Additional Formal Language Results. Figure 12 showcases supplementary results on formal language task, detailing tasks where no model attained a minimum scaled accuracy of 0.3. Although no model achieves proper extrapolation of the task to a larger context length, xLSTM performs best among the evaluated models.

Individual Task Description. The majority of tasks are based on Delétang et al. (2023). We provide the vocabulary size  $|V|$  and the random accuracy  $s_{rand}$  (for accuracy scaling), used in the evaluation. As we evaluate different task lengths each task has a padding token which is used to pad the sequence to the given context length. In Listing 1 there is an example for each task.

- Bucket Sort Given a string of tokens of a sorted alphabet, compute the sorted string.  $|V| = 11$ $s_{\mathrm{rand}} = \frac{1}{|V| - 1}$  
- Cycle Nav Given a string of "movement tokens"  $(+1, -1, \text{STAY})$  compute the end position of the agent with start position 0. The position must be computed modulo the maximum position.

$$
| V | = 9 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 4}
$$

- Even Pairs Given a binary string of  $a$  and  $b$  tokens, compute whether the number of  $ab$  and  $ba$  is even. This task can be solved by checking if the first and last token of the string are equal.

$$
| V | = 3 \quad s _ {\text {r a n d}} = 0. 5
$$

- Majority Given a string of tokens, compute the token that occurred most often in the sequence.

$$
| V | = 6 4 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 1}
$$

- Majority Count Given a string of tokens of an ordered alphabet. Compute the count of the token that occurred most often in the sequence. If the count exceeds the vocab size, the highest vocab token should be outputted.

$$
| V | = 6 4 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 1}
$$

- Missing Duplicate Given a string of tokens. The string is repeated but one of the tokens is masked in the repetition. Output the token that is masked.

$$
| V | = 1 1 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 2}
$$

- Mod Arithmetic (w/o Brackets) Calculate the result — modulo the max number — of the arithmetic operations in the context. The maximum number is the vocabulary size minus the number of special tokens  $(+, -, *, =, [\mathrm{PAD}])$ .

$$
| V | = 1 0 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 5}
$$

- Mod Arithmetic (w Brackets) Calculate the result — modulo the maximum number — of the arithmetic operations in the context. The maximum number is vocabulary size minus the number of special tokens \((+, -, *, =, (\), [PAD])\).

$$
| V | = 1 2 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 7}
$$

- Odds First An string of tokens  $t_1, t_2, t_3, \ldots, t_n$  is given. Output all tokens with and odd index  $(t_1, t_3, \ldots)$  then the token with an even index  $(t_2, t_4, \ldots)$ . Apart from that keep the ordering of the initial string.

$$
| V | = 1 2 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 2}
$$

- Parity Given a binary string of  $a$  and  $b$  tokens,compute if the number of  $b$ 's is even. If the number is even output  $a$  otherwise  $b$ . This is equivalent to sequentially calculating the half-adder sum.

$$
| V | = 3 \quad s _ {\text {r a n d}} = 0. 5
$$

- Repetition Given a string of tokens — repeat it.

$$
| V | = 1 2 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 2}
$$

- Reverse String Given a string of tokens — repeat it in reverse order.

$$
| V | = 1 2 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 2}
$$

- **Stack Manipulation** An initial stack content is given, followed by a sequence of push and pop operations. Compute the stack content after the operations

$$
\left| V \right| = 1 1 \quad s _ {\text {r a n d}} = \frac {1}{\lfloor \frac {| V | - 3}{2} \rfloor}
$$

- Set Given a string of tokens, compute the ordered set of the tokens. Keep the ordering so that tokens that occurred first are also outputted first.

$$
| V | = 1 2 8 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 2}
$$

- Solve Equation Given is an equation with the operators  $\{+, -, *, =, ()\}$ , number, and an unknown variable  $x$ . Compute the value of the variable modulo the max number. The maximum number is vocabulary size minus the number of special tokens  $(+, -, *, =, (), [\mathrm{PAD}], [\mathrm{ACT}])$ .

$$
| V | = 1 4 \quad s _ {\text {r a n d}} = \frac {1}{| V | - 9}
$$

```txt
Bucket Sort  
Sequence: 1 4 8 6 1 1 1 4 6 8  
Cycle Nav  
Sequence: STAY +1 -1 +1 STAY +1 +1 +1 -1 P3  
Even Pairs  
Sequence: a b b a a b a a a  
Majority  
Sequence: 1 7 6 4 3 8 1 7 2 1  
Majority Count  
Sequence: 1 7 6 4 4 8 1 7 2 2  
Missing Duplicate  
Sequence: 4 8 6 2 5 4 8 6 2 [MIS] 5  
Mod Arithmetic (w/o Braces)  
Sequence: 0 - 4 + 0 - 2 = 4 [PAD]  
Mod Arithmetic (w Braces)  
Sequence: ((2) * -2) - (-4 -2)) = 2  
Odds First  
Sequence: 2 7 3 2 6 9 [ACT] 2 3 6 7 2 9  
Parity:  
Sequence: a b b a a b a b  
Repetition  
Sequence: 2 4 8 6 2 [ACT] 2 4 8 6 2  
Reverse String  
Sequence: 2 4 8 6 2 [ACT] 2 6 8 4 2  
Stack Manipulation  
Sequence: ST1 ST1 ST3 POP POP PS3 PS3 [ACT] ST1 ST3 ST3  
Set  
Sequence: 8 6 6 3 5 4 5 3 [ACT] 8 6 3 5 4  
Solve Equation:  
Sequence: ((2 +0) + -x) - (1)) = 2 [ACT] 2
```

Listing 1: Examples of the formal language tasks. Red tokens are evaluated for loss and accuracy metrics, but are padded for the input. The tokens are illustrated in a way that allows easy semantic interpretation for the given task — hence, some tokens are represented by multiple characters.

# B.1.2 Test of xLSTM's Memory Capacities on Associative Recall Tasks.

We test the memory capacity of xLSTM with the Multi-Query Associative Recall task proposed by Arora et al. (2023). Figure 13 illustrates the basic task setup.

Why Multi-Query Associative Recall for Memory Tests of LLM Architectures. Associative Recall (AR), the ability to retrieve a specific value (information) associated with a given key (information), constitutes a key capability for LLM to perform well (Poli et al., 2024; Arora et al., 2023; Olsson et al., 2022). Especially its quality of in-context learning seems to be strongly connected to this capability (Olsson et al., 2022). Arora et al. (2023) attribute performance gaps between early non-Transformer and Transformer language models specifically to performance gaps in associative recall. They argue that prior AR evaluations fall short of capturing these differences and propose MQAR, which can show the AR performance differences that translate to performance differences in language modeling performance. Hence, MQAR is especially suitable to analyze the memory capacity of LLM. Transformer (e.g. Llama) models can be seen as the gold standard for this task as their memory is exponential in the coding dimension (Ramsauer et al., 2021).

Experiment Setup. There are two relevant variables that determine different experimental setups. (1) Context Length (CL): Length of the sequence of one sample — this influences the distances between the key-value definition and the recall. (2) Number Key-Value Pairs (KV): Influences how many key-value pairs the model needs to keep track of. The vocabulary size is always 8192.

In all experiments, we use two blocks (or layers for the pure LSTM) for all models. LSTM (Block) model refers to an architecture where a vanilla LSTM is used instead of self-attention inside a Transformer block.

For each task setup, we train each model with 4 different learning rates (batch size  $>24$ : {1e-2, 2.15e-3, 4.6e-4, 1e-4}, batch size 24: {1e-3, 2.2e-4, 5e-5, 1e-5}). The batch size (BS) changes depending on the context length (CL) (CL=64/128: BS=512; CL=256: BS=256; CL=756: BS=128; CL=1024: BS=96; CL=2048: BS=24). We vary the embedding dimension (Model Dim) between different experiments - different numbers of heads are used accordingly. For each experiment, we generate 100,000 training samples (validation: 3,000 samples) and train for 64 epochs. We apply cosine annealing (min lr: 1e-4 and 1e-5) with  $10\%$  warm-up steps. We use AdamW (Loshchilov & Hutter, 2019) and a weight decay of 0.1 for training.

We conduct three different experiments:

- MQAR-Experiment 1 evaluates, in the same fashion as Arora et al. (2023), a variety of models (Llama, Mamba, Mamba (noWT) - i.e. without weight tying, Retention, Hyena, H3, RWKV-4, RWKV-5, RWKV-6, LSTM, LSTM (Block), xLSTM[0:1], xLSTM[1:0] and xLSTM[1:1]) on increasing task difficulty by increasing the context length and number of key-value pairs simultaneously. We benchmark three parameter settings: CL,KV={(64,4),(128,8),(256,16)}.  
- MQAR-Experiment 2 increases the task difficulty notably and goes beyond previous evaluations on this task. We individually scale the context length (CL={756, 1024, 2048}) and the key-value pairs (KV={48, 96, 256}) and evaluate all combinations. This experiment especially probes the memory capacity because the number of key-value pairs is high. To reduce the computational burden we only evaluate models that perform flawlessly in Experiment 1 — additionally we evaluate Transformer only in the hardest setting (CL=2048) as sanity check, because no performance decrease is expected.  
- MQAR-Experiment 3 analyzes whether the AR capability learned on a certain context length extrapolates to bigger context lengths. For each KV setting of Experiment 2, we use the models (we select the 3 biggest model dimensions) trained on  $\mathrm{CL} = 2048$  and evaluate bigger context lengths ( $\mathrm{CL} = \{4096, 6144, 8192\}$ ).

Extended Results. The result of Experiment 1 can be found in Figure 14. In accordance to the results of Arora et al. (2023) H3, Hyena, RWKV-4 fail to solve the task with a smaller model dimension. In contrast, xLSTM[1:1], xLSTM[1:0], Mamba, RWKV-5 and RWKV-6 are able to solve these settings for all model dimensions. The comparison of xLSTM[0:1] with both original LSTM variants indicates that the exponential gating mechanism improves the AR capabilities of the model. However, both fall short because of the reduced memory capacity compared to xLSTM[1:1] and xLSTM[1:0].

The results of Experiment 2 are presented in Figure 15. Scaling the context length has a low impact on the performance of the models. However, while xLSTM[1:1] and xLSTM[1:0] show no clear decay, both RWKV variants slightly, but consistently lose performance with increasing context lengths. The varying number of key-value pairs, which mainly probes the memory capacity of the non-Transformer models, has a more notable impact across all models. RWKV-5 seems to outperform RWKV-6. The latter fails to learn the task at all in some KV=256 settings. Overall xLSTM[1:1] is the best-performing non-Transformer model — suggesting that it provides enhanced memory capacity, also in long contexts.

Figure 16 shows the extrapolation results from Experiment 3. For xLSTM[1:1], xLSTM[1:0], and Mamba the model performance does not change in the extrapolation setting. The RWKV models (especially RWKV5) degrade slightly with increasing context length. xLSTM[1:1] performs best, as it maintains its superior performance of Experiment 2.

Target


Input


$$
\mathrm {K V} = 4 / \mathrm {C L} = 1 8
$$

Figure 13: Illustration of the MQAR task. Color pairs represent key-value pairs (keys have darker shade). The first part of the sequence defines the key-value pairs for the respective sample. After that, the keys appear randomly according to a power law distribution  ${}^{4}$  . Grey tokens in the input sequence represent a zero token. The "target" sequence contains the value after the respective key appearance - the rest of the tokens are ignored for the accuracy and loss calculation. The model must predict the value tokens given the respective key.

# B.1.3 Test of xLSTM's Long Range Capabilities on the Long Range Arena.

We assess the performance of xLSTM across tasks in the Long Range Arena benchmark (Tay et al., 2021), examining its ability to effectively handle longer context lengths and diverse data types.

Our experiments on Long Range Arena benchmark are composed of five tasks:

- Retrieval: The task is to predict if two documents have a citation link. The dataset of text documents is derived from the ACL Anthology Network (Radev et al., 2009).  
- ListOps: This is a set of modular arithmetic tasks including brackets and lists of numbers, using the operations MIN, MAX, MEDIAN and SUMMOD (modular sum). A particular example is: [MAX 4 3 [MIN 2 3] 1 0 [MEDIAN 1 5 8 9, 2]]  $\rightarrow$  5  
- Image: This task is based on a version of the CIFAR dataset (Krizhevsky, 2009), where images are transformed to a sequence of pixels and this sequence has to be classified into the usual CIFAR classes. We test both a gray-scale (G-Image) and RGB (RGB-Image) version of this dataset, as Orvieto et al. (2023) uses colored images contrary to the standard setup.  
- Pathfinder: The input for this task is a 32x32 gray-scale image, given as pixel sequence, with two dots and several curved lines on it. The task is to predict if the two dots are connected by any of the lines (Linsley et al., 2018).

We omit the Text classification task (Maas et al., 2011), as the language modeling experiments already test this kind of data, and the Pathfinder-X version of Pathfinder.

Experiment Setup. The architectures that are tested in this experiment comprise Llama, Mamba, LSTM, RWKV-4, and xLSTM. LSTM (Block) refers to an architecture where a vanilla LSTM is used inside a post up-projection block (like Transformer with attention replaced by LSTM). For xLSTM we choose the best performing of xLSTM[0:1] or xLSTM[1:0] on the validation set, specifically the former for the Image tasks and the latter for all other ones.

We use the hyperparameter settings of the S5 model (Smith et al., 2022) and Linear Recurrent Unit model (Orvieto et al., 2023), with additional hyperparameter search on learning rates and schedulers for all models. We use two different schedulers: Linear Warm-up Cosine Annealing and Linear Warm-up Cosine Annealing with Restarts. Both learning rate schedulers were evaluated with learning rates of 1e-3, 6e-4 and 1e-4. For the second scheduler, the number of restarts  $(R)$  is set to 3. The model hyperparameters for each dataset are displayed in Table 5.

Results. Table 6 shows the result of experiments on the Long Range Arena benchmark. xLSTM demonstrates consistent strong performance on all of the tasks, suggesting that the proposed architecture is remarkably efficient in handling different aspects of long context problems.

Figure 14: Result of MQAR-Experiment 1. The columns show different task settings (context length and key-value pairs). The rows group related models for better clarity. The  $x$ -axis gives the model size and the  $y$ -axis the validation accuracy.

Figure 15: Result of MQAR-Experiment 2. The columns and rows correspond to different numbers of key-value pairs and the context length respectively. The  $x$ -axis gives the model size and the  $y$ -axis the validation accuracy.

<table><tr><td>Task</td><td>#Blocks</td><td>Embedding 
Dim</td><td>Batch 
Size</td><td>Training 
Steps</td></tr><tr><td>Retrieval</td><td>6</td><td>128</td><td>64</td><td>100k</td></tr><tr><td>ListOps</td><td>8</td><td>128</td><td>32</td><td>80k</td></tr><tr><td>Pathfinder</td><td>6</td><td>192</td><td>64</td><td>500k</td></tr><tr><td>G-Image</td><td>6</td><td>512</td><td>64</td><td>180k</td></tr><tr><td>RGB-Image</td><td>6</td><td>512</td><td>64</td><td>180k</td></tr></table>

Table 5: Long Range Arena model hyperparameters. These are the model hyperparameters used in each of the Long Range Arena tasks. For each model we used the best learning rate and the better of the two learning rate schedulers.

Figure 16: Result of MQAR-Experiment 3 (Extrapolation). All evaluated models were trained on context length 2048 and the number of key-value pairs given by the columns of the plot. The rows show the different context lengths used in the evaluation. The  $x$ -axis gives the model size and the  $y$ -axis the validation accuracy.

<table><tr><td></td><td>Retrieval acc ↑</td><td>ListOps acc ↑</td><td>Pathfinder acc ↑</td><td>G-Image acc ↑</td><td>RGB-Image acc ↑</td><td>Ranking acc ↑</td></tr><tr><td>Random Baseline</td><td>0.500</td><td>0.100</td><td>0.500</td><td>0.100</td><td>0.100</td><td></td></tr><tr><td>Llama</td><td>0.845</td><td>0.379</td><td>0.887</td><td>0.541</td><td>0.629</td><td>5.2</td></tr><tr><td>Mamba</td><td>0.902</td><td>0.325</td><td>0.992</td><td>0.689</td><td>0.765</td><td>2.2</td></tr><tr><td>RWKV-4</td><td>0.898</td><td>0.389</td><td>0.914</td><td>0.691</td><td>0.757</td><td>3.0</td></tr><tr><td>LSTM</td><td>X</td><td>0.275</td><td>X</td><td>0.675</td><td>0.718</td><td>5.4</td></tr><tr><td>LSTM (Block)</td><td>0.880</td><td>0.495</td><td>X</td><td>0.690</td><td>0.756</td><td>3.4</td></tr><tr><td>xLSTM</td><td>0.906</td><td>0.411</td><td>0.919</td><td>0.695</td><td>0.761</td><td>1.6</td></tr></table>

Table 6: Long Range Arena test accuracy. Bold highlights the best performing model, underlined the second best. X denotes models that fail to outperform random baselines. xLSTM is the best of xLSTM[1:0], xLSTM[0:1] based on validation dataset accuracy.

# B.2 Method Comparison and Ablation Study on SlimPajama (15B)

General Training Procedure. We tokenize our datasets using the HuggingFace GPT-2 tokenizer (Radford et al., 2019; Brown et al., 2020) and use this tokenizer for all models in this paper. In general, we try to follow Brown et al. (2020) for the general training setup, i.e. we choose context length 2048 and batch sizes 256 or 512 for our models. We use the AdamW (Loshchilov & Hutter, 2019) optimizer with beta parameters  $(\beta_{1},\beta_{2}) = (0.9,0.95)$  and an epsilon parameter of 1e-5. As learning rate scheduler we use a linear warm-up with 750 steps and cosine decay to  $10\%$  of the peak learning rate. We apply a weight decay of 0.1 to all our models and always exclude the token embedding matrix from weight decay. If not specified otherwise, we do not tie the weights of the token embedding and the language model head. For parallelization, we use PyTorch FSDP in SHARD_GRAD_0P mode with mixed precision in bfloat16, where applicable. For small models we use NO_SHARD. We keep the weights in float32 and reduce the gradients across GPUs in float32. We use torch.compile to speed up models, except for Transformer models as their training curves did not match the non-compiled versions. For xLSTM[7:1], we use positions [3, 5, 7, 40, 42, 44] for sLSTM-based blocks, except for the 125M size, where we use [3, 20] (this is actually a [11:1] ratio). We do not use any positional encoding for our xLSTM models.

Details on Comparison to Other Methods. For the model comparison on 15B training tokens of SlimPajama we train all models with context length 2048 and batch size 256. We use a peak learning rate of 1e-3 for all models for comparability. The learning rate decays over 30k training steps. The models are compared after one epoch at training step 28170. As model implementations we use the original repositories' code for Mamba (Gu & Dao, 2023)  $^{6}$ , RWKV-5, RWKV-6 (Peng et al., 2024)  $^{7}$ . For RWKV-4 we use a cleaned and validated re-implementation based on the original repo and kernels (Peng et al., 2023). In our RWKV-4 implementation we enable weight decay on all parameters except biases, the token embedding weight and all LayerNorm weights. For HGRN (Qin et al., 2023), GLA (Yang et al., 2023), HGRN2 (Qin et al., 2024) we use the a re-implementation flash-linear-attention (Yang & Zhang, 2024) by the authors of GLA (Yang et al., 2023; Yang & Zhang, 2024)  $^{8}$ . For GPT-3 and Llama-like Transformers, we use our own implementations based on PyTorch. Note that for all xLSTMs, Transformers, Mamba and RWKV-4, we use Mixed Precision training with bffloat16 and weights in float32 precision. Following the general training procedure we use torch.compile for all models, except for Transformers and models using the flash-linear-attention library because of compilation problems.

As RWKV-6 performs worse than RWKV-5, we also train a model with peak learning rate 4e-4, as reported in the original repository for 350M parameter models. This model reaches a perplexity of 16.38, worse than the 15.03 for the standard peak learning rate 1e-3 as reported in Table 1. Similarly, we tested the repository learning rates for other model sizes and all performed worse than the ones we also use for xLSTM (see Table 7).

Details on Training Precision for Baselines. For models from flash-linear-attention and RWKV-5/6 models we found that PyTorch automatic mixed precision training did not work, but casting the model weights to float32 initially with FSDP parameter precision bfloat16 led to a working configuration. In this setting, models perform better than in full bfloat16 training, where the weights are casted to bfloat16 initially as well. Full float32 training did not work because of the custom kernels that require bfloat16.

General Details on Ablation Studies. We follow our general training procedure and train all models with context length 2048, batch size 256 and peak learning rate 1e-3. We report perplexity values on the validation set.

<table><tr><td></td><td>Model</td><td>EmbeddingDim</td><td>#Blocks</td><td>#Heads/HeadDim</td><td>#Params M</td><td>Peak LR (15B)</td><td>Peak LR (300B)</td></tr><tr><td rowspan="7">125M</td><td>RWKV-5</td><td>768</td><td>12</td><td>-</td><td>176.5</td><td>3e-3</td><td>-</td></tr><tr><td>RWKV-6</td><td>768</td><td>12</td><td>-</td><td>173.6</td><td>3e-3</td><td>-</td></tr><tr><td>HGRN2</td><td>768</td><td>12</td><td>-</td><td>162.2</td><td>3e-3</td><td>-</td></tr><tr><td>RWKV-4</td><td>768</td><td>12</td><td>-</td><td>169.4</td><td>3e-3</td><td>6e-4</td></tr><tr><td>Llama</td><td>768</td><td>12</td><td>12 / 64</td><td>162.2</td><td>3e-3</td><td>3e-3</td></tr><tr><td>Mamba</td><td>768</td><td>24</td><td>-</td><td>167.8</td><td>3e-3</td><td>3e-3</td></tr><tr><td>xLSTM</td><td>768</td><td>24</td><td>4 / 384</td><td>163.8</td><td>3e-3</td><td>1.5e-3</td></tr><tr><td rowspan="7">350M</td><td>RKWV-5</td><td>1024</td><td>24</td><td>-</td><td>455.7</td><td>1e-3</td><td>-</td></tr><tr><td>RWKV-6</td><td>1024</td><td>24</td><td>-</td><td>441.6</td><td>1e-3</td><td>-</td></tr><tr><td>HGRN2</td><td>1024</td><td>24</td><td>-</td><td>411.4</td><td>1e-3</td><td>-</td></tr><tr><td>RWKV-4</td><td>1024</td><td>24</td><td>-</td><td>430.5</td><td>1e-3</td><td>4e-4</td></tr><tr><td>Llama</td><td>1024</td><td>24</td><td>16 / 64</td><td>406.6</td><td>1.5e-3</td><td>1.5e-3</td></tr><tr><td>Mamba</td><td>1024</td><td>48</td><td>-</td><td>423.1</td><td>1.5e-3</td><td>1.5e-3</td></tr><tr><td>xLSTM</td><td>1024</td><td>48</td><td>4 / 512</td><td>409.3</td><td>1e-3</td><td>7.5e-4</td></tr><tr><td rowspan="7">760M</td><td>RWKV-5</td><td>1536</td><td>24</td><td>-</td><td>947.8</td><td>9e-4</td><td>-</td></tr><tr><td>RWKV-6</td><td>1536</td><td>24</td><td>-</td><td>907.7</td><td>9e-4</td><td>-</td></tr><tr><td>HGRN2</td><td>1536</td><td>24</td><td>-</td><td>834.2</td><td>9e-4</td><td>-</td></tr><tr><td>RWKV-4</td><td>1536</td><td>24</td><td>-</td><td>891.0</td><td>2e-3</td><td>2.5e-4</td></tr><tr><td>Llama</td><td>1536</td><td>24</td><td>16 / 96</td><td>834.1</td><td>1.25e-3</td><td>1.25e-3</td></tr><tr><td>Mamba</td><td>1536</td><td>48</td><td>-</td><td>870.5</td><td>1.25e-3</td><td>1.25e-3</td></tr><tr><td>xLSTM</td><td>1536</td><td>48</td><td>4 / 768</td><td>840.4</td><td>9e-4</td><td>6.25e-4</td></tr><tr><td rowspan="7">1.3B</td><td>RWKV-5</td><td>2048</td><td>24</td><td>-</td><td>1616.0</td><td>9e-4</td><td>-</td></tr><tr><td>RWKV-6</td><td>2048</td><td>24</td><td>-</td><td>1537.5</td><td>9e-4</td><td>-</td></tr><tr><td>HGRN2</td><td>2048</td><td>24</td><td>-</td><td>1439.4</td><td>9e-4</td><td>-</td></tr><tr><td>RWKV-4</td><td>2048</td><td>24</td><td>-</td><td>1515.2</td><td>1e-3</td><td>2e-4</td></tr><tr><td>Llama</td><td>2048</td><td>24</td><td>16 / 128</td><td>1420.4</td><td>1e-3</td><td>1e-3</td></tr><tr><td>Mamba</td><td>2048</td><td>48</td><td>-</td><td>1475.3</td><td>1e-3</td><td>1e-3</td></tr><tr><td>xLSTM</td><td>2048</td><td>48</td><td>4 / 1024</td><td>1422.6</td><td>9e-4</td><td>5e-4</td></tr><tr><td rowspan="7">2.7B</td><td>RWKV-5</td><td>2048</td><td>24</td><td>-</td><td>3194.7</td><td>8e-4</td><td>-</td></tr><tr><td>RWKV-6</td><td>2048</td><td>24</td><td>-</td><td>3021.9</td><td>8e-4</td><td>-</td></tr><tr><td>HGRN2</td><td>2048</td><td>24</td><td>-</td><td>2795.4</td><td>8e-4</td><td>-</td></tr><tr><td>RWKV-4</td><td>2560</td><td>32</td><td>-</td><td>2984.8</td><td>8e-4</td><td>-</td></tr><tr><td>Llama</td><td>2560</td><td>32</td><td>32 / 80</td><td>2779.5</td><td>8e-4</td><td>-</td></tr><tr><td>Mamba</td><td>2560</td><td>64</td><td>-</td><td>2897.2</td><td>8e-4</td><td>-</td></tr><tr><td>xLSTM</td><td>2560</td><td>64</td><td>4 / 1280</td><td>2788.3</td><td>8e-4</td><td>-</td></tr></table>

Table 7: Peak learning rates and model dimensions for scaling law plots.

Additional Ablation Study on Matrix Memory. As default block configuration we use the mLSTM in the pre up-projection block (see Figure 11) and the sLSTM in the post up-projection block (see Figure 10). In this experiment we study combination of mLSTM with different block variants using the xLSTM[1:0] architecture. We compare the mLSTM in a post up-projection block (see Figure 3 and 10) with  $\mathrm{ReLU}^2$  activation function and non-gated feed-forward network to mLSTM in a pre up-projection block with and without a dimension-wise causal convolution. Table 8 shows that the matrix memory benefits from the pre up-projection block structure, and that the convolution within this block is important.

<table><tr><td>Model</td><td>Details</td><td>#Blocks</td><td>Embedding Dim</td><td>#Params M</td><td>SlimPajama (15B) ppl ↓</td></tr><tr><td rowspan="3">xLSTM[1:0]</td><td>Post Up-Projection Block (ReLU2)</td><td>24</td><td>1024</td><td>430.4</td><td>13.90</td></tr><tr><td>Pre Up-Projection Block, No Convolution</td><td>48</td><td>1024</td><td>408.8</td><td>15.41</td></tr><tr><td>Pre Up-Projection Block, With Convolution</td><td>48</td><td>1024</td><td>409.3</td><td>13.43</td></tr></table>

Table 8: Matrix Memory variants. We study different configurations for the matrix memory. Matrix memory in the pre up-projection block performs best and gives xLSTM[1:0]. Notably, it seems that the dimension-wise causal convolution within the pre up-projection block is important.

Details on new xLSTM Components Ablation Study. In Table 2 (top), we show our modifications to the vanilla LSTM that transform the vanilla LSTM into the xLSTM. We start with a large default PyTorch LSTM with 24 layers and 1536 hidden size. Due to a lack of skip-connections and LayerNorms, vanilla LSTMs of this size are not trainable. We then add skip-connections and pre-LayerNorms before each LSTM layer corresponding to a residual architecture. This enables training for LSTMs at this scale. Replacing every second LSTM layer by a non-gated feed-forward network with GeLU activation function (similar to Vaswani et al.), which corresponds to the post up-projection backbone (see Figure 3) further boosts performance. Adding Exponential Gating to this architecture yields the sLSTM as depicted in Figure 10, with another large performance improvement. Finally, adding the best Matrix Memory variant found in Table 8 by replacing some sLSTM blocks with the mLSTM (see Figure 11) gives xLSTM[7:1] with the best performance.

Details on Gating Technique Ablation Study. In Table 2 (bottom), we investigate the effect of trainable and input-dependent gates for mLSTM. The results show that, in contrast to other methods (Katharopoulos et al., 2020; Sun et al., 2023; Qin et al., 2023; Katsch, 2023; Yang et al., 2023; Qin et al., 2024; Peng et al., 2024), having the gates both learnable and input dependent gives the best results.

Details on Scaling Experiments. We follow our general training procedure (see paragraph above) and train all models, including the 1.3B and 2.7B model sizes, with context length 2048 and batch size 256. We use the peak learning rates from Table 7. For Llama and Mamba we use the learning rates reported by Gu & Dao (2023).

# B.3 xLSTM Large Language Models - SlimPajama300B

General Training Procedure. We use the same general training procedure as in Section B.2 with peak learning rates from Table 7. For Llama and Mamba we use the learning rates reported by Gu & Dao (2023). All models are trained with context length 2048. The 125M, 350M and 760M models are trained with batch size 256 for 600k training steps, whereas the 1.3B models are trained with batch size 512 for 300k training steps. We keep the same learning rate scheduler across all models.

Details on Downstream Evaluation. We use the LM Evaluation Harness from EleutherAI (Sutawika et al., 2023) for evaluating the following tasks that measure common sense reasoning: LAMBADA (OpenAI version in LM Evaluation Harness) (Paperno et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), ARC-challenge, ARC-easy (Clark et al., 2018), WinoGrande (Sakaguchi et al., 2021). This selection of downstream tasks is also used in previous work by Gu & Dao (2023).

Following Gu & Dao (2023), we report accuracy for LAMADA, WinoGrande, PIQA, and ARC-easy, and accuracy normalized by sequence length for HellaSwag and ARC-challenge.

We evaluate all models in full float32, full bffloat16 and bffloat16 Mixed Precision with weights in float32. For each model we select the best value respectively.

Details on PALOMA. We use 16 out of the 18 data sources of the PALOMA dataset (Magnusson et al., 2023). We use C4 (Raffel et al., 2019), MC4-EN (Xue et al., 2021), Wikitext-103 (Merit et al., 2017), PennTreebank (Vadas & Curran, 2011), RedPajama (TogetherComputer, 2023), Falcon Refinedweb (Refined Web) (Penedo et al., 2023), Dolma v1.5 (Soldaini et al., 2023), M2D2 S2ORC, M2D2 Wikipedia (Reid et al., 2022), C4-100-Domains (C4 Domains) (Chronopoulou et al., 2022), Dolma-100-Subreddits (Dolma Subreddits) (Soldaini et al., 2023), Dolma-100-Programming Languages (Dolma Coding) (Soldaini et al., 2023; Kocetkov et al., 2022), TwitterAAE (Blodgett et al., 2016; Liang et al., 2023), Manosphere Corpus (Ribeiro et al., 2021), GAB Corpus (Zannettou et al., 2018), 4CHAN Corpus (Papasavva et al., 2020). We leave out ThePile (Gao et al., 2021) and ICE (Greenbaum & Nelson, 1996) as they are not part of Paloma's Huggingface dataset repository<sup>10</sup>. A detailed description of these datasets can be found in Magnusson et al. (2023, Table 2). All models are evaluated in bfloat16 Mixed Precision.

Results on the data sources TwitterAAE, Manosphere, GAB and 4CHAN are reported in Table 9 and for each individual dataset the results are given in Section C.

<table><tr><td></td><td>Model</td><td>#Params M</td><td>Twitter AAE</td><td>Manosphere</td><td>4CHAN</td><td>GAB</td></tr><tr><td rowspan="5">125M</td><td>RWKV-4</td><td>169.4</td><td>265.80</td><td>39.31</td><td>18.48</td><td>53.89</td></tr><tr><td>Llama</td><td>162.2</td><td>277.93</td><td>32.98</td><td>14.03</td><td>56.45</td></tr><tr><td>Mamba</td><td>167.8</td><td>258.17</td><td>32.14</td><td>14.01</td><td>51.58</td></tr><tr><td>xLSTM[1:0]</td><td>163.8</td><td>244.53</td><td>31.45</td><td>13.27</td><td>51.00</td></tr><tr><td>xLSTM[7:1]</td><td>163.7</td><td>248.51</td><td>30.90</td><td>13.45</td><td>50.25</td></tr><tr><td rowspan="5">350M</td><td>RWKV-4</td><td>430.5</td><td>216.17</td><td>30.25</td><td>13.82</td><td>42.25</td></tr><tr><td>Llama</td><td>406.6</td><td>231.09</td><td>25.90</td><td>11.49</td><td>43.04</td></tr><tr><td>Mamba</td><td>423.1</td><td>202.88</td><td>25.24</td><td>11.60</td><td>40.78</td></tr><tr><td>xLSTM[1:0]</td><td>409.3</td><td>200.61</td><td>24.58</td><td>11.20</td><td>39.83</td></tr><tr><td>xLSTM[7:1]</td><td>408.4</td><td>206.25</td><td>24.73</td><td>11.31</td><td>39.86</td></tr><tr><td rowspan="5">760M</td><td>RWKV-4</td><td>891.0</td><td>195.27</td><td>24.66</td><td>12.00</td><td>35.73</td></tr><tr><td>Llama</td><td>834.1</td><td>205.50</td><td>22.69</td><td>10.40</td><td>37.68</td></tr><tr><td>Mamba</td><td>793.2</td><td>182.74</td><td>22.58</td><td>10.47</td><td>36.25</td></tr><tr><td>xLSTM[1:0]</td><td>840.4</td><td>179.74</td><td>21.66</td><td>10.11</td><td>35.33</td></tr><tr><td>xLSTM[7:1]</td><td>839.7</td><td>180.19</td><td>21.78</td><td>10.22</td><td>34.89</td></tr><tr><td rowspan="5">1.3B</td><td>RWKV-4</td><td>1515.2</td><td>174.87</td><td>23.51</td><td>11.34</td><td>33.18</td></tr><tr><td>Llama</td><td>1420.4</td><td>192.52</td><td>20.67</td><td>9.67</td><td>34.84</td></tr><tr><td>Mamba</td><td>1475.3</td><td>171.38</td><td>20.37</td><td>9.80</td><td>32.01</td></tr><tr><td>xLSTM[1:0]</td><td>1422.6</td><td>166.16</td><td>19.94</td><td>9.64</td><td>31.90</td></tr><tr><td>xLSTM[7:1]</td><td>1420.1</td><td>171.36</td><td>20.28</td><td>9.64</td><td>32.17</td></tr></table>

Table 9: Perplexity values per domain.

In order to evaluate the perplexity values on each data source, we split the text documents into sequences of length 2048, which corresponds to the pre-training context length of all models. For documents longer than 2048 tokens we split each document into non-overlapping input sequences. In this case for the last input sequence, we follow the LM Evaluation Harness and fill up the full 2048 token context window with previous tokens, but compute the perplexity only on the remaining tokens.

We compute the token perplexities per data source in Table 4 as the exponential of the negative loglikelihoods per domain weighted by the number of tokens per domain in that data source as it is defined in Magnusson et al. (2023, Equation 1)

# C Detailed Results on PALOMA Language Model Evaluation

We report the perplexity values on each of the 571 subdomains of PALOMA in Table 10. Note that the aggregated perplexity values in Table 4 are not macro averages of the values shown in Table 10.

Table 10: PPL Evaluations: For the 1.3B sized models trained on 300B SlimPajama tokens, these are the detailed evaluation results on the respective validation datasets.  

<table><tr><td>Dataset</td><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>#Params (M)</td><td>1420</td><td>1475</td><td>1515</td><td>1420</td><td>1423</td></tr><tr><td>4chan_meta sep_val-00000000</td><td>9.58</td><td>9.72</td><td>11.37</td><td>9.53</td><td>9.55</td></tr><tr><td>4chan_meta sep_val-00000001</td><td>9.95</td><td>10.06</td><td>11.57</td><td>9.91</td><td>9.88</td></tr><tr><td>4chan_meta sep_val-00000002</td><td>9.42</td><td>9.53</td><td>11.00</td><td>9.40</td><td>9.38</td></tr><tr><td>4chan_meta sep_val-00000003</td><td>9.78</td><td>9.93</td><td>11.48</td><td>9.77</td><td>9.77</td></tr><tr><td>c4_100dom_val_100 Websites.ign.com</td><td>16.22</td><td>15.75</td><td>17.10</td><td>15.67</td><td>15.43</td></tr><tr><td>c4_100dom_val_10 Websites.eventbrite.com</td><td>12.72</td><td>12.33</td><td>13.33</td><td>12.30</td><td>12.12</td></tr><tr><td>c4_100dom_val_11_link.springer.com</td><td>8.66</td><td>8.54</td><td>9.31</td><td>8.42</td><td>8.33</td></tr><tr><td>c4_100dom_val_12 Websites.chicagotribune.com</td><td>12.09</td><td>11.60</td><td>12.49</td><td>11.55</td><td>11.37</td></tr><tr><td>c4_100dom_val_13 Websites.foxnews.com</td><td>9.59</td><td>9.21</td><td>9.83</td><td>9.16</td><td>9.08</td></tr><tr><td>c4_100dom_val_14 Websites.aljazeera.com</td><td>10.97</td><td>10.61</td><td>11.31</td><td>10.50</td><td>10.40</td></tr><tr><td>c4_100dom_val_15 Websites.dailymail.co.uk</td><td>12.42</td><td>11.97</td><td>12.87</td><td>11.85</td><td>11.69</td></tr><tr><td>c4_100dom_val_16 Websites.ncbi.nlm.nih.gov</td><td>7.39</td><td>7.31</td><td>7.98</td><td>7.11</td><td>7.07</td></tr><tr><td>c4_100dom_val_17 Websites.express.co.uk</td><td>11.57</td><td>11.04</td><td>11.84</td><td>10.99</td><td>10.79</td></tr><tr><td>c4_100dom_val_18_en.m.wikipedia.org</td><td>9.28</td><td>8.95</td><td>9.52</td><td>8.89</td><td>8.80</td></tr><tr><td>c4_100dom_val_19_www.cnet.com</td><td>12.61</td><td>12.23</td><td>13.12</td><td>12.09</td><td>11.97</td></tr><tr><td>c4_100dom_val_1_nytimes.com</td><td>13.13</td><td>12.66</td><td>14.04</td><td>12.68</td><td>12.44</td></tr><tr><td>c4_100dom_val_20_webtelegraph.co.uk</td><td>13.71</td><td>13.10</td><td>14.28</td><td>13.06</td><td>12.88</td></tr><tr><td>c4_100dom_val_21_web.theatlantic.com</td><td>14.70</td><td>14.17</td><td>15.54</td><td>14.17</td><td>13.97</td></tr><tr><td>c4_100dom_val_22forums.macrumors.com</td><td>17.77</td><td>17.34</td><td>19.15</td><td>17.22</td><td>16.95</td></tr><tr><td>c4_100dom_val_23_www.oreilly.com</td><td>13.36</td><td>12.99</td><td>14.31</td><td>13.02</td><td>12.88</td></tr><tr><td>c4_100dom_val_24_www.washingtonpost.com</td><td>12.06</td><td>11.58</td><td>12.98</td><td>11.64</td><td>11.41</td></tr><tr><td>c4_100dom_val_25_www.zdnet.com</td><td>13.22</td><td>12.86</td><td>13.80</td><td>12.78</td><td>12.61</td></tr><tr><td>c4_100dom_val_26_webfoxbusiness.com</td><td>9.32</td><td>9.03</td><td>9.58</td><td>8.92</td><td>8.81</td></tr><tr><td>c4_100dom_val_27_web.reuters.com</td><td>10.67</td><td>10.13</td><td>11.16</td><td>10.13</td><td>9.97</td></tr><tr><td>c4_100dom_val_28_web.ibtimes.co.uk</td><td>11.36</td><td>11.01</td><td>11.71</td><td>10.89</td><td>10.76</td></tr><tr><td>c4_100dom_val_29_web.rtf.com</td><td>13.59</td><td>12.96</td><td>14.24</td><td>12.98</td><td>12.74</td></tr><tr><td>c4_100dom_val_2_en.wikipedia.org</td><td>10.75</td><td>10.45</td><td>11.32</td><td>10.32</td><td>10.19</td></tr><tr><td>c4_100dom_val_30_web.prweb.com</td><td>11.18</td><td>10.88</td><td>11.92</td><td>10.83</td><td>10.65</td></tr><tr><td>c4_100dom_val_31_web/deviantart.com</td><td>21.78</td><td>21.05</td><td>22.78</td><td>21.00</td><td>20.69</td></tr><tr><td>c4_100dom_val_32_web.si.com</td><td>11.49</td><td>11.00</td><td>11.92</td><td>10.90</td><td>10.76</td></tr><tr><td>c4_100dom_val_33_web.bbc.com</td><td>9.35</td><td>8.91</td><td>9.41</td><td>8.80</td><td>8.70</td></tr><tr><td>c4_100dom_val_34_github.com</td><td>11.57</td><td>11.49</td><td>12.94</td><td>11.40</td><td>11.28</td></tr><tr><td>c4_100dom_val_35_nypost.com</td><td>14.31</td><td>13.41</td><td>15.29</td><td>13.62</td><td>13.31</td></tr><tr><td>c4_100dom_val_36_itunes.apple.com</td><td>16.49</td><td>15.88</td><td>17.15</td><td>15.98</td><td>15.69</td></tr><tr><td>c4_100dom_val_37_web.instructables.com</td><td>16.75</td><td>16.33</td><td>17.73</td><td>16.28</td><td>15.97</td></tr><tr><td>c4_100dom_val_38_web.youtube.com</td><td>8.42</td><td>8.24</td><td>8.83</td><td>8.22</td><td>8.07</td></tr><tr><td>c4_100dom_val_39_webbooking.com</td><td>8.84</td><td>8.49</td><td>8.83</td><td>8.41</td><td>8.32</td></tr><tr><td>c4_100dom_val_40_web.etsy.com</td><td>11.93</td><td>11.66</td><td>12.66</td><td>11.52</td><td>11.43</td></tr><tr><td>c4_100dom_val_41_web marketwired.com</td><td>7.66</td><td>7.47</td><td>7.88</td><td>7.33</td><td>7.27</td></tr><tr><td>c4_100dom_val_42sites.google.com</td><td>14.23</td><td>13.81</td><td>14.91</td><td>13.68</td><td>13.51</td></tr><tr><td>c4_100dom_val_43_webbaltimoresun.com</td><td>11.57</td><td>11.16</td><td>11.96</td><td>11.09</td><td>10.95</td></tr><tr><td>c4_100dom_val_44_web.agreatertown.com</td><td>13.56</td><td>12.94</td><td>13.57</td><td>12.77</td><td>12.64</td></tr><tr><td>c4_100dom_val_45_web.npr.org</td><td>10.59</td><td>10.30</td><td>11.14</td><td>10.19</td><td>10.12</td></tr><tr><td>c4_100dom_val_46_web.fool.com</td><td>11.03</td><td>10.63</td><td>11.35</td><td>10.56</td><td>10.42</td></tr><tr><td>c4_100dom_val_47 www.tripadvisor.com</td><td>15.80</td><td>15.26</td><td>16.26</td><td>15.10</td><td>14.93</td></tr><tr><td>c4_100dom_val_48 www.bbc.co.uk</td><td>12.55</td><td>12.10</td><td>13.02</td><td>12.00</td><td>11.85</td></tr><tr><td>c4_100dom_val_49 lists.w3.org</td><td>18.75</td><td>18.24</td><td>19.89</td><td>18.05</td><td>17.84</td></tr><tr><td>c4_100dom_val_4 www.latimes.com</td><td>11.88</td><td>11.46</td><td>12.40</td><td>11.39</td><td>11.24</td></tr><tr><td>c4_100dom_val_50 mashable.com</td><td>12.44</td><td>11.95</td><td>12.85</td><td>11.90</td><td>11.76</td></tr><tr><td>c4_100dom_val_51 disneyparksmompanel.disi</td><td>11.99</td><td>11.29</td><td>11.98</td><td>11.16</td><td>11.00</td></tr><tr><td>c4_100dom_val_52 www.cnbc.com</td><td>10.65</td><td>10.32</td><td>10.99</td><td>10.24</td><td>10.10</td></tr><tr><td>c4_100dom_val_53 answers.sap.com</td><td>23.59</td><td>23.09</td><td>25.71</td><td>22.99</td><td>22.55</td></tr><tr><td>c4_100dom_val_54 homestars.com</td><td>14.13</td><td>13.70</td><td>14.51</td><td>13.65</td><td>13.52</td></tr><tr><td>c4_100dom_val_55 www.hindustantimes.com</td><td>12.13</td><td>11.60</td><td>12.74</td><td>11.60</td><td>11.37</td></tr><tr><td>c4_100dom_val_56 www.reference.com</td><td>11.57</td><td>11.04</td><td>11.75</td><td>10.92</td><td>10.79</td></tr><tr><td>c4_100dom_val_57 www.city-data.com</td><td>18.38</td><td>17.94</td><td>19.61</td><td>17.73</td><td>17.62</td></tr><tr><td>c4_100dom_val_58 medium.com</td><td>15.50</td><td>15.09</td><td>16.58</td><td>15.18</td><td>15.01</td></tr><tr><td>c4_100dom_val_59 app-wiringdiagram...</td><td>9.74</td><td>9.10</td><td>9.68</td><td>8.88</td><td>8.75</td></tr><tr><td>c4_100dom_val_5 www.theguardian.com</td><td>14.78</td><td>14.09</td><td>15.47</td><td>14.08</td><td>13.86</td></tr><tr><td>c4_100dom_val_60 www.csmonitor.com</td><td>15.35</td><td>14.85</td><td>15.92</td><td>14.75</td><td>14.57</td></tr><tr><td>c4_100dom_val_61 www.adweek.com</td><td>14.55</td><td>13.95</td><td>15.58</td><td>14.09</td><td>13.81</td></tr><tr><td>c4_100dom_val_62 docs.microsoft.com</td><td>7.69</td><td>7.79</td><td>8.86</td><td>7.68</td><td>7.58</td></tr><tr><td>c4_100dom_val_63 www.yahoo.com</td><td>9.29</td><td>8.88</td><td>9.71</td><td>8.89</td><td>8.77</td></tr><tr><td>c4_100dom_val_64 wwwthesun.co.uk</td><td>12.18</td><td>11.66</td><td>12.74</td><td>11.59</td><td>11.39</td></tr><tr><td>c4_100dom_val_65 www.nydailynews.com</td><td>12.15</td><td>11.60</td><td>12.61</td><td>11.56</td><td>11.36</td></tr><tr><td>c4_100dom_val_66 www.dailystar.co.uk</td><td>10.65</td><td>10.17</td><td>11.03</td><td>10.09</td><td>9.92</td></tr><tr><td>c4_100dom_val_67 fineartamerica.com</td><td>12.06</td><td>11.58</td><td>12.29</td><td>11.46</td><td>11.36</td></tr><tr><td>c4_100dom_val_68 www.kickstarter.com</td><td>13.85</td><td>13.58</td><td>15.38</td><td>13.55</td><td>13.38</td></tr><tr><td>c4_100dom_val_69 uk.reuters.com</td><td>9.54</td><td>9.13</td><td>9.90</td><td>9.07</td><td>8.92</td></tr><tr><td>c4_100dom_val_6_ywww.huffpost.com</td><td>13.45</td><td>13.03</td><td>13.96</td><td>12.99</td><td>12.83</td></tr><tr><td>c4_100dom_val_70 www.insiderpages.com</td><td>13.24</td><td>12.84</td><td>13.55</td><td>12.77</td><td>12.64</td></tr><tr><td>c4_100dom_val_71 www.inquisitr.com</td><td>12.12</td><td>11.58</td><td>12.86</td><td>11.71</td><td>11.38</td></tr><tr><td>c4_100dom_val_72 lists debian.org</td><td>18.18</td><td>17.81</td><td>19.62</td><td>17.67</td><td>17.30</td></tr><tr><td>c4_100dom_val_73 www.straitstimes.com</td><td>11.51</td><td>11.06</td><td>11.91</td><td>10.94</td><td>10.79</td></tr><tr><td>c4_100dom_val_74 www.cbsnews.com</td><td>10.29</td><td>9.91</td><td>10.60</td><td>9.82</td><td>9.72</td></tr><tr><td>c4_100dom_val_75 simple.wikipedia.org</td><td>8.25</td><td>7.85</td><td>8.37</td><td>7.78</td><td>7.67</td></tr><tr><td>c4_100dom_val_76 deadline.com</td><td>14.75</td><td>13.83</td><td>15.48</td><td>13.92</td><td>13.51</td></tr><tr><td>c4_100dom_val_77 www.androidheadlines.con</td><td>11.11</td><td>10.74</td><td>11.43</td><td>10.72</td><td>10.59</td></tr><tr><td>c4_100dom_val_78 www.wired.com</td><td>14.42</td><td>13.88</td><td>15.14</td><td>13.87</td><td>13.68</td></tr><tr><td>c4_100dom_val_79 www.bustle.com</td><td>12.79</td><td>12.33</td><td>13.19</td><td>12.25</td><td>12.09</td></tr><tr><td>c4_100dom_val_7patents.google.com</td><td>7.59</td><td>7.84</td><td>9.33</td><td>7.72</td><td>7.59</td></tr><tr><td>c4_100dom_val_80 premium.wpmudev.org</td><td>16.86</td><td>16.63</td><td>18.13</td><td>16.50</td><td>16.29</td></tr><tr><td>c4_100dom_val_81 www.librarything.com</td><td>14.36</td><td>13.98</td><td>15.42</td><td>13.91</td><td>13.75</td></tr><tr><td>c4_100dom_val_82 mail-archives.apache.org</td><td>5.67</td><td>5.61</td><td>6.17</td><td>5.56</td><td>5.49</td></tr><tr><td>c4_100dom_val_83 scholars.duke.edu</td><td>8.72</td><td>8.43</td><td>9.03</td><td>8.32</td><td>8.21</td></tr><tr><td>c4_100dom_val_84 www.glassdoor.com</td><td>16.64</td><td>15.97</td><td>16.99</td><td>16.00</td><td>15.83</td></tr><tr><td>c4_100dom_val_85 www.pcwworld.com</td><td>12.34</td><td>11.95</td><td>12.95</td><td>11.90</td><td>11.72</td></tr><tr><td>c4_100dom_val_86 www(shutterstock.com</td><td>8.70</td><td>8.89</td><td>10.75</td><td>8.62</td><td>8.52</td></tr><tr><td>c4_100dom_val_87 myemail(constcontact.cc</td><td>14.59</td><td>14.24</td><td>15.32</td><td>14.18</td><td>13.98</td></tr><tr><td>c4_100dom_val_88 www.eventbrite.co.uk</td><td>14.47</td><td>13.99</td><td>14.89</td><td>13.98</td><td>13.79</td></tr><tr><td>c4_100dom_val_89 www.fastcompany.com</td><td>14.24</td><td>13.75</td><td>15.52</td><td>13.82</td><td>13.56</td></tr><tr><td>c4_100dom_val_8 www.businessinsider.com</td><td>10.97</td><td>10.69</td><td>11.35</td><td>10.52</td><td>10.46</td></tr><tr><td>c4_100dom_val_90 www.firstpost.com</td><td>11.71</td><td>11.24</td><td>12.08</td><td>11.12</td><td>10.96</td></tr><tr><td>c4_100dom_val_91 www.entrepreneur.com</td><td>13.10</td><td>12.68</td><td>13.65</td><td>12.72</td><td>12.54</td></tr><tr><td>c4_100dom_val_92 www.breitbart.com</td><td>13.47</td><td>12.67</td><td>14.29</td><td>12.84</td><td>12.56</td></tr><tr><td>c4_100dom_val_93 techcrunch.com</td><td>14.20</td><td>13.68</td><td>15.18</td><td>13.82</td><td>13.58</td></tr><tr><td>c4_100dom_val_94 WWW.Nme.com</td><td>14.12</td><td>13.28</td><td>15.06</td><td>13.43</td><td>13.12</td></tr><tr><td>c4_100dom_val_95 WWW.ndtv.com</td><td>10.66</td><td>10.26</td><td>10.90</td><td>10.10</td><td>10.00</td></tr><tr><td>c4_100dom_val_96 finance.yahoo.com</td><td>9.96</td><td>9.55</td><td>10.22</td><td>9.43</td><td>9.34</td></tr><tr><td>c4_100dom_val_97 archives.lib.state.ma.us</td><td>6.53</td><td>6.12</td><td>7.09</td><td>6.27</td><td>5.85</td></tr><tr><td>c4_100dom_val_98 www.gsmarena.com</td><td>23.21</td><td>22.15</td><td>24.52</td><td>22.10</td><td>21.76</td></tr><tr><td>c4_100dom_val_99 www.lonelyplanet.com</td><td>11.33</td><td>10.92</td><td>12.28</td><td>10.84</td><td>10.69</td></tr><tr><td>c4_100dom_val_9 www.forbes.com</td><td>13.72</td><td>13.31</td><td>14.63</td><td>13.34</td><td>13.13</td></tr><tr><td>c4_en_val-00000000</td><td>14.34</td><td>13.70</td><td>14.87</td><td>13.67</td><td>13.46</td></tr><tr><td>c4_en_val-00000001</td><td>14.86</td><td>14.28</td><td>15.51</td><td>14.21</td><td>14.09</td></tr><tr><td>c4_en_val-00000002</td><td>15.29</td><td>14.71</td><td>15.95</td><td>14.71</td><td>14.51</td></tr><tr><td>c4_en_val-00000003</td><td>12.95</td><td>12.28</td><td>13.32</td><td>12.23</td><td>12.06</td></tr><tr><td>c4_en_val-00000004</td><td>12.56</td><td>12.13</td><td>13.27</td><td>12.05</td><td>11.87</td></tr><tr><td>c4_en_val-00000005</td><td>12.77</td><td>12.35</td><td>13.26</td><td>12.32</td><td>12.18</td></tr><tr><td>dolma-v1_5_val_books</td><td>13.00</td><td>12.44</td><td>13.64</td><td>12.44</td><td>12.27</td></tr><tr><td>dolma-v1_5_val_common-crawl</td><td>16.86</td><td>16.37</td><td>18.00</td><td>16.35</td><td>16.10</td></tr><tr><td>dolma-v1_5_val_pes2o</td><td>9.42</td><td>9.56</td><td>11.25</td><td>9.41</td><td>9.29</td></tr><tr><td>dolma-v1_5_val.reddit.uniform</td><td>23.04</td><td>21.97</td><td>23.84</td><td>22.05</td><td>21.80</td></tr><tr><td>dolma-v1_5_val_stack.uniform</td><td>2.30</td><td>2.33</td><td>2.53</td><td>2.30</td><td>2.29</td></tr><tr><td>dolma-v1_5_val_wiki</td><td>10.86</td><td>10.48</td><td>11.25</td><td>10.41</td><td>10.31</td></tr><tr><td>dolma_100_proglang_val_00_text</td><td>5.61</td><td>6.30</td><td>6.94</td><td>5.67</td><td>5.69</td></tr><tr><td>dolma_100_proglang_val_01markdown</td><td>3.16</td><td>3.16</td><td>3.56</td><td>3.15</td><td>3.11</td></tr><tr><td>dolma_100_proglang_val_02_c</td><td>1.84</td><td>1.91</td><td>2.23</td><td>1.86</td><td>1.85</td></tr><tr><td>dolma_100_proglang_val_03_php</td><td>1.75</td><td>1.75</td><td>1.83</td><td>1.73</td><td>1.72</td></tr><tr><td>dolma_100_proglang_val_04(java</td><td>1.96</td><td>1.99</td><td>2.18</td><td>1.95</td><td>1.95</td></tr><tr><td>dolma_100_proglang_val_05_c++</td><td>2.19</td><td>2.25</td><td>2.53</td><td>2.21</td><td>2.19</td></tr><tr><td>dolma_100_proglang_val_06/python</td><td>2.35</td><td>2.39</td><td>2.62</td><td>2.36</td><td>2.34</td></tr><tr><td>dolma_100_proglang_val_07 javascript</td><td>2.54</td><td>2.59</td><td>2.83</td><td>2.53</td><td>2.53</td></tr><tr><td>dolma_100_proglang_val_08_html</td><td>1.92</td><td>1.94</td><td>2.13</td><td>1.91</td><td>1.91</td></tr><tr><td>dolma_100_proglang_val_09_c#</td><td>2.23</td><td>2.28</td><td>2.45</td><td>2.19</td><td>2.24</td></tr><tr><td>dolma_100_proglang_val_10_yaml</td><td>2.93</td><td>3.01</td><td>3.71</td><td>2.94</td><td>2.92</td></tr><tr><td>dolma_100_proglang_val_11_go</td><td>1.75</td><td>1.78</td><td>1.97</td><td>1.77</td><td>1.75</td></tr><tr><td>dolma_100_proglang_val_12_typesscript</td><td>2.17</td><td>2.20</td><td>2.41</td><td>2.18</td><td>2.16</td></tr><tr><td>dolma_100_proglang_val_13_xml</td><td>2.44</td><td>2.50</td><td>2.78</td><td>2.46</td><td>2.48</td></tr><tr><td>dolma_100_proglang_val_14_css</td><td>2.25</td><td>2.25</td><td>2.34</td><td>2.21</td><td>2.20</td></tr><tr><td>dolma_100_proglang_val_15_jupyter-nb</td><td>1.57</td><td>1.60</td><td>1.75</td><td>1.58</td><td>1.58</td></tr><tr><td>dolma_100_proglang_val_16_rust</td><td>1.96</td><td>2.01</td><td>2.23</td><td>1.97</td><td>1.96</td></tr><tr><td>dolma_100_proglang_val_17 Unity3d-asset</td><td>4.01</td><td>4.17</td><td>4.56</td><td>4.10</td><td>4.05</td></tr><tr><td>dolma_100_proglang_val_18_gettext-catalog</td><td>2.84</td><td>2.87</td><td>3.53</td><td>2.86</td><td>2.83</td></tr><tr><td>dolma_100_proglang_val_19_ruby</td><td>2.41</td><td>2.44</td><td>2.70</td><td>2.39</td><td>2.38</td></tr><tr><td>dolma_100_proglang_val_20_vue</td><td>1.95</td><td>1.95</td><td>2.10</td><td>1.94</td><td>1.93</td></tr><tr><td>dolma_100_proglang_val_21 sql</td><td>2.18</td><td>2.23</td><td>2.46</td><td>2.17</td><td>2.16</td></tr><tr><td>dolma_100_proglang_val_22swift</td><td>1.86</td><td>1.88</td><td>2.04</td><td>1.86</td><td>1.84</td></tr><tr><td>dolma_100_proglang_val_23_kotlin</td><td>2.05</td><td>2.07</td><td>2.29</td><td>2.07</td><td>2.04</td></tr><tr><td>dolma_100_proglang_val_24 Scala</td><td>2.24</td><td>2.28</td><td>2.64</td><td>2.25</td><td>2.23</td></tr><tr><td>dolma_100_proglang_val_25_scss</td><td>2.26</td><td>2.27</td><td>2.38</td><td>2.24</td><td>2.24</td></tr><tr><td>dolma_100_proglang_val_26TEX</td><td>4.04</td><td>4.21</td><td>4.97</td><td>4.10</td><td>4.04</td></tr><tr><td>dolma_100_proglang_val_27_dart</td><td>1.79</td><td>1.82</td><td>2.01</td><td>1.80</td><td>1.78</td></tr><tr><td>dolma_100_proglang_val_28_kicad</td><td>2.57</td><td>2.79</td><td>3.86</td><td>2.68</td><td>2.67</td></tr><tr><td>dolma_100_proglang_val_29_shell</td><td>3.71</td><td>3.74</td><td>4.31</td><td>3.69</td><td>3.63</td></tr><tr><td>dolma_100_proglang_val_30_smali</td><td>1.38</td><td>1.39</td><td>1.45</td><td>1.38</td><td>1.37</td></tr><tr><td>dolma_100_proglang_val_31_lua</td><td>5.65</td><td>6.01</td><td>7.18</td><td>5.33</td><td>5.45</td></tr><tr><td>dolma_100_proglang_val_32_restructuredtext</td><td>4.01</td><td>4.05</td><td>4.66</td><td>3.97</td><td>3.92</td></tr></table>

<table><tr><td>Dataset</td></tr><tr><td>dolma_100_proglang_val_33.perl</td></tr><tr><td>dolma_100_proglang_val_34_diff</td></tr><tr><td>dolma_100_proglang_val_35.ini</td></tr><tr><td>dolma_100_proglang_val_36-jsx</td></tr><tr><td>dolma_100_proglang_val_37 Haskell</td></tr><tr><td>dolma_100_proglang_val_38_gnuplot</td></tr><tr><td>dolma_100_proglang_val_39_postscript</td></tr><tr><td>dolma_100_proglang_val_40_groff</td></tr><tr><td>dolma_100_proglang_val_41_turtle</td></tr><tr><td>dolma_100_proglang_val_42_fortran</td></tr><tr><td>dolma_100_proglang_val_43.makefile</td></tr><tr><td>dolma_100_proglang_val_44(mathematica</td></tr><tr><td>dolma_100_proglang_val_45.pascal</td></tr><tr><td>dolma_100_proglang_val_46_common-lisp</td></tr><tr><td>dolma_100_proglang_val_47_gas</td></tr><tr><td>dolma_100_proglang_val_48_vhdl</td></tr><tr><td>dolma_100_proglang_val_49_julia</td></tr><tr><td>dolma_100_proglang_val_50_edn</td></tr><tr><td>dolma_100_proglang_val_51visual-basic</td></tr><tr><td>dolma_100_proglang_val_52powershell</td></tr><tr><td>dolma_100_proglang_val_53_g-code</td></tr><tr><td>dolma_100_proglang_val_54_ocaml</td></tr><tr><td>dolma_100_proglang_val_55 JAVA-server-p</td></tr><tr><td>dolma_100_proglang_val_56_solidity</td></tr><tr><td>dolma_100_proglang_val_57.graphviz-dot</td></tr><tr><td>dolma_100_proglang_val_58-less</td></tr><tr><td>dolma_100_proglang_val_59_twig</td></tr><tr><td>dolma_100_proglang_val_60.ascidoc</td></tr><tr><td>dolma_100_proglang_val_61 groovy</td></tr><tr><td>dolma_100_proglang_val_62 llvm</td></tr><tr><td>dolma_100_proglang_val_63.hcl</td></tr><tr><td>dolma_100_proglang_val_64.html+erb</td></tr><tr><td>dolma_100_proglang_val_65_erlang</td></tr><tr><td>dolma_100_proglang_val_66_elixir</td></tr><tr><td>dolma_100_proglang_val_67_eagle</td></tr><tr><td>dolma_100_proglang_val_68 arduino</td></tr><tr><td>dolma_100_proglang_val_69 coffeescript</td></tr><tr><td>dolma_100_proglang_val_70_toml</td></tr><tr><td>dolma_100_proglang_val_71_cuda</td></tr><tr><td>dolma_100_proglang_val_72_nix</td></tr><tr><td>dolma_100_proglang_val_73_smalltalk</td></tr><tr><td>dolma_100_proglang_val_74_cmake</td></tr><tr><td>dolma_100_proglang_val_75行動script</td></tr><tr><td>dolma_100_proglang_val_76.glsl</td></tr><tr><td>dolma_100_proglang_val_77_systemverilog</td></tr><tr><td>dolma_100_proglang_val_78_haxe</td></tr><tr><td>dolma_100_proglang_val_79_f#</td></tr><tr><td>dolma_100_proglang_val_80_max</td></tr><tr><td>dolma_100_proglang_val_81_objective-c++</td></tr><tr><td>dolma_100_proglang_val_82_STANDARD-ml</td></tr><tr><td>dolma_100_proglang_val_83_dockerfile</td></tr><tr><td>dolma_100_proglang_val_84 Emacs-lisp</td></tr></table>

<table><tr><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>2.57</td><td>2.62</td><td>3.01</td><td>2.59</td><td>2.55</td></tr><tr><td>2.87</td><td>2.95</td><td>3.43</td><td>2.89</td><td>2.86</td></tr><tr><td>3.91</td><td>4.16</td><td>4.90</td><td>4.05</td><td>3.98</td></tr><tr><td>1.83</td><td>1.84</td><td>1.95</td><td>1.83</td><td>1.82</td></tr><tr><td>2.94</td><td>3.07</td><td>3.73</td><td>3.02</td><td>2.95</td></tr><tr><td>2.65</td><td>2.88</td><td>3.36</td><td>2.81</td><td>2.77</td></tr><tr><td>19.09</td><td>19.52</td><td>19.56</td><td>18.66</td><td>18.64</td></tr><tr><td>6.13</td><td>6.32</td><td>7.45</td><td>6.22</td><td>6.21</td></tr><tr><td>2.35</td><td>2.45</td><td>3.17</td><td>2.39</td><td>2.35</td></tr><tr><td>2.32</td><td>2.39</td><td>2.83</td><td>2.35</td><td>2.31</td></tr><tr><td>2.93</td><td>3.01</td><td>3.51</td><td>2.86</td><td>2.82</td></tr><tr><td>10.34</td><td>11.34</td><td>13.24</td><td>10.49</td><td>10.71</td></tr><tr><td>4.18</td><td>4.81</td><td>5.49</td><td>4.17</td><td>4.27</td></tr><tr><td>2.56</td><td>2.71</td><td>3.32</td><td>2.62</td><td>2.58</td></tr><tr><td>2.49</td><td>2.73</td><td>3.59</td><td>2.57</td><td>2.53</td></tr><tr><td>3.91</td><td>4.06</td><td>4.69</td><td>3.92</td><td>3.90</td></tr><tr><td>3.25</td><td>3.36</td><td>4.05</td><td>3.30</td><td>3.26</td></tr><tr><td>1.99</td><td>2.10</td><td>2.67</td><td>2.04</td><td>2.03</td></tr><tr><td>2.42</td><td>2.49</td><td>2.72</td><td>2.37</td><td>2.38</td></tr><tr><td>4.08</td><td>4.16</td><td>4.50</td><td>3.86</td><td>3.89</td></tr><tr><td>2.26</td><td>2.66</td><td>3.29</td><td>2.44</td><td>2.37</td></tr><tr><td>3.06</td><td>3.29</td><td>4.22</td><td>3.19</td><td>3.13</td></tr><tr><td>2.10</td><td>2.11</td><td>2.31</td><td>2.06</td><td>2.09</td></tr><tr><td>4.09</td><td>4.41</td><td>5.28</td><td>4.05</td><td>4.10</td></tr><tr><td>2.17</td><td>2.48</td><td>3.54</td><td>2.32</td><td>2.29</td></tr><tr><td>2.24</td><td>2.26</td><td>2.33</td><td>2.22</td><td>2.22</td></tr><tr><td>1.81</td><td>1.81</td><td>1.91</td><td>1.80</td><td>1.79</td></tr><tr><td>5.33</td><td>5.50</td><td>6.84</td><td>5.43</td><td>5.34</td></tr><tr><td>2.12</td><td>2.15</td><td>2.41</td><td>2.13</td><td>2.11</td></tr><tr><td>2.26</td><td>2.40</td><td>3.25</td><td>2.31</td><td>2.23</td></tr><tr><td>2.52</td><td>2.56</td><td>2.96</td><td>2.52</td><td>2.48</td></tr><tr><td>2.10</td><td>2.09</td><td>2.23</td><td>2.08</td><td>2.07</td></tr><tr><td>2.84</td><td>2.98</td><td>3.87</td><td>2.88</td><td>2.85</td></tr><tr><td>2.93</td><td>2.99</td><td>3.58</td><td>2.91</td><td>2.90</td></tr><tr><td>5.35</td><td>6.90</td><td>10.75</td><td>5.64</td><td>5.76</td></tr><tr><td>3.37</td><td>3.40</td><td>3.81</td><td>3.28</td><td>3.28</td></tr><tr><td>2.80</td><td>2.85</td><td>3.27</td><td>2.80</td><td>2.77</td></tr><tr><td>7.76</td><td>7.62</td><td>8.44</td><td>7.53</td><td>7.58</td></tr><tr><td>2.15</td><td>2.21</td><td>2.56</td><td>2.19</td><td>2.16</td></tr><tr><td>7.80</td><td>7.84</td><td>9.03</td><td>7.88</td><td>7.83</td></tr><tr><td>9.32</td><td>9.61</td><td>12.60</td><td>9.47</td><td>9.20</td></tr><tr><td>1.87</td><td>1.86</td><td>2.02</td><td>1.84</td><td>1.81</td></tr><tr><td>2.45</td><td>2.54</td><td>2.88</td><td>2.46</td><td>2.46</td></tr><tr><td>2.40</td><td>2.42</td><td>2.72</td><td>2.36</td><td>2.32</td></tr><tr><td>2.53</td><td>2.66</td><td>3.17</td><td>2.58</td><td>2.55</td></tr><tr><td>2.74</td><td>2.81</td><td>3.20</td><td>2.77</td><td>2.76</td></tr><tr><td>2.89</td><td>3.02</td><td>3.53</td><td>2.93</td><td>2.88</td></tr><tr><td>1.59</td><td>1.62</td><td>1.80</td><td>1.61</td><td>1.61</td></tr><tr><td>2.18</td><td>2.19</td><td>2.40</td><td>2.17</td><td>2.16</td></tr><tr><td>3.57</td><td>4.05</td><td>4.79</td><td>3.81</td><td>3.77</td></tr><tr><td>4.08</td><td>4.17</td><td>4.37</td><td>4.01</td><td>4.05</td></tr><tr><td>3.83</td><td>3.83</td><td>4.44</td><td>3.80</td><td>3.72</td></tr></table>

<table><tr><td>Dataset</td><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>dolma_100_proglang_val_85scheme</td><td>2.78</td><td>2.86</td><td>3.40</td><td>2.84</td><td>2.77</td></tr><tr><td>dolma_100_proglang_val_86_clojure</td><td>3.18</td><td>3.30</td><td>4.00</td><td>3.26</td><td>3.17</td></tr><tr><td>dolma_100_proglang_val_87_handlebars</td><td>1.79</td><td>1.79</td><td>1.88</td><td>1.78</td><td>1.78</td></tr><tr><td>dolma_100_proglang_val_88SMARTy</td><td>2.30</td><td>2.35</td><td>2.58</td><td>2.29</td><td>2.30</td></tr><tr><td>dolma_100_proglang_val_89_logos</td><td>2.37</td><td>2.58</td><td>2.98</td><td>2.46</td><td>2.44</td></tr><tr><td>dolma_100_proglang_val_90_stata</td><td>4.67</td><td>5.08</td><td>6.85</td><td>4.85</td><td>4.81</td></tr><tr><td>dolma_100_proglang_val_91_yacc</td><td>2.42</td><td>2.48</td><td>2.87</td><td>2.44</td><td>2.43</td></tr><tr><td>dolma_100_proglang_val_92_nimrod</td><td>2.75</td><td>2.87</td><td>3.63</td><td>2.81</td><td>2.77</td></tr><tr><td>dolma_100_proglang_val_93_tcl</td><td>3.00</td><td>3.16</td><td>3.95</td><td>3.07</td><td>3.02</td></tr><tr><td>dolma_100_proglang_val_94_viml</td><td>5.56</td><td>5.76</td><td>7.21</td><td>5.59</td><td>5.55</td></tr><tr><td>dolma_100_proglang_val_95.asp</td><td>1.79</td><td>1.79</td><td>1.90</td><td>1.77</td><td>1.77</td></tr><tr><td>dolma_100_proglang_val_96(protocol-buffer</td><td>1.32</td><td>1.31</td><td>1.38</td><td>1.31</td><td>1.32</td></tr><tr><td>dolma_100_proglang_val_97_r</td><td>2.80</td><td>2.92</td><td>3.66</td><td>2.86</td><td>2.81</td></tr><tr><td>dolma_100_proglang_val_98_cython</td><td>2.34</td><td>2.39</td><td>2.69</td><td>2.36</td><td>2.35</td></tr><tr><td>dolma_100_proglang_val_99_mediawiki</td><td>2.01</td><td>2.10</td><td>2.48</td><td>2.12</td><td>2.04</td></tr><tr><td>dolma_100_subreddits_val_00_AskReddit</td><td>20.25</td><td>19.29</td><td>20.38</td><td>19.28</td><td>19.14</td></tr><tr><td>dolma_100_subreddits_val_01 POLITICS</td><td>22.08</td><td>20.70</td><td>22.07</td><td>20.83</td><td>20.61</td></tr><tr><td>dolma_100_subreddits_val_02_AmItheAsshole</td><td>22.49</td><td>21.30</td><td>22.89</td><td>21.60</td><td>21.27</td></tr><tr><td>dolma_100_subreddits_val_03_worldnews</td><td>22.57</td><td>21.43</td><td>22.77</td><td>21.50</td><td>21.23</td></tr><tr><td>dolma_100_subreddits_val_04 relatioships</td><td>18.64</td><td>17.80</td><td>18.89</td><td>17.86</td><td>17.67</td></tr><tr><td>dolma_100_subreddits_val_05_Relationships_adv</td><td>19.40</td><td>18.53</td><td>19.68</td><td>18.63</td><td>18.46</td></tr><tr><td>dolma_100_subreddits_val_06_news</td><td>22.49</td><td>21.25</td><td>22.51</td><td>21.49</td><td>21.17</td></tr><tr><td>dolma_100_subreddits_val_07_leagueoflegends</td><td>34.45</td><td>32.41</td><td>35.13</td><td>32.46</td><td>32.04</td></tr><tr><td>dolma_100_subreddits_val_08今天的ailylearned</td><td>22.53</td><td>21.30</td><td>22.68</td><td>21.28</td><td>21.10</td></tr><tr><td>dolma_100_subreddits_val_09_TwoXChromoso</td><td>20.20</td><td>19.16</td><td>20.25</td><td>19.20</td><td>19.02</td></tr><tr><td>dolma_100_subreddits_val_10_personalfinance</td><td>18.62</td><td>17.65</td><td>18.82</td><td>17.73</td><td>17.64</td></tr><tr><td>dolma_100_subreddits_val_11changemyview</td><td>20.02</td><td>19.10</td><td>20.50</td><td>19.17</td><td>18.99</td></tr><tr><td>dolma_100_subreddits_val_12_unpopularopinio</td><td>23.39</td><td>22.16</td><td>23.63</td><td>22.32</td><td>22.04</td></tr><tr><td>dolma_100_subreddits_val_13Movies</td><td>21.62</td><td>20.52</td><td>21.79</td><td>20.64</td><td>20.35</td></tr><tr><td>dolma_100_subreddits_val_14_Games</td><td>22.26</td><td>21.15</td><td>22.52</td><td>21.18</td><td>20.87</td></tr><tr><td>dolma_100_subreddits_val_15_nba</td><td>23.28</td><td>21.93</td><td>23.60</td><td>22.10</td><td>21.85</td></tr><tr><td>dolma_100_subreddits_val_16_pics</td><td>21.84</td><td>20.56</td><td>21.82</td><td>20.64</td><td>20.47</td></tr><tr><td>dolma_100_subreddits_val_17_gaming</td><td>24.45</td><td>23.13</td><td>24.61</td><td>23.15</td><td>22.86</td></tr><tr><td>dolma_100_subreddits_val_18_soccer</td><td>23.38</td><td>22.12</td><td>23.61</td><td>22.19</td><td>22.03</td></tr><tr><td>dolma_100_subreddits_val_19NFL</td><td>19.86</td><td>18.76</td><td>20.17</td><td>18.81</td><td>18.62</td></tr><tr><td>dolma_100_subreddits_val_20_explainlikeimfv</td><td>18.35</td><td>17.21</td><td>18.59</td><td>17.32</td><td>17.03</td></tr><tr><td>dolma_100_subreddits_val_21_conspiracy</td><td>23.86</td><td>22.53</td><td>24.09</td><td>22.67</td><td>22.54</td></tr><tr><td>dolma_100_subreddits_val_22 atheism</td><td>21.23</td><td>20.18</td><td>21.43</td><td>20.23</td><td>20.13</td></tr><tr><td>dolma_100_subreddits_val_23_AskMen</td><td>20.00</td><td>19.04</td><td>20.11</td><td>19.10</td><td>18.94</td></tr><tr><td>dolma_100_subreddits_val_24videos</td><td>22.26</td><td>21.24</td><td>22.51</td><td>21.29</td><td>21.04</td></tr><tr><td>dolma_100_subreddits_val_25-sex</td><td>21.13</td><td>20.13</td><td>21.30</td><td>20.09</td><td>19.98</td></tr><tr><td>dolma_100_subreddits_val_26_raisedbynarcissi</td><td>22.07</td><td>21.08</td><td>22.48</td><td>21.20</td><td>21.02</td></tr><tr><td>dolma_100_subreddits_val_27_NoStupidQuesti</td><td>19.66</td><td>18.59</td><td>19.87</td><td>18.68</td><td>18.52</td></tr><tr><td>dolma_100_subreddits_val_28_DesignTheGam</td><td>35.27</td><td>33.58</td><td>36.13</td><td>33.78</td><td>33.37</td></tr><tr><td>dolma_100_subreddits_val_29_ANime</td><td>23.21</td><td>22.04</td><td>23.46</td><td>22.12</td><td>21.77</td></tr><tr><td>dolma_100_subreddits_val_30_DnD</td><td>28.22</td><td>26.71</td><td>28.78</td><td>26.72</td><td>26.39</td></tr><tr><td>dolma_100_subreddits_val_31_ukpolitics</td><td>22.35</td><td>21.19</td><td>22.80</td><td>21.31</td><td>21.10</td></tr><tr><td>dolma_100_subreddits_val_32 Funny</td><td>20.78</td><td>19.45</td><td>20.70</td><td>19.40</td><td>19.23</td></tr><tr><td>dolma_100_subreddits_val_33_europe</td><td>21.76</td><td>20.59</td><td>22.10</td><td>20.72</td><td>20.52</td></tr><tr><td>dolma_100_subreddits_val_34 canada</td><td>22.44</td><td>21.21</td><td>22.44</td><td>21.30</td><td>21.09</td></tr><tr><td>dolma_100_subreddits_val_35_Christianity</td><td>17.88</td><td>17.02</td><td>18.10</td><td>17.04</td><td>16.94</td></tr><tr><td>dolma_100_subreddits_val_36_SquaredCircle</td><td>25.87</td><td>24.31</td><td>25.83</td><td>24.34</td><td>24.03</td></tr><tr><td>dolma_100_subreddits_val_37_AskWomen</td><td>17.72</td><td>16.81</td><td>17.77</td><td>16.85</td><td>16.72</td></tr><tr><td>dolma_100_subreddits_val_38/legaladvice</td><td>18.66</td><td>17.75</td><td>18.92</td><td>17.74</td><td>17.64</td></tr><tr><td>dolma_100_subreddits_val_39JUSTNOMIL</td><td>24.25</td><td>23.16</td><td>24.86</td><td>23.32</td><td>23.02</td></tr><tr><td>dolma_100_subreddits_val_40_technology</td><td>23.39</td><td>22.09</td><td>23.52</td><td>22.21</td><td>21.95</td></tr><tr><td>dolma_100_subreddits_val_41_IAmA</td><td>19.83</td><td>18.83</td><td>19.86</td><td>18.71</td><td>18.56</td></tr><tr><td>dolma_100_subreddits_val_42_wow</td><td>31.26</td><td>29.25</td><td>31.44</td><td>29.39</td><td>28.82</td></tr><tr><td>dolma_100_subreddits_val_43_Parenting</td><td>20.15</td><td>19.11</td><td>20.43</td><td>19.30</td><td>19.06</td></tr><tr><td>dolma_100_subreddits_val_44_exmormon</td><td>23.12</td><td>21.90</td><td>23.44</td><td>21.99</td><td>21.84</td></tr><tr><td>dolma_100_subreddits_val_45_AdviceAnimals</td><td>22.14</td><td>20.96</td><td>22.14</td><td>20.98</td><td>20.79</td></tr><tr><td>dolma_100_subreddits_val_46_childfree</td><td>21.87</td><td>20.85</td><td>22.13</td><td>20.89</td><td>20.72</td></tr><tr><td>dolma_100_subreddits_val_47_unitedkingdom</td><td>23.27</td><td>22.00</td><td>23.40</td><td>22.00</td><td>21.85</td></tr><tr><td>dolma_100_subreddits_val_48_ffxiv</td><td>32.53</td><td>30.79</td><td>33.33</td><td>31.01</td><td>30.62</td></tr><tr><td>dolma_100_subreddits_val_49_dndnext</td><td>29.67</td><td>28.03</td><td>30.53</td><td>28.26</td><td>27.63</td></tr><tr><td>dolma_100_subreddits_val_50_ADHD</td><td>20.75</td><td>19.83</td><td>21.14</td><td>19.95</td><td>19.78</td></tr><tr><td>dolma_100_subreddits_val_51_loseit</td><td>19.36</td><td>18.39</td><td>19.49</td><td>18.52</td><td>18.33</td></tr><tr><td>dolma_100_subreddits_val_52asoiaf</td><td>25.28</td><td>23.99</td><td>25.63</td><td>23.94</td><td>23.69</td></tr><tr><td>dolma_100_subreddits_val_53_BabyBumps</td><td>20.96</td><td>19.82</td><td>21.11</td><td>19.92</td><td>19.76</td></tr><tr><td>dolma_100_subreddits_val_54_Advice</td><td>19.17</td><td>18.29</td><td>19.35</td><td>18.38</td><td>18.19</td></tr><tr><td>dolma_100_subreddits_val_55_australia</td><td>23.97</td><td>22.51</td><td>24.06</td><td>22.61</td><td>22.40</td></tr><tr><td>dolma_100_subreddits_val_56_CFB</td><td>20.45</td><td>19.41</td><td>20.92</td><td>19.49</td><td>19.23</td></tr><tr><td>dolma_100_subreddits_val_57_offmychest</td><td>19.63</td><td>18.79</td><td>19.77</td><td>18.93</td><td>18.77</td></tr><tr><td>dolma_100_subreddits_val_58.PublicFreakout</td><td>25.96</td><td>24.49</td><td>26.02</td><td>24.65</td><td>24.39</td></tr><tr><td>dolma_100_subreddits_val_59,TrueOffMyChes</td><td>21.53</td><td>20.63</td><td>21.70</td><td>20.73</td><td>20.54</td></tr><tr><td>dolma_100_subreddits_val_60_science</td><td>20.44</td><td>19.46</td><td>20.64</td><td>19.51</td><td>19.38</td></tr><tr><td>dolma_100_subreddits_val_61_magicTCG</td><td>28.82</td><td>26.79</td><td>28.94</td><td>26.69</td><td>26.38</td></tr><tr><td>dolma_100_subreddits_val_62inkelgender</td><td>20.72</td><td>19.86</td><td>21.07</td><td>19.83</td><td>19.62</td></tr><tr><td>dolma_100_subreddits_val_63_DotA2</td><td>34.35</td><td>32.38</td><td>34.74</td><td>32.57</td><td>32.16</td></tr><tr><td>dolma_100_subreddits_val_64_neoliberal</td><td>21.74</td><td>20.59</td><td>22.26</td><td>20.64</td><td>20.45</td></tr><tr><td>dolma_100_subreddits_val_65_whowouldwin</td><td>29.18</td><td>27.81</td><td>30.08</td><td>27.63</td><td>27.30</td></tr><tr><td>dolma_100_subreddits_val_66_depression</td><td>18.28</td><td>17.52</td><td>18.31</td><td>17.50</td><td>17.41</td></tr><tr><td>dolma_100_subreddits_val_67_WTF</td><td>22.30</td><td>21.18</td><td>22.38</td><td>21.17</td><td>20.99</td></tr><tr><td>dolma_100_subreddits_val_68_pathofexile</td><td>40.48</td><td>38.59</td><td>41.43</td><td>38.75</td><td>38.43</td></tr><tr><td>dolma_100_subreddits_val_69_PoliticalDiscuss</td><td>20.01</td><td>18.92</td><td>20.16</td><td>18.97</td><td>18.82</td></tr><tr><td>dolma_100_subreddits_val_70_Libertarian</td><td>22.97</td><td>21.77</td><td>23.15</td><td>21.87</td><td>21.75</td></tr><tr><td>dolma_100_subreddits_val_71_PurplePillDebatu</td><td>24.94</td><td>23.66</td><td>25.44</td><td>23.85</td><td>23.55</td></tr><tr><td>dolma_100_subreddits_val_72_Fitness</td><td>21.57</td><td>20.35</td><td>21.48</td><td>20.34</td><td>20.11</td></tr><tr><td>dolma_100_subreddits_val_73_books</td><td>21.12</td><td>20.02</td><td>21.31</td><td>20.09</td><td>19.82</td></tr><tr><td>dolma_100_subreddits_val_74_dogs</td><td>20.13</td><td>19.12</td><td>20.32</td><td>19.20</td><td>18.92</td></tr><tr><td>dolma_100_subreddits_val_75_pcmasterrace</td><td>23.73</td><td>22.49</td><td>24.02</td><td>22.56</td><td>22.21</td></tr><tr><td>dolma_100_subreddits_val_76Teenagers</td><td>18.37</td><td>16.35</td><td>16.44</td><td>15.56</td><td>17.02</td></tr><tr><td>dolma_100_subreddits_val_77_stopdrinking</td><td>21.08</td><td>20.02</td><td>21.19</td><td>20.17</td><td>19.98</td></tr><tr><td>dolma_100_subreddits_val_78_Overwatch</td><td>30.47</td><td>28.77</td><td>31.13</td><td>29.13</td><td>28.57</td></tr><tr><td>dolma_100_subreddits_val_79電視</td><td>23.97</td><td>22.63</td><td>24.05</td><td>22.75</td><td>22.49</td></tr><tr><td>dolma_100_subreddits_val_80_buildapc</td><td>21.55</td><td>20.22</td><td>21.78</td><td>20.29</td><td>19.98</td></tr><tr><td>dolma_100_subreddits_val_81_askscience</td><td>17.25</td><td>16.39</td><td>17.52</td><td>16.34</td><td>16.11</td></tr><tr><td>dolma_100_subreddits_val_82_programming</td><td>23.66</td><td>22.61</td><td>24.04</td><td>22.55</td><td>22.24</td></tr><tr><td>dolma_100_subreddits_val_83_Guildwars2</td><td>32.98</td><td>31.17</td><td>33.58</td><td>31.39</td><td>30.91</td></tr><tr><td>dolma_100_subreddits_val_84_cars</td><td>22.57</td><td>21.41</td><td>22.73</td><td>21.38</td><td>21.15</td></tr><tr><td>dolma_100_subreddits_val_85formula1</td><td>23.85</td><td>22.65</td><td>24.09</td><td>22.71</td><td>22.49</td></tr><tr><td>dolma_100_subreddits_val_86_sysadmin</td><td>24.23</td><td>22.90</td><td>24.41</td><td>22.96</td><td>22.64</td></tr><tr><td>dolma_100_subreddits_val_87_hockey</td><td>21.46</td><td>20.26</td><td>21.74</td><td>20.37</td><td>20.20</td></tr><tr><td>dolma_100_subreddits_val_88_india</td><td>24.15</td><td>22.92</td><td>24.42</td><td>23.08</td><td>22.68</td></tr><tr><td>dolma_100_subreddits_val_89_SubredditDrama</td><td>19.14</td><td>18.26</td><td>19.63</td><td>18.29</td><td>18.12</td></tr><tr><td>dolma_100_subreddits_val_90_DMAcady</td><td>27.77</td><td>26.31</td><td>28.38</td><td>26.41</td><td>26.00</td></tr><tr><td>dolma_100_subreddits_val_91 Dating_advice</td><td>20.18</td><td>19.27</td><td>20.42</td><td>19.40</td><td>19.21</td></tr><tr><td>dolma_100_subreddits_val_92_Catholicism</td><td>19.11</td><td>18.22</td><td>19.41</td><td>18.17</td><td>18.03</td></tr><tr><td>dolma_100_subreddits_val_93_Drugs</td><td>24.50</td><td>23.29</td><td>24.74</td><td>23.32</td><td>23.12</td></tr><tr><td>dolma_100_subreddits_val_94_trees</td><td>23.56</td><td>22.38</td><td>23.83</td><td>22.41</td><td>22.25</td></tr><tr><td>dolma_100_subreddits_val_95_boardgames</td><td>22.69</td><td>21.48</td><td>23.13</td><td>21.61</td><td>21.38</td></tr><tr><td>dolma_100_subreddits_val_96_Conervative</td><td>22.79</td><td>21.53</td><td>22.97</td><td>21.68</td><td>21.53</td></tr><tr><td>dolma_100_subreddits_val_97_Futurology</td><td>23.55</td><td>22.36</td><td>23.77</td><td>22.37</td><td>22.17</td></tr><tr><td>dolma_100_subreddits_val_98_beyondbump</td><td>21.07</td><td>19.89</td><td>21.22</td><td>20.08</td><td>19.83</td></tr><tr><td>dolma_100_subreddits_val_99weddingplannin</td><td>20.11</td><td>19.01</td><td>20.33</td><td>19.19</td><td>18.96</td></tr><tr><td>falcon-refinedweb_val-00000000</td><td>15.92</td><td>15.46</td><td>17.14</td><td>15.37</td><td>15.22</td></tr><tr><td>falcon-refinedweb_val-0000001</td><td>18.49</td><td>17.91</td><td>19.89</td><td>17.90</td><td>17.71</td></tr><tr><td>falcon-refinedweb_val-0000002</td><td>18.45</td><td>17.90</td><td>19.69</td><td>17.91</td><td>17.68</td></tr><tr><td>falcon-refinedweb_val-0000003</td><td>16.75</td><td>16.23</td><td>17.92</td><td>16.16</td><td>15.89</td></tr><tr><td>falcon-refinedweb_val-0000004</td><td>16.26</td><td>15.66</td><td>17.32</td><td>15.73</td><td>15.41</td></tr><tr><td>falcon-refinedweb_val-0000005</td><td>15.41</td><td>14.96</td><td>16.56</td><td>14.92</td><td>14.74</td></tr><tr><td>gab_val-00000000</td><td>33.19</td><td>30.55</td><td>31.57</td><td>30.73</td><td>30.32</td></tr><tr><td>gab_val-00000001</td><td>35.64</td><td>32.76</td><td>33.96</td><td>32.80</td><td>32.63</td></tr><tr><td>gab_val-00000002</td><td>34.38</td><td>31.68</td><td>32.75</td><td>31.80</td><td>31.65</td></tr><tr><td>gab_val-00000003</td><td>34.86</td><td>32.05</td><td>33.26</td><td>32.20</td><td>32.00</td></tr><tr><td>gab_val-00000004</td><td>36.20</td><td>33.35</td><td>34.58</td><td>33.42</td><td>33.23</td></tr><tr><td>gab_val-00000005</td><td>33.46</td><td>30.82</td><td>31.88</td><td>31.06</td><td>30.72</td></tr><tr><td>gab_val-00000006</td><td>35.76</td><td>32.77</td><td>34.26</td><td>33.04</td><td>32.74</td></tr><tr><td>gab_val-00000007</td><td>35.54</td><td>32.60</td><td>33.76</td><td>32.78</td><td>32.41</td></tr><tr><td>gab_val-00000008</td><td>35.11</td><td>32.03</td><td>33.23</td><td>32.25</td><td>31.86</td></tr><tr><td>gab_val-00000009</td><td>34.13</td><td>31.34</td><td>32.36</td><td>31.50</td><td>31.30</td></tr><tr><td>m2d2_s2orc_unsplit_val_Art</td><td>20.07</td><td>19.80</td><td>21.88</td><td>19.78</td><td>19.44</td></tr><tr><td>m2d2_s2orc_unsplit_val_Philosophy</td><td>14.80</td><td>14.82</td><td>16.77</td><td>14.69</td><td>14.47</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph</td><td>11.70</td><td>11.70</td><td>13.18</td><td>11.52</td><td>11.33</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.CO</td><td>11.47</td><td>11.49</td><td>12.90</td><td>11.37</td><td>11.15</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.EP</td><td>12.76</td><td>12.73</td><td>14.28</td><td>12.60</td><td>12.45</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.GA</td><td>11.70</td><td>11.70</td><td>13.18</td><td>11.52</td><td>11.33</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.HE</td><td>11.85</td><td>11.77</td><td>13.29</td><td>11.62</td><td>11.46</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.IM</td><td>15.36</td><td>15.33</td><td>17.16</td><td>15.21</td><td>14.92</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph.SR</td><td>13.08</td><td>13.08</td><td>14.89</td><td>12.86</td><td>12.70</td></tr><tr><td>m2d2_s2orc_unsplit_val astro-ph_l1</td><td>15.36</td><td>15.33</td><td>17.16</td><td>15.21</td><td>14.92</td></tr><tr><td>m2d2_s2orc_unsplit_val_atom-ph</td><td>12.74</td><td>12.84</td><td>14.44</td><td>12.75</td><td>12.53</td></tr><tr><td>m2d2_s2orc_unsplit_val_chem-ph</td><td>13.20</td><td>13.29</td><td>15.22</td><td>13.14</td><td>12.97</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat</td><td>11.67</td><td>11.78</td><td>13.37</td><td>11.67</td><td>11.50</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.dis-nn</td><td>12.54</td><td>12.67</td><td>14.28</td><td>12.58</td><td>12.38</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.mes-hall</td><td>11.24</td><td>11.50</td><td>13.19</td><td>11.30</td><td>11.10</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.mtrl-sci</td><td>12.19</td><td>12.33</td><td>14.09</td><td>12.18</td><td>11.91</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.other</td><td>11.87</td><td>11.96</td><td>13.55</td><td>11.83</td><td>11.65</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-matquant-gas</td><td>11.67</td><td>11.78</td><td>13.37</td><td>11.67</td><td>11.50</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat柔软</td><td>12.18</td><td>12.23</td><td>13.93</td><td>12.18</td><td>12.02</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.stat-mech</td><td>12.03</td><td>12.14</td><td>13.60</td><td>12.08</td><td>11.89</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.str-el</td><td>10.39</td><td>10.50</td><td>11.98</td><td>10.41</td><td>10.22</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.supr-con</td><td>11.57</td><td>11.66</td><td>13.13</td><td>11.53</td><td>11.30</td></tr><tr><td>m2d2_s2orc_unsplit_val_cond-mat.l1</td><td>12.54</td><td>12.67</td><td>14.28</td><td>12.58</td><td>12.38</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.AI</td><td>11.71</td><td>12.09</td><td>14.20</td><td>12.01</td><td>11.79</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.AR</td><td>13.09</td><td>13.36</td><td>15.30</td><td>13.18</td><td>12.99</td></tr></table>

<table><tr><td>Dataset</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CC</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CE</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CG</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CL</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CR</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CV</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.CY</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.DB</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.DC</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.DL</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.DM</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.DS</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.ET</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.FL</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.GL</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.GR</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.GT</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.HC</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.IR</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.LG</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.LO</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.MA</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.MM</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.MS</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.NA</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.NE</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.NI</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.OH</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.OS</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.PF</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.PL</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.RO</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.SC</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.SD</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.SE</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.SI</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs.SY</td></tr><tr><td>m2d2_s2orc_unsplit_val.cs_l1</td></tr><tr><td>m2d2_s2orc_unsplit_val.econ.EM</td></tr><tr><td>m2d2_s2orc_unsplit_val.econ.TH</td></tr><tr><td>m2d2_s2orc_unsplit_val.econ_11</td></tr><tr><td>m2d2_s2orc_unsplit_val.eess.AS</td></tr><tr><td>m2d2_s2orc_unsplit_val.eess.IV</td></tr><tr><td>m2d2_s2orc_unsplit_val.eess.SP</td></tr><tr><td>m2d2_s2orc_unsplit_val.eess_11</td></tr><tr><td>m2d2_s2orc_unsplit_val.gr-qc</td></tr><tr><td>m2d2_s2orc_unsplit_val.hep-ex</td></tr><tr><td>m2d2_s2orc_unsplit_val.hep-lat</td></tr><tr><td>m2d2_s2orc_unsplit_val.hep-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val.hep-th</td></tr><tr><td>m2d2_s2orc_unsplit_val.math.AC</td></tr><tr><td>m2d2_s2orc_unsplit_val.math.AG</td></tr></table>

<table><tr><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>8.45</td><td>8.81</td><td>10.46</td><td>8.70</td><td>8.54</td></tr><tr><td>13.21</td><td>13.31</td><td>15.01</td><td>13.18</td><td>13.02</td></tr><tr><td>8.39</td><td>8.68</td><td>10.12</td><td>8.59</td><td>8.47</td></tr><tr><td>14.66</td><td>14.75</td><td>16.96</td><td>14.70</td><td>14.47</td></tr><tr><td>14.63</td><td>14.86</td><td>16.72</td><td>14.74</td><td>14.56</td></tr><tr><td>12.68</td><td>12.78</td><td>14.38</td><td>12.66</td><td>12.49</td></tr><tr><td>16.01</td><td>15.93</td><td>17.52</td><td>15.84</td><td>15.67</td></tr><tr><td>11.86</td><td>12.35</td><td>14.66</td><td>12.27</td><td>12.03</td></tr><tr><td>13.60</td><td>14.02</td><td>16.20</td><td>13.79</td><td>13.56</td></tr><tr><td>14.67</td><td>14.83</td><td>17.05</td><td>14.75</td><td>14.50</td></tr><tr><td>8.11</td><td>8.38</td><td>9.84</td><td>8.27</td><td>8.14</td></tr><tr><td>9.63</td><td>9.99</td><td>11.76</td><td>9.88</td><td>9.69</td></tr><tr><td>14.80</td><td>14.95</td><td>17.00</td><td>14.89</td><td>14.67</td></tr><tr><td>9.51</td><td>9.84</td><td>11.64</td><td>9.74</td><td>9.57</td></tr><tr><td>16.51</td><td>16.43</td><td>18.18</td><td>16.38</td><td>16.21</td></tr><tr><td>13.45</td><td>13.60</td><td>15.53</td><td>13.54</td><td>13.29</td></tr><tr><td>9.25</td><td>9.59</td><td>11.34</td><td>9.49</td><td>9.29</td></tr><tr><td>16.76</td><td>16.93</td><td>19.08</td><td>16.84</td><td>16.66</td></tr><tr><td>13.30</td><td>13.46</td><td>15.26</td><td>13.31</td><td>13.21</td></tr><tr><td>10.39</td><td>10.52</td><td>12.14</td><td>10.44</td><td>10.27</td></tr><tr><td>9.75</td><td>10.23</td><td>12.50</td><td>10.03</td><td>9.81</td></tr><tr><td>11.24</td><td>11.65</td><td>14.10</td><td>11.41</td><td>11.19</td></tr><tr><td>13.12</td><td>13.40</td><td>15.29</td><td>13.25</td><td>13.03</td></tr><tr><td>13.98</td><td>14.14</td><td>16.27</td><td>14.11</td><td>13.89</td></tr><tr><td>10.53</td><td>10.80</td><td>12.52</td><td>10.71</td><td>10.47</td></tr><tr><td>13.76</td><td>14.00</td><td>16.10</td><td>13.89</td><td>13.64</td></tr><tr><td>10.00</td><td>10.22</td><td>11.61</td><td>10.04</td><td>9.93</td></tr><tr><td>15.24</td><td>15.43</td><td>17.62</td><td>15.34</td><td>15.10</td></tr><tr><td>14.61</td><td>14.93</td><td>17.35</td><td>14.80</td><td>14.53</td></tr><tr><td>12.60</td><td>12.82</td><td>14.71</td><td>12.70</td><td>12.48</td></tr><tr><td>15.43</td><td>15.74</td><td>18.58</td><td>15.65</td><td>15.40</td></tr><tr><td>13.04</td><td>13.19</td><td>14.95</td><td>13.12</td><td>12.87</td></tr><tr><td>11.10</td><td>11.42</td><td>13.33</td><td>11.30</td><td>11.10</td></tr><tr><td>13.27</td><td>13.42</td><td>15.26</td><td>13.36</td><td>13.13</td></tr><tr><td>17.72</td><td>13.47</td><td>15.46</td><td>13.40</td><td>13.21</td></tr><tr><td>12.03</td><td>12.25</td><td>14.03</td><td>12.19</td><td>11.99</td></tr><tr><td>11.40</td><td>11.79</td><td>13.51</td><td>11.63</td><td>11.39</td></tr><tr><td>8.39</td><td>8.68</td><td>10.12</td><td>8.59</td><td>8.47</td></tr><tr><td>11.62</td><td>11.76</td><td>13.73</td><td>11.68</td><td>11.41</td></tr><tr><td>9.75</td><td>10.16</td><td>11.99</td><td>9.99</td><td>9.88</td></tr><tr><td>9.75</td><td>10.16</td><td>11.99</td><td>9.99</td><td>9.88</td></tr><tr><td>12.05</td><td>12.14</td><td>13.88</td><td>12.09</td><td>11.88</td></tr><tr><td>13.77</td><td>13.89</td><td>15.71</td><td>13.76</td><td>13.54</td></tr><tr><td>11.29</td><td>11.45</td><td>12.94</td><td>11.28</td><td>11.13</td></tr><tr><td>13.77</td><td>13.89</td><td>15.71</td><td>13.76</td><td>13.54</td></tr><tr><td>12.84</td><td>12.99</td><td>14.68</td><td>12.84</td><td>12.71</td></tr><tr><td>10.47</td><td>10.37</td><td>11.61</td><td>10.13</td><td>9.96</td></tr><tr><td>13.13</td><td>13.10</td><td>14.57</td><td>13.02</td><td>12.80</td></tr><tr><td>11.67</td><td>11.81</td><td>13.38</td><td>11.66</td><td>11.45</td></tr><tr><td>11.46</td><td>11.49</td><td>12.71</td><td>11.40</td><td>11.24</td></tr><tr><td>7.08</td><td>7.37</td><td>8.71</td><td>7.26</td><td>7.13</td></tr><tr><td>8.89</td><td>9.27</td><td>11.05</td><td>9.16</td><td>8.95</td></tr></table>

<table><tr><td>Dataset</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.AP</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.AT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.CA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.CO</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.CT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.CV</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.DG</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.DS</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.FA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.GM</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.GN</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.GR</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.GT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.HO</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.KT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.LO</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.MG</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.NA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.NT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.OA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.OC</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.PR</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.QA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.RA</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.RT</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.SG</td></tr><tr><td>m2d2_s2orc_unsplit_val maths.SP</td></tr><tr><td>m2d2_s2orc_unsplit_val maths_11</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin.AO</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin.CD</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin.CG</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin.PS</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin.SI</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin_11</td></tr><tr><td>m2d2_s2orc_unsplit_val_nlin-ex</td></tr><tr><td>m2d2_s2orc_unsplit_val_nclin-th</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.acc-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physicsAo-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.app-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.atm-clus</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.com-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.bio-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.chem-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.class-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.comp-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.data-an</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.ed-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.flu-dyn</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.gen-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.geo-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.hist-ph</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.ins-det</td></tr></table>

<table><tr><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>9.35</td><td>9.53</td><td>10.90</td><td>9.41</td><td>9.35</td></tr><tr><td>8.57</td><td>8.77</td><td>10.16</td><td>8.72</td><td>8.53</td></tr><tr><td>9.18</td><td>9.49</td><td>11.01</td><td>9.36</td><td>9.30</td></tr><tr><td>6.99</td><td>7.33</td><td>8.69</td><td>7.21</td><td>7.08</td></tr><tr><td>9.78</td><td>10.20</td><td>12.04</td><td>10.12</td><td>9.91</td></tr><tr><td>7.81</td><td>8.07</td><td>9.36</td><td>7.99</td><td>7.87</td></tr><tr><td>7.96</td><td>8.18</td><td>9.50</td><td>8.08</td><td>7.98</td></tr><tr><td>7.88</td><td>8.12</td><td>9.61</td><td>8.08</td><td>7.96</td></tr><tr><td>7.71</td><td>7.96</td><td>9.35</td><td>7.88</td><td>7.81</td></tr><tr><td>7.85</td><td>8.15</td><td>9.57</td><td>8.07</td><td>7.93</td></tr><tr><td>6.27</td><td>6.56</td><td>7.82</td><td>6.45</td><td>6.38</td></tr><tr><td>7.39</td><td>7.66</td><td>9.00</td><td>7.51</td><td>7.41</td></tr><tr><td>7.47</td><td>7.71</td><td>9.27</td><td>7.62</td><td>7.47</td></tr><tr><td>14.52</td><td>14.70</td><td>16.52</td><td>14.51</td><td>14.31</td></tr><tr><td>7.54</td><td>7.80</td><td>9.14</td><td>7.70</td><td>7.58</td></tr><tr><td>9.84</td><td>10.41</td><td>12.53</td><td>10.13</td><td>10.03</td></tr><tr><td>8.25</td><td>8.53</td><td>9.99</td><td>8.42</td><td>8.26</td></tr><tr><td>9.85</td><td>10.05</td><td>11.66</td><td>9.95</td><td>9.83</td></tr><tr><td>8.26</td><td>8.51</td><td>9.92</td><td>8.43</td><td>8.31</td></tr><tr><td>7.21</td><td>7.55</td><td>9.07</td><td>7.47</td><td>7.32</td></tr><tr><td>9.70</td><td>10.01</td><td>11.62</td><td>9.85</td><td>9.69</td></tr><tr><td>8.91</td><td>9.20</td><td>10.58</td><td>9.04</td><td>8.99</td></tr><tr><td>8.09</td><td>8.40</td><td>9.93</td><td>8.28</td><td>8.16</td></tr><tr><td>7.18</td><td>7.44</td><td>8.75</td><td>7.39</td><td>7.27</td></tr><tr><td>8.39</td><td>8.71</td><td>10.33</td><td>8.65</td><td>8.49</td></tr><tr><td>8.63</td><td>8.88</td><td>10.36</td><td>8.76</td><td>8.59</td></tr><tr><td>9.39</td><td>9.65</td><td>11.27</td><td>9.52</td><td>9.37</td></tr><tr><td>7.81</td><td>8.07</td><td>9.36</td><td>7.99</td><td>7.87</td></tr><tr><td>11.82</td><td>12.01</td><td>13.77</td><td>11.90</td><td>11.75</td></tr><tr><td>12.73</td><td>12.91</td><td>14.88</td><td>12.87</td><td>12.60</td></tr><tr><td>12.43</td><td>12.75</td><td>14.88</td><td>12.61</td><td>12.44</td></tr><tr><td>11.29</td><td>11.44</td><td>12.86</td><td>11.39</td><td>11.22</td></tr><tr><td>9.44</td><td>9.81</td><td>11.28</td><td>9.64</td><td>9.51</td></tr><tr><td>12.43</td><td>12.75</td><td>14.88</td><td>12.61</td><td>12.44</td></tr><tr><td>13.02</td><td>12.94</td><td>14.61</td><td>12.85</td><td>12.63</td></tr><tr><td>11.65</td><td>11.78</td><td>13.43</td><td>11.68</td><td>11.48</td></tr><tr><td>13.75</td><td>14.01</td><td>16.17</td><td>13.74</td><td>13.58</td></tr><tr><td>13.92</td><td>14.04</td><td>15.91</td><td>13.89</td><td>13.68</td></tr><tr><td>13.70</td><td>13.81</td><td>15.54</td><td>13.62</td><td>13.43</td></tr><tr><td>13.00</td><td>13.13</td><td>15.11</td><td>13.00</td><td>12.74</td></tr><tr><td>12.74</td><td>12.84</td><td>14.44</td><td>12.75</td><td>12.53</td></tr><tr><td>13.30</td><td>13.42</td><td>15.26</td><td>13.32</td><td>13.08</td></tr><tr><td>13.20</td><td>13.29</td><td>15.22</td><td>13.14</td><td>12.97</td></tr><tr><td>11.01</td><td>11.27</td><td>12.85</td><td>11.12</td><td>10.94</td></tr><tr><td>11.23</td><td>11.37</td><td>12.88</td><td>11.26</td><td>11.08</td></tr><tr><td>13.18</td><td>13.33</td><td>14.97</td><td>13.25</td><td>13.00</td></tr><tr><td>12.21</td><td>12.33</td><td>13.88</td><td>12.18</td><td>12.03</td></tr><tr><td>11.81</td><td>11.99</td><td>13.73</td><td>11.81</td><td>11.64</td></tr><tr><td>14.15</td><td>14.39</td><td>16.76</td><td>14.18</td><td>14.03</td></tr><tr><td>14.75</td><td>14.86</td><td>16.81</td><td>14.71</td><td>14.57</td></tr><tr><td>15.57</td><td>15.43</td><td>16.97</td><td>15.40</td><td>15.18</td></tr><tr><td>14.01</td><td>14.16</td><td>16.14</td><td>14.07</td><td>13.79</td></tr></table>

<table><tr><td>Dataset</td><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.med-ph</td><td>14.34</td><td>14.46</td><td>16.50</td><td>14.29</td><td>14.09</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.optics</td><td>12.74</td><td>12.94</td><td>14.64</td><td>12.80</td><td>12.54</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.plasm-ph</td><td>13.65</td><td>13.81</td><td>15.77</td><td>13.69</td><td>13.44</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.pop-ph</td><td>13.80</td><td>13.67</td><td>15.17</td><td>13.60</td><td>13.41</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.soc-ph</td><td>12.79</td><td>12.97</td><td>14.80</td><td>12.83</td><td>12.66</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics.space-ph</td><td>13.00</td><td>13.09</td><td>14.77</td><td>12.94</td><td>12.76</td></tr><tr><td>m2d2_s2orc_unsplit_val_physics_11</td><td>15.57</td><td>15.43</td><td>16.97</td><td>15.40</td><td>15.18</td></tr><tr><td>m2d2_s2orc_unsplit_val_plasm-ph</td><td>13.65</td><td>13.81</td><td>15.77</td><td>13.69</td><td>13.44</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio</td><td>13.69</td><td>13.87</td><td>15.75</td><td>13.75</td><td>13.50</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.BM</td><td>13.28</td><td>13.52</td><td>15.72</td><td>13.41</td><td>13.19</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.CB</td><td>12.06</td><td>12.34</td><td>14.21</td><td>12.19</td><td>11.97</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.GN</td><td>13.21</td><td>11.40</td><td>12.74</td><td>11.32</td><td>11.16</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.MN</td><td>11.96</td><td>11.95</td><td>13.36</td><td>11.90</td><td>11.70</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.NC</td><td>13.69</td><td>13.87</td><td>15.75</td><td>13.75</td><td>13.50</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.OT</td><td>14.90</td><td>14.94</td><td>17.16</td><td>14.92</td><td>14.73</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.PE</td><td>12.57</td><td>12.71</td><td>14.62</td><td>12.69</td><td>12.41</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.QM</td><td>12.49</td><td>12.69</td><td>14.44</td><td>12.56</td><td>12.40</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.SC</td><td>13.68</td><td>13.85</td><td>15.60</td><td>13.75</td><td>13.53</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio.TO</td><td>13.49</td><td>13.53</td><td>15.32</td><td>13.48</td><td>13.33</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-bio_I1</td><td>13.69</td><td>13.87</td><td>15.75</td><td>13.75</td><td>13.50</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.CP</td><td>11.37</td><td>11.61</td><td>13.36</td><td>11.41</td><td>11.28</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.EC</td><td>11.72</td><td>11.89</td><td>13.77</td><td>11.77</td><td>11.63</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.GN</td><td>13.79</td><td>13.91</td><td>15.73</td><td>13.83</td><td>13.61</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.MF</td><td>9.91</td><td>10.21</td><td>11.92</td><td>10.04</td><td>9.90</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.PM</td><td>11.00</td><td>11.31</td><td>13.14</td><td>11.14</td><td>10.94</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.PR</td><td>15.87</td><td>9.25</td><td>10.37</td><td>9.20</td><td>9.03</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.RM</td><td>11.35</td><td>11.49</td><td>13.08</td><td>11.41</td><td>11.22</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.ST</td><td>12.43</td><td>12.46</td><td>14.18</td><td>12.43</td><td>12.26</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin.TR</td><td>12.79</td><td>13.14</td><td>15.32</td><td>12.89</td><td>12.74</td></tr><tr><td>m2d2_s2orc_unsplit_val_q-fin_I1</td><td>13.79</td><td>13.91</td><td>15.73</td><td>13.83</td><td>13.61</td></tr><tr><td>m2d2_s2orc_unsplit_valquant-ph</td><td>11.18</td><td>11.44</td><td>13.18</td><td>11.32</td><td>11.11</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat.AP</td><td>13.37</td><td>13.56</td><td>15.52</td><td>13.42</td><td>13.15</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat.CO</td><td>13.07</td><td>12.56</td><td>14.42</td><td>12.46</td><td>12.24</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat.ME</td><td>11.09</td><td>11.26</td><td>12.91</td><td>11.11</td><td>10.87</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat.ML</td><td>11.13</td><td>11.39</td><td>13.29</td><td>11.23</td><td>11.06</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat.OT</td><td>11.31</td><td>11.55</td><td>13.28</td><td>11.45</td><td>11.24</td></tr><tr><td>m2d2_s2orc_unsplit_val.stat_I1</td><td>13.07</td><td>12.56</td><td>14.42</td><td>12.46</td><td>12.24</td></tr><tr><td>m2d2_s2orc_unsplit_val.supr-con</td><td>11.57</td><td>11.66</td><td>13.13</td><td>11.53</td><td>11.30</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>12.30</td><td>11.90</td><td>12.82</td><td>11.78</td><td>11.66</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>12.13</td><td>11.74</td><td>12.82</td><td>11.63</td><td>11.48</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>14.06</td><td>13.86</td><td>15.17</td><td>13.79</td><td>13.57</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>12.16</td><td>11.80</td><td>12.74</td><td>11.79</td><td>11.55</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>11.75</td><td>11.25</td><td>12.03</td><td>11.17</td><td>11.03</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>10.01</td><td>9.63</td><td>10.36</td><td>9.58</td><td>9.54</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>12.13</td><td>11.85</td><td>12.83</td><td>11.73</td><td>11.58</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Culture_and_the_</td><td>12.36</td><td>12.09</td><td>13.05</td><td>11.99</td><td>11.87</td></tr><tr><td>m2d2_wikipedia_unsplit_val_General_referece</td><td>11.80</td><td>11.46</td><td>12.43</td><td>11.46</td><td>11.30</td></tr><tr><td>m2d2_wikipedia_unsplit_val_General_referece_</td><td>10.52</td><td>10.20</td><td>10.96</td><td>10.12</td><td>9.99</td></tr><tr><td>m2d2_wikipedia_unsplit_val_General_referece_</td><td>11.80</td><td>11.46</td><td>12.43</td><td>11.46</td><td>11.30</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>10.75</td><td>10.47</td><td>11.14</td><td>10.37</td><td>10.30</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>9.64</td><td>9.29</td><td>9.95</td><td>9.27</td><td>9.16</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>10.10</td><td>9.80</td><td>10.43</td><td>9.71</td><td>9.56</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>9.14</td><td>8.83</td><td>9.59</td><td>8.63</td><td>8.54</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>8.91</td><td>8.68</td><td>9.40</td><td>8.61</td><td>8.47</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>10.75</td><td>10.47</td><td>11.14</td><td>10.37</td><td>10.30</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Health_and_fitness</td><td>12.91</td><td>12.49</td><td>13.61</td><td>12.42</td><td>12.28</td></tr><tr><td>m2d2_wikipedia_unsplit_val_History_and_even</td><td>13.65</td><td>13.29</td><td>14.48</td><td>13.20</td><td>13.00</td></tr><tr><td>m2d2_wikipedia_unsplit_val_History_and_even</td><td>11.77</td><td>11.44</td><td>12.36</td><td>11.36</td><td>11.26</td></tr><tr><td>m2d2_wikipedia_unsplit_val_History_and_even</td><td>12.78</td><td>12.41</td><td>13.46</td><td>12.37</td><td>12.12</td></tr><tr><td>m2d2_wikipedia_unsplit_val_History_and_even</td><td>12.36</td><td>11.88</td><td>12.87</td><td>11.79</td><td>11.64</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Human_activites</td><td>12.43</td><td>12.03</td><td>12.98</td><td>11.95</td><td>11.81</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Humanactivites_</td><td>12.43</td><td>12.03</td><td>12.98</td><td>11.95</td><td>11.81</td></tr><tr><td>m2d2_wikipedia_unsplit_val_HumanActivites_</td><td>12.47</td><td>12.05</td><td>13.12</td><td>12.00</td><td>11.82</td></tr><tr><td>m2d2_wikipedia_unsplit_val/Mathematics_and</td><td>12.90</td><td>12.51</td><td>13.79</td><td>12.48</td><td>12.29</td></tr><tr><td>m2d2_wikipedia_unsplit_val/Mathematics_and_</td><td>8.24</td><td>8.26</td><td>9.37</td><td>8.28</td><td>8.06</td></tr><tr><td>m2d2_wikipedia_unsplit_val/Mathematics_and_</td><td>13.21</td><td>12.87</td><td>13.90</td><td>12.85</td><td>12.67</td></tr><tr><td>m2d2_wikipedia_unsplit_val/Mathematics_and_</td><td>12.90</td><td>12.51</td><td>13.79</td><td>12.48</td><td>12.29</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Natural_and_phys</td><td>9.19</td><td>8.22</td><td>8.81</td><td>7.97</td><td>7.96</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Natural_and_phys</td><td>10.97</td><td>10.70</td><td>11.53</td><td>10.64</td><td>10.51</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Natural_and_phys</td><td>11.69</td><td>11.36</td><td>12.28</td><td>11.22</td><td>11.05</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Natural_and_phys</td><td>10.43</td><td>10.11</td><td>10.95</td><td>10.00</td><td>9.82</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Natural_and_phys</td><td>11.48</td><td>11.09</td><td>11.93</td><td>10.98</td><td>10.90</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Philosophy_and_t</td><td>11.83</td><td>11.72</td><td>13.04</td><td>11.60</td><td>11.45</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Philosophy_and_t</td><td>12.00</td><td>11.61</td><td>12.66</td><td>11.57</td><td>11.43</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Philosophy_and_t</td><td>10.94</td><td>10.61</td><td>11.34</td><td>10.56</td><td>10.42</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Religion_and_bel</td><td>12.81</td><td>12.45</td><td>13.44</td><td>12.38</td><td>12.19</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Religion_and_bel</td><td>11.11</td><td>10.80</td><td>11.66</td><td>10.71</td><td>10.58</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Religion_and_bel</td><td>11.46</td><td>11.06</td><td>11.86</td><td>10.95</td><td>10.85</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Religion_and_bel</td><td>12.38</td><td>12.03</td><td>12.94</td><td>11.91</td><td>11.79</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Society_and_socia</td><td>10.53</td><td>10.24</td><td>11.03</td><td>10.16</td><td>10.05</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Society_and_socia</td><td>10.47</td><td>10.16</td><td>10.95</td><td>10.14</td><td>10.04</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Society_and_socia</td><td>12.48</td><td>12.13</td><td>13.02</td><td>12.07</td><td>11.93</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Technology_and_</td><td>8.51</td><td>8.18</td><td>8.66</td><td>7.93</td><td>7.88</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Technology_and_</td><td>12.45</td><td>12.07</td><td>13.00</td><td>12.03</td><td>11.88</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Technology_and_</td><td>13.62</td><td>13.23</td><td>14.56</td><td>13.18</td><td>12.97</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Technology_and_</td><td>13.00</td><td>12.72</td><td>13.87</td><td>12.64</td><td>12.43</td></tr><tr><td>m2d2_wikipedia_unsplit_val_Technology_and_</td><td>14.34</td><td>13.90</td><td>15.20</td><td>13.94</td><td>13.73</td></tr><tr><td>manosphere_meta sep_val_avfm</td><td>19.42</td><td>19.27</td><td>21.88</td><td>19.64</td><td>19.18</td></tr><tr><td>manosphere_meta sep_val.incels</td><td>11.26</td><td>12.18</td><td>21.40</td><td>11.51</td><td>11.29</td></tr><tr><td>manosphere_meta sep_val_mgtow</td><td>24.83</td><td>24.27</td><td>27.50</td><td>24.12</td><td>23.80</td></tr><tr><td>manosphere_meta sep_val_puaforum</td><td>24.22</td><td>23.85</td><td>26.52</td><td>23.86</td><td>23.52</td></tr><tr><td>manosphere_meta sep_val_red-pill-talk</td><td>34.59</td><td>33.90</td><td>37.26</td><td>33.90</td><td>33.27</td></tr><tr><td>manosphere_meta sep_val_reddit</td><td>20.63</td><td>19.78</td><td>21.10</td><td>19.94</td><td>19.58</td></tr><tr><td>manosphere_meta sep_val_rooshv</td><td>22.46</td><td>22.17</td><td>24.78</td><td>22.01</td><td>21.69</td></tr><tr><td>manosphere_meta sep_val_the attractsion</td><td>20.85</td><td>20.57</td><td>23.17</td><td>20.57</td><td>20.20</td></tr><tr><td>mc4_val-00000000</td><td>8.35</td><td>8.41</td><td>10.02</td><td>8.23</td><td>8.15</td></tr><tr><td>mc4_val-00000001</td><td>12.17</td><td>11.97</td><td>13.58</td><td>11.74</td><td>11.64</td></tr><tr><td>mc4_val-00000002</td><td>9.96</td><td>10.06</td><td>11.96</td><td>9.86</td><td>9.67</td></tr><tr><td>mc4_val-00000003</td><td>11.38</td><td>11.29</td><td>12.77</td><td>11.12</td><td>11.00</td></tr><tr><td>mc4_val-00000004</td><td>11.96</td><td>11.64</td><td>13.03</td><td>11.50</td><td>11.35</td></tr><tr><td>ptb_val</td><td>15.92</td><td>16.65</td><td>19.37</td><td>16.00</td><td>15.92</td></tr><tr><td>redpajama_val_arxiv</td><td>5.15</td><td>5.28</td><td>5.78</td><td>5.12</td><td>5.09</td></tr><tr><td>redpajama_val_books</td><td>12.91</td><td>12.71</td><td>13.60</td><td>12.61</td><td>12.50</td></tr><tr><td>redpajama_val_c4</td><td>13.01</td><td>12.51</td><td>13.55</td><td>12.49</td><td>12.27</td></tr></table>

Dataset

redpajama_val_commoncrawl

redpajama_val_github

redpajama_val_stackexchange

redpajama_val_wikipedia

twitterAAE_HELM.fixed_val_AA

twitterAAE_HELM.fixed_val_white

wiktext_103_val

<table><tr><td>Llama</td><td>Mamba</td><td>RWKV-4</td><td>xLSTM[7:1]</td><td>xLSTM[1:0]</td></tr><tr><td>10.90</td><td>10.56</td><td>11.70</td><td>10.52</td><td>10.35</td></tr><tr><td>1.66</td><td>1.66</td><td>1.75</td><td>1.65</td><td>1.64</td></tr><tr><td>3.73</td><td>3.72</td><td>4.03</td><td>3.68</td><td>3.63</td></tr><tr><td>4.64</td><td>4.38</td><td>4.68</td><td>4.35</td><td>4.29</td></tr><tr><td>346.98</td><td>302.79</td><td>310.30</td><td>301.65</td><td>289.97</td></tr><tr><td>118.62</td><td>107.34</td><td>109.13</td><td>107.65</td><td>105.13</td></tr><tr><td>11.74</td><td>11.76</td><td>13.73</td><td>11.32</td><td>11.41</td></tr></table>

# Footnotes:

Page 30: 1https://python.org $^{2}$ https://pytorch.org <sup>3</sup>https://docs.nvidia.com/cuda/archive/12.1.0/ 
Page 34: 4The keys are distributed on the "evaluation part" of the sequence given a power-law distribution. This is motivated by similar structures in natural language text. 
Page 39: $^{5}$ https://huggingface.co/docs/transformers/en/model_doc/gpt2  
 $^{6}$ https://github.com/state-spaces/mamba  
 $^{7}$ https://github.com/BlinkDL/RWKV-LM/  
 $^{8}$ https://github.com/sustcsonglin/flash-linear-attention  
 $^{9}$ https://github.com/BlinkDL/RWKV-LM/blob/64b7fe4c66cff7da37019630268075b0558f6dc5/RWKV-v5/train.py#L44 
Page 42: 10https://huggingface.co/datasets/allenai/paloma 
