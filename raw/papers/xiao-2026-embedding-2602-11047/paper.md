Embedding Inversion via Conditional Masked Diffusion Language Models
====================================================================

Han Xiao  
Jina AI by Elastic  
han.xiao@jina.ai

###### Abstract

We frame embedding inversion as conditional masked diffusion, recovering all tokens in parallel through iterative denoising rather than sequential autoregressive generation. A masked diffusion language model is conditioned on the target embedding via adaptive layer normalization, requiring only 8 forward passes through a 78M parameter model with no access to the target encoder. On 32-token sequences across three embedding models, the method achieves up to 81.3% token accuracy. Source code and live demo are available at [https://github.com/jina-ai/embedding-inversion-demo](https://github.com/jina-ai/embedding-inversion-demo "").

1 Introduction
--------------

Text embeddings power modern retrieval systems, and production deployments routinely treat them as safe, anonymized representations. Vec2Text*(Morris et al., [2023] as much as text"))* challenged this assumption by recovering 92% of 32-token sequences from their embeddings using a T5 encoder-decoder with iterative correction. Subsequent work has expanded the attack surface: ALGEN*(Chen et al., [2025])* enables cross model inversion with few-shot alignment, and Zero2Text*(Kim et al., [2026])* achieves training free inversion via LLM priors and online regression.

These methods share a common design: they generate tokens autoregressively, then iteratively re-embed the hypothesis to compute a correction signal. This creates two practical bottlenecks. First, each correction step requires a forward pass through the target embedding model, making the attack cost proportional to the number of iterations. Vec2Text typically requires over 20 iterations per sequence. Second, the autoregressive backbone accumulates errors left-to-right, with no mechanism to revise earlier tokens based on later context.

We propose an alternative formulation: embedding inversion as conditional masked diffusion. Starting from a fully masked sequence, a denoising model iteratively reveals tokens at all positions in parallel, conditioned on the target embedding vector via adaptive layer normalization. Correction is built into the diffusion process itself: each step refines all positions simultaneously using global context, without ever reembedding the current hypothesis. The embedding vector enters only through AdaLN modulation, making the approach encoder agnostic: the same architecture applies to any embedding model without alignment training. We validate on three encoders with different architectures and dimensionalities, achieving up to 81.3% token recovery on 32-token sequences with no access to the target encoder at inference time.

<img src='figures/architecture.png' alt='Refer to caption' title='' width='476' height='258' />

*Figure 1: Architecture of the Conditional Masked Diffusion Language Model. The embedding vector is projected and injected into each transformer layer via AdaLN conditioning. The model predicts original tokens at masked positions through iterative denoising.*

2 Related Work
--------------

### 2.1 Embedding Inversion Attacks

Embedding inversion emerged as a research area with Vec2Text*(Morris et al., [2023] as much as text"))*, which demonstrated that T5 encoder-decoder models could recover 92% exact matches on 32-token sequences through hypothesis generation followed by iterative correction. The correction mechanism computes embedding distances and refines outputs through multiple forward passes, but requires compatible embedding architectures and suffers from autoregressive error accumulation.

The field has advanced rapidly with methods addressing Vec2Text’s architectural constraints. ALGEN*(Chen et al., [2025])* introduced few-shot cross model alignment, demonstrating that embedding spaces can be aligned with only 1k training samples through one-step optimization, enabling inversion across incompatible architectures. Zero2Text*(Kim et al., [2026])* achieved training free inversion using LLM priors combined with online ridge regression, eliminating the need for paired training data entirely. On MS MARCO, Zero2Text achieved 1.8$\times$ ROUGE-L improvement over baselines in black-box cross-domain settings. Together, these methods show that embedding inversion generalizes across architectures and data regimes. Our work contributes the first diffusion-based approach, replacing sequential generation and explicit correction with parallel masked denoising.

### 2.2 Discrete Diffusion Models

Discrete diffusion began with D3PM*(Austin et al., [2021])*, which extended continuous diffusion to categorical distributions through absorbing state processes. Masked Diffusion Language Models*(Sahoo et al., [2024])* simplified this framework by using uniform masking with log-linear noise schedules, achieving competitive language modeling performance while enabling parallel generation. The field has since diversified: Score Entropy Discrete Diffusion*(Lou et al., [2024])* introduced entropy-based scoring, providing improved sample quality through better noise scheduling. Constrained Discrete Diffusion*(Cardei et al., [2025])* added constraint satisfaction mechanisms for controlled generation tasks.

Our conditional MDLM builds on this foundation, adapting masked diffusion to the embedding inversion task through adaptive layer normalization conditioning.

### 2.3 Conditional Diffusion

Conditioning mechanisms for diffusion models have evolved primarily in continuous domains. Classifier-free guidance*(Ho and Salimans, [2022])* enables conditional generation by training a single model with dropped conditioning signals, then interpolating predictions at inference. Classifier guidance*(Dhariwal and Nichol, [2021])* uses external classifier gradients to steer generation toward desired attributes. For vision tasks, Diffusion Transformers*(Peebles and Xie, [2023])* introduced adaptive layer normalization that modulates layer normalization parameters based on conditioning signals, providing fine-grained control over feature representations at each transformer layer. We adapt AdaLN to discrete text generation, using it to inject embedding information into each denoising step. This conditioning mechanism is architecture agnostic, working with any embedding model without requiring alignment training or model specific modifications, in contrast to Vec2Text’s T5-specific architecture or ALGEN’s explicit alignment procedure.

3 Method
--------

We use the following notation throughout: $\mathbf{x}\=(x_{1},\ldots,x_{n})$ denotes a token sequence of length $n$ from vocabulary $\mathcal{V}$; $\mathbf{e}\in\mathbb{R}^{d}$ denotes the embedding vector; $t\in[0,1]$ denotes the diffusion timestep with $t\=0$ being fully unmasked and $t\=1$ being fully masked; $\theta$ denotes the model parameters; $\mathbf{c}\in\mathbb{R}^{D_{h}}$ denotes the projected conditioning vector with hidden dimension $D_{h}$; $x_{t}$ denotes the masked sequence at timestep $t$; $x_{0}$ denotes the original unmasked sequence.

### 3.1 Problem Formulation

Given an embedding function $f:\mathcal{V}^{n}\to\mathbb{R}^{d}$ and embedding vector $\mathbf{e}\=f(\mathbf{x})$, we seek to recover the original sequence by maximizing the conditional probability:

|  | $\hat{\mathbf{x}}\=\arg\max_{\mathbf{x}^{\prime}}p_{\theta}(\mathbf{x}^{\prime}|\mathbf{e})$ |  | (1) |
| --- | --- | --- | --- |

where $p_{\theta}(\mathbf{x}|\mathbf{e})$ is modeled using masked diffusion with adaptive layer normalization conditioning.

### 3.2 Masked Diffusion Process

Following MDLM*(Sahoo et al., [2024])*, we define a forward noising process that gradually masks tokens according to a noise schedule. For each token position $i$ at timestep $t$, the forward transition is:

|  | $q(x_{t,i}|x_{0,i})\=\begin{cases}x_{0,i}\&\text{with probability }\alpha_{t}\\ [\text{MASK}]\&\text{with probability }1-\alpha_{t}\end{cases}$ |  | (2) |
| --- | --- | --- | --- |

where $x_{t,i}$ is the token at position $i$ and timestep $t$, $x_{0,i}$ is the original token, and $\alpha_{t}$ is the survival probability. We use the log-linear schedule $\alpha_{t}\=e^{-\lambda t}$ with $\lambda\=5.0$, which concentrates masking in later timesteps while preserving structure in early denoising stages. The reverse process learns to predict the original token $x_{0,i}$ at each masked position given the partially masked sequence $x_{t}$, timestep $t$, and conditioning embedding $\mathbf{e}$. The model outputs a categorical distribution over the vocabulary:

|  | $p_{\theta}(x_{0,i}|x_{t},t,\mathbf{e})\=\text{Categorical}(\text{softmax}(\mathbf{z}_{i}))$ |  | (3) |
| --- | --- | --- | --- |

where $\mathbf{z}_{i}\in\mathbb{R}^{|\mathcal{V}|}$ are the logits for position $i$ produced by the transformer network parameterized by $\theta$. The model predicts all positions in parallel, conditioned on the global context provided by the embedding. We minimize the Rao-Blackwellized ELBO with $1/t$ weighting:

|  | $\mathcal{L}(\theta)\=\mathbb{E}_{t\sim\text{Uniform}[0,1]}\mathbb{E}_{\mathbf{x}_{0}\sim\mathcal{D}}\mathbb{E}_{x_{t}\sim q(x_{t}|x_{0})}\left[\frac{1}{t}\sum_{i:x_{t,i}\=[\text{MASK}]}-\log p_{\theta}(x_{0,i}|x_{t},t,\mathbf{e})\right]$ |  | (4) |
| --- | --- | --- | --- |

where $\mathcal{D}$ is the data distribution, the sum is over masked positions only, and the $1/t$ weighting upweights the low-noise regime ($t\to 0$), where few tokens remain masked and precise reconstruction matters most.

### 3.3 Model Architecture

Our model consists of three components: embedding projection, transformer backbone, and adaptive layer normalization conditioning (Figure[1]). The input embedding $\mathbf{e}\in\mathbb{R}^{d}$, where $d$ is determined by the target encoder, is projected to the transformer hidden dimension $D_{h}$ via a two-layer MLP:

|  | $\mathbf{c}\=\mathbf{W}_{2}\cdot\text{GELU}(\mathbf{W}_{1}\mathbf{e}+\mathbf{b}_{1})+\mathbf{b}_{2}$ |  | (5) |
| --- | --- | --- | --- |

where $\mathbf{W}_{1}\in\mathbb{R}^{D_{h}\times d}$, $\mathbf{W}_{2}\in\mathbb{R}^{D_{h}\times D_{h}}$, and $\mathbf{b}_{1},\mathbf{b}_{2}\in\mathbb{R}^{D_{h}}$ are learned parameters. We use an 8-layer transformer with 12 attention heads and FFN dimension $4D_{h}$. In our experiments, $D_{h}\=768$, yielding FFN dimension 3072. Input and output embeddings are weight-tied to reduce parameters given the large vocabulary size $|\mathcal{V}|\=50257$.

Following DiT*(Peebles and Xie, [2023])*, we condition each transformer layer on both the timestep $t$ and the embedding vector $\mathbf{c}$ via adaptive layer normalization. For each layer $\ell$, we compute modulation parameters:

|  | $\displaystyle\gamma_{t}^{(\ell)},\beta_{t}^{(\ell)}$ | $\displaystyle\=\text{MLP}_{t}^{(\ell)}(t)$ |  | (6) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\gamma_{c}^{(\ell)},\beta_{c}^{(\ell)}$ | $\displaystyle\=\text{MLP}_{c}^{(\ell)}(\mathbf{c})$ |  | (7) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\gamma^{(\ell)}$ | $\displaystyle\=\gamma_{t}^{(\ell)}+\gamma_{c}^{(\ell)}$ |  | (8) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\beta^{(\ell)}$ | $\displaystyle\=\beta_{t}^{(\ell)}+\beta_{c}^{(\ell)}$ |  | (9) |
| --- | --- | --- | --- | --- |

where $\text{MLP}_{t}^{(\ell)}$ and $\text{MLP}_{c}^{(\ell)}$ are single-layer MLPs that output vectors of dimension $D_{h}$. The layer normalization at layer $\ell$ is then modulated:

|  | $\text{AdaLN}(\mathbf{h}^{(\ell)})\=\gamma^{(\ell)}\odot\frac{\mathbf{h}^{(\ell)}-\mu(\mathbf{h}^{(\ell)})}{\sigma(\mathbf{h}^{(\ell)})}+\beta^{(\ell)}$ |  | (10) |
| --- | --- | --- | --- |

where $\mathbf{h}^{(\ell)}\in\mathbb{R}^{n\times D_{h}}$ is the input to layer $\ell$, $\mu(\cdot)$ and $\sigma(\cdot)$ compute mean and standard deviation over the hidden dimension, and $\odot$ denotes element-wise multiplication. This formulation allows the conditioning signal and timestep to independently modulate the layer normalization at each depth, providing fine-grained control over feature representations.

### 3.4 Decoding Strategies

We consider four strategies for generating tokens from the trained model.

Sequential greedy decoding unmasks tokens left to right:

|  | $x_{i}\=\arg\max_{v\in\mathcal{V}}p_{\theta}(v\mid x_{<i},[\text{MASK}]^{n-i},\mathbf{e},t)$ |  | (11) |
| --- | --- | --- | --- |

where $t\=(n-i)/n$ is the fraction of remaining masked tokens. This produces coherent text but sacrifices the parallel nature of diffusion.

Euler sampling applies the Euler method to the reverse diffusion process, starting from $x_{1}\=[\text{MASK}]^{n}$ with uniform timesteps from $t{\=}1$ to $t{\=}0$:

|  | $\hat{x}_{0,i}\sim p_{\theta}(x_{0,i}\mid x_{t},t,\mathbf{e})\quad\forall\,i$ |  | (12) |
| --- | --- | --- | --- |

sampling all positions simultaneously at each step.

Euler with remasking adds a correction mechanism: after each Euler step, a fraction $\tau$ of positions with the lowest confidence $\max_{v}p_{\theta}(v\mid x_{t},t,\mathbf{e})$ are re-masked:

|  | $x_{t^{\prime},i}\=\begin{cases}\hat{x}_{0,i}\&\text{if position }i\text{ is not in the bottom-}\tau\text{ fraction}\\ [\text{MASK}]\&\text{otherwise}\end{cases}$ |  | (13) |
| --- | --- | --- | --- |

allowing subsequent steps to refine uncertain predictions. We find $\tau\=0.05$ optimal (Table[3]).

Two-stage decoding first generates a hypothesis via sequential greedy decoding, then refines it using Euler sampling initialized at this hypothesis rather than a fully masked sequence.

4 Experimental Results
----------------------

We train on 2M samples from C4*(Raffel et al., [2020])*, filtered to 32 tokens. We use the GPT-2 tokenizer with vocabulary size 50,257. Training uses batch size 400 for 200K steps with AdamW optimizer at learning rate $10^{-4}$ and EMA decay 0.9999. We employ a log-linear noise schedule with $\lambda\=5.0$ following*Sahoo et al. ([2024])*. Timesteps are sampled uniformly from $[0,1]$. Embeddings are computed using the target encoder and cached. We evaluate on three embedding models with different architectures and dimensionalities: jina-embeddings-v3*(Sturua et al., [2025])* with 570M parameters and 1024-dimensional embeddings, Qwen3-Embedding-0.6B with 600M parameters and 1024-dimensional embeddings, and EmbeddingGemma-300m with 300M parameters and 768-dimensional embeddings. We train separate models for each encoder using multilingual data from mC4 to assess generalization across embedding spaces.

The complete model has approximately 270M parameters due to the large vocabulary embeddings, but only 78M trainable parameters consisting of the 8 transformer layers, embedding projection MLP, and AdaLN conditioning MLPs.

Table[1] shows results across all three embedding encoders using sequential greedy decoding, which provides the highest token accuracy. Qwen3-Embedding achieves the best performance at 81.3% token accuracy, followed by EmbeddingGemma at 78.8% and jina-v3 at 76.0%. All models are trained on multilingual data from mC4.

*Table 1: Performance across embedding encoders using sequential greedy decoding. All trained on 2M multilingual samples from mC4. Best checkpoint selected by validation loss.*

| Encoder | Token Acc. | Steps | Val Loss | Vocab | Embed Dim |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 81.3% | 72.5K | 1.317 | 152K | 1024 |
| EmbeddingGemma-300m | 78.8% | 49.5K | 1.55 | 262K | 768 |
| jina-embeddings-v3 | 76.0% | 62.5K | 1.60 | 250K | 1024 |

Table[2] compares four decoding strategies across all three encoders on 10 languages. Cosine similarity is averaged over the same sentence translated into English, Chinese, German, Japanese, French, Spanish, Korean, Russian, Arabic, and Portuguese. Sequential greedy consistently achieves the highest similarity across encoders.

*Table 2: Average cosine similarity across decoding strategies and encoders, evaluated on 10 languages per encoder.*

| Decoding Method | jina-embeddings-v3 | Qwen3-Embedding | EmbeddingGemma |
| --- | --- | --- | --- |
| Sequential Greedy | 0.715 | 0.585 | 0.621 |
| Euler Sampling | 0.667 | 0.556 | 0.604 |
| Euler + Remasking | 0.665 | 0.584 | 0.595 |
| Two-Stage | 0.667 | 0.591 | 0.605 |

Euler with remasking at 0.05 improves over vanilla Euler by 2.6 percentage points in token accuracy. Two-stage decoding achieves highest exact match at 13.1%. Baselines confirm that embedding conditioning is essential: random tokens achieve 0.02% accuracy, while unconditional LM achieves 2.1% despite high fluency with BLEU score 89.3.

Table[3] shows optimal performance at remask probability 0.05 for Euler sampling with adaptive remasking. Higher rates discard correct predictions, lower rates provide insufficient correction.

*Table 3: Effect of remasking probability on Euler sampling performance.*

| Re-mask Prob. | Token Acc. | Cosine Sim. | BLEU |
| --- | --- | --- | --- |
| 0.00 (no re-mask) | 65.2% | 0.81 | 38.7 |
| 0.05 | 67.8% | 0.82 | 42.1 |
| 0.10 | 66.3% | 0.81 | 40.2 |
| 0.20 | 63.7% | 0.80 | 37.1 |

<img src='x1.png' alt='Refer to caption' title='' width='660' height='581' />

*(a) Token accuracy*

<img src='x2.png' alt='Refer to caption' title='' width='660' height='581' />

*(b) Validation loss*

*Figure 2: Training dynamics across three embedding encoders on 2M multilingual samples. Qwen3-Embedding reaches 81.3% token accuracy at 72.5K steps with validation loss 1.32. All models show diminishing returns beyond 50K steps, suggesting architectural improvements rather than extended training as the path to further gains.*

5 Conclusion
------------

We presented embedding inversion via conditional masked diffusion, achieving 81.3% token accuracy across three embedding models with a 78M parameter decoder that requires no access to the target encoder. The progression from Vec2Text, which requires a compatible encoder architecture, to ALGEN, which needs alignment training, to Zero2Text, which queries the target API, to our work, which needs none of these, demonstrates that inversion attacks are becoming more accessible, not less. Production systems that cache or transmit embedding vectors under the assumption of irreversibility should treat them as sensitive data requiring protection equivalent to the original text.

Current limitations include the restriction to 32-token sequences, whereas real documents are substantially longer. Scaling to longer sequences via hierarchical diffusion or sliding window approaches is a natural next step. Incorporating language model priors through classifier-free guidance could further improve reconstruction quality, particularly for low-frequency tokens. On the defense side, encrypted vector search and inversion-resistant embeddings through adversarial training or differential privacy on the embedding space remain important open problems for security-critical deployments.

References
----------

* J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg (2021)Structured denoising diffusion models in discrete state-spaces.In NeurIPS,Vol. 34,  pp. 17981–17993.Cited by: [§2.2].
* M. Cardei, J. K. Christopher, T. Hartvigsen, B. R. Bartoldson, B. Kailkhura, and F. Fioretto (2025)Constrained language generation with discrete diffusion models.arXiv preprint arXiv:2503.09790.Cited by: [§2.2].
* Y. Chen, Q. Xu, and J. Bjerva (2025)ALGEN: few-shot inversion attacks on textual embeddings via cross-model alignment and generation.In ACL,Cited by: [§1],[§2.1].
* P. Dhariwal and A. Nichol (2021)Diffusion models beat gans on image synthesis.In NeurIPS,Vol. 34,  pp. 8780–8794.Cited by: [§2.3].
* J. Ho and T. Salimans (2022)Classifier-free diffusion guidance.In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications,Cited by: [§2.3].
* D. Kim, D. Kang, K. Lee, H. Baek, and B. B. Kang (2026)Zero2Text: zero-training cross-domain inversion attacks on textual embeddings.arXiv preprint arXiv:2602.01757.Cited by: [§1],[§2.1].
* A. Lou, C. Meng, and S. Ermon (2024)Discrete diffusion modeling by estimating the ratios of the data distribution.In ICML, pp. 32819–32848.Cited by: [§2.2].
* J. X. Morris, V. Kuleshov, V. Shmatikov, and A. M. Rush (2023)Text embeddings reveal (almost) as much as text.In EMNLP, pp. 12448–12460.Cited by: [§1],[§2.1].
* W. Peebles and S. Xie (2023)Scalable diffusion models with transformers.In ICCV, pp. 4195–4205.Cited by: [§2.3],[§3.3].
* C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020)Exploring the limits of transfer learning with a unified text-to-text transformer.Journal of Machine Learning Research 21 (140),  pp. 1–67.Cited by: [§4].
* S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov (2024)Simple and effective masked diffusion language models.In NeurIPS,Vol. 37.Cited by: [§2.2],[§3.2],[§4].
* S. Sturua, I. Mohr, M. K. Akram, M. Günther, B. Wang, M. Krimmel, F. Wang, G. Mastrapas, A. Koukounas, N. Wang, and H. Xiao (2025)Jina-embeddings-v3: multilingual embeddings with task lora.In ECIR,Cited by: [§4].
