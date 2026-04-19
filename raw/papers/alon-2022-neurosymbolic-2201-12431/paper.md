# Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval

Uri Alon Frank F. Xu Junxian He Sudipta Sengupta Dan Roth Graham Neubig

$^{1}$ Language Technologies Institute, Carnegie Mellon University  $^{2}$ Amazon AWS  $^{3}$ AWS AI Labs

{ualon,fangzhex,junxianh,gneubig}@cs.cmu.edu {sudipta,drot}@amazon.com

# Abstract

Retrieval-based language models (R-LM) model the probability of natural language text by combining a standard language model (LM) with examples retrieved from an external datastore at test time. While effective, a major bottleneck of using these models in practice is the computationally costly datastore search, which can be performed as frequently as every time step. In this paper, we present RETOMATON - retrieval automaton - which approximates the datastore search, based on (1) saving pointers between consecutive datastore entries, and (2) clustering of entries into "states". This effectively results in a weighted finite automaton built on top of the datastore, instead of representing the datastore as a flat list. The creation of the automaton is unsupervised, and a RETOMATON can be constructed from any text collection: either the original training corpus or from another domain. Traversing this automaton at inference time, in parallel to the LM inference, reduces its perplexity by up to 1.85, or alternatively saves up to  $83\%$  of the nearest neighbor searches over kNN-LM (Khandelwal et al., 2020) without hurting perplexity. Our code and trained models are available at https://github.com/neulab/retomaton.

# 1. Introduction

Retrieval-based language models (R-LMs) have recently been shown to improve over standard neural models in a variety of tasks such as unconditional language modeling (Guu et al., 2018; He et al., 2020), machine translation (Zhang et al., 2018; Gu et al., 2018; Khandelwal et al., 2021),

$^{1}$ Language Technologies Institute, Carnegie Mellon University  $^{2}$ Amazon AWS  $^{3}$ Amazon. Correspondence to: Uri Alon <ualon@cs.cmu.edu>.

Proceedings of the  $39^{th}$  International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

question answering (Karpukhin et al., 2020; Ram et al., 2021), and code generation (Hayati et al., 2018; Hashimoto et al., 2018). The key ingredient of R-LMs is their ability to utilize training examples at test time without having to rely on the information encoded in the model's weights only.

In these models, the retrieval component first searches for nearest neighbor examples in an external datastore; then, the base model references these examples during the prediction. This fusion of language models (LMs) and retrieval improves the base language model from several perspectives, including higher accuracy (Xu et al., 2021), domain adaptability (Jiang et al., 2021), and reduced size (Borgeaud et al., 2021). Further, the retrieved examples provide information regarding the provenance of the model's predictions, and retrieval allows for modifying the dataset without retraining the model. Nevertheless, the most critical bottleneck of these models is their frequent search over the datastore, which hinders the use of R-LMs in practical settings.

$k$ -Nearest Neighbors Language Model One prominent example of such a retrieval-based model is  $k\mathrm{NN}$ -LM (Grave et al., 2017; Khandelwal et al., 2020), which predicts a token by linearly interpolating the base LM's output with a nonparametric nearest neighbor distribution. This distribution is constructed by searching for the  $k$ -nearest neighbors ( $k\mathrm{NN}$ ) in the datastore and weighting them according to their distance to the current test context. Notably, this  $k$ -nearest neighbor search is performed for every generated test token, introducing severe inference overhead, since this search is significantly slower than the LM's standard "forward pass".

Our Approach: RETOMATON Our main insight is that retrieved neighbors at the current time step also hint at the neighbors that will be retrieved at future time steps, and can thus save repetitive searches later. Specifically, we construct a weighted finite automaton (WFA) on top of an existing datastore, by keeping pointers between datastore entries and clustering similar entries, in a completely unsupervised way. This automaton allows sporadic, infrequent, kNN searches, and a much cheaper traversal of the automaton at other time steps. We call our model RETOMATON - retrieval automaton. RETOMATON is illustrated in Figure 1.

Figure 1: An illustration of RETOMATON. Given a context (1) ("The U.S."), a  $k$ -nearest neighbor search (2) returns the nearest datastore entries. Every datastore entry (  $\bullet$  in the figure) is a 3-tuple of (key, value, pointer) (3), where the key is the LM's hidden state and the value is the target token as in Khandelwal et al. (2020); the pointer points to the datastore entry that appears next in the corpus. Close datastore entries are clustered together, and form an automaton state (O,O,O). The pointers of the clustered entries form the state's possible transitions. At inference time, the model decodes (4) while performing multiple parallel traversals  $(\rightarrow, \rightarrow, \rightarrow)$  on the automaton to find useful datastore entries, instead of performing a full kNN search. Dashed arrows  $(-\rightarrow)$  denote allowed automaton transitions that were not taken during the current decoding.

Concretely, applying RETOMATON to a strong WIKITEXT-103 LM using only the original training set allows for saving  $81\%$  of the kNN searches of Khandelwal et al. (2020) without hurting perplexity, or alternatively, reducing perplexity by 1.85 without saving searches. In both cases, we do not perform any additional training other than clustering. We also show that RETOMATON allows for effective domain adaptation, by simply constructing a RETOMATON for a different domain. When we construct RETOMATON on top of a fine-tuned LM, we decrease the perplexity by more than  $17\%$  compared to just fine-tuning. Finally, we perform a thorough ablation study, separating the contributions of pointers and clustering, and analyzing the tradeoff between coarse- vs. fine-grained clustering. We believe that these results suggest a promising direction for the neuro-symbolic synergy of neural models and symbolic automata.

# 2. Background: the kNN-LM Model

$k\mathrm{NN}$  -LM (Khandelwal et al., 2020) is a language model that estimates the probability of the next token by interpolating an already-trained base LM with a  $k\mathrm{NN}$  distribution. The  $k\mathrm{NN}$  distribution is provided by searching for the  $k$  nearest neighbors in an external datastore and weighting them according to their negative distance.

Datastore Creation Given a context sequence of tokens  $c^{(t)} = (w^{(1)},\dots,w^{(t - 1)})$ , the base LM estimates  $p_{LM}(w|c^{(t)})$ , the distribution over the target token  $w$ . Let  $f$  be the function that maps a context  $c$  to a fixed-length vector using an already-trained LM. For example,  $f(c)$  can be the output of a Transformer LM's last self-attention layer. kNN-LM constructs the datastore using a single forward

pass over a text collection, which can include the training set that the base LM was trained on, or not. Given the  $i$ -th example  $(c_i, w_i)$  in the text collection  $\mathcal{D}$ , the key-value pair  $(f(c_i), w_i)$  is defined such that  $f(c_i)$  is the key, and  $w_i$  is the value. The datastore  $(\mathcal{K}, \mathcal{V})$  is the set of all pairs constructed from the examples in the text collection  $\mathcal{D}$ :

$$
(\mathcal {K}, \mathcal {V}) = \left\{\left(f \left(c _ {i}\right), w _ {i}\right) \mid \left(c _ {i}, w _ {i}\right) \in \mathcal {D} \right\} \tag {1}
$$

Inference At test time, given a test context  $c$ , the model queries the datastore  $(\mathcal{K},\mathcal{V})$  to retrieve the  $k$ -nearest neighbors  $\mathcal{N}$  of  $f(c)$ , according to a distance function  $dist$  between  $f(c)$  and every  $f(c_i)$  in the datastore.  $dist$  is typically the squared  $\ell_2$  distance. These nearest neighbor pairs form a distribution over the target token  $w$ , where the probability of every vocabulary item is computed proportionally to the exponent of its negative distance, while summing over all its occurrences in retrieved values of  $\mathcal{N}$ :

$$
p _ {k \mathrm {N N}} (w \mid c) \propto
$$

$$
\sum_ {\left(f \left(c _ {i}\right), w _ {i}\right) \in \mathcal {N}} \mathbb {1} _ {w = w _ {i}} \exp (- d i s t (f (c), f (c _ {i}))) \tag {2}
$$

The  $k\mathrm{NN}$ -LM's next token probability is then the interpolation of  $p_{LM}$  and  $p_{k\mathrm{NN}}$ , using a scalar hyperparameter  $\lambda$ :

$$
p (w \mid c) = \lambda p _ {k \mathrm {N N}} (w \mid c) + (1 - \lambda) p _ {L M} (w \mid c) \tag {3}
$$

Notice that the  $k$ -nearest neighbor search is performed for every target token. This introduces severe overhead, since the search is significantly slower than the LM's standard forward pass. In the next section, we show how we can avoid the search in the majority of the time steps.

Figure 2: An illustration of our automaton creation process. The text  $\mathcal{D}$  contains two sentences: (i) "The U.S. president is Joe Biden"; and (ii) "The 46th president of the U.S. is Joseph Robinette Biden". Each datastore entry is a pair of a key vector encoding the prefix and a value representing the next token, such as  $\boxed{\circ\circ\bullet\text{president}}$ . We save a pointer from every entry to its successor in the text, and we cluster close key vectors to form automaton states ( $①$  and  $②$ ) that share pointers.

# 3. Building Automata from Datastores

To save kNN searches, we build a weighted finite automaton (WFA) on top of the kNN-LM dataset. Then, we traverse the automaton to estimate the next nearest neighbors. We will demonstrate the automaton creation process using Figure 2 as a running example.

# 3.1. Definitions

Given a finite vocabulary  $\Sigma$  and the set  $\Sigma^{*}$  of finite sequences over  $\Sigma$ , a trained autoregressive LM defines a distribution  $p_{LM}$  over  $\Sigma$  for any given context  $c \in \Sigma^{*}$ . Given such an LM having  $f: \Sigma^{*} \to \mathbb{R}^{d}$  and a text collection  $\mathcal{D}$ , we can create a datastore  $(\mathcal{K}, \mathcal{V})$  as detailed in Section 2. As shown in Figure 2, this results in a set of key-value entries such as  $\bigcirc \bigcirc \bigcirc \bigcirc \text{president}$ , where the key is the LM encoding of every prefix in  $\mathcal{D}$ , and the value is the following token.

Weighted Finite Automaton Our automaton is a tuple  $A = \langle Q, \Sigma, q_0, \delta, \phi \rangle$  such that  $Q$  is a finite set of states,  $\Sigma$  is the LM's vocabulary,  $q_0 \in Q$  is the initial state,  $\delta: Q \times \Sigma \to \mathbb{P}(Q)$  is a transition function where  $\mathbb{P}(Q)$  denotes the power set of  $Q$ , and  $\phi: Q \times \mathbb{R}^d \times \Sigma \to \mathbb{R}$  is a transition weight function. Unlike standard WFAs, the transition weights are dynamic: notice that  $\phi$  depends also on a vector in  $\mathbb{R}^d$ , in addition to a previous state and an input token. Also, note that  $\delta$  is defined such that we transition into a set of states.

# 3.2. Constructing the Automaton

**Pointers** Our main insight is that during the creation of the datastore, we can keep a pointer from every datastore entry to the entry that appears next in the text  $\mathcal{D}$ .

Imagine that at time  $t$ , the test context is  $c^{(t)} = (w^{(1)},\dots,w^{(t - 1)})$ , and the model retrieves a datastore entry  $(f(c_{i}),w_{i})$ . If after the interpolation with the base LM (Equation (3)), the model's generated token  $w^{(t)}$  (by sampling or argmax) is equal to  $w_{i}$ , then the entry  $(f(c_{i + 1}),w_{i + 1})$  is likely to be a useful entry at time  $t + 1$

because  $c_{i}$  was a near neighbor of  $c^{(t)}$ , and both these contexts were followed by the same token  $w_{i} = w^{(t)}$ . That is, our underlying assumption can be formulated as:

$$
f \left(c _ {i}\right) \approx f \left(c ^ {(t)}\right) \Longrightarrow f \left(c _ {i} \cdot w\right) \approx f \left(c ^ {(t)} \cdot w\right) \tag {4}
$$

where  $\approx$  denotes vector similarity, and  $c^{(t)}\cdot w$  is the continuation of the context  $c^{(t)}$  using the token  $w$ .

Thus, given the  $i$ -th example  $(c_i, w_i) \in \mathcal{D}$ , instead of keeping only the key-value pair  $(f(c_i), w_i)$  as in Khandelwal et al. (2020), we save every datastore entry as  $(f(c_i), w_i, p_i) \in (\mathcal{K}, \mathcal{V}, \mathcal{P})$ , where  $p_i$  is a pointer to the next entry, or the next entry's index in the datastore. This is illustrated as arrows in Figure 2, where every entry has a pointer to the entry that followed it in the text  $\mathcal{D}$ . For example, the entry  $\bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc$  points to  $\bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \bigcirc \big○$

States If every entry had only a single outgoing pointer, the automaton would only capture n-grams that appeared in the text verbatim, and thus will not be able to generalize to unseen phrases. For example in Figure 2, the sequence "The 46th president of the U.S. is Joe Biden" would not be captured, because the prefix "The 46th president of the U.S." appeared in one sentence, and the suffix "Joe Biden" appeared in another sentence. Thus, to allow entries to share their pointers, we cluster entries having close keys into an automaton state. A state includes all entries in the cluster, and the entries' pointers are the state's allowed outgoing transitions. For example, in Figure 2, the entry  $(\bullet \bullet \bullet ,i s)$  is clustered with another entry having the same value,  $(\bullet \bullet \bullet ,i s)$  (surrounded by  $0$  and marked as  $①$  ). Furthermore, we can also cluster similar contexts that do not have the same value, such as cluster  $②$  containing  $(\bullet \bullet \bullet ,J o e)$  and  $(\bullet \bullet \bullet ,J o s e p h)$ , which allows capturing the phrase "Jospeh Biden".

This step can be performed using any clustering algorithm such as  $k$ -means. In this case, the choice of  $k_{\mathrm{clus}}$  determines the final number of states  $|Q| = k_{\mathrm{clus}}$  (ignoring the initial state  $q_0$ ). We experiment with different values of  $k_{\mathrm{clus}}$  and clustering algorithms in Section 6.

Transition States Given a source state  $q \in Q$  and an input token  $w \in \Sigma$ , we define the set of allowable next states as the clusters of entries that are pointed to by a datastore entry in  $q$  having the value  $w$ . In other words, we follow pointers of datastore entries from  $q$  whose value is  $w$ , and take the resulting entries' clusters. Formally, let  $\pi: (\mathcal{K}, \mathcal{V}, \mathcal{P}) \to Q$  be a function that maps every datastore entry into its containing state, and the function  $\pi^{-1}: Q \to \mathbb{P}((\mathcal{K}, \mathcal{V}, \mathcal{P}))$ , which maps every state into the set of datastore entries contained in it. Let  $\rho: \mathcal{P} \to (\mathcal{K}, \mathcal{V}, \mathcal{P})$ , be the "dereferencing" operator, which returns a datastore entry given a pointer that points to it. We can now define the allowed transitions as:

$$
\delta (q, w) = \left\{\pi (\rho (p _ {i})) | (\cdot , w _ {i}, p _ {i}) \in \pi^ {- 1} (q), w _ {i} = w \right\} \tag {5}
$$

This is illustrated in Figure 2 as the outgoing pointers of cluster  $②$ , which allow transitioning from  $②$  to different states given the input tokens  $w = \text{"Joe"}$  or  $w = \text{"Joseph"}$ . This results in a (currently non-weighted) finite-state automaton, whose nodes are clusters of datastore entries, and whose edges represent the successiveness of entries in  $\mathcal{D}$ , where successiveness is shared among clustered entries.

# 3.3. Traversal of the Automaton

At test time, given a test context  $c^{(t)}$ , we traverse the automaton while visiting multiple states at every time step, marked in Figure 1 as  $\rightarrow$ ,  $\rightarrow$ ,  $\rightarrow$ . A traversal begins with a full kNN search of the datastore to retrieve the  $k$ -nearest neighbors  $\mathcal{N}^{(t)}$  of  $f(c^{(t)})$ . The initial traversal states  $S^{(t)} \subseteq Q$  are the union of the states to which these  $k$ -nearest neighbors belong:  $S^{(t)} = \bigcup_{e \in \mathcal{N}^{(t)}} \pi(e)$ .<sup>1</sup>

In the next time step  $t + 1$ , given a token  $w^{(t)} \in \Sigma$  that was generated (by argmax or sampling) by the model at time  $t$ , we compute the union of all valid transitions from states  $q \in S^{(t)}$ . We define  $\hat{\delta} : \mathbb{P}(Q) \times \Sigma \to \mathbb{P}(Q)$  as follows:

$$
\hat {\delta} (\mathcal {S}, w) = \bigcup_ {q \in \mathcal {S}} \delta (q, w) \tag {6}
$$

The decision of whether to continue traversing or start a new traversal can be made in several ways. We experimented with several alternatives, but found the most intuitive way to simply be whether the number of new states is greater than or equal to a threshold  $\tau$ . That is, we continue traversing if  $|\hat{\delta} (\mathcal{S}^{(t)},w^{(t)})|\geq \tau$ , or start a new traversal otherwise.

When we continue traversing, we take the new states as our states in the next step. When we start a new traversal, we perform a new kNN search resulting in  $\mathcal{N}^{(t + 1)}$ , but also

include the remaining states we obtained so far:

$$
\mathcal {S} ^ {(t + 1)} = \tag {7}
$$

$$
\left\{ \begin{array}{l l} \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) & | \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) | \geq \tau \\ \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) \cup \bigcup_ {e \in \mathcal {N} ^ {(t + 1)}} \pi (e) & \text {o t h e r w i s e} \end{array} \right.
$$

Varying  $\tau$  allows us to control the trade-off between higher accuracy with frequent traversal restarts, and thus frequent kNN searches (high  $\tau$ ), versus lower accuracy with rare kNN searches, which saves time (low  $\tau$ ). For additional intuition for  $\tau$ , see Appendix A.

Transition Weights Given a set of states  $S$ , a test context  $c$  and an input token  $w \in \Sigma$ , we define the transition weight from every  $q \in S$  similarly to Equation (2), except that we sum exponents of negative distances between  $f(c)$  and all entries whose value is  $w$  that are contained the state  $q$ ; then, we normalize across all states in  $S^{(t)}$ :

$$
\phi (q, c, w) = \sum_ {\left(k _ {i}, w _ {i}, \cdot\right) \in \pi^ {- 1} (q)} \mathbb {1} _ {w = w _ {i}} \exp \left(- d i s t (f (c), k _ {i})\right) \tag {8}
$$

$$
p _ {\text {a u t o}} (w \mid c, \mathcal {S}) \propto \sum_ {q \in \mathcal {S}} \phi (q, c, w) \tag {9}
$$

Finally, we interpolate  $p_{\mathrm{auto}}$  with  $p_{LM}$ :

$$
p (w \mid c, \mathcal {S}) = \lambda p _ {\text {a u t o}} (w \mid c, \mathcal {S}) + (1 - \lambda) p _ {L M} (w \mid c) \tag {10}
$$

# 4. Experimental Setup

We evaluate RETOMATON in two different settings: (i) standard autoregressive language modeling, where the datastore is constructed from the same training corpus that the base LM was trained on ("in-training datastore"); and (ii) domain adaptation, where the datastore is constructed from a different domain than the base LM was trained on. Our experiments are fully reproducible and our code is available at https://github.com/neulab/retomaton.

Implementation We base our experiments on the original kNN-LM implementation that uses the FAISS (Johnson et al., 2019) library to perform kNN search. We also use FAISS for the one-time  $k$ -means clustering.

Hyperparameters We used the same settings as the baseline implementations without any special tuning of our model, and always matched the settings to conduct a fair evaluation. We saved half precision (fp16) datastore keys as He et al. (2021). For WIKITEXT-103, which creates a datastore of 103M entries, we use  $k$ -means clustering with  $k_{\mathrm{clus}} = 1\mathrm{M}$ . For Law-MT, which creates a datastore of 19M entries, we use  $k_{\mathrm{clus}} = 200\mathrm{K}$ , which maintains an average cluster size of  $\sim 100$  in both datasets. We analyze the difference between clustering algorithms and the number of clusters in Section 6. Additional details are provided in Appendix B.

Metrics The main metric that we focus on is perplexity with respect to the fraction of saved searches (FoSS). In our preliminary experiments, we found that measuring wall-clock time is difficult to reproduce, is brittle to temporary hardware and system load, and is affected by the specific kNN retrieval library. Contrarily, FoSS does not depend on hardware and engineering factors and is thus completely reproducible. In Appendix C, we empirically analyze FoSS with respect to the saved wall-clock time and we show that they are identical up to an additive constant that depends on the hardware and the specific settings. Thus, FoSS serves as a good proxy to the saved wall-clock time.

We control FoSS in RETOMATON by running with different values of the  $\tau \in [1,\infty)$  threshold, as detailed in Section 3.3 and Appendix A. Higher  $\tau$  results in frequent restarts and thus lower FoSS.

# 4.1. In-Domain Datastore

Data Following Khandelwal et al. (2020), we use WIKIText-103 (Merit et al., 2017), which is a standard benchmark for autoregressive language modeling, having 103M/250K/250K tokens from Wikipedia in its training/validation/test sets, respectively.

Model We use a Transformer (Vaswani et al., 2017) as our base LM, trained by Khandelwal et al. (2020) following the architecture and settings of Baevski & Auli (2019) with adaptive inputs, adaptive softmax (Joulin et al., 2017), and a large 267K word-level vocabulary. This base LM consists of 16 layers, each with 16 self-attention heads, 1024-dimensional hidden states, 4096-dimensional feedforward layers, amounting to 247M parameters.

# 4.2. Domain Adaptation

Data Following He et al. (2021), we use the English part of Law-MT, which is an English-German translation dataset for the law domain, released by Koehn & Knowles (2017) and resplit by Aharoni & Goldberg (2020). The training set consists of 19M tokens, which we use to build a datastore and an automaton (Section 5.2), or fine-tune the base LM and then build a datastore and automaton (Section 5.3).

Model Following He et al. (2021), as our base LM we use a 12-layer, 1536-dimensional transformer with a vocabulary of 42K subword units (Sennrich et al., 2016), amounting to 656M parameters and trained by Ng et al. (2019) on WMT News Crawl (Barrault et al., 2019).

# 4.3. Baselines

$kNN - LM$  We compare RETOMATON to  $kNN - LM$  using the original code and hyperparameters of Khandelwal et al.

Figure 3: Experiments on WIKITEXT-103, where the data-tore is created from the same training set that the base LM was trained on. RETOMATON reduces perplexity across all FoSS values, and even reduces perplexity when FoSS=0.

(2020). The only difference in hyperparameters is that following He et al. (2021), we use the faster approximate  $k$  NN distances provided by FAISS, rather than recomputing them, in our model and in the baselines. We vary the FoSS in  $k$  NN-LM by uniformly selecting a certain fraction of time steps in which we skip the  $k$  NN search and use only the base LM ( $p_{LM}$ ), identically to the "Random" baseline in He et al. (2021). That is,  $k$  NN-LM with FoSS=0 is the standard  $k$  NN-LM, and  $k$  NN-LM with FoSS=1.0 is the base LM.

ADAPTRET We also compare RETOMATON to a retrieval-saving approach that is orthogonal to ours. ADAPTRET is the Adaptive Retrieval approach of He et al. (2021), which trains a light MLP to predict at each time step whether the base LM is "confident enough" to use its output only, or should it perform a kNN search. The main conceptual difference between RETOMATON and ADAPTRET is that RETOMATON skips kNN searches but still computes the interpolation with the non-parametric distribution  $p_{\mathrm{auto}}$  for every token, using the automaton (Equation (10)). In contrast, when ADAPTRET skips a kNN search, it also skips the interpolation with the  $p_{k\mathrm{NN}}$  distribution of Equation (3) entirely, and backs-off to rely on the base LM solely. For a detailed discussion of the conceptual differences between RETOMATON and ADAPTRET, see Section 8.1.

# 5. Results

# 5.1. In-Domain Datastore

We experiment with creating a datastore and an automaton from the same data that the base LM was trained on.

Figure 4: Experiments for domain adaptation, where the datastore is constructed from Law-MT.

Figure 3 shows how RETOMATON reduces the perplexity on WIKITEXT-103 across different FoSS rates. Specifically, RETOMATON saves  $81\%$  of the kNN searches while matching the perplexity of kNN-LM. If we do not perform clustering ("w/o clustering"), we can still save more than  $60\%$  of the kNN searches while matching kNN-LM. Compared to ADAPTRET, RETOMATON saves  $60\%$  of the searches while matching the best perplexity of ADAPTRET.

Surprisingly, even when we do not attempt to save any searches (FoSS=0), RETOMATON reduces perplexity from 16.65 (kNN-LM) and 16.35 (ADAPTRET) to 16.08. The explanation for this is that even when RETOMATON performs kNN search on every step, as kNN-LM, it includes the pointers from the previous time step (Equation (7)), which are more likely to be correct than the general retrieved nearest neighbors. Some neighbors may be included twice – both as retrieved kNNs, and as pointers from the previous time step; this case is equivalent to increasing the weight of a subset of the retrieved kNNs, which are more likely to be correct.

# 5.2. Domain Adaptation

We also experiment with domain adaptation, where the base LM was trained on newspapers, and the models are tested on law documents, using datastore and automaton that were constructed from law data as well, as detailed in Section 4.2.

Figure 4 shows how RETOMATON reduces the perplexity on Law-MT from 12.34 (kNN-LM) and 12.01 (ADAPTRET) to 10.49 when search is performed every step (FoSS=0). As we increase the fraction of saved searches, RETOMATON shows a very gentle ascent in perplexity, while the perplexity of kNN-LM increases exponentially.

The high perplexity of the ADAPTRET baseline is caused by the high perplexity of the base LM (106.56): in time steps where ADAPTRET does not perform search, its output is identical to the base LM's probability. That is, in domain adaptation, where the base LM performs poorly, interpolating it with the  $p_{k\mathrm{NN}}$  distribution (Equation (3)) is crucial. Thus, approximating the  $p_{k\mathrm{NN}}$  distribution using an automaton is much more effective than pruning it using ADAPTRET, while kNN searches are saved in both cases.

It is also interesting to notice the difference between datasets. In particular, we find that RETOMATON provides a stronger effect in Law-MT, reflected in the very gentle ascent in Figure 4, over its effect in WIKITEXT-103. We believe that one major reason is n-gram repetitiveness between the training and the validation sets. As shown in Figures 12 and 13, there is much higher repetitiveness of n-grams in Law-MT over WIKITEXT-103. For example,  $21\%$  of the 5-grams in the validation set of WIKITEXT-103 were seen in the training data; in contrast, in Law-MT  $-62\%$  of the 5-grams in the validation set were seen during training.

# 5.3. Improving Fine-Tuning

Table 1: Experiments using a base LM that was fine-tuned on Law-MT. Numbers denote perplexity, and the relative reduction over the fine-tuned LM is shown in parentheses.  

<table><tr><td>Model</td><td colspan="2">FoSS=0</td><td colspan="2">FoSS=0.5</td></tr><tr><td>fine-tuned LM</td><td></td><td>8.61</td><td></td><td></td></tr><tr><td>kNN-LM</td><td>7.93</td><td>(↓7.9%)</td><td>8.25</td><td>(↓4.2%)</td></tr><tr><td>ADAPTRET</td><td>7.81</td><td>(↓9.2%)</td><td>7.91</td><td>(↓8.1%)</td></tr><tr><td>RETOMATON</td><td>7.10</td><td>(↓17.5%)</td><td>7.15</td><td>(↓17.0%)</td></tr></table>

In Section 5.2 we used RETOMATON to domain-adapt a base LM that was trained on a different domain; however, can RETOMATON improve a fine-tuned base LM?

We fine-tuned the base LM of Section 5.2 on the Law-MT training set, and recreated a datastore and an automaton using the fine-tuned model. As Table 1 shows, while kNN-LM and ADAPTRET reduce the perplexity compared to the fine-tuned model from 8.61 to 7.93 and 7.81, respectively, RETOMATON further reduces perplexity to 7.10, which is a relative reduction of more than  $17.5\%$ . This shows how RETOMATON can strengthen even fine-tuned models.

# 6. Ablation Study

Pointers vs. clustering The main two contributions in RETOMATON are the use of pointers and the clustering. Here, we tease apart the contribution of each of these.

The w/o clustering model is an ablation of RETOMATON,

Figure 5: Analysis of the number of clusters on the validation set of WIKTEXT-103. Additional clustering runs can be found in Appendix D (Figure 9).

Figure 6: Analysis of the number of clusters on the validation set of Law-MT. A larger version of this figure can be found in Appendix D (Figure 10).

<table><tr><td>length=3</td><td>length=6</td><td>length=10</td></tr><tr><td>, and the</td><td>the Streets Have No Name &quot;</td><td>. As a result, there was not a single</td></tr><tr><td>, but the</td><td>Department of Transportation ( MDOT )&quot;</td><td>and some were moved to new locations. Before its</td></tr><tr><td>roughly bounded by</td><td>In the United States ,</td><td>but it was not until the following day that a</td></tr><tr><td>in the first</td><td>the end of the song ,</td><td>the end of the Second World War was completed in</td></tr><tr><td>a number of</td><td>not occur until 27 May 1915</td><td>to lack of evidence; however, the decision was</td></tr><tr><td>, when the</td><td>fired the shots that caused the</td><td>the Streets Have No Name &#x27; is more like the</td></tr></table>

Table 2: Some of the sequences from the WIKITEXT-103 validation set that our automaton captured without performing kNN search. We selected length=10 sequences that did not appear in the training data.

which spares the clustering preprocessing step, and uses only pointers. As shown in Figure 3, even the  $w/o$  clustering model achieves lower perplexity than the other baselines, with a lowest perplexity of 16.12 at FoSS=0. Up to FoSS=0.4,  $w/o$  clustering performs only slightly worse than the base RETOMATON. Starting from FoSS=0.7, the  $w/o$  clustering model almost consolidates with ADAPTRET.

From these experiments, we conjecture that RETOMATON's performance in few saved searches (FoSS  $< 0.4$ ) stems mostly from keeping pointers, which provides the most significant boost. Starting from FoSS  $= 0.7$ , the gap between w/o clustering and the base RETOMATON shows the contribution of clustering, which allows longer sequences of consecutive steps without performing kNN search.

Clustering Granularity The only hyperparameter that RETOMATON introduces is the choice of the number of clusters. A higher number of clusters results in smaller, fine-grained clusters. A low number of clusters results in larger, coarse-grained clusters, where every cluster is possibly more noisy.

Here, we vary the number of clusters and also experiment with the "greedy" clustering algorithms from He et al. (2021). The advantage of the greedy algorithm is that it is computationally cheaper: it requires searching for each datastore entry's nearest neighbors, and then performing a single pass of merging, while  $k$ -means requires multiple iterations over the entire datastore.

The results of these experiments are shown in Figure 5 for WIKTEXT-103 and Figure 6 for Law-MT. As shown in Figure 5,  $k = 500\mathrm{K}$  means and  $k = 1\mathrm{M}$  means achieve similar perplexities, while  $k = 100\mathrm{K}$  is too coarse-grained. The greedy algorithm presents an interesting tradeoff, as it achieves lower perplexity than the others at FoSS=0, but degrades as FoSS increases, since each cluster is smaller. Figure 6 shows a similar tradeoff in Law-MT: the most fine-grained clustering using  $k = 400\mathrm{K}$  means performs best for FoSS=0, but achieves a higher perplexity than others at FoSS>0.7. Additional clustering runs are shown in Appendix D.

<table><tr><td>Training Sequence from: https://en.wikipedia.org/wiki/Oh,_What_a_Knight!</td></tr><tr><td>The writer of the scenario is unknown , but it was most likely Lloyd Lonergan . He was an experienced newspaperman employed by The New York Evening World while … A surviving film still gives the possibility of identifying three of the actors in the film …</td></tr><tr><td>Validation Sequence from: https://en.wikipedia.org/wiki/Home_Made_Mince_Pie</td></tr><tr><td>The writer of the scenario is unknown , but it was most likely Lloyd Lonergan . He was an experienced newspaperman employed by The New York Evening World while … A surviving film still gives the possibility of identifying eight actors …</td></tr></table>

Figure 7: An example for a 236-token long sequence that RETOMATON was able to capture from the WIKITEXT-103 training set and apply to the validation set. Although most of the paragraph appears as-is in both sets, they have different endings, as these are articles about different silent films from 1910. This shows how RETOMATON allows to dynamically retrieve chunks of text from the training data using a single kNN search, rather than a single token at a time as kNN-LM.

# 7. Qualitative Analysis

What are the sequences of tokens that RETOMATON captured and predicted consecutively, without kNN search?

Table 2 shows examples of sequences among those that were given the highest probability ( $p_{\text{auto}}$  in Equation (9)) from the validation set of WIKITEXT-103. Naturally, short sequences (length=3) are usually common 3-grams such as “, and the”. As length increases (to 6 and 10 tokens), the sequences become more specific; for example, two of them contain part of the name of the song “Where the Streets Have No Name” by the band U2. Nevertheless, we selected sequences into the list of length=10 such that none of them appeared as-is in the training set, to show that RETOMATON does not only memorize n-grams, but instead, clustering allows it to compose multiple small n-grams into longer n-grams.

Figure 7 shows a 217-token long passage that appears in both training and validation sets, but with different endings. RETOMATON can predict this passage consecutively, without performing search. This shows how RETOMATON can retrieve single tokens from the datastore, but it also can adaptively construct much longer chunks, if needed.

Figure 11 (Appendix D) shows a histogram of the lengths of these sequences. The vast majority  $(98\%)$  of the validation set tokens are included in n-grams having  $n > 1$ , which either started or continued an automaton traversal.

Figure 8 shows a random sample of states and transitions from the automaton constructed from the training set of WIKITEXT-103. Further exploration of the automaton can be performed using our publicly available code.

# 8. Related Work

# 8.1. Comparison to ADAPTRET (He et al., 2021)

The closest work to ours is ADAPTRET (He et al., 2021). ADAPTRET saves kNN searches as well, but it suffers from conceptual weaknesses compared to our work:

When the model performs kNN search: ADAPTRET uses

only the neighbors retrieved by the  $k\mathrm{NN}$  search. RETOMA-TON uses the set of retrieved nearest neighbors as well, but also includes the remaining pointers from the previous time step (Equation (7)). Apparently, these remaining pointers have a higher likelihood of predicting the correct token than the general set of retrieved nearest neighbors.

When the model does not perform kNN search: ADAP-TRET skips the interpolation of  $p_{kNN}$  with  $p_{LM}$ , and uses  $p_{LM}$  solely. In contrast, RETOMATON still computes the interpolation of  $p_{\mathrm{auto}}$  with  $p_{LM}$ , thanks to its pointers. As an interesting direction for future work, we expect that learning a dynamic interpolation factor  $\lambda$ , similarly to ADAPTRET, will even further improve RETOMATON's results.

Data Efficiency ADAPTRET requires training its MLP on a dataset that is disjoint from the corpus that the datastore was built from, to prevent overfitting. Thus, He et al. had to train their MLP on the validation set, which is not data-efficient - spending  $90\%$  of the original validation set for additional training. In contrast, our approach is completely unsupervised, and thus does not require additional data.

# 8.2. Retrieval and Neuro-Symbolic Methods

Granularity of Retrieval While Khandelwal et al. (2020) and Yogatama et al. (2021) retrieve a token at a time step, other work retrieved a sentence (Hashimoto et al., 2018; Gu et al., 2018; Rubin et al., 2021), a prototype (Guu et al., 2018; He et al., 2020), or a chunk (Guu et al., 2020; Borgeaud et al., 2021). RETOMATON implicitly generalizes these approaches by dynamically constructing the retrieved sequence, essentially being able to retrieve individual tokens as well as constructing search-free longer passages.

Hybrid Models Combining n-grams (Neubig & Dyer, 2016) and automata (Rijhwani et al., 2021) with neural language models has usually led to "static", count-based, transitions weights. In contrast, states in our automaton are based on hidden representations of the neural LM, which allows RETOMATON to dynamically weigh transitions. Other work scored automata with RNNs (Rastogi et al., 2016; Lin et al.,

Figure 8: A random sample of the automaton constructed from the training set of WIKTEXT-103

2019), or constructed RNNs from automata (Schwartz et al., 2018; Peng et al., 2018); RETOMATON differs from these approaches by providing with retrieved instances from a datastore, instead of enforcing structure on the neural LM.

Leveraging Structure in kNN-LMs Meng et al. (2022) train a graph neural network (GNN) on top of the encoded test contexts and the retrieved examples. Differently from our work, their GNN requires additional training, while our approach is completely unsupervised. Applying RETOMATON on top of their approach is an interesting future direction which is expected to further improve their results.

Automata Extraction The extraction of automata from neural networks goes back to Giles et al. (1992) and Omlin & Giles (1996), mainly for synthetic and simple regular languages. Later, Weiss et al. (2018; 2019) scaled up the extraction to larger GRU and LSTM architectures. In this work, we do not only extract an automaton, but also combine it with a neural LM to improve the LM's accuracy.

# 9. Conclusion

We presented RETOMATON - retrieval automaton. RETOMATON approximates a  $k\mathrm{NN}$  search over an external

corpus by clustering similar neighbors into automaton states, and keeping pointers from previously found neighbors, which form the transition between states. This results in a weighted finite automaton, which allows approximating the nearest neighbors in most of the time steps, instead of performing a  $k$ NN search at every step.

Empirically, traversing the automaton at inference time saves up to  $83\%$  of the kNN searches in both in-domain and domain adaptation settings, and reduces the perplexity of strong LMs, even after they were fine-tuned.

These results suggest a promising direction for the neurosymbolic synergy of neural models with symbolic automata. We believe that the principles and the methods presented in this paper are also applicable to other R-LMs, including phrase- and chunk-based retrieval models. To these ends, we make all our code, data, and models publicly available.

# Acknowledgments

We thank Lucio Dery and Vincent Hellendoorn for the helpful discussions and thorough feedback. We are also grateful to the anonymous reviewers for their useful comments and suggestions.

# References

Aharoni, R. and Goldberg, Y. Unsupervised domain clusters in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7747-7763, 2020.  
Baevski, A. and Auli, M. Adaptive input representations for neural language modeling. In International Conference on Learning Representations, 2019.  
Barrault, L., Bojar, O., Costa-jussa, M. R., Federmann, C., Fishel, M., Graham, Y., Haddow, B., Huck, M., Koehn, P., Malmasi, S., Monz, C., Muller, M., Pal, S., Post, M., and Zampieri, M. Findings of the 2019 conference on machine translation (WMT19). In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pp. 1-61, Florence, Italy, August 2019. Association for Computational Linguistics. doi: 10.18653/v1/W19-5301. URL https://aclanthology.org/W19-5301.  
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Driessche, G. v. d., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions of tokens. arXiv preprint arXiv:2112.04426, 2021.  
Chen, Q., Wang, H., Li, M., Ren, G., Li, S., Zhu, J., Li, J., Liu, C., Zhang, L., and Wang, J. SPTAG: A library for fast approximate nearest neighbor search, 2018. URL https://github.com/Microsoft/SPTAG.  
Giles, C. L., Miller, C. B., Chen, D., Chen, H.-H., Sun, G.-Z., and Lee, Y.-C. Learning and extracting finite state automata with second-order recurrent neural networks. Neural Computation, 4(3):393-405, 1992.  
Grave, E., Cisse, M. M., and Joulin, A. Unbounded cache model for online language modeling with open vocabulary. Advances in neural information processing systems, 30, 2017.  
Gu, J., Wang, Y., Cho, K., and Li, V. O. Search engine guided neural machine translation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., and Kumar, S. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pp. 3887-3896. PMLR, 2020.  
Guu, K., Hashimoto, T. B., Oren, Y., and Liang, P. Generating sentences by editing prototypes. Transactions of the Association for Computational Linguistics, 6:437-450, 2018.

Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. REALM: Retrieval-augmented language model pretraining. arXiv preprint arXiv:2002.08909, 2020.  
Hashimoto, T. B., Guu, K., Oren, Y., and Liang, P. A retrieve-and-edit framework for predicting structured outputs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 10073-10083, 2018.  
Hayati, S. A., Olivier, R., Avvaru, P., Yin, P., Tomasic, A., and Neubig, G. Retrieval-based neural code generation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 925-930, 2018.  
He, J., Berg-Kirkpatrick, T., and Neubig, G. Learning sparse prototypes for text generation. Advances in Neural Information Processing Systems, 33, 2020.  
He, J., Neubig, G., and Berg-Kirkpatrick, T. Efficient nearest neighbor language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5703-5714, 2021.  
Jiang, Q., Wang, M., Cao, J., Cheng, S., Huang, S., and Li, L. Learning kernel-smoothed machine translation with retrieved examples. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7280-7290, 2021.  
Johnson, J., Douze, M., and Jégou, H. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 2019.  
Joulin, A., Cissé, M., Grangier, D., Jégou, H., et al. Efficient softmax approximation for gpus. In International conference on machine learning, pp. 1302-1310. PMLR, 2017.  
Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769-6781, 2020.  
Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=HklBjCEKvH.  
Khandelwal, U., Fan, A., Jurafsky, D., Zettlemoyer, L., and Lewis, M. Nearest neighbor machine translation. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=7wCBOfJ8hJM.

Koehn, P. and Knowles, R. Six challenges for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation, pp. 28-39, 2017.  
Lin, C.-C., Zhu, H., Gormley, M. R., and Eisner, J. Neural finite-state transducers: Beyond rational relations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 272-283, 2019.  
Meng, Y., Zong, S., Li, X., Sun, X., Zhang, T., Wu, F., and Li, J. GNN-LM: Language modeling based on global contexts via GNN. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=BS491-B5Bql.  
Merit, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.  
Neubig, G. and Dyer, C. Generalizing and hybridizing count-based and neural language models. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1163-1172, 2016.  
Ng, N., Yee, K., Baevski, A., Ott, M., Auli, M., and Edunov, S. Facebook fair's wmt19 news translation task submission. In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pp. 314-319, 2019.  
Omlin, C. W. and Giles, C. L. Extraction of rules from discrete-time recurrent neural networks. Neural networks, 9(1):41-52, 1996.  
Peng, H., Schwartz, R., Thomson, S., and Smith, N. A. Rational recurrences. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1203-1214, 2018.  
Ram, O., Shachaf, G., Levy, O., Berant, J., and Globerson, A. Learning to retrieve passages without supervision. arXiv preprint arXiv:2112.07708, 2021.  
Rastogi, P., Cotterell, R., and Eisner, J. Weighting finite-state transductions with neural context. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 623-633, 2016.  
Rijhwani, S., Rosenblum, D., Anastasopoulos, A., and Neubig, G. Lexically-aware semi-supervised learning forOCR post-correction. Transactions of the Association for Computational Linguistics, 9:1285-1302, 2021.  
Rubin, O., Herzig, J., and Berant, J. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021.

Schwartz, R., Thomson, S., and Smith, N. A. Bridging cnns, rnns, and weighted finite-state machines. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 295-305, 2018.  
Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1715-1725, 2016.  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 6000-6010, 2017.  
Weiss, G., Goldberg, Y., and Yahav, E. Extracting automata from recurrent neural networks using queries and counterexamples. In International Conference on Machine Learning, pp. 5247-5256. PMLR, 2018.  
Weiss, G., Goldberg, Y., and Yahav, E. Learning deterministic weighted automata with queries and counterexamples. Advances in Neural Information Processing Systems, 32: 8560-8571, 2019.  
Xu, F. F., He, J., Neubig, G., and Hellendoorn, V. J. Capturing structural locality in non-parametric language models. arXiv preprint arXiv:2110.02870, 2021.  
Yogatama, D., de Masson d'Autume, C., and Kong, L. Adaptive semiparametric language models. Transactions of the Association for Computational Linguistics, 9:362-373, 2021.  
Zhang, J., Utiyama, M., Sumita, E., Neubig, G., and Nakamura, S. Guiding neural machine translation with retrieved translation pieces. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1325-1335, 2018.

# A. Intuition for  $\pmb{\tau}$

For brevity, we repeat Equation (7) here:

$$
\mathcal {S} ^ {(t + 1)} = \left\{ \begin{array}{l l} \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) & | \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) | \geq \tau \\ \hat {\delta} \left(\mathcal {S} ^ {(t)}, w ^ {(t)}\right) \cup \bigcup_ {e \in \mathcal {N} ^ {(t + 1)}} \pi (e) & \text {o t h e r w i s e} \end{array} \right.
$$

Intuitively, the larger  $|\hat{\delta}(\mathcal{S}^{(t)}, w^{(t)})|$  is, the more "correct entries" - entries whose values were equal to  $w^{(t)}$  that we had at time  $t$ , and the more we can rely on their pointers at time  $t + 1$ . Thus, if  $|\hat{\delta}(\mathcal{S}^{(t)}, w^{(t)})| \geq \tau$ , it means that we have more states to continue traversing to, and we can avoid the kNN search. If  $|\hat{\delta}(\mathcal{S}^{(t)}, w^{(t)})| < \tau$ , it means that many of the entries at time  $t$  were "incorrect" (their value was not equal to  $w^{(t)}$ ), and the set of new states  $\hat{\delta}(\mathcal{S}^{(t)}, w^{(t)})$  is small (or even empty); in this case, we perform a new kNN search and restart an automaton traversal.

The minimal value is  $\tau = 1$ , which means that as long as we have at least one automaton state to continue traversing from, we use it without performing  $k\mathrm{NN}$  search. The maximal value is  $\tau = \infty$ , which means that  $|\hat{\delta} (S^{(t)},w^{(t)})|$  will always be lower than  $\tau$ , and thus we will perform  $k\mathrm{NN}$  search at every step.

# B. Evaluation Details

Implementation and Hyperparameters We used the exact hyperparameters of (Khandelwal et al., 2020) including  $k_{\mathrm{neigh}} = 1024$  (the number of retrieved nearest neighbors when performing a full search) and the same FAISS (Johnson et al., 2019) kNN library. Following He et al. (2021), we loaded the index to the GPU, and used half precision (fp16) datastore keys.

During the automaton traversal, when reaching automaton states – there might be too many datastore entries in it, which can make the computation slow. We thus always computed Equation (9) over at most max_knns = 1024 datastore entries (1024 datastore entries overall, in the union of all states  $q \in S^{(t)}$ ), preferring entries that were directly pointed by pointers from the previous state  $(\rho(p))$ , and otherwise choosing members from clusters randomly. We chose 1024 to match the number of nearest neighbors retrieved when performing a full search  $k_{\mathrm{neigh}} = 1024$ , although these numbers are not coupled to each other and denote different things.

Hardware We ran all experiments on 32 CPU cores, and RTX 3090 or v100 GPUs. Since our main metric is the fraction of saved searches (FoSS), a different GPU will not change our results. The experiments in Appendix C (Figure 14) were performed on the same machine using a RTX 3090 GPU.

# C. Fraction of Saved Searches (FoSS) vs. Wall-clock Saved Time

In our experiments in Section 5, we reported perplexity compared to FoSS (the fraction of saved searches). The other alternative of measuring wall-clock time is difficult to reproduce, is brittle to temporary hardware and system load, and affected by the specific kNN retrieval library such as FAISS as used in Khandelwal et al. (2020), ScaNN (Guo et al., 2020) as used in Borgeaud et al. (2021), or SPTAG (Chen et al., 2018), etc. Further, it depends on factors that are orthogonal to our contribution, such as whether the RAM is large enough to store the datastore, and the random-access reading latency of the hard-drive.

FoSS, in contrast, does not depend on hardware, engineering factors, or temporary system load. FoSS is also completely reproducible while wall-clock time is not. Thus, FoSS serves as a good proxy to wall-clock time that enables reproducible experiments. Here, we perform an empirical analysis of FoSS with respect to the saved wall-clock time.

Figure 14 shows a comparison between the fraction of saved wall-clock time and FoSS. As shown, the saved wall-clock time and FoSS are identical up to an additive constant that depends on the hardware and the specific settings. Searching for  $k$ -nearest neighbors using a CPU FAISS index results in a curve that is very close to the optimal  $y = x$ , meaning that almost the entire reduction in searches is translated directly into saving of wall-clock time. Using a GPU index without clustering (only pointers) results in a penalty of  $17\%$ , but the curve is almost parallel to the optimal  $y = x$ . Using a GPU index with clustering results in a penalty of  $24\%$ , and begins to be beneficial in terms of wall-clock time starting from FoSS=0.32.

The experiments in Figure 14 were performed in the setting that we load the datastore to memory, to prevent the hard drive's latency from being the bottleneck. Another option is to approximate the key vectors using the FAISS index, but currently

FAISS's reconstruct API is implemented only for a CPU index (rather than a GPU index),² and for a single key ID at a time,³ and thus does not support batching.

We expect that as the datastore size increases to the scales of Borgeaud et al. (2021), and as the number of neighbors retrieved increases ( $k_{\mathrm{neigh}}$ ) – the more pressure that will be put on the  $k$ NN search, the more of a bottleneck that it will become, and the larger relative benefit that saving  $k$ NN searches will provide to wall-clock time.

# D. Additional Results

Comparison of Datasets Figure 12 and Figure 13 show the overlap of n-grams between the training and validation set of WIKTEXT-103 and Law-MT. As shown, for all values of n, more n-grams from the validation set were seen in the training set in Law-MT compared to WIKTEXT-103. We see this as the explanation for the better scaling of RETOMATON on Law-MT, where the perplexity only gently increases as we increase FoSS, compared WIKTEXT-103.

Ablation Study Figure 9 and Figure 10 show results for different clustering algorithms and granularities, on WIKITEXT-103 and Law-MT, respectively. These figures are similar to Figure 5 and Figure 6, except that we include here more runs of  $k$ -means with more values of  $k_{\mathrm{clus}}$  and more runs of the greedy clustering of He et al. (2021).

$^{2}$ https://github.com/facebookresearch/faiss/issues/314  
 $^{3}$ https://github.com/facebookresearch/faiss/issues/1163

Figure 9: Analysis of the number of clusters on the validation set of WIKTEXT-103.

Figure 10: Analysis of the number of clusters on the validation set of Law-MT.

Figure 11: Histogram of the lengths of sequences that were predicted consecutively, without kNN search, in WIKITEXT-103.

Figure 12: The fraction of  $n$ -gram types in the validation set that appeared verbatim in the training set in each dataset, for different values of  $n$ .

Figure 13: The fraction of  $n$ -gram occurrences in the validation set that appeared verbatim in the training set in each dataset, for different values of  $n$ .

Figure 14: A comparison between the fraction of saved wall-clock time vs. FoSS, the fraction of saved searches. The fraction of saved wall-clock time was computed relatively to the baseline kNN-LM.

# Footnotes:

Page 3: Formally, we start every traversal from the initial state  $q_{0}$ , perform a kNN search to retrieve  $\mathcal{N}^{(t)}$ , and then make an  $\epsilon$ -transition (transitioning without consuming an input token) into  $\mathcal{S}^{(t)}$ . 
