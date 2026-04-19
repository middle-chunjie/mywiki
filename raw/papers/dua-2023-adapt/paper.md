# To Adapt or to Annotate: Challenges and Interventions for Domain Adaptation in Open-Domain Question Answering

Dheeru Dua<sup>1*</sup> Emma Strubell<sup>2,3</sup> Sameer Singh<sup>1</sup> Pat Verga<sup>2</sup>

<sup>1</sup>University of California Irvine <sup>2</sup> Google Research <sup>3</sup> Carnegie Melon University

# Abstract

Recent advances in open-domain question answering (ODQA) have demonstrated impressive accuracy on standard Wikipedia style benchmarks. However, it is less clear how robust these models are and how well they perform when applied to real-world applications in drastically different domains. While there has been some work investigating how well ODQA models perform when tested for out-of-domain (OOD) generalization, these studies have been conducted only under conservative shifts in data distribution and typically focus on a single component (ie. retrieval) rather than an end-to-end system. In response, we propose a more realistic and challenging domain shift evaluation setting and, through extensive experiments, study end-to-end model performance. We find that not only do models fail to generalize, but high retrieval scores often still yield poor answer prediction accuracy. We then categorize different types of shifts and propose techniques that, when presented with a new dataset, predict if intervention methods are likely to be successful. Finally, using insights from this analysis, we propose and evaluate several intervention methods which improve end-to-end answer F1 score by up to  $\sim 24$  points.

# 1 Introduction

General-purpose open-domain question answering (Chen et al., 2017; Lee et al., 2019) is an important task that necessitates reading and understanding a large number of documents and succinctly answering a given question. It is especially crucial in fields such as Biomedicine, Legal, News, etc., where a huge number of documents are added everyday and domain expertise is necessary to understand these documents.

Recently, there have been great advancements and successes in open-domain question answering


Figure 1: Top: Average Reader performance for Baseline and best interventional or augmentation setup on top. Bottom: The difference between baseline and performance (end-to-end F1) after introducing interventions, averaged over datasets exhibiting specific shift types.

models (ODQA). The state of the art ODQA systems perform a two-stage pipeline process (Izacard et al., 2022): 1) given a question, a context retriever (Karpukhin et al., 2020; Izacard et al., 2021; Raffel et al., 2020) selects relevant passages and 2) a question answering model, also known as reader (Izacard and Grave, 2020) answers the given question based on the retrieved passages. This decoupling allows for independent advancements in domain generalization and adaptation of general-purpose context retrievers (Thakur et al., 2021) and question answering (Fisch et al., 2019) models.

A general purpose ODQA model should be resilient to changes in document, question and answer distributions. However, existing works seldom study the effectiveness of a model trained on a particular source domain and applied to a new target domain. In this work, we ask the following questions:

1. How well do current state-of-the-art ODQA methods perform when tested on varying degrees of data shift and under what conditions they fail?  
2. Given a small set of labeled examples in the new target domain, can we predict whether existing intervention schemes would be useful in adapting from a given source model or would it better to collect annotations in the target domain?  
3. What interventions or adaptation strategies can we perform to improve ODQA performance in OOD testing?

Following the above research questions we make four primary contribution in this work.

First, in Section 2 we aggregate a set of seven ODQA datasets, spanning five different domains for evaluating domain generalization of an ODQA model trained on general purpose. In Section 4, we use this test-bed to show that most SotA ODQA models fail to generalize, and go on to analyze the failure modes for OOD generalization. For example, we observe that the retriever model's performance is quite sensitive to the type of entities and length of passages seen at training, and additionally, in  $\sim 65\%$  of cases where the answer string appears in the retrieval list (one of the most commonly used metrics for retrieval accuracy), the context does not justify the answer.

Second, in Section 3, we propose a generalizability test, that determines the type of dataset shift in the target datasets with only a few labeled examples in target domain. This gives us an idea of how likely it is for a model trained in the source domain to adapt to an unseen target domain. In Figure 1, we observe that target datasets which are close to source domain and exhibit 'No shift', do not show much improvement with zero or few shot data augmentation. While the target datasets, that are very different from source data and exhibit 'Full shift' need examples generated in a few shot way that capture the underlying target domain to adapt to the target dataset. We consider few-shot examples as proxy for target data distribution. Zero-shot data augmentation techniques yield best adaptation under 'Label shit' and 'Covariate shift'.

Third, in Section 5.1, we analyze the performance impact of various intervention schemes, such as heuristic data augmentations and language model generated pseudo-data, without relying on

any labeled target data. We observe that zero-shot data interventions yield a relatively high improvement in performance (up to  $15\%$  in F1) for some shift types, while others do not see these gains.

Finally, in Section 5.2, we propose a simple and effective technique for few-shot language model data generation requiring only a handful of examples from the target domain. While many existing works have leveraged question generation models for creating additional training data, these models are typically trained on the source domain data and suffer the same generalization shortcomings. Instead, inspired by the strong performance of large language models (LLM) for summarization, we generate sentences by prompting the LLM with a handful of examples from the target domain. We convert the generated sentences into cloze style QA pairs and show that this technique is especially effective when zero shot adaptation methods fail to capture the target domain distribution, yielding improvements of up to  $24\%$  in F1.

# 2 Background and Setup

An open-domain (ODQA) model learns interactions among three random variables: question  $(\mathbb{Q})$ , answer  $(\mathbb{A})$  and context  $(\mathbb{C})$ . For a given  $q \in \mathbb{Q}$ , first the retriever  $\mathcal{R}$  returns a set of passages,  $c_{q} = \mathcal{R}(q,\mathbb{C})$ . These passages are then sent to an answering model  $\mathcal{M}$  (also known as reader) to obtain the final answer,  $\hat{a} \gets \mathcal{M}(a|q,c_{q})$ .

In our experiments, we follow prior work and compute retriever performance as Accuracy at  $\mathrm{K(Acc@k)}$ , which computes if the oracle answer is found in the top- $k$  retrieved passages<sup>1</sup>. We set  $k = 100$  in all of our experiments. To measure the reader performance, we compute token-level  $F_{1}$  between the oracle answer and prediction from the answering model<sup>2</sup>.

# 2.1 Datasets

In this work, we test the generalizability of a model trained on a source domain to seven datasets in five vastly different target domains.

Source Domain: For all of our experiments, our source domain is English Wikipedia along with the supervised data from NaturalQuestions

(NQ) (Kwiatkowski et al., 2019) and BoolQ (Clark et al., 2019). We treat this domain as our source as it used for the vast majority of current work in ODQA (and many other areas of language research).

In addition to the supervised training data from NQ and BoolQ, we add additional cloze style questions derived from the QA pairs in NQ. For each qa pair, we retrieve a sentence from Wikipedia with the highest BM25 similarity score. We then convert the retrieved sentence into a cloze-style question by replacing the answer string in the sentence with sentinel markers (Raffel et al., 2020) $^3$ .

Target Domains: We consider five vastly different domains (Stackoverflow, Reddit, Pubmed, Japanese Statute Law codes, CNN/Dailymail and Wikipedia) as our target corpora and re-purpose seven open-domain QA and/or reading comprehension datasets for our evaluations (Figure ??). The datasets are Quasar-S (Dhingra et al., 2017), Quasar-T (Dhingra et al., 2017), SearchQA (Dunn et al., 2017) and BioASQ (Balikas et al., 2015) which were introduced as ODQA datasets over Stackoverflow, Reddit,Wikipedia and Pubmed corpus respectively.

Additionally, we re-purpose NewsQA (Trischler et al., 2016) and CliCR (Suster and Daelemans, 2018) as ODQA datasets. These datasets were originally introduced as reading comprehension evaluations and constructed by retrieving a set of passages for the given question from Pubmed and CNN/Dailymail corpus. We also re-purpose COLIEE (Rabelo et al., 2022), originally an entailment based QA dataset, by transforming the examples into boolean questions and retrieving passages from a Japanese Statute Law corpus. End-to-end performance of ODQA models trained on target QA pairs with BM25 retrievals from the target corpus (UB-Ret, Figure 5), indicates that these datasets can be reasonably re-purposed for our ODQA setup. Figure 2 shows some examples from target datasets.

# 2.2 Models

Retrievers: We compare four diverse retrievers: 1) BM25 (Robertson and Spärck Jones, 1994) (sparse and unsupervised), 2) Contriever (semi-supervised with MS-MARCO) (Izacard et al., 2021) 3) Dense

Passage Retriever (DPR) (Karpukhin et al., 2020) and the state-of-the-art model 4) Spider (Ram et al., 2021) (supervised with NaturalQuestions).

Reader: We use the state-of-the-art T5-large based fusion-in-decoder (FiD) model (Izacard and Grave, 2020) which encodes top 100 documents in parallel. The representation are concatenated and then decoded to generate the final answer.

# 3 Categorizing Data Shift Types

There are many aspects that determine in what ways and to what extent one data distribution differs from another. Having a better understanding of this spectrum of possibilities would enable us to predict whether a new dataset would be compatible with an existing model and, if not, what types of interventions would be required in order to enable the model to adapt to the new domain.

In this section, we define a taxonomy of shift types for ODQA based on the distributions of the sub-components of the problem (answer, question, and context distributions), and develop of technique for categorizing target domains amongst those shift types. While we find in later sections that the type of shift often influences the effectiveness of an interventional scheme (See sections 5 and 5.2), we also find that actually determining the type of shift is quite challenging.

# 3.1 Types of dataset shift

Each domain contains both an input distribution (questions, contexts) and output distribution (answers). The compatibility - or lack-there-of - over these two sub-distributions lead to four possible settings.

No shift both the input and output distributions between the source and target domain match.

Label shift (Storkey et al., 2009) occurs when the input distributions of the source and target domains match, i.e.,  $p_{s}(x_{s}) = p_{t}(x_{t})$  while the output label distribution given the input between source and target domain does not match,  $p_{s}(y_{s}|x_{s}) \neq p_{t}(y_{t}|x_{t})$ .

Covariate shift (Zadrozny, 2004) occurs when the source and target input distributions do not match i.e.,  $p_{s}(x_{s}) \neq p_{t}(x_{t})$  while the output label distributions matches  $p_{s}(y_{s}|x_{s}) = p_{t}(y_{t}|x_{t})$ .

Full shift occurs when both the source and target input and output distributions do not match.

<table><tr><td>Dataset, Corpus</td><td>#ques, #docs</td><td>Passage</td><td>Question-Answer</td></tr><tr><td>BioASQ, Pubmed</td><td>5k, 30M</td><td>Parkinson&#x27;s disease (PD) is one of the most common degenerative disorders of the central nervous system that produces motor and non-motor symptoms. The majority of cases are idiopathic and characterized by the presence of Lewy bodies.</td><td>Q: Which disease of the central nervous system is characterized by the presence of Lewy bodies? A: Parkinson&#x27;s disease</td></tr><tr><td>ClioCR, Pubmed</td><td>90k, 30M</td><td>Detailed history and examination ruled out the above causes except the exposure to high altitude as a cause for koilonychia in our patient. Exposure to high altitude is a known aetiology for koilonychias, also described by some authors as “Ladakhi koilonychia”.</td><td>Q: _ is a known cause of koilonychia, described by some as Ladakhi koilonychia. A: High altitude exposure</td></tr><tr><td>Quasar-S, Stackover-flow</td><td>30k, 1.5M</td><td>I have a mixed integer quadratic program MIQP which I would like to solve using SCIP. The program is in the form such that on fixing the integer variables the problem turns out to be a linear program.</td><td>Q: scip – an software package for solving mixed integer _ problems A: linear-programming</td></tr><tr><td>Quasar-T, Reddit</td><td>30k, 2M</td><td>Because of widespread immunization , tetanus is now rare. Another name for tetanus is lockjaw.</td><td>Q: Lockjaw is another name for which disease A: tetanus</td></tr><tr><td>NewsQA, Dailymail</td><td>70k, 0.5M</td><td>Former boxing champion Vernon Forrest, 38, was shot and killed in southwest Atlanta, Georgia, on July 25.</td><td>Q: Where was Forrest killed ? A: in southwest Atlanta, Georgia</td></tr><tr><td>SearchQA, Wikipedia</td><td>70k, 20M</td><td>The Dangerous Summer and The Garden of Eden. Written in 1959 while Hemingway was in Spain on commission for Life...</td><td>Q: While he was in Spain in 1959, he wrote “The Dangerous Summer”, a story about rival bullfighters A: Hemingway</td></tr><tr><td>COLIEE, Japanese Legal Codes</td><td>886, 1k</td><td>A manifestation of intention based on fraud or duress is voidable. If a third party commits a fraud inducing a first party to make a manifestation of intention to a second party, that manifestation of intention is voidable only if the second party knew or could have known that fact. The rescission of a manifestation of intention induced by fraud under the provisions of the preceding two paragraphs may not be duly asserted against a third party in good faith acting without negligence.</td><td>Q: Is it true: A person who made a manifestation of intention which was induced by duress emanated from a third party may rescind such manifestation of intention on the basis of duress, only if the other party knew or was negligent of such fact. A: No</td></tr></table>

Figure 2: Examples from datasets with context and question-answer pairs from different domains.

Figure 3: Generalizability test: At first level, the farther the target distribution from uniform as compared to gold, the closer it is to the source. At second level, the gradual increase from left to right in the leaf nodes depicts increase in difference between distance of reference (source) from uniform and distance of target from uniform. The lower the difference (i.e, the left branch at final depth), the closer is the target to source.

# 3.2 Calculating Shift in ODQA

Most existing works consider classification setups where it is easy to compute the input and output distributions. However, in our setting, we lack a consistent method for computing these distributions which often require large amounts of labeled target data to train a target model for comparison. As an alternative, we determine the type of dataset shift by estimating whether the source model contains useful information about the input and output distributions of the target dataset when compared with an uninformative uniform prior.

We characterize shift in ODQA as a two-step process. We first compute the input distribution, i.e., the joint question and context distribution using unnormalized (energy) scores from a dense retriever (Karpukhin et al., 2020) to quantify the compatibility between a given question and a context via  $\mathcal{R}(q,c)$ . Then, obtain the likelihood of the gold context for a given question by normalizing the energy scores from the retriever over a set of contexts. This computation over the entire corpus can be very expensive and results in a low entropy distribution. To address this, we sample a set of contexts,  $C$ , from the entire corpus  $\mathbb{C}$ .

$$
p (q, c _ {g}) = \frac {\mathcal {R} (q , c _ {g})}{\sum_ {c _ {k} \in C} \mathcal {R} (q , c _ {k})} \tag {1}
$$

In the second step, we test if the output distributions match by computing the likelihood of generating an oracle answer given a question and the relevant contexts. We use global normalization (Goyal et al., 2019) for computing the probability distribution over a set of answer spans. Ideally, the normalization should be computed over all possible answer spans in the corpus which is intractable. We instead sample a set of answer spans to approximate the normalizer.

$$
p \left(a _ {g} \mid q, c _ {q}\right) = \frac {\prod_ {t} \mathcal {M} \left(a _ {g} ^ {t} \mid a _ {g} ^ {<   t} , q , c _ {q}\right)}{\sum_ {a _ {k} \in \mathcal {A}} \prod_ {t} \mathcal {M} \left(a _ {k} ^ {t} \mid a _ {k} ^ {<   t} , q , c _ {q}\right)} \tag {2}
$$

# 3.3 Predicting type of dataset shift

Adapting or fine-tuning a pre-trained source model to match the target domain, can be formulated in a Bayesian framework. The source model acts as a prior which when exposed to interventional data, that estimates the likelihood of target domain,

results in a (fine-tuned) posterior distribution.

$$
\begin{array}{l} \underbrace {p (\theta_ {t} | x _ {t})} _ {\text {p o s t e r i o r}} = \underbrace {p (\theta_ {s} ; x _ {s})} _ {\text {p r i o r}} \underbrace {p (x _ {t} | \theta_ {s} ; x _ {s})} _ {\text {l i k e l i h o o d}} \\ = p _ {s} \left(x _ {s}\right) p _ {t} \left(x _ {l} \mid x _ {s}\right) \\ \end{array}
$$

To analyze the type of dataset shift, we devise a generalizability test, where we compare the prior distribution to an uninformative prior like the uniform distribution. In particular, if the source model is closer to the uniform distribution when compared with the oracle distribution it does not have the reasoning ability (informative signal) to understand the target domain. We assume we have access to a few labeled examples in target domain for evaluation.

Input/Retriever Distribution: In the first stage, we compute the input distribution using retriever scores by following Eq. 1. Then, for a given question, we compute the distance of the input distribution of the target domain from the uniform distribution,  $d_u^t$  and average the distances over the set of examples in the evaluation set. Similarly, we also compute the distance from the gold distribution as  $d_g^t$ . If  $d_u^t > d_u^g$ , we conclude that the distance of a target distribution is far from the uniform distribution and closer to the gold distribution, indicating that the source distribution is likely compatible with the target domain (Figure 3). Since we do not assume access to labeled target domain data for training, this compatibility measure is used as a proxy to infer that  $p_s(q, c) \approx p_t(q, c_t)$ . We use this notation as a way to interpret that source and target distributions are compatible and not necessarily equal.

<table><tr><td>Dataset</td><td>Retriever</td><td>Reader</td><td>Shift</td></tr><tr><td>BioASQ</td><td>0.3027</td><td>0.1765</td><td>Label</td></tr><tr><td>CliCR</td><td>-0.8839</td><td>0.2352</td><td>Full</td></tr><tr><td>Quasar-S</td><td>-0.6697</td><td>0.0767</td><td>Covar.</td></tr><tr><td>Quasar-T</td><td>0.2016</td><td>0.1694</td><td>Label</td></tr><tr><td>NewsQA</td><td>-0.1967</td><td>0.1800</td><td>Full</td></tr><tr><td>SearchQA</td><td>0.6165</td><td>-0.0063</td><td>No</td></tr></table>

Table 1: Wasserstein distance: Computed over 100 examples labeled examples from target domain. The reference of source domain model has  $d_u^r = 0.2925$

Output/Reader Distribution: In the second stage, we follow a similar procedure to characterize for the output distribution. To analyze the compatibility between the output answer distribution and a uniform distribution, we need to compute

a probability distribution over a set of answers similar to stage 1. However, the conditional answer generation model is not trained with a contrastive loss like the retriever leading to the answer likelihood distribution having a higher entropy. Also, the support set of answers used for normalization contains only grammatically correct answer spans making the likelihood scores attenuated. To deal with these issues, we use a reference answer conditional distribution to de-bias the likelihood scores with a threshold. We treat the source distribution as our reference and compute the distance from the uniform and gold distributions with respect to the source distribution on 100 examples from the validation set of the source domain. To infer if  $p_{s}(a|q,c) \approx p_{t}(a|q,c)$ , we determine the difference between the distance of reference distribution from uniform and distance of target distribution from uniform. If this difference is close to 0, we conjecture that the  $p_{s}(a|q,c)$  and  $p_{t}(a|q,c)$  are compatible.

In Figure 3, we can see that the dataset SearchQA falls under the "No shift" category, hence, we conjecture that it will observe minimal improvements under most data intervention schemes, as the source is already able to capture the target distribution well (Section 5.1, 5.2). We also conjecture that datasets falling under the category of "Label shift" and "Covariate shift" are more amenable to zero-shot data interventions, however, "Full shift" would benefit most from few-shot examples or collecting annotations in the target domain. We consider few shot augmentations as a proxy for annotating examples in the target domain because the augmentations are generated with supervision from target data.

# 4 How Well do Models Generalize?

In this section, we want to first get a sense of how well existing SotA ODQA models perform when tested OOD. We test the OOD performance of source-trained models on target domain validation sets and, when they fail, analyze what caused those errors.

# 4.1 End-to-End Zero-shot Generalization

in Figure 5, we test the end-to-end domain adaption performance of three model variants:

Source: a fully source domain trained model with BM25 retrieved documents, demonstrating zero-shot generalization performance.

Upperbound-Reader a target domain trained

reader model with contexts retrieved by BM25 – the overall strongest retriever.

Upperbound-Retriever a target domain trained reader model with gold contexts to approximate upper-bound performance.

Overall, when testing models on the new target domains we observe large performance drops. This is especially true when the target corpus differs from Wikipedia, such as in Quasar-S (stackoverflow) and CliCR (pubmed), even though the model requires similar reading capabilities to those needed in the source domain.

Interestingly, even though the BM25 retriever accuracy is relatively high on the target datasets, (for example,  $\sim 83\%$  Acc@100 on Quasar-S), that accuracy does not translate to strong reader performance and therefore, overall QA accuracy ( $\sim 11\%$  F1 on Quasar-S, Figure 5).

To understand the performance gap, we manually sample 50 prediction from each target dataset where retrieved passages contain the oracle answer but the reader produced an incorrect prediction. We observe that in around  $65\%$  cases, the Acc@100 metric yields a false positive, where the passage contains an exact string match of the correct answer, but the context does not actually answer the given question. For example, the question "What is the name of the office used by the president in the white house?" and answer "oval", the retrieved passage "A tunnel was dug into the White House connecting the Oval Office to a location in the East Wing...." is credited to able to answer the question. This shows that end-to-end performance is crucial in understanding improvements in retrievers which is often ignored.

# 4.2 Retriever Generalization

To analyze model performance further, we compare the zero-shot generalization performance of four different retrieval models in figure 4: BM25, Contriever, Spider and DPR.

One observation we find is that Spider, the best performing model on the source domain, exhibits an improvement on SearchQA  $(\sim 1\%)$  (which uses the same underlying source Wikipedia domain), but shows large drops in performance when applied to the target datasets:  $\sim 40\%$  on NewsQA,  $\sim 28\%$  on Quasar-T and, Quasar-S.

To understand the reason for such an enormous performance drop, we sample 50 random incorrect predictions from Spider for manual analysis.

Figure 4: Retriever performance (Acc@100) without any interventions on target domain corpus

We observe two major failure modes. First, we find that dense models are sensitive to changes in the length of contexts. When exposed to documents with heterogeneous lengths that differ from those that they were trained on, models tend to over retrieve shorter contexts. To quantify the sensitivity to changes in lengths on source domains itself, we pool passages from all target corpus into a combined index. We observe that performance of Spider when exposed to this combined index reduces by  $\sim 15\%$  and restricting the minimum length of contexts to be 50 words alleviates the problem and recovers the original performance. The second common failure mode occurs due to changes in distribution of entity types from source to target domain, for instance words like "plant" in question "Which is produced in plants of narora kakrapar tarapur" refers to "power plant" in the Wikipedia domain, while in case of Pubmed "plant" often refers to living organic matter (Sciavolino et al., 2021). This is more evident in Spider which uses an auxiliary loss that encourages documents with shared recurring spans (mostly entities) to be closer to each other. This skews model learning to entities seen during training. Overall, BM25, being an unsupervised method, shows the most competitive

Figure 5: Reader performance on target validation set without any interventions. SearchQA, Quasar-S and Quasar-T do not have gold passage annotations so both upperbound are same. The majority voting baseline on COLIEE is 50.95

performance across all the domains.

# 5 Interventions for Improving Adaption

In the previous section, we hypothesize which target datasets are easily adapted to by a source domain model. Based on the generalizability test, our conjecture was that datasets with a less severe shift like Quasar-S, Quasar-T, and BioASQ would show marked performance improvements with zero-shot adaptation when compared with datasets like CliCR and NewsQA. In the following experiments we observe an average performance improvement of about  $8.5\%$  F1 on datasets with label shift and covariate shift as compared to  $3.5\%$  F1 on datasets with full shift.

# 5.1 Zero-shot Adaptation Methods

We perform a series of controlled zero-shot data intervention methods, where we consider the effect of change in distribution of each random variable: question  $(\mathbb{Q})$ , answer  $(\mathbb{A})$  and context  $(\mathbb{C})$  one at a time, while keeping the other two fixed. Our zero-shot interventions utilize only unsupervised (i.e., no question-answer pair annotations) data from the target domain but use source domain data in various ways to generate examples in the target domain.

Varying context distribution To test the effect of change in context distribution, we pool all pas

sages from each dataset into a single document index. In figure 6, we observe that learned models like Spider are sensitive to out-of-domain distractors, especially when a target dataset is based on the source domain corpus (Wikipedia). For instance, SearchQA and NQ both suffer a performance drop of about  $\sim 15\%$ . On the other hand, unsupervised BM25 is much more robust and has a consistent performance even when exposed to a larger pool of documents with the one exception being the legal domain in COLIEE which is a very small index and loses representation when combined with much bugger datasets.

Figure 6: Retriever Performance (Acc@100): Varying context distribution by creating a combined document index

Additionally, in Figure 7 we show that the FiD reader is not as sensitive as the retriever to changes in context distribution (target vs combined) as we observe only a drop of  $3\%$  in F1 for NewsQA in worst case scenario.

Varying answer distribution Many works (Gururangan et al., 2018; Dua et al., 2020; Jiang and Bansal, 2019) have shown that unanticipated bias in answer prior distribution can introduce spurious correlations in model learning. In this experiment, we vary the answer distribution by changing the sampling distribution over plausible answer spans. First, we extract and annotate coarse grain entity

Figure 7: Reader Performance (F1): Effect of change in context distribution with BM25 retrievals from the combined index.

types from the target corpus using spaCy4. We then use this coarse-grain entity type information as a set of classes to sample entities to act as cloze-style answers. We choose 50k entities with four different sampling strategies: most frequent, uniformly sampled from entity type categories, randomly sampled from various entity type categories and sampling in proportion to entity type distribution of answers in training set of target dataset.

We choose BioASQ to perform these controlled experiments because the source model has a reasonable end-to-end performance on BioASQ even when retrieving passages from the source domain Wikipedia corpus (Figure 7), suggesting that the source corpus contains sufficient information for answering many BioASQ questions. This allows us to use the Wikipedia corpus alone for retrieval, which is useful to control for fixed passage distribution and gauge the impact of the answer distribution in isolation.

In Table 2, we show that choosing the answer distribution proportional to the uniform distribution across entity type categories boosts retriever performance compared to random sampling, allowing the model to capture all types of answers and generalize better to unseen answer distributions. On the other hand, the best reader model performance is achieved when we know the correct answer distribution of the target dataset upfront, as we see in Table 3. While this demonstrates that answer

priors influence reader performance, in an unsupervised setup we will not have this true distribution. Therefore, we adopt the second best technique, i.e., uniform sampling from across the entity type categories for other experiments in the paper (Table 5).

<table><tr><td>Augmentations</td><td>Acc@100</td></tr><tr><td>Random</td><td>45.35</td></tr><tr><td>Uniform</td><td>50.02</td></tr><tr><td>Most frequent</td><td>39.33</td></tr><tr><td>BioASQ train answers</td><td>47.48</td></tr></table>

To understand the impact of the pre-training vs fine-tuning corpus, we also compare the performance of the FiD reader initialized from T5 pretrained on common-crawl dataset(C4) compared to one that was pre-trained on pubmed articles. After pretraining, both models are then fine-tuned on our source domain data. In this case, we observe that fine-tuning on a domain that differs from that used in pre-training results in deterioration of model performance.

Table 2: Answer distribution: Retriver performance on BioASQ  

<table><tr><td>Augmentations</td><td>C4</td><td>Pubmed</td></tr><tr><td>Random</td><td>33.50</td><td>33.51</td></tr><tr><td>Uniform</td><td>39.07</td><td>35.97</td></tr><tr><td>Most frequent</td><td>38.18</td><td>34.90</td></tr><tr><td>BioASQ train answers</td><td>41.33</td><td>36.71</td></tr></table>

Varying question distribution To vary the question distribution, we augment the source domain with augmentations generated from the target domain using two different methods. Our first approach uses a question generation (Subramanian et al., 2017) model trained on the source domain to generate a question given a passage and an answer. This question generation model can be applied to a new target passage and a plausible answer span (entity mention) from the passage (Shakeri et al., 2020; Krishna and Iyyer, 2019; Song et al., 2018; Klein and Nabi, 2019). We refer to this method as "Standard QGen" in table 4 and 5. Our second approach, which has been less explored previously, converts a sentence in the target corpus to a fill-in-the-blank style cloze question (Taylor, 1953) by masking a plausible answer span (entity mention) in the sentence. We refer to this method as "Cloze QA".

In order to sample answers for which we should curate Standard and Cloze QA pairs, we follow the previous subsection and sample answer spans uniformly based on an entity type distribution from the target corpus. We then query our combined index to create a dataset containing cloze style questions aligned with relevant documents. We use these same sampled answers to generate standard QGen QA pairs as well.

We combine this augmented data with our initial source domain data to train a DPR retriever (Table 4) and a FiD reader (Table 5). We observe similar average performance across both intervention types in retriever and reader models. However, cloze QA pairs are computationally much more efficient to generate as they do not require any additional question generation models.

Table 3: Answer distribution: Reader performance on BioASQ  

<table><tr><td></td><td>Baseline</td><td>Cloze QA</td><td>Standard QGen</td></tr><tr><td>ClioR</td><td>23.87</td><td>24.88</td><td>23.99</td></tr><tr><td>BioASQ</td><td>50.41</td><td>48.04</td><td>45.45</td></tr><tr><td>Quasar-S</td><td>50.37</td><td>66.87</td><td>68.21</td></tr><tr><td>Quasar-T</td><td>54.77</td><td>53.93</td><td>55.57</td></tr><tr><td>NewsQA</td><td>12.54</td><td>18.79</td><td>15.22</td></tr><tr><td>SearchQA</td><td>63.03</td><td>52.97</td><td>54.77</td></tr><tr><td>COLIEE</td><td>61.47</td><td>60.55</td><td>57.80</td></tr></table>

# 5.2 Few-shot Generalizability and Adapatability

In section 5, we saw that zero-shot adaptation does not work well in cases where the target domain distribution is very far from the source domain. As hypothesized in section 3, we would expect improvements from few-shot interventions in the "Full Shift" datasets NewsQA and CliCR to be more effective than the zero-shot interventions from Section 5. In this section, we find that to be true in addition to the largely across-the-board effectiveness of few-shot interventions.

Table 4: Retriever performance: Comparing two types of question formats for augmentation  

<table><tr><td></td><td>Baseline</td><td>Cloze QA</td><td>Standard QGen</td></tr><tr><td>BioASQ</td><td>45.38</td><td>49.41</td><td>46.43</td></tr><tr><td>CliCR</td><td>6.126</td><td>7.340</td><td>10.56</td></tr><tr><td>Quasar-S</td><td>10.24</td><td>21.79</td><td>17.47</td></tr><tr><td>Quasar-T</td><td>34.92</td><td>41.99</td><td>44.73</td></tr><tr><td>NewsQA</td><td>18.57</td><td>21.20</td><td>12.71</td></tr><tr><td>SearchQA</td><td>34.60</td><td>38.80</td><td>37.27</td></tr><tr><td>COLIEE</td><td>46.79</td><td>54.17</td><td>62.38</td></tr></table>

Table 5: Reader performance: Comparing two types of question formats for augmentation

Few-shot Data Generation Zero-shot interventions like question generation models are trained on the source domain and inevitably do not produce generations that are fully compatible with the target domain, leading to degradation when the source and target domains differ drastically. An alternative approach would be to train a question generation model with a few examples from the target domain. However, in practice it is difficult to adapt or fine-tune a question generation and answering model (for validating QA pair correctness) with only a handful of examples.

To alleviate this problem, we propose a few shot technique that prompts a LLM (Chowdhery et al., 2022) to generate a sentence given a passage. We use eight seed examples from the target domain to generate additional training data to help bootstrap adaptation in the target domain. We observe that it is easier for large language models to condition on a single variable (context) and compress (Goyal et al., 2022) multiple facts from the passage into a single sentence, as compared to conditioning on a context and answer span together. Moreover, in section 5.1 we observed that augmentation with cloze style QA pairs yielded similar performance to using question-formatted QA pairs, offering evidence that the precise format is not as important as the content itself.

We prompt the model in the following format, "After reading the article, «context» the doctor said «sentence». For pubmed articles. For other target corpus we replace doctor with engineer, journalist and poster for stackoverflow, dailymail and reddit respectively. To filter out invalid sentences, we apply three simple heuristics and remove any generation that 1) includes a number, 2) does not repeat part of the passage verbatim, and 3) has less than  $75\%$  word set overlap with the passage (after removing stopwords). To gauge the precision of our generations, we manually sampled 20 generated sentences for each dataset and found that they were correct more than  $70\%$  of the time.

To test the retriever performance, we train a DPR model with NaturalQuestions and around  $\sim 8\mathrm{k - }10\mathrm{k}$  examples, containing pairs of original passage and generated sentence. We compare this model with original source domain DPR model in Table 6. We observe performance improvements of upto  $\sim 18\%$  in NewsQA and  $\sim 21\%$  in Quasar-S.

Comparison to Few-Shot Closed-Book Rather than use a LLM and few-shot prompting to gen

<table><tr><td></td><td>Baseline</td><td>DataGen</td></tr><tr><td>ClioCR</td><td>23.87</td><td>29.06</td></tr><tr><td>BioASQ</td><td>50.41</td><td>51.36</td></tr><tr><td>Quasar-S</td><td>50.37</td><td>71.93</td></tr><tr><td>Quasar-T</td><td>54.77</td><td>55.47</td></tr><tr><td>NewsQA</td><td>12.54</td><td>22.69</td></tr><tr><td>SearchQA</td><td>63.03</td><td>63.35</td></tr><tr><td>COLIEE</td><td>73.39</td><td>82.23</td></tr></table>

erate data, one could alternatively use the same model and examples to answer questions directly. We next test to what extent the LLM can perform closed-book QA by prompting the same model as used in our data generation with 8 examples that demonstrate how to answer questions in the target domain. In Table 7, we observe that the LLM does well on datasets with trivia style factual questions, like SearchQA and Quasar-T, but in other cases does not perform as well. The few-shot data augmentation trained model, on the other hand, performs better across a wider range of domains and datasets with the improvements upto  $\sim 24\%$  in F1 on Quasar-S when compared with baseline.

Table 6: Retriever Acc@100 with target specific few shot augmentations (DataGen).  

<table><tr><td>Reader Params</td><td>Baseline (770M)</td><td>Closed-Book (540B)</td><td>DataGen (770M)</td></tr><tr><td>BioASQ</td><td>45.38</td><td>32.02</td><td>50.64</td></tr><tr><td>CliCR</td><td>6.126</td><td>10.84</td><td>19.42</td></tr><tr><td>Quasar-S</td><td>10.24</td><td>23.75</td><td>34.19</td></tr><tr><td>Quasar-T</td><td>34.92</td><td>55.32</td><td>45.86</td></tr><tr><td>NewsQA</td><td>18.57</td><td>8.67</td><td>23.37</td></tr><tr><td>SearchQA</td><td>34.60</td><td>61.53</td><td>37.65</td></tr><tr><td>COLIEE</td><td>46.79</td><td>53.02</td><td>61.11</td></tr></table>

Table 7: Reader: F1 performance with target specific few-shot augmentations (DataGen). Both Close-Book and DataGen use eight examples from the target domain. Few-shot closed-book performance on NQ with eight examples is 36.71

In Figure 8, we show qualitative examples generated by our few-shot method depicting that they are able to compose facts from multiple sentences.

# 6 Related Work

Most existing works on domain adaption in question answering consider generalization of the retriever or reader in isolation.

Domain generalization in readers: The most popular work in generalization in reading comprehension was introduced as part of the

<table><tr><td>Dataset, Corpus</td><td>Passage</td><td>Generated Sentence</td></tr><tr><td>BioASQ, Pubmed</td><td>Herceptin is widely used in treating Her2-overexpressing breast cancer. However, the application of Herceptin in prostate cancer is still controversial.... This implies that targeting Her2 by both radio- and immunotherapy might be a potential strategy for treating patients with androgen-independent prostate cancer...</td><td>Herceptin is a breast cancer drug that has been used in treating prostate cancer.</td></tr><tr><td>ClioCR, Pubmed</td><td>An infant was admitted with symptoms of diarrhoea and vomiting. After initial improvement she unexpectedly died. Postmortem confirmed a diagnosis of cytomegalovirus (CMV) enterocolitis. The authors report this case and review other published cases of immunocompetent infants who presented with this infection. Clinicians should consider stool CMV PCR test or referral for endoscopy and biopsy in young babies who present with profuse and prolonged episodes of diarrhoea.</td><td>Immunocompetent infants can present with CMV enterocolitis.</td></tr><tr><td>Quasar-S, Stackover-flow</td><td>I’ve recently found scalp-bindgen from a Gitter room on Scala Native. Seems like at the present point in time they are developing a tool for generating Scala bindings for C header-files. Are there plans for generating Scala bindings for Objective-C and C++ too...</td><td>scalabindgen – scalp-bindgen is a tool that generates scalp bindings for C header files.</td></tr><tr><td>Quasar-T, Reddit</td><td>Interview With Gary James’ Interview With Marshall Lytle of Bill Haley’s Comets It can be safely said that “Rock Around The Clock” was the song by the group Bill Haley And His Comets that started the Rock ’n Roll movement. Still performing today, he spoke about those early days of Rock ’n Roll and his appreciation for what it meant to him.</td><td>Bill Haley and his comets made rock and roll music</td></tr><tr><td>NewsQA, CNN/Dailymail</td><td>The Kardashianians are already a staple on E! Network . But they’ve chosen the month of November to assert their dominance on the book world. Kourtney, Kim, and Khloe’s first novel,” Dollhouse ,” hits shelves today . “Dollhouse,” the first fiction endeavor from the Kardashianians, follows sisters Kamille, Kassidy, ...</td><td>The Kardashianians released a new book called ‘Dollhouse’.</td></tr><tr><td>SearchQA, Wikipedia</td><td>Charles Henry Dow was an American journalist who co-founded Dow Jones and Company with Edward Jones and Charles Bergstresser. Dow also founded The Wall Street Journal, which has become one of the most respected financial publications in the world... In 1877, he published a History of Steam Navigation between New York and...</td><td>Charles Henry Dow, an American journalist, founded The Wall Street Journal in 1882.</td></tr></table>

Figure 8: Examples of data generated from few-shot prompting.

MRQA (Fisch et al., 2019) challenge, which focuses on transfer of learning from multiple source datasets to unseen target datasets. This multi-task learning setup requires model to perform reasoning at test time that may be unseen at training. It is used as a way to discern what type of reasoning abilities learned at training time are more beneficial for generalization to a cohort of unseen reasoning abilities. However, in this work, we focus on generalization capabilities of an end-to-end ODQA setup to be able read and understand passages in new domain and not the abilities to perform unseen reasoning.

Domain generalization in retrievers: A recent line of work that test domain generalization of retrievers (Petroni et al., 2020; Ram et al., 2021; Izacard et al., 2022) focuses on conservative changes to source domain, for instance testing generalization performance of model trained Natural Questions to WebQuestions, TriviaQA – all of which use the same Wikipedia corpus. Another line of work follows a recently proposed retrieval benchmark, BEIR (Thakur et al., 2021) that tests generalizability of a general purpose retriever to different corpus/domains. Moreover, it considers only re

triever performance in isolation and not end-to-end ODQA performance which can be a brittle metric. Also, it examines the ability of a general purpose retriever to generalize to various domains out-of-the-box and not necessarily how to adapt to a new domain.

Domain adaptation work in retrievers (Dai et al., 2022) generate passages in a few shot manner given the query but this does not require the answer (entities) to be correct in the generated passage. (Ma et al., 2020) performs a zero-shot adaptation with noisy labels as it is difficult to train a QA val-. idator in target domain. (Siriwardhana et al., 2022) utilizes examples from target domain in a transfer learning setup while we work in zero to few shot setting.

# 7 Conclusion

In this work we investigated domain generalization in open domain question answering and presented four main contributions. First, we analysed the problems with existing ODQA model and investigate their failure modes. Second, we explored various zero-shot and few-shot data interventions

to improve a model's ability to generalize to an unseen target domain. Finally, we described a taxonomy of dataset shift types that provides an way to approximate how effective a source domain trained model can be adapted towards a new target domain.

# References

Georgios Balikas, Anastasia Krithara, Ioannis Partalas, and George Paliouras. 2015. Bioasq: A challenge on large-scale biomedical semantic indexing and question answering. In International Workshop on Multimodal Retrieval in the Medical Domain, pages 26-39. Springer.  
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading wikipedia to answer open-domain questions. arXiv preprint arXiv:1704.00051.  
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.  
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044.  
Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B Hall, and Ming-Wei Chang. 2022. Promptagator: Few-shot dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755.  
Bhuwan Dhingra, Kathryn Mazaitis, and William W Cohen. 2017. Quasar: Datasets for question answering by search and reading. arXiv preprint arXiv:1707.03904.  
Dheeru Dua, Sameer Singh, and Matt Gardner. 2020. Benefits of intermediate annotations in reading comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5627-5634.  
Matthew Dunn, Levent Sagun, Mike Higgins, V Ugur Guney, Volkan Cirik, and Kyunghyun Cho. 2017. Searchqa: A new q&a dataset augmented with context from a search engine. arXiv preprint arXiv:1704.05179.  
Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. 2019. Mrqa 2019 shared task: Evaluating generalization in reading comprehension. arXiv preprint arXiv:1910.09753.  
Kartik Goyal, Chris Dyer, and Taylor Berg-Kirkpatrick. 2019. An empirical investigation of global and local

normalization for recurrent neural sequence models using a continuous relaxation to beam search. arXiv preprint arXiv:1904.06834.  
Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2022. News summarization and evaluation in the era of gpt-3. arXiv preprint arXiv:2209.12356.  
Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R Bowman, and Noah A Smith. 2018. Annotation artifacts in natural language inference data. arXiv preprint arXiv:1803.02324.  
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.  
Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282.  
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299.  
Yichen Jiang and Mohit Bansal. 2019. Avoiding reasoning shortcuts: Adversarial evaluation, training, and model development for multi-hop qa. arXiv preprint arXiv:1906.07132.  
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.  
Tassilo Klein and Moin Nabi. 2019. Learning to answer by learning to ask: Getting the best of gpt-2 and bert worlds. arXiv preprint arXiv:1911.02365.  
Kalpesh Krishna and Mohit Iyyer. 2019. Generating question-answer hierarchies. arXiv preprint arXiv:1906.02622.  
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466.  
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. arXiv preprint arXiv:1906.00300.

Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan McDonald. 2020. Zero-shot neural passage retrieval via domain-targeted synthetic question generation. arXiv preprint arXiv:2004.14503.  
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. 2020. Kilt: a benchmark for knowledge intensive language tasks. arXiv preprint arXiv:2009.02252.  
Juliano Rabelo, Randy Goebel, Mi-Young Kim, Yoshi-nobu Kano, Masaharu Yoshioka, and Ken Satoh. 2022. Overview and discussion of the competition on legal information extraction/entailment (col-ee) 2021. The Review of Socionetwork Strategies, 16(1):111-133.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1-67.  
Ori Ram, Gal Shachaf, Omer Levy, Jonathan Berant, and Amir Globerson. 2021. Learning to retrieve passages without supervision. arXiv preprint arXiv:2112.07708.  
Stephen E Robertson and Karen Sparck Jones. 1994. Simple, proven approaches to text retrieval. Technical report, University of Cambridge, Computer Laboratory.  
Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. 2021. Simple entity-centric questions challenge dense retrievers. arXiv preprint arXiv:2109.08535.  
Siamak Shakeri, Cicero Nogueira dos Santos, Henry Zhu, Patrick Ng, Feng Nan, Zhiguo Wang, Ramesh Nallapati, and Bing Xiang. 2020. End-to-end synthetic data generation for domain adaptation of question answering systems. arXiv preprint arXiv:2010.06028.  
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. 2022. Improving the domain adaptation of retrieval augmented generation (rag) models for open domain question answering. arXiv preprint arXiv:2210.02627.  
Linfeng Song, Zhiguo Wang, Wael Hamza, Yue Zhang, and Daniel Gildea. 2018. Leveraging context information for natural question generation. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 569-574.  
Amos Storkey et al. 2009. When training and test sets are different: characterizing learning transfer. Dataset shift in machine learning, 30:3-28.

Sandeep Subramanian, Tong Wang, Xingdi Yuan, Saizheng Zhang, Yoshua Bengio, and Adam Trischler. 2017. Neural models for key phrase detection and question generation. arXiv preprint arXiv:1706.04560.  
Simon Šuster and Walter Daelemans. 2018. Click: A dataset of clinical case reports for machine reading comprehension. arXiv preprint arXiv:1803.09720.  
Wilson L Taylor. 1953. "cloze procedure": A new tool for measuring readability. Journalism quarterly, 30(4):415-433.  
Nandan Thakur, Nils Reimers, Andreas Rückle, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663.  
Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. 2016. Newsqa: A machine comprehension dataset. arXiv preprint arXiv:1611.09830.  
Bianca Zadrozny. 2004. Learning and evaluating classifiers under sample selection bias. In Proceedings of the twenty-first international conference on Machine learning, page 114.

# Footnotes:

Page 0: *This work was done when the first author was an intern at Google Research. 
Page 1: <sup>1</sup>The only exception is the COLIEE dataset which primarily contains boolean (yes/no) answers so we instead use oracle passages to compute Acc@100 2We do not consider the other common reader metric of exact-match to reduce the occurrences of minor dataset annotation guidelines leading to a 0 score for a reasonable answer. 
Page 2: 3We use cloze augmentation for training reader models because some target datasets contain cloze-style questions, keeping the question distribution consistent across different experimental setups. We do not perform this augmentation for retrievers because we observed a performance drop. 
Page 7: 4https://spacy.io/ 
