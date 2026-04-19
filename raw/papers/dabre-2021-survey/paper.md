A Brief Survey of Multilingual Neural Machine Translation
=========================================================

Raj Dabre  
NICT  
Kyoto, Japan  
raj.dabre@nict.go.jp  
\&Chenhui Chu*  
Institute for Datability Science  
Osaka University  
Osaka, Japan.  
chu@ids.osaka-u.ac.jp  
\&Anoop Kunchukuttan*  
Microsoft AI \& Research,  
Hyderabad, India.  
ankunchu@microsoft.com  
equal contribution

###### Abstract

We present a survey on multilingual neural machine translation (MNMT), which has gained a lot of traction in the recent years. MNMT has been useful in improving translation quality as a result of knowledge transfer. MNMT is more promising and interesting than its statistical machine translation counterpart because end-to-end modeling and distributed representations open new avenues. Many approaches have been proposed in order to exploit multilingual parallel corpora for improving translation quality. However, the lack of a comprehensive survey makes it difficult to determine which approaches are promising and hence deserve further exploration. In this paper, we present an in-depth survey of existing literature on MNMT. We categorize various approaches based on the resource scenarios as well as underlying modeling principles. We hope this paper will serve as a starting point for researchers and engineers interested in MNMT.

1 Introduction
--------------

Neural machine translation (NMT) *Cho et al. ([2014](#bib.bib20 "")); Sutskever et al. ([2014](#bib.bib88 "")); Bahdanau et al. ([2015](#bib.bib5 ""))* has become the dominant paradigm for MT in academic research as well as commercial use *Wu et al. ([2016](#bib.bib104 ""))*. NMT has shown state-of-the-art performance for many language pairs *Bojar et al. ([2017](#bib.bib11 ""), [2018](#bib.bib10 ""))*. Its success can be mainly attributed to the use of distributed representations of language, enabling end-to-end training of an MT system. Unlike statistical machine translation (SMT) systems *Koehn et al. ([2007](#bib.bib53 ""))*, separate lossy components like word aligners, translation rule extractors and other feature extractors are not required.
The dominant NMT approach is the Embed - Encode - Attend - Decode paradigm. Recurrent neural network (RNN) *Bahdanau et al. ([2015](#bib.bib5 ""))*, convolutional neural network (CNN) *Gehring et al. ([2017](#bib.bib38 ""))* and self-attention *Vaswani et al. ([2017](#bib.bib95 ""))* architectures are popular approaches based on this paradigm. For a more detailed exposition of NMT, we refer readers to some prominent tutorials *Neubig ([2017](#bib.bib71 "")); Koehn ([2017](#bib.bib52 ""))*.

While initial research on NMT started with building translation systems between two languages, researchers discovered that the NMT framework can naturally incorporate multiple languages. Hence, there has been a massive increase in work on MT systems that involve more than two languages *Dong et al. ([2015](#bib.bib30 "")); Firat et al. ([2016a](#bib.bib35 "")); Zoph and Knight ([2016](#bib.bib108 "")); Cheng et al. ([2017](#bib.bib19 "")); Johnson et al. ([2017](#bib.bib48 "")); Chen et al. ([2017](#bib.bib17 ""), [2018b](#bib.bib18 "")); Neubig and Hu ([2018](#bib.bib72 ""))* etc. We refer to NMT systems handling translation between more than one language pair as multilingual NMT (MNMT) systems. The ultimate goal MNMT research is to develop one model for translation between all possible languages by effective use of available linguistic resources.

<img src='x1.png' alt='Refer to caption' title='' width='484' height='210' />

*Figure 1: MNMT research categorized according to resource scenarios and underlying modeling principles.*

MNMT systems are desirable because training models with data from many language pairs might help acquire knowledge from multiple sources *Zoph and Knight ([2016](#bib.bib108 ""))*. Moreover, MNMT systems tend to generalize better due to exposure to diverse languages, leading to improved translation quality. This particular phenomenon is known as knowledge transfer *Pan and Yang ([2010](#bib.bib77 ""))*.
Knowledge transfer has been strongly observed for translation between low-resource languages, which have scarce parallel corpora or other linguistic resources but have benefited from data in other languages *Zoph et al. ([2016](#bib.bib109 ""))*. In addition, MNMT systems will be compact, because a single model handles translations for multiple languages *Johnson et al. ([2017](#bib.bib48 ""))*. This can reduce the deployment footprint, which is crucial for constrained environment like mobile phones or IoT devices. It can also simplify the large-scale deployment of MT systems.
Most importantly, we believe that the biggest benefit of doing MNMT research is getting better insights into and answers to an important question in natural language processing: how do we build distributed representations such that similar text across languages have similar representations?

There are multiple MNMT scenarios based on available resources and studies have been conducted for the following scenarios (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ A Brief Survey of Multilingual Neural Machine Translation")111Please see the supplementary material for papers related to each category.):

Multiway Translation. The goal is constructing a single NMT system for one-to-many *Dong et al. ([2015](#bib.bib30 ""))*, many-to-one *Lee et al. ([2017](#bib.bib62 ""))* or many-to-many *Firat et al. ([2016a](#bib.bib35 ""))* translation using parallel corpora for more than one language pair.

Low or Zero-Resource Translation. For most of the language pairs in the world, there are small or no parallel corpora, and three main directions have been studied for this scenario. Transfer learning: Transferring translation knowledge from a high-resource language pair to improve the translation of a low-resource language pair *Zoph et al. ([2016](#bib.bib109 ""))*. Pivot translation: Using a high-resource language (usually English) as a pivot to translate between a language pair *Firat et al. ([2016a](#bib.bib35 ""))*. Zero-shot translation: Translating between language pairs without parallel corpora *Johnson et al. ([2017](#bib.bib48 ""))*.

Multi-Source Translation. Documents that have been translated into more than one language might, in the future, be required to be translated into another language. In this scenario, existing multilingual redundancy in the source side can be exploited for multi-source translation *Zoph and Knight ([2016](#bib.bib108 ""))*.

Given these benefits, scenarios and the tremendous increase in the work on MNMT in recent years, we undertake this survey paper on MNMT to systematically organize the work in this area. To the best of our knowledge, no such comprehensive survey on MNMT exists. Our goal is to shed light on various MNMT scenarios, fundamental questions in MNMT, basic principles, architectures, and datasets of MNMT systems.
The remainder of this paper is structured as follows:
We present a systematic categorization of different approaches to MNMT in each of the above mentioned scenarios to help understand the array of design choices available while building MNMT systems (Sections [2](#S2 "2 Multiway NMT ‣ A Brief Survey of Multilingual Neural Machine Translation"), [3](#S3 "3 Low or Zero-Resource MNMT ‣ A Brief Survey of Multilingual Neural Machine Translation"), and [4](#S4 "4 Multi-Source NMT ‣ A Brief Survey of Multilingual Neural Machine Translation")). We put the work in MNMT into a historical perspective with respect to multilingual MT in older MT paradigms (Section [5](#S5 "5 Multilingualism in Older Paradigms ‣ A Brief Survey of Multilingual Neural Machine Translation")). We also describe popular multilingual datasets and the shared tasks that focus on multilingualism (Section [6](#S6 "6 Datasets and Resources ‣ A Brief Survey of Multilingual Neural Machine Translation")). In addition, we compare MNMT with domain adaptation for NMT, which tackles the problem of improving low-resource in-domain translation (Section [7](#S7 "7 Connections with Domain Adaptation ‣ A Brief Survey of Multilingual Neural Machine Translation")).
Finally, we share our opinions on future research directions in MNMT (Section [8](#S8 "8 Future Research Directions ‣ A Brief Survey of Multilingual Neural Machine Translation")) and conclude this paper (Section [9](#S9 "9 Conclusion ‣ A Brief Survey of Multilingual Neural Machine Translation")).

2 Multiway NMT
--------------

The goal is learning a single model for $l$ language pairs $(s_{i},t_{i})\in\mathbf{L}$ $(i\=1\textrm{ to }l)$, where $\mathbf{L}\subset S\times T$, and $S,T$ are sets of source and target languages respectively. $S$ and $T$ need not be mutually exclusive. Parallel corpora are available for these $l$ language pairs. One-many, many-one and many-many NMT models have been explored in this framework.
Multiway translation systems follow the standard paradigm in popular NMT systems. However, this architecture is adapted to support multiple languages. The wide ranges of possible architectural choices is exemplified by two highly contrasting prototypical approaches.

### 2.1 Prototypical Approaches

Complete Sharing. *Johnson et al. ([2017](#bib.bib48 ""))* proposed a highly compact model where all languages share the same embeddings, encoder, decoder, and attention mechanism. A common vocabulary, typically subword-level like byte pair encoding (BPE) *Sennrich et al. ([2016b](#bib.bib84 ""))*, is defined across all languages. The input sequence includes a special token (called the language tag) to indicate the target language. This enables the decoder to correctly generate the target language, though all target languages share the same decoder parameters. The model has minimal parameter size as all languages share the same parameters; and achieves comparable/better results w.r.t. bilingual systems. But, a massively multilingual system can run into capacity bottlenecks *Aharoni et al. ([2019](#bib.bib1 ""))*. This is a black-box model, which can use an off-the-shelf NMT system to train a multilingual system. *Ha et al. ([2016](#bib.bib43 ""))* proposed a similar model, but they maintained different vocabularies for each language.

This architecture is particularly useful for related languages, because they have high degree of lexical and syntactic similarity *Sachan and Neubig ([2018](#bib.bib81 ""))*. Lexical similarity can be further utilized by (a) representing all languages in a common script using script conversion *Dabre et al. ([2018](#bib.bib27 "")); Lee et al. ([2017](#bib.bib62 ""))* or transliteration (*Nakov and Ng ([2009](#bib.bib70 ""))* for multilingual SMT), (b) using a common subword-vocabulary across all languages e.g. character *Lee et al. ([2017](#bib.bib62 ""))* and BPE *Nguyen and Chiang ([2017](#bib.bib73 ""))*, (c) representing words by both character encoding and a latent embedding space shared
by all languages *Wang et al. ([2019](#bib.bib100 ""))*.

*Pinnis et al. ([2018](#bib.bib78 ""))* and *Lakew et al. ([2018a](#bib.bib59 ""))* have compared RNN, CNN and the self-attention based architectures for MNMT. They show that self-attention based architectures outperform the other architectures in many cases.

Minimal Sharing.
On the other hand, *Firat et al. ([2016a](#bib.bib35 ""))* proposed a model comprised of separate embeddings, encoders and decoders for each language. By sharing attention across languages, they show improvements over bilingual models. However, this model has a large number of parameters. Nevertheless, the number of parameters only grows linearly with the number of languages, while it grows quadratically for bilingual systems spanning all the language pairs in the multiway system.

### 2.2 Controlling Parameter Sharing

In between the extremities of parameter sharing exemplified by the above mentioned models, lies an array of choices. The degree of parameter sharing depends on the divergence between the languages involved *Sachan and Neubig ([2018](#bib.bib81 ""))* and can be controlled at various layers of the MNMT system. Sharing encoders among multiple languages is very effective and is widely used *Lee et al. ([2017](#bib.bib62 "")); Sachan and Neubig ([2018](#bib.bib81 ""))*. *Blackwood et al. ([2018](#bib.bib9 ""))* explored target language, source language and pair specific attention parameters. They showed that target language specific attention performs better than other attention sharing configurations. For self-attention based NMT models, *Sachan and Neubig ([2018](#bib.bib81 ""))* explored various parameter sharing strategies. They showed that sharing the decoder self-attention and encoder-decoder inter-attention parameters is useful for linguistically dissimilar languages. *Zaremoodi et al. ([2018](#bib.bib105 ""))* further proposed a routing network to dynamically control parameter sharing learned from the data. Designing the right sharing strategy is important to maintaining a balance between model compactness and translation accuracy.

Dynamic Parameter or Representation Generation. Instead of defining the parameter sharing protocol a priori, *Platanios et al. ([2018](#bib.bib79 ""))* learned the degree of parameter sharing from the data. This is achieved by defining the language specific model parameters as a function of global parameters and language embeddings. This approach also reduces the number of language specific parameters (only language embeddings), while still allowing each language to have its own unique parameters for different network layers. In fact, the number of parameters is only a small multiple of the compact model (the multiplication factor accounts for the language embedding size) *Johnson et al. ([2017](#bib.bib48 ""))*, but the language embeddings can directly impact the model parameters instead of the weak influence that language tags have.

Universal Encoder Representation. Ideally, multiway systems should generate encoder representations that are language agnostic. However, the attention mechanism sees a variable number of encoder representations depending on the sentence length (this could vary for translations of the same sentence). To overcome this, an attention bridge network generates a fixed number of contextual representations that are input to the attention network *Lu et al. ([2018](#bib.bib63 "")); Vázquez et al. ([2018](#bib.bib96 ""))*. *Murthy et al. ([2018](#bib.bib68 ""))* pointed out that the contextualized embeddings are word order dependent, hence not language agnostic.

Multiple Target Languages. This is a challenging scenario because parameter sharing has to be balanced with the capability to generate sentences in each target language. *Blackwood et al. ([2018](#bib.bib9 ""))* added the language tag to the beginning as well as end of sequence to avoid its attenuation in a left-to-right encoder. *Wang et al. ([2018](#bib.bib101 ""))* explored multiple methods for supporting target languages: (a) target language tag at beginning of the decoder, (b) target language dependent positional embeddings, and (c) divide hidden units of each decoder layer into shared and language-dependent ones. Each of these methods provide gains over *Johnson et al. ([2017](#bib.bib48 ""))*, and combining all gave the best results.

### 2.3 Training Protocols

Joint Training. All the available languages pairs are trained jointly to minimize the mean negative log-likelihood for each language pair. As some language pairs would have more data than other languages, the model may be biased. To avoid this, sentence pairs from different language pairs are sampled to maintain a healthy balance. Mini-batches can be comprised of a mix of samples from different language pairs *Johnson et al. ([2017](#bib.bib48 ""))* or the training schedule can cycle through mini-batches consisting of a language pair only *Firat et al. ([2016a](#bib.bib35 ""))*.
For architectures with language specific layers, the latter approach is convenient to implement.

Knowledge Distillation. In this approach suggested by *Tan et al. ([2019](#bib.bib89 ""))*, bilingual models are first trained for all language pairs involved. These bilingual models are used as teacher models to train a single student model for all language pairs. The student model is trained using a linear interpolation of the standard likelihood loss as well as distillation loss that captures the distance between the output distributions of the student and teacher models. The distillation loss is applied for a language pair only if the teacher model shows better translation accuracy than the student model on the validation set. This approach shows better results than joint training of a black-box model, but training time increases significantly because bilingual models also have to be trained.

3 Low or Zero-Resource MNMT
----------------------------

An important motivation for MNMT is to improve or support translation for language pairs with scarce or no parallel corpora, by utilizing training data from high-resource language pairs. In this section, we will discuss the MNMT approaches that specifically address the low or zero-resource scenario.

### 3.1 Transfer Learning

Transfer learning *Pan and Yang ([2010](#bib.bib77 ""))* has been widely explored to address low-resource translation, where knowledge learned from a high-resource language pair is used to improve the NMT performance on a low-resource pair.

Training. Most studies have explored the following setting: the high-resource and low-resource language pairs share the same target language. *Zoph et al. ([2016](#bib.bib109 ""))* first showed that transfer learning can benefit low-resource language pairs. First, they trained a parent model on a high-resource language pair. The child model is initialized with the parent’s parameters wherever possible and trained on the small parallel corpus for the low-resource pair. This process is known as fine-tuning. They also studied the effect of fine-tuning only a subset of the child model’s parameters (source and target embeddings, RNN layers and attention). The initialization has a strong regularization effect in training the child model. *Gu et al. ([2018b](#bib.bib41 ""))* used the model agnostic meta learning (MAML) framework *Finn et al. ([2017](#bib.bib34 ""))* to learn appropriate parameter initialization from the parent pair(s) by taking the child pair into consideration. Instead of fine-tuning, both language pairs can also be jointly trained *Gu et al. ([2018a](#bib.bib40 ""))*.

Language Relatedness. *Zoph et al. ([2016](#bib.bib109 ""))* and *Dabre et al. ([2017b](#bib.bib29 ""))* have empirically shown that language relatedness between the parent and child source languages has a big impact on the possible gains from transfer learning. *Kocmi and Bojar ([2018](#bib.bib50 ""))* showed that transfer learning improves low-resource language translation, even when neither the source nor the target languages are shared between the resource-rich and poor language pairs. Further investigation is needed to understand the gains in translation quality in this scenario. *Neubig and Hu ([2018](#bib.bib72 ""))* used language relatedness to prevent overfitting when rapidly adapting pre-trained MNMT model for low-resource scenarios. *Chaudhary et al. ([2019](#bib.bib15 ""))* used this approach to translate 1,095 languages to English.

Lexical Transfer. *Zoph et al. ([2016](#bib.bib109 ""))* randomly initialized the word embeddings of the child source language, because those could not be transferred from the parent. *Gu et al. ([2018a](#bib.bib40 ""))* improved on this simple initialization by mapping pre-trained monolingual embeddings of the parent and child sources to a common vector space. On the other hand, *Nguyen and Chiang ([2017](#bib.bib73 ""))* utilized the lexical similarity between related source languages using a small subword vocabulary. *Lakew et al. ([2018b](#bib.bib60 ""))* dynamically updated the vocabulary of the parent model with the low-resource language pair before transferring parameters.

Syntactic Transfer. *Gu et al. ([2018a](#bib.bib40 ""))* proposed to encourage better transfer of contextual representations from parents using a mixture of language experts network. *Murthy et al. ([2018](#bib.bib68 ""))* showed that reducing the word order divergence between source languages via pre-ordering is beneficial in extremely low-resource scenarios.

### 3.2 Pivoting

Zero-resource NMT was first explored by *Firat et al. ([2016a](#bib.bib35 ""))*, where a multiway NMT model was used to translate from Spanish to French using English as a pivot language. This pivoting was done either at run time or during pre-training.

Run-Time Pivoting. *Firat et al. ([2016a](#bib.bib35 ""))* involved a pipeline through paths in the multiway model, which first translates from French to English and then from English to Spanish. They also experimented with using the intermediate English translation as an additional source for the second stage.

Pivoting during Pre-Training. *Firat et al. ([2016b](#bib.bib36 ""))* used the MNMT model to first translate the Spanish side of the training corpus to English which in turn is translated into French. This gives a pseudo-parallel French-Spanish corpus where the source is synthetic and the target is original. The MNMT model is fine tuned on this synthetic data and this enables direct French to Spanish translation. *Firat et al. ([2016b](#bib.bib36 ""))* also showed that a small clean parallel corpus between French and Spanish can be used for fine tuning and can have the same effect as a pseduo-parallel corpus which is two orders of magnitude larger.
Pivoting models can be improved if they are jointly trained as shown by *Cheng et al. ([2017](#bib.bib19 ""))*. Joint training was achieved by either forcing the pivot language’s embeddings to be similar or maximizing the likelihood
of the cascaded model on a small source-target parallel
corpus. *Chen et al. ([2017](#bib.bib17 ""))* proposed teacher-student learning for pivoting where they first trained a pivot-target NMT model and used it as a teacher to guide the behaviour of a source-target NMT model.

### 3.3 Zero-Shot

The approaches proposed so far involve pivoting or synthetic corpus generation, which is a slow process due to its two-step nature. It is more interesting, and challenging, to enable translation between a zero-resource pair without explicitly involving a pivot language during decoding or for generating pseudo-parallel corpora.
This scenario is known as zero-shot NMT.
Zero-shot NMT also requires a pivot language but it is only used during training without the need to generate pseudo-parallel corpora.

Training. Zero-shot NMT was first demonstrated by *Johnson et al. ([2017](#bib.bib48 ""))*. However, this zero-shot translation method is inferior to pivoting. They showed that the context vectors (from attention) for unseen language pairs differ from the seen language pairs, possibly explaining the degradation in translation quality. *Lakew et al. ([2017](#bib.bib61 ""))* tried to overcome this limitation by augmenting the training data with the pseudo-parallel unseen pairs generated by iterative application of the same zero-shot translation. *Arivazhagan et al. ([2018](#bib.bib2 ""))* included explicit language invariance losses in the optimization function to encourage parallel sentences to have the same representation. Reinforcement learning for zero-shot learning was explored by *Sestorain et al. ([2018](#bib.bib85 ""))* where the dual learning framework was combined with rewards from language models.

Corpus Size. Work on translation for Indian languages showed that zero-shot works well only when the training corpora are extremely large *Mattoni et al. ([2017](#bib.bib64 ""))*. As the corpora for most Indian languages contain fewer than 100k sentences, the zero-shot approach is rather infeasible despite linguistic similarity. *Lakew et al. ([2017](#bib.bib61 ""))* confirmed this in the case of European languages where small training corpora were used. *Mattoni et al. ([2017](#bib.bib64 ""))* also showed that zero-shot translation works well only when the training corpora are large, while *Aharoni et al. ([2019](#bib.bib1 ""))* show that massively multilingual models are beneficial for zeroshot translation.

Language Control. Zero-shot NMT tends to translate into the wrong language at times and *Ha et al. ([2017](#bib.bib44 ""))* proposed to filter the output of the softmax so as to force the model to translate into the desired language.

4 Multi-Source NMT
-------------------

If the same source sentence is available in multiple languages then these sentences can be used together to improve the translation into the target language. This technique is known as multi-source MT *Och and Ney ([2001](#bib.bib76 ""))*.
Approaches for multi-source NMT can be extremely useful for creating N-lingual (N $>$ 3) corpora such as Europarl *Koehn ([2005](#bib.bib51 ""))* and UN *Ziemski et al. ([2016b](#bib.bib107 ""))*.
The underlying principle is to leverage redundancy in terms of source side linguistic phenomena expressed in multiple languages.

Multi-Source Available. Most studies assume that the same sentence is available in multiple languages. *Zoph and Knight ([2016](#bib.bib108 ""))* showed that a multi-source NMT model using separate encoders and attention networks for each source language outperforms single source models.
A simpler approach concatenated multiple source sentences and fed them to a standard NMT model *Dabre et al. ([2017a](#bib.bib26 ""))*, with performance comparable to *Zoph and Knight ([2016](#bib.bib108 ""))*.
Interestingly, this model could automatically identify the boundaries between different source languages and simplify the training process for multi-source NMT. *Dabre et al. ([2017a](#bib.bib26 ""))* also showed that it is better to use linguistically similar source languages, especially in low-resource scenarios. Ensembling of individual source-target models is another beneficial approach, for which *Garmash and Monz ([2016](#bib.bib37 ""))* proposed several methods with different degrees of parameterization.

Missing Source Sentences. There can be missing source sentences in multi-source corpora. *Nishimura et al. ([2018b](#bib.bib75 ""))* extended *Zoph and Knight ([2016](#bib.bib108 ""))* by representing each “missing” source language with a dummy token. *Choi et al. ([2018](#bib.bib21 ""))* and *Nishimura et al. ([2018a](#bib.bib74 ""))* further proposed to use MT generated synthetic sentences, instead of a dummy token for the missing source languages.

Post-Editing. Instead of having a translator translate from scratch, multi-source NMT can be used to generate high quality translations. The translations can then be post-edited, a process that is less labor intensive and cheaper compared to translating from scratch. Multi-source NMT has been used for post-editing where the translated sentence is used as an additional source, leading to improvements *Chatterjee et al. ([2017](#bib.bib14 ""))*.

5 Multilingualism in Older Paradigms
------------------------------------

One of the long term goals of the MT community is the development of architectures that can handle more than two languages.

#### RBMT.

To this end, rule-based systems (RBMT) using an interlingua were explored widely in the past. The interlingua is a symbolic semantic, language-independent representation for natural language text *Sgall and Panevová ([1987](#bib.bib86 ""))*. Two popular interlinguas are UNL *Uchida ([1996](#bib.bib93 ""))* and AMR *Banarescu et al. ([2013](#bib.bib6 ""))* Different interlinguas have been proposed in various systems like KANT *E. H. Nyberg and Carbonell ([1997](#bib.bib33 ""))*, UNL, UNITRAN *Dorr ([1987](#bib.bib31 ""))* and DLT *Witkam ([2006](#bib.bib102 ""))*. Language specific analyzers converted language input to interlingua, while language specific decoders converted the interlingua into another language. To achieve an unambiguous semantic representation, a lot of linguistic analysis had to be performed and many linguistic resources were required. Hence, in practice, most interlingua systems were limited to research systems or translation in specific domains and could not scale to many languages. Over time most MT research focused on building bilingual systems.

#### SMT.

Phrase-based SMT (PBSMT) systems *Koehn et al. ([2003](#bib.bib55 ""))*, a very successful MT paradigm, were also bilingual for the most part. Compared to RBMT, PBSMT requires less linguistic resources and instead requires parallel corpora. However, like RBMT, they work with symbolic, discrete representations making multilingual representation difficult. Moreover, the central unit in PBSMT is the phrase, an ordered sequence of words (not in the linguistic sense). Given its arbitrary structure, it is not clear how to build a common symbolic representation for phrases across languages. Nevertheless, some shallow forms of multilingualism have been explored in the context of: (a) pivot-based SMT, (b) multi-source PBSMT, and (c) SMT involving related languages.

Pivoting. Popular solutions are: chaining source-pivot and pivot-target systems at decoding *Utiyama and Isahara ([2007](#bib.bib94 ""))*, training a source-target system using synthetic data generated using target-pivot and pivot-source systems *Gispert and Marino ([2006](#bib.bib39 ""))*, and phrase-table triangulation pivoting source-pivot and pivot-target phrase tables *Utiyama and Isahara ([2007](#bib.bib94 "")); Wu and Wang ([2007](#bib.bib103 ""))*.

Multi-source. Typical approaches are: re-ranking outputs from independent source-target systems *Och and Ney ([2001](#bib.bib76 ""))*, composing a new output from independent source-target outputs *Matusov et al. ([2006](#bib.bib65 ""))*, and translating a combined input representation of multiple sources using lattice networks over multiple phrase tables *Schroeder et al. ([2009](#bib.bib82 ""))*.

Related languages. For multilingual translation with multiple related source languages, the typical approaches involved script unification by mapping to a common script such as Devanagari *Banerjee et al. ([2018](#bib.bib7 ""))* or transliteration *Nakov and Ng ([2009](#bib.bib70 ""))*. Lexical similarity was utilized using subword-level translation models *Vilar et al. ([2007](#bib.bib97 "")); Tiedemann ([2012a](#bib.bib91 "")); Kunchukuttan and Bhattacharyya ([2016](#bib.bib56 ""), [2017](#bib.bib57 ""))*. Combining subword-level representation and pivoting for translation among related languages has been explored *(Henríquez et al., [2011](#bib.bib45 ""); Tiedemann, [2012a](#bib.bib91 ""); Kunchukuttan et al., [2017](#bib.bib58 ""))*. Most of the above mentioned multilingual systems involved either decoding-time operations, chaining black-box systems or composing new phrase-tables from existing ones.

#### Comparison with MNMT.

While symbolic representations constrain a unified multilingual representation, distributed universal language representation using real-valued vector spaces makes multilingualism easier to implement in NMT. As no language specific feature engineering is required for NMT, making it possible to scale to multiple languages. Neural networks provide flexibility in experimenting with a wide variety of architectures, while advances in optimization techniques and availability of deep learning toolkits make prototyping faster.

6 Datasets and Resources
------------------------

MNMT requires parallel corpora in similar domains across multiple languages.

Multiway. Commonly used publicly available multilingual parallel corpora are the TED corpus *Mauro et al. ([2012](#bib.bib66 ""))*, UN Corpus *Ziemski et al. ([2016a](#bib.bib106 ""))* and those from the European Union like Europarl, JRC-Aquis, DGT-Aquis, DGT-TM, ECDC-TM, EAC-TM *Steinberger et al. ([2014](#bib.bib87 ""))*. While these sources are primarily comprised of European languages, parallel corpora for some Asian languages is accessible through the WAT shared task *Nakazawa et al. ([2018](#bib.bib69 ""))*. Only small amount of parallel corpora are available for many languages, primarily from movie subtitles and software localization strings *Tiedemann ([2012b](#bib.bib92 ""))*.

Low or Zero-Resource. For low or zero-resource NMT translation tasks, good test sets are required for evaluating translation quality. The above mentioned multilingual parallel corpora can be a source for such test sets. In addition, there are other small parallel datasets like the FLORES dataset for English-{Nepali,Sinhala} *Guzmán et al. ([2019](#bib.bib42 ""))*, the XNLI test set spanning 15 languages *Conneau et al. ([2018b](#bib.bib25 ""))* and the Indic parallel corpus *Birch et al. ([2011](#bib.bib8 ""))*. The WMT shared tasks *Bojar et al. ([2018](#bib.bib10 ""))* also provide test sets for some low-resource language pairs.

Multi-Source. The corpora for multi-source NMT have to be aligned across languages. Multi-source corpora can be extracted from some of the above mentioned sources. The following are widely used for evaluation in the literature: Europarl *Koehn ([2005](#bib.bib51 ""))*, TED *Tiedemann ([2012b](#bib.bib92 ""))*, UN *Ziemski et al. ([2016b](#bib.bib107 ""))*. The Indian Language Corpora Initiative (ILCI) corpus *Jha ([2010](#bib.bib47 ""))* is a 11-way parallel corpus of Indian languages along with English. The Asian Language Treebank *Thu et al. ([2016](#bib.bib90 ""))* is a 9-way
parallel corpus of South-East Asian languages along with English, Japanese and Bengali. The MMCR4NLP project *Dabre and Kurohashi ([2017](#bib.bib28 ""))* compiles language family grouped multi-source corpora and provides standard splits.

Shared Tasks. Recently, shared tasks with a focus on multilingual translation have been conducted at IWSLT *Cettolo et al. ([2017](#bib.bib12 ""))*, WAT *Nakazawa et al. ([2018](#bib.bib69 ""))* and WMT *Bojar et al. ([2018](#bib.bib10 ""))*; so common benchmarks are available.

7 Connections with Domain Adaptation
------------------------------------

High quality parallel corpora are limited to specific domains.
Both, vanilla SMT and NMT perform poorly for domain specific translation in low-resource scenarios *Duh et al. ([2013](#bib.bib32 "")); Koehn and Knowles ([2017](#bib.bib54 ""))*.
Leveraging out-of-domain parallel corpora and in-domain monolingual corpora for in-domain translation is known as domain adaptation for MT *Chu and Wang ([2018](#bib.bib23 ""))*.

As we can treat each domain as a language, there are many similarities and common approaches between MNMT and domain adaptation for NMT.
Therefore, similar to MNMT, when using out-of-domain parallel corpora for domain adaptation, multi-domain NMT and transfer learning based approaches *Chu et al. ([2017](#bib.bib22 ""))* have been proposed for domain adaptation.
When using in-domain monolingual corpora, a typical way of doing domain adaptation is generating a pseduo-parallel corpus by back-translating target in-domain monolingual corpora *Sennrich et al. ([2016a](#bib.bib83 ""))*, which is similar to the pseduo-parallel corpus generation in MNMT *Firat et al. ([2016b](#bib.bib36 ""))*.

There are also many differences between MNMT and domain adaptation for NMT. While pivoting is a popular approach for MNMT *Cheng et al. ([2017](#bib.bib19 ""))*, it is unsuitable for domain adaptation.
As there are always vocabulary overlaps between different domains, there are no zero-shot translation *Johnson et al. ([2017](#bib.bib48 ""))* settings in domain adaptation. In addition, it not uncommon to write domain specific sentences in different styles and so
multi-source approaches *Zoph and Knight ([2016](#bib.bib108 ""))* are not applicable either.
On the other hand, data selection approaches in domain adaptation that select out-of-domain sentences which are similar to in-domain sentences *Wang et al. ([2017a](#bib.bib98 ""))* have not been applied to MNMT. In addition, instance weighting approaches *Wang et al. ([2017b](#bib.bib99 ""))* that interpolate in-domain and out-of-domain models have not been studied for MNMT. However, with the development of cross-lingual sentence embeddings, data selection and instance weighting approaches might be applicable for MNMT in the near future.

8 Future Research Directions
----------------------------

While exciting advances have been made in MNMT in recent years, there are still many interesting directions for exploration.

Language Agnostic Representation Learning. A core question that needs further investigation is: how do we build encoder and decoder representations that are language agnostic? Particularly, the questions of word-order divergence between the source languages and variable length encoder representations have received little attention.

Multiple Target Language MNMT. Most current efforts address multiple source languages.
Multiway systems for multiple low-resource target languages need more attention. The right balance between sharing representations vs. maintaining the distinctiveness of the target language for generation needs exploring.

Explore Pre-training Models. Pre-training embeddings, encoders and decoders have been shown to be useful for NMT *Ramachandran et al. ([2017](#bib.bib80 ""))*. How pre-training can be incorporated into different MNMT architectures, is an important as well. Recent advances in cross-lingual word *Klementiev et al. ([2012](#bib.bib49 "")); Mikolov et al. ([2013](#bib.bib67 "")); Chandar et al. ([2014](#bib.bib13 "")); Artetxe et al. ([2016](#bib.bib3 "")); Conneau et al. ([2018a](#bib.bib24 "")); Jawanpuria et al. ([2019](#bib.bib46 ""))* and sentence embeddings *Conneau et al. ([2018b](#bib.bib25 "")); Chen et al. ([2018a](#bib.bib16 "")); Artetxe and Schwenk ([2018](#bib.bib4 ""))* could provide directions for this line of investigation.

Related Languages, Language Registers and Dialects. Translation involving related languages, language registers and dialects can be further explored given the importance of this use case.

Code-Mixed Language. Addressing intra-sentence multilingualism i.e. code mixed input and output, creoles and pidgins is an interesting research direction. The compact MNMT models can handle code-mixed input, but code-mixed output remains an open problem *Johnson et al. ([2017](#bib.bib48 ""))*.

Multilingual and Multi-Domain NMT. Jointly tackling multilingual and multi-domain translation is an interesting direction with many practical use cases. When extending an NMT system to a new language, the parallel corpus in the domain of interest may not be available. Transfer learning in this case has to span languages and domains.

9 Conclusion
------------

MNMT has made rapid progress in the recent past. In this survey, we have covered literature pertaining to the major scenarios we identified for multilingual NMT: multiway, low or zero-resource (transfer learning, pivoting, and zero-shot approaches) and multi-source translation. We have systematically compiled the principal design approaches and their variants, central MNMT issues and their proposed solutions along with their strengths and weaknesses. We have put MNMT in a historical perspective w.r.t work on multilingual RBMT and SMT systems. We suggest promising and important directions for future work. We hope that this survey paper could significantly promote and accelerate MNMT research.

References
----------

* Aharoni et al. (2019)Roee Aharoni, Melvin Johnson, and Orhan Firat. 2019.Massively Multilingual Neural Machine Translation.In *NAACL (to appear)*.
* Arivazhagan et al. (2018)Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Roee Aharoni, Melvin Johnson, and
Wolfgang Macherey. 2018.The missing ingredient in zero-shot neural machine translation.
* Artetxe et al. (2016)Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2016.Learning principled bilingual mappings of word embeddings while
preserving monolingual invariance.In *Proceedings of the Conference on Empirical Methods in
Natural Language Processing*, pages 2289–2294.
* Artetxe and Schwenk (2018)Mikel Artetxe and Holger Schwenk. 2018.[Massively multilingual
sentence embeddings for zero-shot cross-lingual transfer and beyond](http://arxiv.org/abs/1812.10464 "").*CoRR*, abs/1812.10464.
* Bahdanau et al. (2015)Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015.Neural machine translation by jointly learning to align and
translate.In *In Proceedings of the 3rd International Conference on
Learning Representations (ICLR 2015)*, San Diego, USA. International
Conference on Learning Representations.
* Banarescu et al. (2013)Laura Banarescu, Claire Bonial, Shu Cai, Madalina Georgescu, Kira Griffitt, Ulf
Hermjakob, Kevin Knight, Philipp Koehn, Martha Palmer, and Nathan Schneider.
2013.[Abstract meaning
representation for sembanking](http://www.aclweb.org/anthology/W13-2322 "").In *Proceedings of the 7th Linguistic Annotation Workshop and
Interoperability with Discourse*, pages 178–186, Sofia, Bulgaria.
Association for Computational Linguistics.
* Banerjee et al. (2018)Tamali Banerjee, Anoop Kunchukuttan, and Pushpak Bhattacharyya. 2018.Multilingual Indian Language Translation System at WAT 2018:
Many-to-one Phrase-based SMT.In *5th Workshop on Asian Language Translation*.
* Birch et al. (2011)Lexi Birch, Chris Callison-Burch, Miles Osborne, and Matt Post. 2011.The indic multi-parallel corpus.<http://homepages.inf.ed.ac.uk/miles/babel.html>.
* Blackwood et al. (2018)Graeme Blackwood, Miguel Ballesteros, and Todd Ward. 2018.[Multilingual neural
machine translation with task-specific attention](http://aclweb.org/anthology/C18-1263 "").In *Proceedings of the 27th International Conference on
Computational Linguistics*, pages 3112–3122. Association for Computational
Linguistics.
* Bojar et al. (2018)Ondřej Bojar, Christian Federmann, Mark Fishel, Yvette Graham, Barry
Haddow, Philipp Koehn, and Christof Monz. 2018.[Findings of the 2018
conference on machine translation (WMT18)](http://aclweb.org/anthology/W18-6401 "").In *Proceedings of the Third Conference on Machine Translation:
Shared Task Papers*, pages 272–303. Association for Computational
Linguistics.
* Bojar et al. (2017)Ondřej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry
Haddow, Shujian Huang, Matthias Huck, Philipp Koehn, Qun Liu, Varvara
Logacheva, Christof Monz, Matteo Negri, Matt Post, Raphael Rubino, Lucia
Specia, and Marco Turchi. 2017.[Findings of the
2017 conference on machine translation (WMT17)](http://www.aclweb.org/anthology/W17-4717 "").In *Proceedings of the Second Conference on Machine
Translation*, pages 169–214, Copenhagen, Denmark. Association for
Computational Linguistics.
* Cettolo et al. (2017)Mauro Cettolo, Marcello Federico, Luisa Bentivogli, Jan Niehues, Sebastian
Stüker, Katsuhito Sudoh, Koichiro Yoshino, and Christian Federmann. 2017.Overview of the IWSLT 2017 evaluation campaign.In *IWSLT*.
* Chandar et al. (2014)Sarath Chandar, Stanislas Lauly, Hugo Larochelle, Mitesh Khapra, Balaraman
Ravindran, Vikas C Raykar, and Amrita Saha. 2014.An autoencoder approach to learning bilingual word representations.In *Proceedings of the Advances in Neural Information Processing
Systems*, pages 1853–1861.
* Chatterjee et al. (2017)Rajen Chatterjee, M. Amin Farajian, Matteo Negri, Marco Turchi, Ankit
Srivastava, and Santanu Pal. 2017.[Multi-source neural
automatic post-editing: Fbk’s participation in the WMT 2017 ape shared
task](https://doi.org/10.18653/v1/W17-4773 "").In *Proceedings of the Second Conference on Machine
Translation*, pages 630–638. Association for Computational Linguistics.
* Chaudhary et al. (2019)Aditi Chaudhary, Siddharth Dalmia, Junjie Hu, Xinjian Li, Austin Matthews,
Aldrian Obaja Muis, Naoki Otani, Shruti Rijhwani, Zaid Sheikh, Nidhi Vyas,
Xinyi Wang, Jiateng Xie, Ruochen Xu, Chunting Zhou, Peter J. Jansen, Yiming
Yang, Lori Levin, Florian Metze, Teruko Mitamura, David R. Mortensen, Graham
Neubig, Eduard Hovy, Alan W Black, Jaime Carbonell, Graham V. Horwood,
Shabnam Tafreshi, Mona Diab, Efsun S. Kayi, Noura Farra, and Kathleen
McKeown. 2019.[The ARIEL-CMU Systems for
LoReHLT18](http://arxiv.org/abs/1902.08899 "").*CoRR*, abs/1902.08899.
* Chen et al. (2018a)Xilun Chen, Ahmed Hassan Awadallah, Hany Hassan, Wei Wang, and Claire Cardie.
2018a.[Zero-resource multilingual
model transfer: Learning what to share](http://arxiv.org/abs/1810.03552 "").*CoRR*, abs/1810.03552.
* Chen et al. (2017)Yun Chen, Yang Liu, Yong Cheng, and Victor O.K. Li. 2017.[A teacher-student
framework for zero-resource neural machine translation](https://doi.org/10.18653/v1/P17-1176 "").In *Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pages 1925–1935.
Association for Computational Linguistics.
* Chen et al. (2018b)Yun Chen, Yang Liu, and Victor O. K. Li. 2018b.Zero-resource neural machine translation with multi-agent
communication game.In *AAAI*, pages 5086–5093. AAAI Press.
* Cheng et al. (2017)Yong Cheng, Qian Yang, Yang Liu, Maosong Sun, and Wei Xu. 2017.[Joint training for
pivot-based neural machine translation](https://doi.org/10.24963/ijcai.2017/555 "").In *Proceedings of the Twenty-Sixth International Joint
Conference on Artificial Intelligence, IJCAI-17*, pages 3974–3980.
* Cho et al. (2014)KyungHyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014.On the properties of neural machine translation: Encoder-decoder
approaches.In *Eighth Workshop on Syntax, Semantics and Structure in
Statistical Translation*.
* Choi et al. (2018)Gyu Hyeon Choi, Jong Hun Shin, and Young Kil Kim. 2018.[Improving a
multi-source neural machine translation model with corpus extension for
low-resource languages](http://aclweb.org/anthology/L18-1144 "").In *Proceedings of the Eleventh International Conference on
Language Resources and Evaluation (LREC-2018)*. European Language Resource
Association.
* Chu et al. (2017)Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2017.[An empirical comparison
of domain adaptation methods for neural machine translation](https://doi.org/10.18653/v1/P17-2061 "").In *Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers)*, pages 385–391.
Association for Computational Linguistics.
* Chu and Wang (2018)Chenhui Chu and Rui Wang. 2018.[A survey of domain
adaptation for neural machine translation](http://aclweb.org/anthology/C18-1111 "").In *Proceedings of the 27th International Conference on
Computational Linguistics*, pages 1304–1319. Association for Computational
Linguistics.
* Conneau et al. (2018a)Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and
Hervé Jégou. 2018a.Word translation without parallel data.In *Proceedings of the International Conference on Learning
Representations*.URL: <https://github.com/facebookresearch/MUSE>.
* Conneau et al. (2018b)Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R.
Bowman, Holger Schwenk, and Veselin Stoyanov. 2018b.XNLI: Evaluating Cross-lingual Sentence Representations.In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*. Association for Computational Linguistics.
* Dabre et al. (2017a)Raj Dabre, Fabien Cromieres, and Sadao Kurohashi. 2017a.Enabling multi-source neural machine translation by concatenating
source sentences in multiple languages.In *Proceedings of MT Summit XVI, vol.1: Research Track*, pages
96–106.
* Dabre et al. (2018)Raj Dabre, Anoop Kunchukuttan, Atsushi Fujita, and Eiichiro Sumita. 2018.NICT’s participation in WAT 2018: Approaches using
multilingualism and recurrently stacked layers.In *5th Workshop on Asian Language Translation*.
* Dabre and Kurohashi (2017)Raj Dabre and Sadao Kurohashi. 2017.Mmcr4nlp: Multilingual multiway corpora repository for natural
language processing.*arXiv preprint arXiv:1710.01025*.
* Dabre et al. (2017b)Raj Dabre, Tetsuji Nakagawa, and Hideto Kazawa. 2017b.[An empirical study of
language relatedness for transfer learning in neural machine translation](http://aclweb.org/anthology/Y17-1038 "").In *Proceedings of the 31st Pacific Asia Conference on Language,
Information and Computation*, pages 282–286. The National University
(Phillippines).
* Dong et al. (2015)Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. 2015.[Multi-task learning for
multiple language translation](https://doi.org/10.3115/v1/P15-1166 "").In *Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers)*, pages 1723–1732.
Association for Computational Linguistics.
* Dorr (1987)Bonnie J. Dorr. 1987.UNITRAN: An Interlingua Approach to Machine Translation.In *Proceedings of the 6th Conference of the American
Association of Artificial Intelligence*.
* Duh et al. (2013)Kevin Duh, Graham Neubig, Katsuhito Sudoh, and Hajime Tsukada. 2013.[Adaptation data
selection using neural language models: Experiments in machine translation](http://www.aclweb.org/anthology/P13-2119 "").In *Proceedings of the 51st Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers)*, pages 678–683,
Sofia, Bulgaria.
* E. H. Nyberg and Carbonell (1997)T. Mitamura E. H. Nyberg and J. Carbonell. 1997.The KANT Machine Translation System: From R\&D to Initial
Deployment.In *Proceedings of LISA (The Library and Information Services in
Astronomy) Workshop on Integrating Advanced Translation Technology*.
* Finn et al. (2017)Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017.[Model-agnostic
meta-learning for fast adaptation of deep networks](http://proceedings.mlr.press/v70/finn17a.html "").In *Proceedings of the 34th International Conference on Machine
Learning*, volume 70 of *Proceedings of Machine Learning Research*,
pages 1126–1135, International Convention Centre, Sydney, Australia. PMLR.
* Firat et al. (2016a)Orhan Firat, Kyunghyun Cho, and Yoshua Bengio. 2016a.[Multi-way, multilingual
neural machine translation with a shared attention mechanism](https://doi.org/10.18653/v1/N16-1101 "").In *Proceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies*, pages 866–875. Association for Computational Linguistics.
* Firat et al. (2016b)Orhan Firat, Baskaran Sankaran, Yaser Al-Onaizan, Fatos T. Yarman Vural, and
Kyunghyun Cho. 2016b.[Zero-resource
translation with multi-lingual neural machine translation](https://doi.org/10.18653/v1/D16-1026 "").In *Proceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing*, pages 268–277. Association for Computational
Linguistics.
* Garmash and Monz (2016)Ekaterina Garmash and Christof Monz. 2016.[Ensemble learning for
multi-source neural machine translation](http://aclweb.org/anthology/C16-1133 "").In *Proceedings of COLING 2016, the 26th International
Conference on Computational Linguistics: Technical Papers*, pages 1409–1418.
The COLING 2016 Organizing Committee.
* Gehring et al. (2017)Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin.
2017.[Convolutional sequence to sequence learning](http://proceedings.mlr.press/v70/gehring17a.html "").In *Proceedings of the 34th International Conference on Machine
Learning*, volume 70 of *Proceedings of Machine Learning Research*,
pages 1243–1252, International Convention Centre, Sydney, Australia. PMLR.
* Gispert and Marino (2006)Adri‘a De Gispert and Jose B Marino. 2006.Catalan-English statistical machine translation without parallel
corpus: bridging through Spanish.In *In Proc. of 5th International Conference on Language
Resources and Evaluation (LREC)*.
* Gu et al. (2018a)Jiatao Gu, Hany Hassan, Jacob Devlin, and Victor O.K. Li. 2018a.[Universal neural
machine translation for extremely low resource languages](https://doi.org/10.18653/v1/N18-1032 "").In *Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers)*, pages 344–354. Association for
Computational Linguistics.
* Gu et al. (2018b)Jiatao Gu, Yong Wang, Yun Chen, Victor O. K. Li, and Kyunghyun Cho.
2018b.[Meta-learning for
low-resource neural machine translation](http://aclweb.org/anthology/D18-1398 "").In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pages 3622–3631. Association for Computational
Linguistics.
* Guzmán et al. (2019)Francisco Guzmán, Peng-Jen Chen, Myle Ott, Juan Pino, Guillaume Lample,
Philipp Koehn, Vishrav Chaudhary, and Marc’Aurelio Ranzato. 2019.Two New Evaluation Datasets for Low-Resource Machine Translation:
Nepali-English and Sinhala-English.*arXiv preprint arXiv:1902.01382*.
* Ha et al. (2016)Thanh-Le Ha, Jan Niehues, and Alexander H. Waibel. 2016.Toward multilingual neural machine translation with universal encoder
and decoder.In *Proceedings of the 13th International Workshop on Spoken
Language Translation*.
* Ha et al. (2017)Thanh-Le Ha, Jan Niehues, and Alexander H. Waibel. 2017.Effective strategies in zero-shot neural machine translation.In *IWSLT*.
* Henríquez et al. (2011)C. Henríquez, M. R. Costa-jussá, R. E. Banchs, L. Formiga, and J. B. Mari no. 2011.Pivot Strategies as an Alternative for Statistical Machine
Translation Tasks Involving Iberian Languages.In *Workshop on ICL NLP Tasks*.
* Jawanpuria et al. (2019)Pratik Jawanpuria, Arjun Balgovind, Anoop Kunchukuttan, and Bamdev Mishra.
2019.Learning multilingual word embeddings in latent metric space: a
geometric approach.*Transaction of the Association for Computational Linguistics
(TACL)*.
* Jha (2010)Girish Nath Jha. 2010.The TDIL Program and the Indian Langauge Corpora Intitiative
(ILCI).In *LREC*.
* Johnson et al. (2017)Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng
Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado,
Macduff Hughes, and Jeffrey Dean. 2017.[Google’s multilingual
neural machine translation system: Enabling zero-shot translation](http://aclweb.org/anthology/Q17-1024 "").*Transactions of the Association for Computational Linguistics*,
5:339–351.
* Klementiev et al. (2012)Alexandre Klementiev, Ivan Titov, and Binod Bhattarai. 2012.Inducing crosslingual distributed representations of words.In *Proceedings of the International Conference on Computational
Linguistics: Technical Papers*, pages 1459–1474.
* Kocmi and Bojar (2018)Tom Kocmi and Ondřej Bojar. 2018.[Trivial transfer
learning for low-resource neural machine translation](http://www.aclweb.org/anthology/W18-6325 "").In *Proceedings of the Third Conference on Machine Translation,
Volume 1: Research Papers*, pages 244–252, Belgium, Brussels. Association
for Computational Linguistics.
* Koehn (2005)Philipp Koehn. 2005.[Europarl: A
Parallel Corpus for Statistical Machine Translation](http://mt-archive.info/MTS-2005-Koehn.pdf "").In *Conference Proceedings: the tenth Machine Translation
Summit*, pages 79–86, Phuket, Thailand. AAMT, AAMT.
* Koehn (2017)Philipp Koehn. 2017.[Neural machine translation](http://arxiv.org/abs/1709.07809 "").*CoRR*, abs/1709.07809.
* Koehn et al. (2007)Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello
Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard
Zens, Chris Dyer, Ondřej Bojar, Alexandra Constantin, and Evan Herbst.
2007.[Moses: Open
source toolkit for statistical machine translation](http://www.aclweb.org/anthology/P/P07/P07-2045 "").In *Proceedings of the 45th Annual Meeting of the Association
for Computational Linguistics Companion Volume Proceedings of the Demo and
Poster Sessions*, pages 177–180, Prague, Czech Republic. Association for
Computational Linguistics.
* Koehn and Knowles (2017)Philipp Koehn and Rebecca Knowles. 2017.[Six challenges for
neural machine translation](http://www.aclweb.org/anthology/W17-3204 "").In *Proceedings of the First Workshop on Neural Machine
Translation*, pages 28–39, Vancouver. Association for Computational
Linguistics.
* Koehn et al. (2003)Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003.Statistical phrase-based translation.In *Proceedings of the 2003 Conference of the North American
Chapter of the Association for Computational Linguistics on Human Language
Technology-Volume 1*, pages 48–54. Association for Computational
Linguistics.
* Kunchukuttan and Bhattacharyya (2016)Anoop Kunchukuttan and Pushpak Bhattacharyya. 2016.Orthographic Syllable as basic unit for SMT between Related
Languages.In *Proceedings of the Conference on Empirical Methods in
Natural Language Processing*.
* Kunchukuttan and Bhattacharyya (2017)Anoop Kunchukuttan and Pushpak Bhattacharyya. 2017.Learning variable length units for SMT between related languages
via Byte Pair Encoding.In *First Workshop on Subword and Character level models in
NLP*.
* Kunchukuttan et al. (2017)Anoop Kunchukuttan, Maulik Shah, Pradyot Prakash, and Pushpak Bhattacharyya.
2017.[Utilizing lexical
similarity between related, low-resource languages for pivot-based smt](http://aclweb.org/anthology/I17-2048 "").In *Proceedings of the Eighth International Joint Conference on
Natural Language Processing (Volume 2: Short Papers)*, pages 283–289. Asian
Federation of Natural Language Processing.
* Lakew et al. (2018a)Surafel Melaku Lakew, Mauro Cettolo, and Marcello Federico. 2018a.[A comparison of
transformer and recurrent neural networks on multilingual neural machine
translation](http://aclweb.org/anthology/C18-1054 "").In *Proceedings of the 27th International Conference on
Computational Linguistics*, pages 641–652. Association for Computational
Linguistics.
* Lakew et al. (2018b)Surafel Melaku Lakew, Aliia Erofeeva, Matteo Negri, Marcello Federico, and
Marco Turchi. 2018b.Transfer learning in multilingual neural machine translation with
dynamic vocabulary.In *IWSLT*.
* Lakew et al. (2017)Surafel Melaku Lakew, Quintino F. Lotito, Matteo Negri, Marco Turchi, and
Marcello Federico. 2017.Improving zero-shot translation of low-resource languages.In *IWSLT*.
* Lee et al. (2017)Jason Lee, Kyunghyun Cho, and Thomas Hofmann. 2017.[Fully character-level
neural machine translation without explicit segmentation](http://aclweb.org/anthology/Q17-1026 "").*Transactions of the Association for Computational Linguistics*,
5:365–378.
* Lu et al. (2018)Yichao Lu, Phillip Keung, Faisal Ladhak, Vikas Bhardwaj, Shaonan Zhang, and
Jason Sun. 2018.[A neural interlingua
for multilingual machine translation](http://aclweb.org/anthology/W18-6309 "").In *Proceedings of the Third Conference on Machine Translation:
Research Papers*, pages 84–92. Association for Computational Linguistics.
* Mattoni et al. (2017)Giulia Mattoni, Pat Nagle, Carlos Collantes, and Dimitar Shterionov. 2017.Zero-shot translation for indian languages with sparse data.In *Proceedings of MT Summit XVI, Vol.2: Users and Translators
Track*, pages 1–10.
* Matusov et al. (2006)Evgeny Matusov, Nicola Ueffing, and Hermann Ney. 2006.Computing consensus translation for multiple machine translation
systems using enhanced hypothesis alignment.In *11th Conference of the European Chapter of the Association
for Computational Linguistics*.
* Mauro et al. (2012)Cettolo Mauro, Girardi Christian, and Federico Marcello. 2012.Wit3: Web inventory of transcribed and translated talks.In *Conference of European Association for Machine Translation*,
pages 261–268.
* Mikolov et al. (2013)Tomas Mikolov, Quoc V Le, and Ilya Sutskever. 2013.Exploiting similarities among languages for machine translation.Technical report, arXiv preprint arXiv:1309.4168.
* Murthy et al. (2018)V. Rudra Murthy, Anoop Kunchukuttan, and Pushpak Bhattacharyya. 2018.[Addressing word-order
divergence in multilingual neural machine translation for extremely low
resource languages](http://arxiv.org/abs/1811.00383 "").*CoRR*, abs/1811.00383.
* Nakazawa et al. (2018)Toshiaki Nakazawa, Shohei Higashiyama, Chenchen Ding, Raj Dabre, Anoop
Kunchukuttan, Win Pa Pa, Isao Goto, Hideya Mino, Katsuhito Sudoh, and Sadao
Kurohashi. 2018.Overview of the 5th workshop on asian translation.In *Proceedings of the 5th Workshop on Asian Translation
(WAT2018)*.
* Nakov and Ng (2009)Preslav Nakov and Hwee Tou Ng. 2009.Improved statistical machine translation for resource-poor languages
using related resource-rich languages.In *Proceedings of the 2009 Conference on Empirical Methods in
Natural Language Processing*.
* Neubig (2017)Graham Neubig. 2017.[Neural machine translation
and sequence-to-sequence models: A tutorial](http://arxiv.org/abs/1703.01619 "").*CoRR*, abs/1703.01619.
* Neubig and Hu (2018)Graham Neubig and Junjie Hu. 2018.[Rapid adaptation of
neural machine translation to new languages](http://aclweb.org/anthology/D18-1103 "").In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pages 875–880. Association for Computational
Linguistics.
* Nguyen and Chiang (2017)Toan Q. Nguyen and David Chiang. 2017.[Transfer learning
across low-resource, related languages for neural machine translation](http://aclweb.org/anthology/I17-2050 "").In *Proceedings of the Eighth International Joint Conference on
Natural Language Processing (Volume 2: Short Papers)*, pages 296–301. Asian
Federation of Natural Language Processing.
* Nishimura et al. (2018a)Yuta Nishimura, Katsuhito Sudoh, Graham Neubig, and Satoshi Nakamura.
2018a.[Multi-source neural machine
translation with data augmentation](https://arxiv.org/abs/1810.06826 "").In *15th International Workshop on Spoken Language Translation
(IWSLT)*, Brussels, Belgium.
* Nishimura et al. (2018b)Yuta Nishimura, Katsuhito Sudoh, Graham Neubig, and Satoshi Nakamura.
2018b.[Multi-source neural
machine translation with missing data](http://aclweb.org/anthology/W18-2711 "").In *Proceedings of the 2nd Workshop on Neural Machine
Translation and Generation*, pages 92–99. Association for Computational
Linguistics.
* Och and Ney (2001)Franz Josef Och and Hermann Ney. 2001.Statistical multi-source translation.In *Proceedings of MT Summit*, volume 8, pages 253–258.
* Pan and Yang (2010)Sinno Jialin Pan and Qiang Yang. 2010.[A survey on transfer
learning](https://doi.org/10.1109/TKDE.2009.191 "").*IEEE Trans. on Knowl. and Data Eng.*, 22(10):1345–1359.
* Pinnis et al. (2018)Mārcis Pinnis, Matīss Rikters, and Rihards Krišlauks. 2018.Training and Adapting Multilingual NMT for Less-resourced and
Morphologically Rich Languages.In *Proceedings of the Eleventh International Conference on
Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan. European
Language Resources Association (ELRA).
* Platanios et al. (2018)Emmanouil Antonios Platanios, Mrinmaya Sachan, Graham Neubig, and Tom Mitchell.
2018.[Contextual parameter
generation for universal neural machine translation](http://aclweb.org/anthology/D18-1039 "").In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pages 425–435. Association for Computational
Linguistics.
* Ramachandran et al. (2017)Prajit Ramachandran, Peter J. Liu, and Quoc V. Le. 2017.Unsupervised pretraining for sequence to sequence learning.In *EMNLP*.
* Sachan and Neubig (2018)Devendra Sachan and Graham Neubig. 2018.[Parameter sharing
methods for multilingual self-attentional translation models](http://aclweb.org/anthology/W18-6327 "").In *Proceedings of the Third Conference on Machine Translation:
Research Papers*, pages 261–271. Association for Computational Linguistics.
* Schroeder et al. (2009)Josh Schroeder, Trevor Cohn, and Philipp Koehn. 2009.Word lattices for multi-source translation.In *Proceedings of the 12th Conference of the European Chapter
of the Association for Computational Linguistics*, pages 719–727.
Association for Computational Linguistics.
* Sennrich et al. (2016a)Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016a.[Improving neural
machine translation models with monolingual data](http://www.aclweb.org/anthology/P16-1009 "").In *Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pages 86–96, Berlin,
Germany. Association for Computational Linguistics.
* Sennrich et al. (2016b)Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b.[Neural machine
translation of rare words with subword units](http://www.aclweb.org/anthology/P16-1162 "").In *Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pages 1715–1725,
Berlin, Germany. Association for Computational Linguistics.
* Sestorain et al. (2018)Lierni Sestorain, Massimiliano Ciaramita, Christian Buck, and Thomas Hofmann.
2018.[Zero-shot dual machine
translation](http://arxiv.org/abs/1805.10338 "").*CoRR*, abs/1805.10338.
* Sgall and Panevová (1987)Petr Sgall and Jarmila Panevová. 1987.[Machine translation,
linguistics, and interlingua](https://doi.org/10.3115/976858.976876 "").In *Proceedings of the Third Conference on European Chapter of
the Association for Computational Linguistics*, EACL ’87, pages 99–103,
Stroudsburg, PA, USA. Association for Computational Linguistics.
* Steinberger et al. (2014)Ralf Steinberger, Mohamed Ebrahim, Alexandros Poulis, Manuel Carrasco-Benitez,
Patrick Schlüter, Marek Przybyszewski, and Signe Gilbro. 2014.An overview of the European Union’s highly multilingual parallel
corpora.*Language Resources and Evaluation*, 48(4):679–707.
* Sutskever et al. (2014)Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014.[Sequence
to sequence learning with neural networks](http://dl.acm.org/citation.cfm?id=2969033.2969173 "").In *Proceedings of the 27th International Conference on Neural
Information Processing Systems*, NIPS’14, pages 3104–3112, Cambridge, MA,
USA. MIT Press.
* Tan et al. (2019)Xu Tan, Yi Ren, Di He, Tao Qin, and Tie-Yan Liu. 2019.Multilingual neural machine translation with knowledge distillation.In *International Conference on Learning Representations*.
* Thu et al. (2016)Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew M Finch, and Eiichiro Sumita.
2016.Introducing the asian language treebank (ALT).In *LREC*.
* Tiedemann (2012a)Jörg Tiedemann. 2012a.Character-based pivot translation for under-resourced languages and
domains.In *Proceedings of the 13th Conference of the European Chapter
of the Association for Computational Linguistics*.
* Tiedemann (2012b)Jörg Tiedemann. 2012b.Parallel data, tools and interfaces in opus.In *Proceedings of the Eight International Conference on
Language Resources and Evaluation (LREC’12)*, Istanbul, Turkey. European
Language Resources Association (ELRA).
* Uchida (1996)H. Uchida. 1996.UNL: Universal Networking Language – An Electronic Language for
Communication, Understanding, and Collaboration.In *UNU/IAS/UNL Center*.
* Utiyama and Isahara (2007)Masao Utiyama and Hitoshi Isahara. 2007.A Comparison of Pivot Methods for Phrase-Based Statistical Machine
Translation.In *Conference of the North Americal Chapter of the Association
for Computational Linguistics*, pages 484–491.
* Vaswani et al. (2017)Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017.[Attention
is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf "").In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, editors, *Advances in Neural
Information Processing Systems 30*, pages 5998–6008. Curran Associates, Inc.
* Vázquez et al. (2018)Raúl Vázquez, Alessandro Raganato, Jörg Tiedemann, and
Mathias Creutz. 2018.[Multilingual NMT with a
language-independent attention bridge](http://arxiv.org/abs/1811.00498 "").*CoRR*, abs/1811.00498.
* Vilar et al. (2007)David Vilar, Jan-T Peter, and Hermann Ney. 2007.Can we translate letters?In *Proceedings of the Second Workshop on Statistical Machine
Translation*.
* Wang et al. (2017a)Rui Wang, Andrew Finch, Masao Utiyama, and Eiichiro Sumita. 2017a.[Sentence embedding for
neural machine translation domain adaptation](http://aclweb.org/anthology/P17-2089 "").In *Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers)*, pages 560–566,
Vancouver, Canada. Association for Computational Linguistics.
* Wang et al. (2017b)Rui Wang, Masao Utiyama, Lemao Liu, Kehai Chen, and Eiichiro Sumita.
2017b.Instance weighting for neural machine translation domain adaptation.In *Proceedings of the 2017 Conference on Empirical Methods in
Natural Language Processing*, pages 1482–1488, Copenhagen, Denmark.
* Wang et al. (2019)Xinyi Wang, Hieu Pham, Philip Arthur, and Graham Neubig. 2019.[Multilingual
neural machine translation with soft decoupled encoding](https://openreview.net/forum?id=Skeke3C5Fm "").In *International Conference on Learning Representations*.
* Wang et al. (2018)Yining Wang, Jiajun Zhang, Feifei Zhai, Jingfang Xu, and Chengqing Zong. 2018.[Three strategies to
improve one-to-many multilingual translation](http://aclweb.org/anthology/D18-1326 "").In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pages 2955–2960. Association for Computational
Linguistics.
* Witkam (2006)T. Witkam. 2006.History and Heritage of the DLT (Distributed Language Translation)
project.In *Utrecht, The Netherlands: private publication*.
* Wu and Wang (2007)Hua Wu and Haifeng Wang. 2007.Pivot language approach for phrase-based statistical machine
translation.*Machine Translation*, 21(3):165–181.
* Wu et al. (2016)Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner,
Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws,
Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian,
Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick,
Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016.[Google’s neural machine
translation system: Bridging the gap between human and machine translation](http://arxiv.org/abs/1609.08144 "").*CoRR*, abs/1609.08144.
* Zaremoodi et al. (2018)Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari. 2018.[Adaptive knowledge
sharing in multi-task learning: Improving low-resource neural machine
translation](http://aclweb.org/anthology/P18-2104 "").In *Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers)*, pages 656–661.
Association for Computational Linguistics.
* Ziemski et al. (2016a)Michal Ziemski, Marcin Junczys-Dowmunt, and Bruno Pouliquen.
2016a.The united nations parallel corpus v1. 0.In *LREC*.
* Ziemski et al. (2016b)Michał Ziemski, Marcin Junczys-Dowmunt, and Bruno Pouliquen.
2016b.The United Nations Parallel Corpus v1.0.In *Proceedings of the Tenth International Conference on
Language Resources and Evaluation (LREC 2016)*, Paris, France. European
Language Resources Association (ELRA).
* Zoph and Knight (2016)Barret Zoph and Kevin Knight. 2016.[Multi-source neural
translation](https://doi.org/10.18653/v1/N16-1004 "").In *Proceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies*, pages 30–34. Association for Computational Linguistics.
* Zoph et al. (2016)Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016.[Transfer
learning for low-resource neural machine translation](http://aclweb.org/anthology/D/D16/D16-1163.pdf "").In *Proceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4,
2016*, pages 1568–1575.
