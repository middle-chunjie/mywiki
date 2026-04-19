ProCIS: A Benchmark for Proactive Retrieval in Conversations
============================================================

Chris SamarinasUniversity of Massachusetts AmherstAmherstMAUnited States[csamarinas@cs.umass.edu](mailto:csamarinas@cs.umass.edu)andHamed ZamaniUniversity of Massachusetts AmherstAmherstMAUnited States[zamani@cs.umass.edu](mailto:zamani@cs.umass.edu)

(2024)

###### Abstract.

The field of conversational information seeking, which is rapidly gaining interest in both academia and industry, is changing how we interact with search engines through natural language interactions. Existing datasets and methods are mostly evaluating *reactive* conversational information seeking systems that solely provide response to every query from the user. We identify a gap in building and evaluating *proactive* conversational information seeking systems that can monitor a multi-party human conversation and proactively engage in the conversation at an opportune moment by retrieving useful resources and suggestions. In this paper, we introduce a large-scale dataset for proactive document retrieval that consists of over 2.8 million conversations. We conduct crowdsourcing experiments to obtain high-quality and relatively complete relevance judgments through depth-k pooling. We also collect annotations related to the parts of the conversation that are related to each document, enabling us to evaluate proactive retrieval systems. We introduce normalized proactive discounted cumulative gain (npDCG) for evaluating these systems, and further provide benchmark results for a wide range of models, including a novel model we developed for this task. We believe that the developed dataset, called ProCIS, paves the path towards developing proactive conversational information seeking systems.

conversational search; proactive search systems; dialogue systems

††journalyear: 2024††copyright: rightsretained††conference: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 14–18, 2024; Washington, DC, USA††booktitle: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA††doi: 10.1145/3626772.3657869††isbn: 979-8-4007-0431-4/24/07††ccs: Information systems Test collections

1. Introduction
----------------

Conversational information seeking (CIS) has been identified as an emerging subfield of research with information retrieval and related disciplines *(Zamani et al., [2023])*. CIS systems are revolutionizing the way users seek and access information through natural language dialogues *(Hattimare et al., [2023]; Samarinas et al., [2024])*. The advent of large language models (LLMs), such as LLaMA *(Touvron et al., [2023])*, Mistral *(Jiang et al., [2023])*, and GPT-4 *(OpenAI, [2023])*, in the past year has opened up new opportunities for improving these systems.

Existing models for CIS are generally designed based on a “query-response” paradigm, where the user starts the interaction by submitting a search query and the system responds with a search result (e.g., a ranked-list of passages or documents that may include snippets, a direct natural language answer to the submitted query, one or more entity cards, or a combination of them). In this paradigm, users can interact with the search results and/or try a different query which will be answered by another search result. This process repeats until the user terminates the search session. Popular conversational question answering benchmarks, such as QuAC *(Choi et al., [2018])* and CoQA *(Reddy et al., [2019])*, and TREC Conversational Assistance Track (CAsT) *(Owoicho et al., [2022])* have been mainly focusing on evaluating CIS systems based on such paradigm. However, this is not an optimal interaction design for CIR systems and they must be able to perform *mixed-initiative interactions* *(Aliannejadi et al., [2021])*. Asking clarifying questions *(Zamani et al., [2020c], [a], [b]; Aliannejadi et al., [2019]; Braslavski et al., [2017]; Rao and Daumé III, [2018]; Stoyanchev et al., [2014]; Sekulić et al., [2022])* is a type of mixed-initiative interaction that has been extensively studied in the context of CIS. For more information about mixed-initiative CIS, refer to *(Zamani et al., [2023], Chapter 6)*.

This paper focuses on *proactive interactions* as another type of mixed-initiative CIS *(Wadhwa and Zamani, [2021])*. Proactive CIS, despite numerous applications, has been relatively under-explored. A reason for this is lack of data for training and evaluating proactive systems. This paper attempts to bridge this gap and presents a new large-scale data collection for proactive CIS. To this aim, we focus on *document retrieval as proactive contextual suggestion to multi-party human conversations*.

Imagine multiple users are interacting with each other and a CIS agent is monitoring this conversation. This can be a conversation in a forum, in a group chat such as Slack channel,111<https://slack.com/> or a bot listening to an ongoing conversation between people in a meeting room.222In some of these scenarios, there are privacy considerations that should be taken into account. These issues are outside the scope of this work, yet important regardless. While users are conversing, an agent may engage in the conversation by providing a useful suggestion or by verifying the factually of the claims in the conversation. Such proactive informational interactions are of interest to this work. Therefore, we introduce the task of proactive document retrieval that given a sequence of utterances decides whether to engage in the conversation with a retrieval result list or not.

To build a large-scale data collection for this work, we use a Wikipedia dump with about 5 million English articles as a knowledge source, i.e., retrieval collection, that a CIS agent can use to proactive engage in a conversation. We then collected Reddit threads333We obtain Reddit threads from <https://github.com/ArthurHeitmann/arctic_shift>. Any use of the data must be in accordance with the Reddit’s term of services. in which multiple users are interacting with each other about a topic and there exists a link to Wikipedia articles in that thread. We assume that such threads are likely to be informational and we can build systems that find relevant Wikipedia pages once they are excluded from the conversation. After careful filtering and data cleaning, we end up with a large-scale dataset with over 2.8 million conversations, called *ProCIS*. We provide multiple data splits for training, development, and testing. To have a reliable test set, we conduct crowdsourcing experiments to annotate documents for each conversation. We not only collect relevance judgment, but also ask annotators to highlight the parts of conversation that are related to each relevant document. This enables us to evaluate proactive systems by knowing what documents provide useful suggestions to each conversation utterance. The crowdsourcing experiments are conducted using Amazon Mechanical Turk and the depth-$k$ pooling approach is applied to construct the document pool for annotation.

We further provide an evaluation methodology for evaluating proactive document retrieval systems. We introduce normalized proactive discounted cumulative gain ($npDCG$) for evaluating such methods. We also adopt the developed dataset for *reactive* document retrieval methods for contextual suggestion. There can also be real-world applications for such reactive scenarios, where users explicitly ask a bot to engage in a multi-party human conversation and provide useful suggestions. We evaluate a term-matching retrieval model, a neural sparse retrieval model, a single-vector dense retrieval model, and a multi-vector dense retrieval model on the evaluated benchmarks. We also introduce a novel approach that is employing LLMs for query generation and result filtering.

We believe the benchmark results presented in this paper smooth the path towards developing advanced proactive CIS systems. We release the data, code, and benchmark results for research purposes: <https://github.com/algoprog/ProCIS>.

2. Related Work
----------------

This section reviews the literature on the three main areas related to our work.

### 2.1. Proactive Search Systems

Proactive search systems, designed to anticipate user needs and provide relevant information without explicit queries, are gaining attention due to their potential to enhance user experience and improve search efficiency. A study on the effectiveness of proactive search systems proposed a framework to evaluate such systems based on the correlation between the expected and predicted outcomes *(Sen et al., [2018])*. This study demonstrated the potential of proactive search systems to recommend documents that could help users accomplish their tasks without explicit queries.

Another work introduced the concept of *information fostering*, a proactive approach that predicts potential issues and provides help to overcome them *(Shah, [2018])*. This approach goes beyond recommending queries and documents and suggests strategies and people, thereby offering a more comprehensive support system for information seekers.
The use of Wikipedia concepts in proactive information retrieval was explored in a study on improving retrieval on noisy text *(Ahmed and Bulathwela, [2022])*. The study demonstrated the potential of Wikipedia concepts to provide relevance signals and improve precision in proactive information retrieval.

A study on proactive search support in conversations demonstrated how a proactive search agent could augment conversations by retrieving and presenting information related to the conversation *(Andolina et al., [2018])*. The study highlighted the potential of proactive search systems to affect the topical structure of conversations and reduce the need for explicit search activity. Recently, *Wadhwa and Zamani ([2021])* highlighted the opportunities and challenges in proactive interactions in conversational information seeking.

This work complements the literature on proactive search by building a large-scale dataset, introducing an evaluation methodology, and presenting benchmark results. For more information about proactive search systems, refer to the tutorial recently presented by *Liao et al. ([2023])*.

*Table 1. Statistics of each data split in the ProCIS dataset.*

|  | train | dev | future-dev | test |
| --- | --- | --- | --- | --- |
| Total conversations | 2,830,107 | 4165 | 3385 | 100 |
| Total posts | 1,893,201 | 4165 | 3385 | 100 |
| Number of subreddits covered | 34,785 | 1563 | 1659 | 100 |
| Total unique users in the conversations | 2,284,841 | 10,896 | 7,920 | 309 |
| Average number of turns | 5.41 ($\pm$ 7.81) | 4.91 ($\pm$ 3.60) | 4.48 ($\pm$ 3.30) | 4.49 ($\pm$ 1.60) |
| Average number of words per conversation | 406.01 ($\pm$ 774.67) | 359.19 ($\pm$ 734.95) | 325.36 ($\pm$ 609.58) | 173.85 ($\pm$ 101.22) |
| Average number of words per turn | 70.54 ($\pm$ 82.38) | 68.77 ($\pm$ 74.80) | 72.55 ($\pm$ 85.37) | 41.58 ($\pm$ 26.49) |
| Average number of Wikipedia links per conversation | 1.71 ($\pm$ 2.46) | 1.90 ($\pm$ 3.03) | 1.15 ($\pm$ 0.57) | 1.15 ($\pm$ 0.46) |
| Average number of unique users per conversation | 3.17 ($\pm$ 1.41) | 2.93 ($\pm$ 1.16) | 2.88 ($\pm$ 1.11) | 3.41 ($\pm$ 1.39) |
| Average number of comments per user | 6.71 ($\pm$ 462.74) | 1.88 ($\pm$ 8.21) | 1.92 ($\pm$ 12.93) | 1.45 ($\pm$ 2.49) |

### 2.2. Conversational Information Seeking

CIS is a rapidly evolving field that mainly focuses on retrieving information in the context of conversations *(Mo et al., [2024])*. The Conversational Assistance Track (CAsT) *(Dalton et al., [2019])* was a significant initiative in this direction, aiming to facilitate Conversational Information Seeking (CIS) research and create a large-scale reusable test collection for conversational search systems.

Several studies have proposed models and theories to address the challenges unique to CIS. For instance, one study proposed a theoretical model of a conversational system that implements a small set of properties derived from past work on human conversations *(Radlinski and Craswell, [2017])*. Another study proposed a Conversational Dense Retrieval model that learns contextualized embeddings for multi-turn conversational queries *(Yu et al., [2021])*.

The role of retrieval in CIS has also been explored. One study introduced an open-retrieval conversational question answering task, where evidence is retrieved from a large collection before extracting answers *(Qu et al., [2020])*. Another study proposed a pipeline for passage retrieval in a conversational search setting, comprising conversational term selection and multi-view reranking *(Kumar and Callan, [2020])*.

Datasets play a crucial role in the development and evaluation of conversational search models. Several studies have introduced new datasets to facilitate research in this area. For instance, one study introduced MANtIS, a large-scale dataset containing multi-domain and grounded information seeking dialogues *(Penha et al., [2019])*. Another study created a dataset, OR-QuAC, to facilitate research on open-retrieval conversational question answering *(Qu et al., [2020])*. The dataset created by *Ros et al. ([2023])* is perhaps the most similar data to ProCIS. It is also based on Reddit thread, however, ProCIS has multiple advantages in comparison; it is more than an order of magnitude larger in terms of both training examples and corpus size, it has a carefully annotated test set and it uses a corpus of clean Wikipedia articles which can be more useful for research experiments that involve specific concepts. In a similar vein, the RCD-2020 track *(Ganguly et al., [2020])* introduced another small-scale dataset. However, the conversations in this dataset are derived from short movie dialogues, resulting in brief and simplistic exchanges. Consequently, the dataset is primarily suitable for limited evaluation purposes rather than comprehensive conversational modeling.

CIS is a complex and challenging task that requires comprehensive understanding of the conversational inputs, effective query reformulation, and efficient retrieval methods. The studies discussed in this section have made significant contributions towards addressing these challenges. However, there is still much work to be done, particularly in the area of proactive retrieval in conversations, which is the focus of our work.

### 2.3. Information Filtering

Information filtering is a closely related area to proactive conversational information seeking. The TREC Filtering Track *(Robertson and Soboroff, [2003])* and CLEF INFILE *(Besançon et al., [2009])* focused on the multiple filtering tasks including adaptive filtering, where systems aim to select relevant documents from a stream of incoming documents based on a user’s profile. These tracks have contributed to the development of effective filtering techniques based on threshold optimization *(Arampatzis, [2001])* and profile adaptation *(Gauch et al., [2007]; Zamani and Shakery, [2018])*.

<img src='extracted/2405.06460v1/images/annotation-ui.png' alt='Refer to caption' title='' width='419' height='201' />

*Figure 1. The online annotation interface for the crowd-sourcing of the test set.*

3. Data Collection
-------------------

Our goal is to construct a large-scale dataset that enables proactive retrieval research for multi-party human conversations. To this aim, we focus on Wikipedia as the retrieval corpus for providing information proactively and use Reddit threads to obtain multi-party human conversations. In order to focus on the conversation threads that are likely to benefit from information seeking, we only collect conversations that contain one or more hyperlinks to a Wikipedia article. To gather this data, we utilize the publicly available dumps of Reddit posts from pushshift.io,444Currently available at: <https://github.com/ArthurHeitmann/arctic_shift> collected from 2005 until 2022. We applied several filters to ensure the data quality, such as removing not-suitable-for-work (NSFW) content, posts with external links, or embedded media. We also excluded non-English content and removed HTML formatting and link mentions. We then sampled nested threads where each comment in the chain is the child of the previous comment.

As the document corpus, we used a pre-processed dump of 5,315,384 Wikipedia articles,555Available at [https://github.com/tscheepers/Wikipedia-Summary-Dataset](https://github.com/tscheepers/Wikipedia-Summary-Dataset "") and we mapped all the extracted Wikipedia links in the collected Reddit threads to the articles in this corpus. We further excluded the threads with links to Wikipedia articles that do not exist in the corpus. After applying all the filters, the final dataset comprised 2,830,107 conversations from 1,893,201 unique posts.

#### Data Splits

We split the data to four subsets: train, dev, future-dev, and test. The three subsets of train, dev, and test are split randomly, while the future-dev set only contains conversations that follow the conversations in the training set chronologically. This split can be used for evaluating the generalization capabilities of retrieval models in potentially new emerging concepts and topics not seen during training. The test split was sampled from 100 unique random subreddits, all from posts with at least a Reddit score of 20 to ensure high quality. The relevance annotation in train, dev, and future-dev is sparse and sometimes noisy due to topic shifts. We assume a Wikipedia article is relevant to a conversation, if its URL is mentioned in the conversation thread. After some analysis through crowdsourcing, we found that 63% of the mentioned Wikipedia articles actually provide useful context to the conversation. This enables us to construct a large-scale dataset with sparse noisy annotation. According to lessons learned from the MS MARCO collection *(Nguyen et al., [2016])*, large-scale sparse annotations can be quite impactful in training retrieval models. To ensure comprehensive evaluation, we conduct pooling and collect human annotations for the documents in the test set. Therefore, even though the test set contains only 100 conversations, it has on average 8.02 relevant documents per conversation, much higher than the 1.15 (potentially irrelevant) links on average that the users mentioned originally on Reddit. Table [1] presents the statistics of ProCIS.

<img src='extracted/2405.06460v1/images/comment_lengths.png' alt='Refer to caption' title='' width='598' height='449' />

*Figure 2. Utterance length (# tokens) distribution in ProCIS.*

<img src='extracted/2405.06460v1/images/thread_lengths.png' alt='Refer to caption' title='' width='598' height='449' />

*Figure 3. Conversation length (# tokens) distribution in ProCIS.*

<img src='extracted/2405.06460v1/images/turns.png' alt='Refer to caption' title='' width='598' height='449' />

*Figure 4. Conversation length (# turns) distribution in ProCIS.*

*Table 2. The 60 most frequent categories (subreddits) in the ProCIS training set.*

| Subreddit | Frequency | Subreddit | Frequency | Subreddit | Frequency |
| --- | --- | --- | --- | --- | --- |
| AskReddit | 500,419 (38.19%) | neoliberal | 12,751 (0.97%) | UnresolvedMysteries | 7062 (0.54%) |
| askscience | 58,527 (4.47%) | AskAnAmerican | 12,664 (0.97%) | exmormon | 7040 (0.54%) |
| explainlikeimfive | 55,063 (4.20%) | CapitalismVSocialism | 12,529 (0.96%) | AskMen | 6753 (0.52%) |
| IAmA | 44,315 (3.38%) | CFB | 12,232 (0.93%) | OutOfTheLoop | 6745 (0.51%) |
| changemyview | 34,106 (2.60%) | soccer | 12,179 (0.93%) | hockey | 6700 (0.51%) |
| atheism | 31,717 (2.42%) | math | 12,108 (0.92%) | books | 6636 (0.51%) |
| politics | 28,388 (2.17%) | bestof | 11,737 (0.90%) | Jokes | 6546 (0.50%) |
| DebateReligion | 25,419 (1.94%) | anime | 11,244 (0.86%) | unitedkingdom | 6164 (0.47%) |
| Christianity | 24,170 (1.84%) | leagueoflegends | 10,680 (0.82%) | reddit.com | 6087 (0.46%) |
| PoliticalDiscussion | 22,467 (1.71%) | europe | 10,650 (0.81%) | learnprogramming | 5989 (0.46%) |
| Showerthoughts | 20,035 (1.53%) | Fitness | 9375 (0.72%) | buildapc | 5945 (0.45%) |
| SubredditDrama | 18,751 (1.43%) | whowouldwin | 9198 (0.70%) | cars | 5928 (0.45%) |
| NoStupidQuestions | 18,286 (1.40%) | DebateAChristian | 9066 (0.69%) | talesfromtechsupport | 5836 (0.45%) |
| history | 17,200 (1.31%) | Libertarian | 8889 (0.68%) | asoiaf | 5804 (0.44%) |
| AskHistorians | 16,709 (1.28%) | nba | 8883 (0.68%) | space | 5566 (0.42%) |
| conspiracy | 15,466 (1.18%) | india | 8628 (0.66%) | guns | 5445 (0.42%) |
| nfl | 14,688 (1.12%) | tipofmytongue | 7795 (0.59%) | NeutralPolitics | 5423 (0.41%) |
| unpopularopinion | 13,828 (1.06%) | DebateAnAtheist | 7718 (0.59%) | ukpolitics | 5351 (0.41%) |
| AskEurope | 13,206 (1.01%) | AskTrumpSupporters | 7564 (0.58%) | Drugs | 5306 (0.40%) |
| movies | 12,899 (0.98%) | Bitcoin | 7160 (0.55%) | LifeProTips | 5277 (0.40%) |

#### Relevance Assessment for the Test Set

Reliable evaluation requires complete relevance annotation. In order to collect relevance annotations for the constructed test set, we follow the pooling guideline from TREC. For each conversation in the test split, we created 5 pools of up to 10 document candidates from the following models: BM25, SPLADE, ANCE, ColBERT and LMGR with GPT-4 (refer to Section [6] for LMGR). The reason for choosing these models is to obtain a diverse pool of documents. BM25 is a term-matching model, while SPLADE is a neural sparse retrieval model, ANCE and ColBERT are single- and multi-vector dense retrieval models, and LMGR is an effective generative model for retrieval. The supervised neural models were fine-tuned on the ProCIS training set before producing the pools.

Once the pools are constructed, we designed a careful crowdsourcing experiment on Amazon’s Mechanical Turk for data annotation. Each Human Intelligence Task (HIT) asks the crowdworkers to do the following:

1. (1)

    Understand the Conversation: read the Reddit post title, content, and comments one by one to understand the main topic and subtopics of the conversation. The crowdworker should read each utterance and click on ‘next’ to see the next utterance.

2. (2)

    Summarize the whole conversation in 1-2 sentences covering the main topics and themes.

3. (3)

    Read and annotate each document by selecting one of the three relevance levels: provides useful context, partially relevant or irrelevant (see below).

4. (4)

    Highlight evidence: In case of a relevant or partially relevant document, highlight all the sentences of the conversation that are related to the document.

These steps ensure that the crowdworkers spend enough time to thoroughly understand each conversation and help us easily detect and discard low quality annotations. To facilitate the annotation process, we built a custom interface with the following features: it first displays annotation instructions, then expects the worker to read the comments one at a time, expects a brief summary with a minimum required length of six words and then displays documents one by one for annotation. For each document, it asks for a relevance level option, and in case of relevance or partial relevance, the worker is required to highlight some text in the conversation (see Figure[1]). We ask for three-level graded relevance annotation with respect to the following definitions:

1. (1)

    Provides Useful Context (Label 2): The document is directly related to the conversation. It provides substantial useful context, background, or additional information that enhances understanding of the conversation. This could include articles about the main topic, key individuals, events, or concepts mentioned in the conversation.

2. (2)

    Partially Relevant (Label 1): The document is somewhat related to the conversation. It may provide some context or additional information, but it is not directly tied to the main topic of the conversation. This could include articles about broader themes, tangential topics, or peripheral concepts mentioned in the conversation.

3. (3)

    Irrelevant (Label 0): The document is not related to the conversation in any way. It does not provide any useful context or additional information to the conversation.

In each HIT, we display a conversation between 3 and 10 utterances and up to 500 words long. In total we created 1000 HITs in 8 batches and collected 4207 relevance assessments with supporting evidence. Each HIT was completed by 3 different workers. We limited the HITs to adult workers from the US, UK, Australia and Ireland, with over 98% approval rate who have completed at least 5,000 assignments. The inter-annotator agreement was 0.6482 in Fleiss’ $\kappa$ score. For the final labels we used majority voting. If two annotators picked partially relevant and relevant as labels for one document and the third one irrelevant, we assigned relevant as the final label. The ideal positions of the annotated documents were identified based on the earliest position of the highlighted supported evidence from the 3 workers for each HIT. The total annotation cost was $3500 for $1.16 per HIT. The annotation interface is released for transparency and future usage.

4. Data Analysis
-----------------

In this section, we deep dive into the ProCIS dataset to analyze its characteristics and understand its potential impact on the research community.

#### Utterance Length

We analyze the distribution of comment lengths in the ProCIS dataset (Figure [4]). We tokenize the data using segtok tokenizer from the NLTK library,666<https://www.nltk.org> and found that 95% of the comments have up to 200 tokens in length. As presented in Table[1], the average comment length is $70.54\pm 82.38$. This shows that the lengths of comments can vary significantly, and sometimes can be quite long. This insight is essential because it denotes that modeling conversations require long context modeling. It also highlights the need for retrieval models to be able to handle varying lengths of comments in order to effectively address the needs of users.

#### Conversation Length

Around 95% of the conversations in the ProCIS dataset have up to 700 tokens in length and consist of 5.41 turns on average with the standard deviation of 7.81 (see Figures [4] and [4] and Table[1]). However, a small percentage of conversations can span over thousands of tokens and sometimes more than 20 turns. This demonstrates the complexity and dynamics of real conversations. It emphasizes the need for retrieval models to be capable of maintaining context and coherence across multiple conversational turns while providing relevant information in a timely manner. The test set, by design, has shorter conversations, so that it is easier to get good relevance assessments from crowdworkers.

#### Diversity of Topics

ProCIS covers a broad range of topics, with the most popular categories being politics, religion, sports, finance, science, and general discussions (see Table[2]). This extensive topical diversity reinforces the importance of developing retrieval models that can handle open-domain conversations. It also showcases the potential for ProCIS to serve as a benchmark for evaluating the performance of retrieval models across different domains. In Figure [5], we can see a clustered t-SNE *(Hinton and Roweis, [2002])* projection of all the 100 subreddits covered in the test set. We use a pre-trained sentence embedding model777[https://hf.co/sentence-transformers/all-mpnet-base-v2](https://hf.co/sentence-transformers/all-mpnet-base-v2 "") for encoding their descriptions and agglomerative clustering to identify broader groups. From the visualization we can see all the topic clusters covered in the test set.

<img src='extracted/2405.06460v1/images/subreddit_clusters.png' alt='Refer to caption' title='' width='598' height='586' />

*Figure 5. t-SNE visualization of the categories (subreddits) in the ProCIS test set.*

5. Task Formulation
--------------------

In conversational information seeking systems, such as conversational question answering or conversational passage retrieval as modeled by the TREC CAsT Track *(Owoicho et al., [2022])*, the last user utterance is often a query or a question, and the goal is to retrieve a list of passages or produce an answer to the question in the last utterance in the context of the conversation. Orthogonal to these research problems, we target *document retrieval as a form of contextual suggestion*. This means that the user utterances are not queries or questions and we do not aim at retrieving documents in order to answer them. Instead, we aim at retrieving documents that can add value to an ongoing conversation. Imagine there is a multi-party human conversation, and a retrieval engine can occasionally provide resources to contribute to the conversation. In this context, we identify two different retrieval tasks, as follows:

1. (1)

    Reactive document retrieval for contextual suggestion in conversation: During an ongoing multi-party human conversation, imagine any of the involved parties can hit a button to ask for recommendation or useful resources. These resources are explicit answers to any question but can contribute to the ongoing conversation. Since this document retrieval model is initiated by pushing a button, we call this a *reactive* model.

2. (2)

    Proactive document retrieval for contextual suggestion in conversation: Imagine an agent is monitoring an ongoing multi-party human conversation and at any turn in the conversation can jump in and provide a useful suggestion by retrieving documents. These systems are more challenging than reactive systems, as they need to additionally decide when is a good time to proactively engage with the conversation.

#### Evaluating Reactive Document Retrieval Models

Given a conversation as a sequence of utterances $U\={u_{1},u_{2},\cdots,u_{m}}$ and a corpus of Wikipedia articles $D\={d_{1},d_{2},\cdots,d_{n}}$. The goal of reactive document retrieval is to develop a retrieval model $M$ that takes a conversation up to turn $k$ (i.e., ${u_{1},u_{2},\cdots,u_{k}}$ and returns a ranked list of documents. We aim at providing useful suggestions that can contribute to any of the utterances in the given conversation.

For evaluating reactive retrieval models, we assume the user asks for contextual suggestion at the end of the conversation, thus we use the whole conversation as query and retrieve documents. We then calculate standard retrieval metrics, such as nDCG@k, MRR, MAP, and Recall@k, to evaluate the system. Given the nature of conversational information seeking, precision-oriented metrics (with small $k$ values) are more important.

#### Evaluating Proactive Document Retrieval Models

In proactive document retrieval, for each turn in a conversation, the goal is to either wait (do nothing) or retrieve a list of documents.
We evaluate proactive document retrieval as follows. At every conversation turn, a proactive retrieval system can make a binary decision: whether to retrieve and show a result list to the users or not. If the system decides to pass and not engage in the conversation, there is no cost to the users and also no gain is obtained. Therefore, we only evaluate the system when it presents retrieval results to the users. We can adopt different ranking metrics for this purpose, but in this paper, we only adopt nDCG as follows. Assume that a proactive retrieval model returns a result list $D_{i}\={d_{i1},d_{i2},...,d_{ik}}$ after observing the $i$th utterance in the conversation, i.e., $u_{i}$. Assume that $r_{ij}\in{0,1,2}$ denotes the relevance label associated with $d_{ij}$,

| (1) |  | $pDCG\=\frac{1}{Z}\sum_{i\=1}^{n}\mathbbm{1}{|D_{i}|>0}*DCG(D_{i}\setminus% \bigcup_{i^{\prime}\=1}^{i-1}{D_{i^{\prime}}})$ |  |
| --- | --- | --- | --- |

where $\mathbbm{1}{|D_{i}|>0}$ means if the proactive retrieval model engages in the conversation by retrieving a result list, i.e., if the result list is not empty. $Z$ is a normalization term and is equal to the number of times that the proactive retrieval model returns a result list, meaning $Z\=\sum_{i\=1}^{n}\mathbbm{1}{|D_{i}|>0}$. The notation $D_{i}\setminus\bigcup_{i^{\prime}\=1}^{i-1}{D_{i^{\prime}}}$ means the result list returned in response to the $i$th utterance excluding any result list that has been presented to the users before. The reason for this decision is that we assume there is no value in retrieving the same document over and over in the same dialogue and we do not want to reward a retrieval model that returns the same relevant document at multiple turns. Discounted Cumulative Gain (DCG) is calculated as:

|  | $DCG(D_{i})\=\sum_{j\=1}^{k}\frac{\text{rel}(r_{ij})}{\log(j+1)}$ |  |
| --- | --- | --- |

where $r_{ij}$ is defined as follows:

|  | $\text{rel}(r_{ij})\=\begin{cases}r_{ij}\times\frac{1}{\log(1+i-(l-1))}\&\text{if%  }i\geq l\\ 0\&\text{if }i<l\end{cases}$ |  |
| --- | --- | --- |

where $l$ is the perfect utterance number for document $d_{ij}$ to appear. In fact, if document $d_{ij}$ adds value to a conversation, turn $l$ is the first utterance in which this document becomes relevant. Therefore, if $i<l$ the document is non-relevant. Otherwise, if $i\=l$, then, the model should not be penalized, thus $\text{rel}(r_{ij})\=r_{ij}\times\frac{1}{\log(2)}\=r_{ij}$. If $i>l$, this means that the model has a delay in presenting a relevant document to its users and thus needs to be penalized. Inspired by the DCG formulation, we use logarithm as a concave penalization function. We next normalize $pDCG$ values as follows:

| (2) |  | $npDCG\=\frac{pDCG}{ipDCG}$ |  |
| --- | --- | --- | --- |

where $ipDCG$ denotes ideal $pDCG$ and represents the highest value that $pDCG$ can obtain for a given utterance. Therefore, $npDCG\in[0,1]$.
Note that $ipDCG$ is obtained by a model that for every turn retrieves all relevant documents, sorted by their relevance score, for that turn and if there is no relevant document associated with that turn, it skips retrieval. Figure[6] presents a toy example to explain how npDCG values are computed.

<img src='extracted/2405.06460v1/images/npdcg.png' alt='Refer to caption' title='' width='281' height='230' />

*Figure 6. Examples of rank lists and their npDCG scores based on various positions of the doc with ideal position in utterance 3 (with dashed border). Green, gray and red represent relevant, partially relevant and irrelevant docs respectively.*

<img src='extracted/2405.06460v1/images/lmgr.png' alt='Refer to caption' title='' width='461' height='208' />

*Figure 7. An overview of the Language Model Grounded Retrieval (LMGR) framework.*

6. Retrieval Methods
---------------------

To provide benchmark results, we evaluate a wide range of methods on our datasets. Four of them are well-known existing methods from different categories, such as term-matching, neural sparse retrieval, neural single-vector dense retrieval, and neural multi-vector dense retrieval. We also introduce a novel approach specifically designed for this task that deals with a long sequence as input, called LMGR.

BM25 is a widely used ranking function that improves upon TF-IDF by incorporating term frequency saturation and document length normalization. It is the most common baseline in information retrieval tasks. For BM25 retrieval in our experiments, we used tantivy.888[https://github.com/quickwit-oss/tantivy](https://github.com/quickwit-oss/tantivy "")

SPLADE is a neural retrieval model that uses sparse representations for documents and queries. It is based on explicit sparsity regularization and a log-saturation effect on term weights, leading to highly sparse representations. This model offers a trade-off between effectiveness and efficiency, and is trained end-to-end in a single stage *(Formal et al., [2021])*. For SPLADE, as well as ColBERT below, we relied on the implementations from the tevatron library*(Gao et al., [2022])* for our experiments.

ANCE is a dual-encoder neural retrieval model that uses dense representations to measure the similarity between queries and documents. It addresses the discrepancy between the data distribution used in training and testing by using an Approximate Nearest Neighbor (ANN) index of the corpus to select more realistic negative training instances *(Xiong et al., [2021])*. We used the official implementation999<https://github.com/microsoft/ANCE> for training.

ColBERT is multi-vector dense retrieval model. It introduces a late interaction architecture that independently encodes the query and the document using BERT*(Devlin et al., [2019])* and then employs a cheap yet powerful interaction step that models their similarity *(Khattab and Zaharia, [2020])*.

Language Model Grounded Retrieval. The Language Model Grounded Retrieval (LMGR) framework, which we propose for this task, is inspired by the two-stage zero-shot entity linking approach of the BLINK model *(Wu et al., [2020])*. For each entity mentioned in a text, BLINK first retrieves some candidates using a dense retrieval model and then re-ranks them with a cross-encoder to pick the final result. Large Language Models have a good memorization of the Wikipedia corpus and can be very effective in generating relevant concepts to given complex queries. However, they can hallucinate and produce entities that are not actually in the Wikipedia corpus, or they can generate titles that are not exact matches. Applying a linking methodology like BLINK can solve this issue.

The LMGR framework consists of three stages: top-n candidate generation, top-k candidate retrieval, and grounding (Figure [7]).

* •

    Candidate Generation In the first stage, the LMGR framework uses a large language model (LLM) to generate top-n candidates. The LLM is trained to predict the next word in a sentence, given the previous words. This allows it to generate a list of potential candidates using some separation format which could be relevant to the conversation. The LLM is capable of understanding the context of the conversation and generating candidates that are not only relevant but also diverse, addressing the complexity and diversity of open-domain conversations to a high extent. For each candidate, we prompt the LLM to generate pairs of title and one sentence descriptions.

* •

    Candidate Retrieval The second stage involves top-k candidate retrieval from each generated candidate from the LLM. This stage is crucial for identifying the corpus items that are closest to the generated candidate. For this stage, we employ dense retrieval using pre-trained sentence embeddings[7] from a corpus of Wikipedia title-description pairs, where description is limited to the first sentence of each Wikipedia article.

* •

    Grounding The third stage of the LMGR framework is the final candidate selection or grounding. In this stage, the LLM or a re-ranker is used to select the final candidate from the top-k candidates retrieved in the previous stage. The selected candidate is the one that the LLM or re-ranker determines to be the one most likely describing the same concept with the generated candidate.

In our experiments, we used the same LLM for both candidate generation and grounding. The LLM we used is based on the OpenChat-3.5 *(Wang et al., [2023])*, which is a fine-tuned version of Mistral-7B *(Jiang et al., [2023])*. We prompt the model to generate up to 20 candidates and experiment with retrieving and grounding 1, 3 and 5 results.

### 6.1. Proactive Retrieval

In addition to the retrieval methods mentioned above, we also employ a proactive retrieval approach that decides when to retrieve documents based on the current utterance and the conversation history. This is achieved using a binary classifier based on DeBERTa *(He et al., [2023])* that predicts whether a given utterance requires document retrieval or not. The classifier is trained on pairs sampled in a balanced way from the training set. For each positive pair (an utterance with associated relevant documents), we randomly sample an utterance without associated relevant documents to create a negative pair. This balanced sampling helps the classifier learn to distinguish between utterances that require retrieval and those that do not.

The proactive retrieval classifier is applied before the actual retrieval methods. If the classifier predicts that an utterance requires retrieval, the selected retrieval method (e.g. ColBERT or LMGR) is then used to fetch relevant documents. This approach helps to reduce unnecessary retrieval for utterances that do not require additional information, thereby improving the efficiency and effectiveness of the overall system.

7. Benchmark Results
---------------------

#### Reactive Retrieval

Table[3] shows the experimental results for reactive retrieval. BM25 is the worst performing model for this task and is expected, because term matching is far from sufficient to provide articles with useful context. Only for frequent entity mentions in a conversation this model might work to an extent. From the supervised neural models, ColBERT has the best performance, however it still struggles to achieve good recall metrics in the reactive setting. On the other hand, our proposed LMGR framework outperforms all the other models by a big margin in reactive retrieval, achieving very promising recall metrics as well with recall@5 up to 28.53% and recall@20 53.06%. This is an insightful finding, because this is one of the first retrieval tasks where zero-shot LLMs outperform existing state-of-the-art retrieval models by such a big margin. This showcases the potential of using LLMs directly for retrieval instead of using traditional techniques that involve vector representation and scoring. Generative retrieval models *(Tay et al., [2022]; Zeng et al., [2024])* are already following this path.

#### Proactive Retrieval

Table[4] shows the experimental results for proactive retrieval. In contrast to the reactive retrieval setting, traditional retrieval models seem to outperform LMGR in proactive retrieval. This shows that LMGR is only suited for high-level understanding of a conversation and without fine-tuning can be unstable for proactive retrieval. The supervised neural models, particularly ColBERT, demonstrate better performance in the proactive setting compared to LMGR. This suggests that while LLMs excel in understanding the context and retrieving relevant articles reactively, they may struggle to anticipate future information needs without further adaptation to the specific task of proactive retrieval.

*Table 3. Experimental results for reactive retrieval on the ProCIS test set. The top section is for baselines and the bottom is our proposed LMGR framework. The superscript * denotes statistically significant improvements compared to all the baselines. $k$ is the number of retrieved candidates. Note that LMGR produces up to 20 results.*

| Model | nDCG@5 | nDCG@20 | nDCG@100 | MRR | MAP | R@5 | R@20 | R@100 | R@1K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.0654 | 0.0754 | 0.0969 | 0.1561 | 0.0395 | 0.0410 | 0.0687 | 0.1202 | 0.2266 |
| SPLADE | 0.1605 | 0.1578 | 0.1575 | 0.4752 | 0.0752 | 0.0946 | 0.1343 | 0.1432 | 0.2946 |
| ANCE | 0.1854 | 0.1912 | 0.2240 | 0.4902 | 0.0984 | 0.0989 | 0.1635 | 0.2517 | 0.4316 |
| ColBERT | 0.2091 | 0.2094 | 0.2383 | 0.5679 | 0.1113 | 0.1117 | 0.1778 | 0.2649 | 0.4564 |
| LMGR, k\=1 | 0.2638 | 0.3678 | - | 0.6187 | 0.2000 | 0.2116 | 0.4091 | - | - |
| LMGR, k\=3 | 0.2714 | 0.3986 | - | 0.6132 | 0.2198 | 0.2354 | 0.4614 | - | - |
| LMGR, k\=5 | 0.3408∗ | 0.4524∗ | - | 0.6300∗ | 0.2663∗ | 0.2853∗ | 0.5306∗ | - | - |

*Table 4. Experimental results for proactive retrieval on the ProCIS test set using a DeBERTa-base proactive classifier. npDCG is the metric we defined for conversational proactive retrieval evaluation. The superscript * denotes statistically significant improvements compared to all the baselines. Note that LMGR produces up to 20 results.*

| Model | npDCG@5 | npDCG@20 | npDCG@100 |
| --- | --- | --- | --- |
| BM25 | 0.0229 | 0.0337 | 0.0405 |
| SPLADE | 0.1305 | 0.1440 | 0.1542 |
| ANCE | 0.1508 | 0.1792 | 0.2061 |
| ColBERT | 0.1719 | 0.1944 | 0.2172 |
| LMGR, k\=1 | 0.0574 | 0.1445 | - |
| LMGR, k\=3 | 0.0613 | 0.1527 | - |
| LMGR, k\=5 | 0.0781 | 0.1840 | - |

8. Future Research Directions
------------------------------

The ProCIS dataset serves as a foundation and starting point for future research in proactive CIS. However, there are several promising avenues for future work that can be explored.

Development of Proactive Retrieval Models The unique nature of conversations, with their length, complexity, and domain diversity, presents a challenge for traditional retrieval models. Future research could focus on developing new models that are specifically designed to handle these challenges. For instance, the LMGR framework currently relies on large language models (LLMs), which can be computationally expensive. However, it’s possible that smaller more cost-effective language models, trained on synthetic data, could perform just as well, if not better. Future research could also focus more on pro-activeness, exploring when to retrieve documents and how many to select.

Improvement of Dense Retrieval Models The potential for improvement in dense retrieval models is another promising area for future research. For example, a conversation encoder and a document encoder could be pre-trained separately and then fine-tuned on the target dataset. This could improve the model’s ability to understand and respond to conversational context. Additionally, pre-training techniques for the conversation encoder, such as response ranking or response masking tasks, could be investigated. These techniques could further enhance the encoder’s understanding of conversational dynamics.

Advanced Pooling Methods To capture the content of longer conversations more effectively, advanced pooling methods could be explored. Techniques such as averaging, attention mechanisms over utterance-level representations, and content filtering could be employed. These methods could help to distill the most important information from lengthy conversations, improving the model’s ability to respond appropriately.

Explainability The utility of suggested concepts could be improved by generating explanations that clarify their relevance and importance within the context of the conversation. This could help users to better understand why certain concepts are being suggested, improving their overall experience.

Query Generation In addition to concept suggestions, the generation of natural language queries could be explored. This could further facilitate the information seeking process, making it easier for users to find the information they need.

Synthetic Data Generation The potential of LLMs for synthetic data generation is another area that could be investigated. Understanding how synthetic annotations can improve the performance of retrieval and generative models could be beneficial. This could lead to more accurate and efficient models.

Generative Retrieval Models Finally, experimenting with generative retrieval models *(Tay et al., [2022]; Zeng et al., [2024])* for this task could also be a promising direction for future work. These models could potentially provide more accurate and relevant responses, improving the overall user experience.

9. Conclusions
---------------

In this paper, we have introduced ProCIS, a large-scale dataset for proactive conversational information seeking collected from Reddit threads and enriched with external links to Wikipedia articles. ProCIS addresses a significant gap in the research landscape, providing a standardized benchmark for the development and evaluation of proactive retrieval models in the context of open-domain conversations. The dataset consists of over 2.8 million multi-party conversations, offering a rich resource for exploring the complexities and challenges of proactive IR in conversational settings.

We also proposed the Language Model Grounded Retrieval framework (LMGR) as a baseline for this new task. Despite being a zero-shot method in our current experiments, LMGR outperforms existing ad-hoc retrieval models by a significant margin in the reactive setting, showing that after optimization this might be a very effective approach in the proactive setting as well. This also showcases the potential of using LLMs directly for retrieval instead of using traditional techniques that involve embedding and scoring.

The ProCIS dataset represents a significant step forward in the advancement of conversational agents capable of proactively seeking out and providing useful information to users. We hope that this dataset will inspire further research in the area of proactive conversational search and lead to the emergence of new techniques and approaches that will enhance user experiences in conversations and unlock the full potential of conversational LLM agents in various domains.

Acknowledgments
---------------

This work was supported in part by the Center for Intelligent Information Retrieval, in part by the Office of Naval Research contract number N000142212688, in part by NSF grant number 2143434, in part by the Amazon Alexa Prize Competition, and in part by an award from Adobe. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.

References
----------

* (1)
* Ahmed and Bulathwela (2022)Tabish Ahmed and Sahan Bulathwela. 2022.Towards Proactive Information Retrieval in Noisy Text with Wikipedia Concepts.*arXiv preprint arXiv:2210.09877* (2022).
* Aliannejadi et al. (2021)Mohammad Aliannejadi, Leif Azzopardi, Hamed Zamani, Evangelos Kanoulas, Paul Thomas, and Nick Craswell. 2021.Analysing Mixed Initiatives and Search Strategies during Conversational Search. In *Proceedings of the 30th ACM International Conference on Information \& Knowledge Management* (Virtual Event, Queensland, Australia) *(CIKM ’21)*. Association for Computing Machinery, New York, NY, USA, 16–26.[https://doi.org/10.1145/3459637.3482231](https://doi.org/10.1145/3459637.3482231 "")
* Aliannejadi et al. (2019)Mohammad Aliannejadi, Hamed Zamani, Fabio Crestani, and W. Bruce Croft. 2019.Asking Clarifying Questions in Open-Domain Information-Seeking Conversations. In *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval* (Paris, France) *(SIGIR’19)*. Association for Computing Machinery, New York, NY, USA, 475–484.[https://doi.org/10.1145/3331184.3331265](https://doi.org/10.1145/3331184.3331265 "")
* Andolina et al. (2018)Salvatore Andolina, Valeria Orso, Hendrik Schneider, Khalil Klouche, Tuukka Ruotsalo, Luciano Gamberini, and Giulio Jacucci. 2018.Investigating Proactive Search Support in Conversations. In *Proceedings of the 2018 Designing Interactive Systems Conference* (Hong Kong, China) *(DIS ’18)*. Association for Computing Machinery, New York, NY, USA, 1295–1307.[https://doi.org/10.1145/3196709.3196734](https://doi.org/10.1145/3196709.3196734 "")
* Arampatzis (2001)Avi Arampatzis. 2001.Unbiased sd threshold optimization, initial query degradation, decay, and incrementality, for adaptive document filtering.. In *TREC*.
* Besançon et al. (2009)Romaric Besançon, Stéphane Chaudiron, Djamel Mostefa, Olivier Hamon, Ismaïl Timimi, and Khalid Choukri. 2009.Overview of CLEF 2008 INFILE Pilot Track. In *Evaluating Systems for Multilingual and Multimodal Information Access*, Carol Peters, Thomas Deselaers, Nicola Ferro, Julio Gonzalo, Gareth J. F. Jones, Mikko Kurimo, Thomas Mandl, Anselmo Peñas, and Vivien Petras (Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 939–946.
* Braslavski et al. (2017)Pavel Braslavski, Denis Savenkov, Eugene Agichtein, and Alina Dubatovka. 2017.What Do You Mean Exactly? Analyzing Clarification Questions in CQA. In *Proceedings of the 2017 Conference on Conference Human Information Interaction and Retrieval* (Oslo, Norway) *(CHIIR ’17)*. Association for Computing Machinery, New York, NY, USA, 345–348.[https://doi.org/10.1145/3020165.3022149](https://doi.org/10.1145/3020165.3022149 "")
* Choi et al. (2018)Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018.QuAC: Question Answering in Context. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, Brussels, Belgium, 2174–2184.[https://doi.org/10.18653/v1/D18-1241](https://doi.org/10.18653/v1/D18-1241 "")
* Dalton et al. (2019)Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2019.TREC CAsT 2019: The conversational assistance track overview.*TREC* (2019).
* Devlin et al. (2019)Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186.[https://doi.org/10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423 "")
* Formal et al. (2021)Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021.SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval* (Virtual Event, Canada) *(SIGIR ’21)*. Association for Computing Machinery, New York, NY, USA, 2288–2292.[https://doi.org/10.1145/3404835.3463098](https://doi.org/10.1145/3404835.3463098 "")
* Ganguly et al. (2020)Debasis Ganguly, Dipasree Pal, Manisha Verma, and Procheta Sen. 2020.Overview of RCD-2020, the FIRE-2020 track on Retrieval from Conversational Dialogues. In *FIRE 2020: Forum for Information Retrieval Evaluation, Hyderabad, India, December 16-20, 2020*, Prasenjit Majumder, Mandar Mitra, Surupendu Gangopadhyay, and Parth Mehta (Eds.). ACM, 33–36.[https://doi.org/10.1145/3441501.3441518](https://doi.org/10.1145/3441501.3441518 "")
* Gao et al. (2022)Luyu Gao, Xueguang Ma, Jimmy J. Lin, and Jamie Callan. 2022.Tevatron: An Efficient and Flexible Toolkit for Dense Retrieval.*ArXiv* abs/2203.05765 (2022).
* Gauch et al. (2007)Susan Gauch, Mirco Speretta, Aravind Chandramouli, and Alessandro Micarelli. 2007.*User Profiles for Personalized Information Access*.Springer Berlin Heidelberg, Berlin, Heidelberg, 54–89.[https://doi.org/10.1007/978-3-540-72079-9_2](https://doi.org/10.1007/978-3-540-72079-9_2 "")
* Hattimare et al. (2023)Amit Hattimare, Arkin Dharawat, Yelman Khan, Yen-Chieh Lien, Chris Samarinas, George Z. Wei, Yulin Yang, and Hamed Zamani. 2023.MarunaBot: Multi-Modal Augmentation for Large Language Models with Applications to Task-Oriented Dialogues. In *Alexa Prize TaskBot Challenge 2 Proceedings*.[https://www.amazon.science/alexa-prize/proceedings/marunabot-v2-towards-end-to-end-multi-modal-task-oriented-dialogue-systems](https://www.amazon.science/alexa-prize/proceedings/marunabot-v2-towards-end-to-end-multi-modal-task-oriented-dialogue-systems "")
* He et al. (2023)Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023.DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net.[https://openreview.net/pdf?id\=sE7-XhLxHA](https://openreview.net/pdf?id=sE7-XhLxHA "")
* Hinton and Roweis (2002)Geoffrey E Hinton and Sam Roweis. 2002.Stochastic neighbor embedding.*Advances in neural information processing systems* 15 (2002).
* Jiang et al. (2023)Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023.Mistral 7B.arXiv:2310.06825 [cs.CL]
* Khattab and Zaharia (2020)Omar Khattab and Matei Zaharia. 2020.ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval* (Virtual Event, China) *(SIGIR ’20)*. Association for Computing Machinery, New York, NY, USA, 39–48.[https://doi.org/10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075 "")
* Kumar and Callan (2020)Vaibhav Kumar and Jamie Callan. 2020.Making Information Seeking Easier: An Improved Pipeline for Conversational Search. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 3971–3980.[https://doi.org/10.18653/v1/2020.findings-emnlp.354](https://doi.org/10.18653/v1/2020.findings-emnlp.354 "")
* Liao et al. (2023)Lizi Liao, Grace Hui Yang, and Chirag Shah. 2023.Proactive Conversational Agents. In *Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining* (Singapore, Singapore) *(WSDM ’23)*. Association for Computing Machinery, New York, NY, USA, 1244–1247.[https://doi.org/10.1145/3539597.3572724](https://doi.org/10.1145/3539597.3572724 "")
* Mo et al. (2024)Fengran Mo, Chen Qu, Kelong Mao, Tianyu Zhu, Zhan Su, Kaiyu Huang, and Jian-Yun Nie. 2024.History-Aware Conversational Dense Retrieval.arXiv:2401.16659 [cs.IR]
* Nguyen et al. (2016)Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016.MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.(November 2016).[https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/](https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/ "")
* OpenAI (2023)OpenAI. 2023.GPT-4 Technical Report.arXiv:2303.08774 [cs.CL]
* Owoicho et al. (2022)Paul Owoicho, Jeffrey Dalton, Mohammad Aliannejadi, Leif Azzopardi, Johanne R Trippas, and Svitlana Vakulenko. 2022.TREC CAsT 2022: Going beyond user ask and system retrieve with initiative and response generation.*NIST Special Publication* (2022), 500–338.
* Penha et al. (2019)Gustavo Penha, Alexandru Balan, and Claudia Hauff. 2019.Introducing mantis: a novel multi-domain information seeking dialogues dataset.*arXiv preprint arXiv:1912.04639* (2019).
* Qu et al. (2020)Chen Qu, Liu Yang, Cen Chen, Minghui Qiu, W. Bruce Croft, and Mohit Iyyer. 2020.Open-Retrieval Conversational Question Answering. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval* (Virtual Event, China) *(SIGIR ’20)*. Association for Computing Machinery, New York, NY, USA, 539–548.[https://doi.org/10.1145/3397271.3401110](https://doi.org/10.1145/3397271.3401110 "")
* Radlinski and Craswell (2017)Filip Radlinski and Nick Craswell. 2017.A Theoretical Framework for Conversational Search. In *Proceedings of the 2017 Conference on Conference Human Information Interaction and Retrieval* (Oslo, Norway) *(CHIIR ’17)*. Association for Computing Machinery, New York, NY, USA, 117–126.[https://doi.org/10.1145/3020165.3020183](https://doi.org/10.1145/3020165.3020183 "")
* Rao and Daumé III (2018)Sudha Rao and Hal Daumé III. 2018.Learning to Ask Good Questions: Ranking Clarification Questions using Neural Expected Value of Perfect Information. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics, Melbourne, Australia, 2737–2746.[https://doi.org/10.18653/v1/P18-1255](https://doi.org/10.18653/v1/P18-1255 "")
* Reddy et al. (2019)Siva Reddy, Danqi Chen, and Christopher D. Manning. 2019.CoQA: A Conversational Question Answering Challenge.*Transactions of the Association for Computational Linguistics* 7 (2019), 249–266.[https://doi.org/10.1162/tacl_a_00266](https://doi.org/10.1162/tacl_a_00266 "")
* Robertson and Soboroff (2003)S Robertson and Ian Soboroff. 2003.The TREC-2002 Filtering Track Report.[https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id\=50769](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50769 "")
* Ros et al. (2023)Kevin Ros, Matthew Jin, Jacob Levine, and ChengXiang Zhai. 2023.Retrieving Webpages Using Online Discussions. In *Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval* (Taipei, Taiwan) *(ICTIR ’23)*. Association for Computing Machinery, New York, NY, USA, 159–168.[https://doi.org/10.1145/3578337.3605139](https://doi.org/10.1145/3578337.3605139 "")
* Samarinas et al. (2024)Chris Samarinas, Pracha Promthaw, Atharva Nijasure, Hansi Zeng, Julian Killingback, and Hamed Zamani. 2024.Simulating Task-Oriented Dialogues with State Transition Graphs and Large Language Models.arXiv:2404.14772 [cs.CL]
* Sekulić et al. (2022)Ivan Sekulić, Mohammad Aliannejadi, and Fabio Crestani. 2022.Evaluating Mixed-initiative Conversational Search Systems via User Simulation. In *Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining* (Virtual Event, AZ, USA) *(WSDM ’22)*. Association for Computing Machinery, New York, NY, USA, 888–896.[https://doi.org/10.1145/3488560.3498440](https://doi.org/10.1145/3488560.3498440 "")
* Sen et al. (2018)Procheta Sen, Debasis Ganguly, and Gareth Jones. 2018.Procrastination is the Thief of Time: Evaluating the Effectiveness of Proactive Search Systems. In *The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval* (Ann Arbor, MI, USA) *(SIGIR ’18)*. Association for Computing Machinery, New York, NY, USA, 1157–1160.[https://doi.org/10.1145/3209978.3210114](https://doi.org/10.1145/3209978.3210114 "")
* Shah (2018)Chirag Shah. 2018.Information fostering-being proactive with information seeking and retrieval: Perspective paper. In *Proceedings of the 2018 conference on human information interaction \& retrieval*. 62–71.
* Stoyanchev et al. (2014)Svetlana Stoyanchev, Alex Liu, and Julia Hirschberg. 2014.Towards Natural Clarification Questions in Dialogue Systems. In *AISB ’14*, Vol. 20.
* Tay et al. (2022)Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W Cohen, and Donald Metzler. 2022.Transformer Memory as a Differentiable Search Index. In *Advances in Neural Information Processing Systems*, Vol. 35. Curran Associates, Inc., 21831–21843.[https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf "")
* Touvron et al. (2023)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023.LLaMA: Open and Efficient Foundation Language Models.arXiv:2302.13971 [cs.CL]
* Wadhwa and Zamani (2021)Somin Wadhwa and Hamed Zamani. 2021.Towards System-Initiative Conversational Information Seeking.. In *DESIRES*. 102–116.
* Wang et al. (2023)Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2023.Openchat: Advancing open-source language models with mixed-quality data.*arXiv preprint arXiv:2309.11235* (2023).
* Wu et al. (2020)Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2020.Scalable Zero-shot Entity Linking with Dense Entity Retrieval. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 6397–6407.[https://doi.org/10.18653/v1/2020.emnlp-main.519](https://doi.org/10.18653/v1/2020.emnlp-main.519 "")
* Xiong et al. (2021)Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021.Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net.[https://openreview.net/forum?id\=zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln "")
* Yu et al. (2021)Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, and Zhiyuan Liu. 2021.Few-Shot Conversational Dense Retrieval. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval* (Virtual Event, Canada) *(SIGIR ’21)*. Association for Computing Machinery, New York, NY, USA, 829–838.[https://doi.org/10.1145/3404835.3462856](https://doi.org/10.1145/3404835.3462856 "")
* Zamani et al. (2020a)Hamed Zamani, Susan Dumais, Nick Craswell, Paul Bennett, and Gord Lueck. 2020a.Generating Clarifying Questions for Information Retrieval. In *Proceedings of The Web Conference 2020* (Taipei, Taiwan) *(WWW ’20)*. Association for Computing Machinery, New York, NY, USA, 418–428.[https://doi.org/10.1145/3366423.3380126](https://doi.org/10.1145/3366423.3380126 "")
* Zamani et al. (2020b)Hamed Zamani, Gord Lueck, Everest Chen, Rodolfo Quispe, Flint Luu, and Nick Craswell. 2020b.MIMICS: A Large-Scale Data Collection for Search Clarification. In *Proceedings of the 29th ACM International Conference on Information \& Knowledge Management*. Association for Computing Machinery, New York, NY, USA, 3189–3196.[https://doi.org/10.1145/3340531.3412772](https://doi.org/10.1145/3340531.3412772 "")
* Zamani et al. (2020c)Hamed Zamani, Bhaskar Mitra, Everest Chen, Gord Lueck, Fernando Diaz, Paul N. Bennett, Nick Craswell, and Susan T. Dumais. 2020c.Analyzing and Learning from User Interactions for Search Clarification. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval* (Virtual Event, China) *(SIGIR ’20)*. Association for Computing Machinery, New York, NY, USA, 1181–1190.[https://doi.org/10.1145/3397271.3401160](https://doi.org/10.1145/3397271.3401160 "")
* Zamani and Shakery (2018)Hamed Zamani and Azadeh Shakery. 2018.A language model-based framework for multi-publisher content-based recommender systems.*Information Retrieval Journal* 21 (2018), 369–409.[https://doi.org/10.1007/s10791-018-9327-0](https://doi.org/10.1007/s10791-018-9327-0 "")
* Zamani et al. (2023)Hamed Zamani, Johanne R. Trippas, Jeff Dalton, and Filip Radlinski. 2023.Conversational Information Seeking.*Foundations and Trends® in Information Retrieval* 17, 3-4 (2023), 244–456.[https://doi.org/10.1561/1500000081](https://doi.org/10.1561/1500000081 "")
* Zeng et al. (2024)Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, and Hamed Zamani. 2024.Scalable and Effective Generative Information Retrieval. In *Proceedings of The Web Conference 2024* (Singapore, Singapore) *(WWW ’24)*. Association for Computing Machinery, New York, NY, USA.
