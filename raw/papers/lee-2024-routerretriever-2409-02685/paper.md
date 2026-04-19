RouterRetriever: Routing over a Mixture of Expert Embedding Models
==================================================================

Hyunji Lee1 Luca Soldaini2 Arman Cohan2, 3 Minjoon Seo1 Kyle Lo2Work performed during internship at Ai2.

###### Abstract

Information retrieval methods often rely on a single embedding model trained on large, general-domain datasets like MSMARCO. While this approach can produce a retriever with reasonable overall performance, they often underperform models trained on domain-specific data when testing on their respective domains. Prior work in information retrieval has tackled this through multi-task training, but the idea of routing over a mixture of domain-specific expert retrievers remains unexplored despite the popularity of such ideas in language model generation research.
In this work, we introduce RouterRetriever, a retrieval model that leverages a mixture of domain-specific experts by using a routing mechanism to select the most appropriate expert for each query. RouterRetriever is lightweight and allows easy addition or removal of experts without additional training.
Evaluation on the BEIR benchmark demonstrates that RouterRetriever outperforms both models trained on MSMARCO (+2.1 absolute nDCG@10) and multi-task models (+3.2). This is achieved by employing our routing mechanism, which surpasses other routing techniques (+1.8 on average) commonly used in language modeling.
Furthermore, the benefit generalizes well to other datasets, even in the absence of a specific expert on the dataset. RouterRetriever is the first work to demonstrate the advantages of routing over a mixture of domain-specific expert embedding models as an alternative to a single, general-purpose embedding model, especially when retrieving from diverse, specialized domains.

\faGithub

Code [github/amy-hyunji/RouterRetriever](https://github.com/amy-hyunji/RouterRetriever "")  
<img src='extracted/6233898/figures/hugging-face_1f917.png' alt='[Uncaptioned image]' title='' width='14' height='14' /> Weights [hf.co/amy-hyunji/RouterRetriever](https://huggingface.co/amy-hyunji-lee/routerretriever "")

1 Introduction
--------------

Domain-specific retrievers have been shown to outperform general-purpose retrievers for specialized retrieval settings*(Izacard et al. [2021]; Bonifacio et al. [2022])*, even in cases where domain-specific training data is only available at a much smaller scale than general-domain datasets like MSMARCO*(Campos et al. [2016])*.
Yet, developing and maintaining separate retrieval systems for each specialized retrieval domain can be costly compared to simply maintaining a single general-purpose MSMARCO-trained model.
Even employing multi-task training, which combines both MSMARCO as well as domain-specific data, to improve performance of the single model setup can be expensive when considering the need for full model retraining whenever a new target retrieval domain emerges, and may not always preserve performance uniformly across all target domains*(Wang et al. [2023]; Lee et al. [2023])*.
Research has largely focused on improving the performance of these single model setups through data construction*(Wang et al. [2021]; Ma et al. [2020])* and domain adaptation*(Xin et al. [2021]; Fang et al. [2024])*.
But less attention has been paid to what can be done if we afford ourselves to use *multiple* specialized retrieval models.

In this work, we introduce RouterRetriever, a retrieval model that leverages a mixture of domain-specific experts with a routing mechanism to select the most suitable expert for each instance. RouterRetriever consists of a shared base retrieval model and multiple LoRA*(Hu et al. [2021])* components which serves as experts for specific domains.
During training, each expert is trained on a domain-specific dataset while sharing the same frozen base model.
Thus the expert component captures domain-specific knowledge and extracts embeddings tailored to the domain.
At inference time, as shown in Figure[1], our routing method determines the most appropriate expert by calculating the average similarity between the query and a set of pilot embeddings representing each expert.
The expert with the highest similarity score is selected, and the corresponding domain-specific embedding is generated using the chosen expert. RouterRetriever is lightweight, as it only requires the training of a parameter-efficient LoRA module for each expert, resulting in a minimal increase in parameters. Additionally, RouterRetriever offers significant flexibility: unlike maintaining a single model that requires retraining when domains are added or removed, RouterRetriever simply adds or removes experts without the need for further training of the rest of the model.

We demonstrate the effectiveness of RouterRetriever on the BEIR benchmark*(Thakur et al. [2021])* through a series of experiments with various combinations of target domains:
First, RouterRetriever routing between only domain-specific experts outperforms both training a single model on the same dataset in a multi-task manner and training a single model on MSMARCO only.
Second, we observe that routing techniques from language modeling research don’t necessarily translate to our retrieval setting, which motivates us to introduce a new, specialized routing technique based on embedding similarities.
Third, RouterRetriever consistently improves performance as we add new experts, whereas multi-task training tends to show performance degradation after a certain number of target domains are included.
In fact, even without an expert defined for a target domain, RouterRetriever can outperform a single general-purpose model on those unseen domains.

<img src='x1.png' alt='Refer to caption' title='' width='830' height='478' />

*Figure 1: RouterRetriever: ① Given a query, we first extract its embedding using a base encoder. We then calculate an average similarity between the query embedding (black dot) and the pilot embeddings for each expert (orange dots for Expert A, red dots for Expert B, and blue dots for Expert C). The expert with the highest average similarity (Expert A in this case) is selected. ② The final query embedding is then produced by passing the query to Expert Encoder A, which consists of the base encoder combined with the selected expert LoRA module.*

We also conduct analysis to better understand the factors behind these performance benefits.
First, to understand data scaling benefits when training each individual expert, we find in-domain performance rapidly increases as we increase training set size, whereas out-of-domain performance often may not see any improvement with scale; hence, the need to use multiple experts.
Second, we show that RouterRetriever continually improves as the number of experts increases but quickly runs into diminishing returns.
These improvements aren’t just with respect to overall performance, but also with respect to performance stability across domains.
Third, when analyzing routing errors, RouterRetriever tends to have a sparser expert selection compared to our instance-level oracle setup.

2 Related Works
---------------

#### Domain Specific Retriever

Substantial research on retrieval models aims to improve performance on domain-specific tasks.
One approach focuses on dataset augmentation. As domain-specific training datasets are often unavailable and can be costly to construct, researchers have developed methods that either train models in an unsupervised manner*(Lee, Chang, and Toutanova [2019]; Gao, Yao, and Chen [2021]; Gao and Callan [2021])* or fine-tune models on pseudo-queries generated for domain-specific datasets*(Bonifacio et al. [2022]; Ma et al. [2020]; Wang et al. [2021])*.
Another approach is developing domain-specific embeddings. A common approach is training in a multi-task manner over domain-specific datasets*(Lin et al. [2023b]; Wang et al. [2021])*.
Recent works have aimed to improve domain-specific retrievers by developing instruction-following retrieval models*(Asai et al. [2022]; Weller et al. [2024]; Oh et al. [2024]; Su et al. [2022]; Wang et al. [2023])*; instruction contains such domain knowledge.
Another example is *Fang et al. ([2024])* which trains a soft token for domain-specific knowledge.
While these methods also aim to generate high-quality domain-specific embeddings, they focus on incorporating domain-specific knowledge into the input and processing it with a *single* embedding model.
In contrast, RouterRetriever employs a mixture of *multiple* embedding models, encoding domain knowledge directly into their parametric representations to produce more effective embeddings.

#### Routing Techniques

Various works have focused on developing domain-specific experts and routing mechanisms to improve general performance in generation tasks.
One approach simultaneously trains experts and the routing mechanism*(Sukhbaatar et al. [2024]; Muqeeth et al. [2024])*.
Another line of work includes post-hoc techniques that do not require additional training for routing.
Some approaches use the model itself as the knowledge source by training it on domain-specific knowledge*(Feng et al. [2023])*, incorporate domain-specific knowledge in the token space*(Belofsky [2023]; Shen et al. [2024])*, or select the most relevant source from a sampled training dataset of each domain*(Ye et al. [2022]; Jang et al. [2023])*.
Routing techniques have also been investigated for improving generation quality in retrieval-augmented generation tasks; *Mallen et al. ([2022])* explores routing to decide whether to use external knowledge and *Jeong et al. ([2024])* focuses on routing to choose among different retrieval approaches.
In our work, we observe that directly adapting routing techniques from generation tasks to retrieval does not yield optimal performance.
To address this, we introduce a routing technique specifically tailored to retrieval tasks.
In information retrieval, *(Lin et al. [2023a])* introduces a technique that decomposes long and complex queries into sub-queries, which are then routed to specialized expert retrievers. Unlike our work, which employs lightweight components as experts, they rely on separate, individual expert models.
Further, while they assign sub-queries to different experts in a rules-based manner, our method processes the entire query and applies dynamic routing.

3 Router Retriever
------------------

*Algorithm 1  Constructing Pilot Embedding Library*

0:Domain-specific training datasets $D_{1},\dots,D_{T}$, experts $\mathcal{E}\={e_{1},\dots,e_{T}}$

1:Initialize empty map $\mathcal{P}\={}$ for the pilot embedding library

2: foreach dataset $D_{i}$ in ${D_{1},\dots,D_{T}}$do

3:Initialize an empty list $\mathcal{L}_{i}\=[\ ]$

4: foreach instance $x_{j}$ in $D_{i}$do

5:$e_{\text{max}}\=\arg\max_{e_{i}\in\mathcal{E}}\text{Performance}(e_{i},x_{j})$

6:Add pair $(x_{j},e_{\text{max}})$ to $\mathcal{L}_{i}$

7: end for

8: foreach expert $e_{m}$ in $\mathcal{E}$do

9:${Group}_{m}\={x_{j}\mid e_{\text{max}}\=e_{m}\text{ for }(x_{j},e_{\text{max}})%
\text{ in }\mathcal{L}_{i}}$

10: if${Group}_{m}$ is not emptythen

11:$\mathbf{c}_{m}\=\text{Centroid}(\text{BaseEncoder}({Group}_{m})$)

12:$\mathcal{P}[e_{m}]$.append($\mathbf{c}_{m}$)

13: end if

14: end for

15: end for

16: Output: Pilot embeddings library $\mathcal{P}$

In this section, we introduce RouterRetriever, a retrieval model composed of a base retrieval model and multiple domain-specific experts. As shown in Figure[1], for a given input query, ① the most appropriate embedding is selected using a routing mechanism. Then, ② the query embedding is generated by passing the query through the selected expert alongside the base encoder.

During training, we fix the base retrieval model and only finetune the specialized experts, one for each target domain using domain-specific training data.
We use Contriever*(Izacard et al. [2021])* as our base encoder, and our experts are parameter-efficient LoRA*(Hu et al. [2021])* modules.
We also pre-compute for each domain a set of representative *pilot embeddings* that will help us route queries to appropriate experts.
We refer to the mappings between pilot embeddings to the associated trained expert for each domain as the *pilot embedding library*.
This overall process is only performed once.
During inference, when given an input query, a *routing mechanism* determines the appropriate expert by calculating the similarity score between the input query embedding and the pilot embeddings in the pilot embedding library, and then choosing the expert with the highest average similarity score.
This design allows for the flexible addition or removal of domain-specific experts without requiring any further training of the routing mechanism.
To get into specifics:

#### Experts

For each domain $D_{i}$, where $i\=1,\ldots,T$ and $T$ is the total number of domains, we train an expert LoRA module $e_{i}$ using the corresponding domain dataset. After the training step, we have a total of $T$ different experts, $\mathcal{E}\={e_{1},e_{2},\ldots,e_{T}}$, with each expert $e_{i}$ specialized for a specific domain.

#### Pilot Embedding Library

To construct the pilot embedding library, given a domain-specific training dataset $D_{i}\={x_{1},\dots,x_{n}}$ where $x_{j}$ is an instance in $D_{i}$, we perform inference using all experts $\mathcal{E}$ to identify which expert provides the most suitable representative embedding for each instance as shown in Algorithm[1]. For each instance $x_{j}$, we select $e_{\text{max}}$, the expert that demonstrates the highest performance, defined as $e_{\text{max}}\=\arg\max_{e_{i}\in\mathcal{E}}\text{Performance}(e_{i},x_{j})$. This process produces pairs $(x_{j},e_{\text{max}})$ for all instances in the dataset $D_{i}$.
The intuition here is that $e_{max}$ for $x_{j}$ doesn’t have to be the expert trained on the source $D_{i}$ that contains $x_{j}$.

Next, with the constructed pairs $(x_{j},e_{\text{max}})$, we group them by the ones that have the same $e_{\text{max}}$.
This results in $T$ groups, one for each domain (${Group}_{m},m\=1,\cdots,T$), where each ${Group}_{m}$ contains list of instances $x_{j}$ all sharing the same $e_{max}$.
If the ${Group}_{m}$ is not empty, we extract all embeddings for instances in the group using the base encoder (BaseModel), and
calculate the average, or centroid, embedding $\mathbf{c}_{m}$, which is taken as the pilot embedding for the domain111We also experiment with $k$-means clustering of different numbers of $k$, the number of centroid embeddings, and having a single centroid embedding ($k$\=1) yields the highest performance, as additional centroids often act as distractors. Details are in Appendix[C.1].
This results in one pilot embedding per group, yielding a maximum of $T$ pilot embeddings for the training dataset $D_{i}$. Each of these embeddings is associated with a different expert, representing the most suitable one for that domain.
When ${Group}_{m}$ is empty, we skip this step, so the number of pilot embeddings for $D_{i}$ could be less than $T$.

By repeating this process across all domain-specific training datasets $D_{1},\dots,D_{T}$, we obtain a maximum of $T^{2}$ pilot embeddings: $T$ domain-specific training datasets times $T$ groups per dataset.

#### Routing Mechanism

Given an input query, we calculate the similarity between the query embedding extracted from the base encoder and the $T^{2}$ pilot embeddings in the pilot embedding library. We then average the similarity scores for $T$ pilot embeddings associated with the same expert, resulting in a mean similarity score for each expert. The expert corresponding to the highest mean similarity score is selected as the most suitable embedding model.

4 Experimental Setup
--------------------

#### Baselines

We compare the performance of RouterRetriever with a single base encoder model trained on the same dataset in a Multi-Task manner and a single base encoder model trained on a large-scale general-domain dataset MSMARCO. Following previous works*(Muqeeth et al. [2024]; Jang et al. [2023])*, we also evaluate two oracle settings: DatasetOracle and InstanceOracle222DatasetOracle and InstanceOracle correspond to Best Individual and Oracle, respectively, in prior works *Jang et al. ([2023])* and *Muqeeth et al. ([2024])*. The DatasetOracle setting is a dataset-level oracle that routes all queries in a dataset to the expert with the highest average performance for that dataset, while the InstanceOracle setting is an instance-level oracle that routes each individual instance to its best-performing expert.

We also conduct experiments with various other routing techniques commonly used in language modeling tasks: ExpertClassifierRouter*(Shen et al. [2024])*, ClassificationHeadRouter*(Muqeeth et al. [2024])*, and DatasetRouter*(Ye et al. [2022]; Jang et al. [2023])*.
ExpertClassifierRouter employs a binary classifier for each expert to calculate the probability of selecting that expert. The expert with the highest probability is chosen for the final selection.
ClassificationHeadRouter uses a single classifier layer to determine the appropriate expert for each instance.
DatasetRouter is the most similar to RouterRetriever, as it selects the expert by retrieving the instance with the highest similarity score. However, there are two key differences: RouterRetriever uses the predicted label, whereas DatasetRouter relies on the original dataset label. Also, RouterRetriever incorporates a clustering step to group instances, while DatasetRouter randomly samples 100 instances from the training dataset.
Further details of baselines and training methods are in Appendix[A.1].

#### Dataset

We use the provided training333In our early experiments, we noted that some training datasets in BEIR were so small that domain-specific models were underperforming MSMARCO on those target domains simply due to a lack of training data. To conduct a proper study of routing over experts, we had to first ensure that the respective experts were reasonably well-trained.
As such, for our experiments, we also use the generated queries provided by BEIR at <https://huggingface.co/BeIR>. and test sets in the BEIR benchmark*(Thakur et al. [2021])*.
We inspect BEIR domain splits using embeddings from our base encoder, a pre-trained Contriever model, in Figure[2].
We observe that datasets like MSMARCO*(Campos et al. [2016])* and ArguAna*(Wachsmuth, Syed, and Stein [2018])* tend to have widely dispersed embeddings, indicative of their “general-domain” nature, while other datasets like HotpotQA*(Yang et al. [2018])*, NFCorpus*(Boteva et al. [2016])*, SciFact*(Wadden et al. [2020])*, and FiQA*(Maia et al. [2018])* tend to have compact and tightly clustered instances, indicative of their “domain-specific” nature.
Some like Quora*(Iyer, Dandekar, and Csernai [2017])* have high dispersion in their queries but have tightly clustered contexts.
Where needed, we may use acronyms for datasets: ArguAna (AR), Quora (QU), MSMARCO (MS), HotpotQA (HO), SciFact (SF), NFCorpus (NF), FiQA (FI), SciDocs (SD), and TREC-COVID (TR).

<img src='x2.png' alt='Refer to caption' title='' width='830' height='830' />

<img src='x3.png' alt='Refer to caption' title='' width='830' height='830' />

*Figure 2: TSNE visualization of contriever embeddings for queries (left) and contexts (right) when sampled 100 instances from each dataset.
We see high dispersion “general-domain” datasets like ArguAna and MSMARCO (blue) while “domain-specific” datasets like HotPotQA (green), NFCorpus (grey), SciFact (pink), and FiQA (purple) are tightly clustered. Datasets like Quora (yellow) have disperse queries but compact contexts.*

|  |  | MSMARCO | Quora | ArguAna | HotpotQA | NFCorpus | SciFact | FiQA | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Retrievers | Single model on MSMARCO | 25.7 | 84.1 | 37.2 | 57.6 | 31.7 | 67.2 | 28.8 | 47.5 |
| | Single model with Multi-Task | 22.4 | 82.0 | 36.9 | 52.1 | 32.9 | 69.4 | 28.9 | 46.4 |
| RouterRetriever (w/o MSMARCO expert) | 22.2 | 83.6 | 39.5 | 59.5 | 33.4 | 76.0 | 30.5 | 49.3 |
| Routing | ExpertClassifierRouter | 23.8 | 82.5 | 37.9 | 53.1 | 31.5 | 67.1 | 29.1 | 46.4 |
| | ClassificationHeadRouter | 22.6 | 83.4 | 38.5 | 52.8 | 32.7 | 69.6 | 28.2 | 46.8 |
| DatasetRouter | 23.6 | 83.9 | 37.3 | 58.4 | 33.1 | 73.4 | 29.9 | 48.5 |
| RouterRetriever (w/ MSMARCO expert) | 23.0 | 83.8 | 38.6 | 59.9 | 33.4 | 77.6 | 30.8 | 49.6 |
| Oracles | DatasetOracle | 25.7 | 84.5 | 40.2 | 59.9 | 34.4 | 79.8 | 32.2 | 50.9 |
| | InstanceOracle | 34.5 | 89.9 | 48.5 | 66.6 | 39.0 | 85.4 | 39.6 | 57.6 |

*Table 1: Retrievers: When trained on the same dataset size, RouterRetriever consistently outperforms single model baselines (MSMARCO and Multi-Task) in terms of nDCG@10 on BEIR benchmark. Routing: RouterRetriever also surpasses various standard routing techniques commonly used in language modeling. Oracles: RouterRetriever achieves performance comparable to the DatasetOracle model. InstanceOracle indicates room for future work in router improvements.*

#### Hyperparameters

We use the pre-trained Contriever*(Izacard et al. [2021])* as our base encoder and train experts (LoRA) according to the settings in *Lee et al. ([2023])*, with a rank of 8, an alpha of 32 per expert, thereby training approximately 0.5% of the parameters (about 1M parameters) per expert.
For training, we adopt the few-shot hyperparameters from *Izacard et al. ([2021])*: a learning rate of 1e-4, a batch size of 256 with in-batch negatives, and a maximum of 500 epochs with early stopping.
For brevity, we focus on presenting results for which experts are applied only to the query encoder, keeping the context encoder frozen.
We include the results of applying experts to the context encoder in Appendix[B.1].

5 Results
---------

### 5.1 Overall Performance

Table[1] shows the performance of RouterRetriever using seven domain-specific experts compared to baseline models, evaluated on test sets of corresponding experts. RouterRetriever outperforms both single model baselines—Multi-Task training over the same training data mix as well as training only on MSMARCO—even without an MSMARCO expert.444For fair comparison, we ensure the number of training instances of used RouterRetriever and Multi-Task mix does not exceeds the number of training instances in MSMARCO. And including an MSMARCO expert further improves performance for most domains.
These results underscore the importance of having separate embedding models (experts) for each domain and dynamically selecting the most appropriate expert for each query rather than relying on a single model to handle multiple domains.
We include additional results with different combinations of experts in Appendix[B.2].

### 5.2 Comparing Different Routing Techniques

We experiment with different routing techniques commonly used in language modeling and compare them to our proposed routing mechanism. Results in Table[1] show that the routing technique used in RouterRetriever consistently achieves the highest performance.
In fact, ClassificationHeadRouter and ExpertClassifierRouter approaches tend to underperform compared to simply using a single retriever trained solely on MSMARCO. DatasetRouter, which is the closest to RouterRetriever, tends to show higher performance than the single model trained on MSMARCO but also often still shows lower performance than RouterRetriever.
These results suggest that routing techniques developed for language modeling may not generalize well to information retrieval. We hypothesize that the differences in the effectiveness of routing techniques between language modeling and information retrieval can be explained from the following perspective:
In language modeling, routing decisions are often made at the token level, which allows for greater flexibility and reduces the impact of any single choice. However, in information retrieval, where a single representative embedding is required, the choice of expert is made only once per instance. This makes the process more vulnerable to the routing technique used and thus requires greater care to ensure precise routing.
We achieve this by designing a routing mechanism around embedding similarities, which leans into the strengths of our encoder models.
We are excited to see more sophisticated routing methods inspired by retrieval-specific designs in future works, especially closing the gap to InstanceOracle performance.

|  | w/ Experts | w/o Experts |
| --- | --- | --- |
| Single model on MSMARCO | 47.5 | 31.6 |
| Single model with Multi-Task | 46.4 | 31.2 |
| RouterRetriever (w/ MSMARCO expert) | 49.6 | 31.9 |
| DatasetOracle | 50.9 | 34.2 |
| InstanceOracle | 57.6 | 41.5 |

*Table 2: RouterRetriever not only shows high performance (nDCG@10) for datasets with trained experts but it also generalizes to those without experts. “w/ Experts” results are taken from Table[1]. “w/o Experts” averages results on seven other BEIR test sets that lack training sets.*

### 5.3 Zero-shot Generalization to Unseen Domains

We’ve demonstrated RouterRetriever’s effective use of domain-specific training data, but what of unseen test sets that don’t have corresponding trained experts?
In Table[2], we evaluate our models for true zero-shot generalization to seven more BEIR test sets: Touche-2020*(Thakur et al. [2024])*, Climate-FEVER*(Diggelmann et al. [2020])*, DBPedia*(Hasibi et al. [2017])*, NaturalQuestions*(Kwiatkowski et al. [2019])*, FEVER*(Thorne et al. [2018])*, TREC-COVID*(Roberts et al. [2020])*, and SciDocs*(Cohan et al. [2020])*.
We find the benefits of RouterRetriever extend beyond the datasets for which specific experts were trained. We provide dataset-specific results in Appendix[B.3].

### 5.4 Training and Inference Efficiency

RouterRetriever achieves high training efficiency by using parameter-efficient LoRA experts, which account for only 0.5% of the parameters per expert.
This makes the addition of new experts insignificant in terms of total parameter count.
It uses the same amount of training data as any multi-task approach.
However, unlike multi-task training which requires retraining the entire model when adding, removing, or changing domains, RouterRetriever allows for these modifications without additional training, as our routing technique is training-free.
However, during inference, computing the query embedding involves two forward passes: first to identify the appropriate expert (routing), and second to generate the final query embedding. Improving the computation efficiency of this routing technique is a direction for future work. Detailed analysis over the efficiency in Appendix[B.4].

6 Analysis
----------

### 6.1 Impact of Dataset Size when Training Experts

<img src='x4.png' alt='Refer to caption' title='' width='830' height='727' />

*Figure 3: Single expert performance (nDCG@10; y-axis) against number of training instances (x-axis). Each line color represents the training dataset used, and each plot is a BEIR test dataset.
As we increase training set size, in-domain performance increases rapidly, but may not transfer to improved out-of-domain performance.*

Figure[3] shows the relationship between the amount of training data and the performance of a single expert retriever.
For in-domain evaluation datasets, performance generally improves as the number of training instances increases. However, in out-of-domain evaluation datasets, simply increasing the number of training samples does not necessarily lead to better performance.
Interestingly, when testing out-of-domain, experts perform better when trained on general domains (e.g., ArguAna and MSMARCO) compared to domain-specific experts (e.g., SciFact and NFcorpus). We attribute this to the broader coverage of general-domain datasets, as illustrated in Figure[2].
These results suggest that while a larger training dataset is generally beneficial for expert in-domain performance, broad coverage and diversity of the training dataset have a more significant impact on out-of-domain performance. More details on the performance of each expert are in Appendix[C.2].

<img src='x5.png' alt='Refer to caption' title='' width='830' height='564' />

*Figure 4: Average nDCG@10 (y-axis) by the number of experts (x-axis) for various models. RouterRetriever tends to show improved performance as the number of experts increases, outperforming a single MSMARCO-trained model even with just three experts despite less training data.*

<img src='x6.png' alt='Refer to caption' title='' width='831' height='521' />

*Figure 5: Average instance-level oracle routing performance nDCG@10 (y-axis) by the number of available experts (x-axis). The improvement rate tends to be high when adding experts initially followed by diminishing returns.*

### 6.2 Impact of Number of Experts

Figure[4] shows the relationship between the number of experts and the performance of RouterRetriever.
It outperforms the single MSMARCO-trained model even with just three experts, indicating that despite not having as diverse or large a training dataset as MSMARCO, the advantage of having multiple embedding models and the ability to select the most suitable one leads to better performance.
The performance of multi-task training tends to fluctuate as the number of domains (experts) increases. We hypothesize that with a large number of domains, the model struggles to find the optimal embedding for general cases due to high variance across training datasets.

Yet, Figure[4] also shows diminishing returns in RouterRetriever as we increase the number of experts, but a consistent increase in DatasetOracle.
To further study whether this is due to the need for better routing, we experiment with InstanceOracle but varying the pool of available experts.
Figure[5] shows as the number of experts increases, InstanceOracle performance also improves quickly before encountering some diminishing returns.
In repeating these experiments in Appendix[C.3], we found this is true regardless of the expert combinations or order in which they’re added.
Overall, we interpret the results to mean that RouterRetriever’s routing technique tends to be more distracted as more experts are added, which could motivate future work on scaling router fidelity closer to InstanceOracle to handle the higher complexity of more experts.

Table[3] shows results from experiments in which we sequentially add experts to RouterRetriever.
At low expert counts, each addition of a new expert can dramatically change performance across tasks that previously had an in-domain expert.
For example, adding a HotpotQA expert caused performance on ArguAna to drop from 40.1 to 38.5 and SciFact from 76.7 to 72.2, while causing the expected improvement in HotpotQA to increase from 55.3 to 59.2.
At higher expert counts, these side-effects are much more muted.
For example, adding SciDocs or TREC-COVID experts to a seven-expert RouterRetriever improves performance for SciDocs (14.8 to 16.3) and TREC-COVID (44.9 to 56.2), but doesn’t change overall performance for the other categories.

<img src='x7.png' alt='Refer to caption' title='' width='747' height='448' />

*(a) InstanceOracle*

<img src='x8.png' alt='Refer to caption' title='' width='747' height='448' />

*(b) RouterRetriever*

*Figure 6:  For each evaluation dataset (y-axis), how often is each expert chosen (x-axis)? Darker cells mean more frequent selection. Diagonal entries mean in-domain selection. (a) While “general” experts like AR and MS appear well-suited to tackle instances from other datasets (darker columns), instances from certain datasets like SF and NF must be routed to the in-domain expert (sparse columns with single dark concentration). (b) RouterRetriever has sparser routing behavior, tending towards following dataset boundaries, which explains similar results as DatasetOracle.*

| Start | Addition | AR | NF | SF | FI | HO | QU | MS | SD | TR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unstable additions with few experts | | | | | | | | |  |  |
| AR, NF, SF, FI | - | 40.1 | 32.3 | 76.7 | 30.7 | 55.3 | 83.2 | 22.1 | 15.1 | 43.1 |
| AR, NF, SF, FI | + HotpotQA | 38.5 | 33.0 | 72.2 | 27.9 | 59.2 | 82.7 | 22.3 | 15.9 | 43.6 |
| AR, NF, SF, FI, HO | + Quora | 39.5 | 33.4 | 76.0 | 30.5 | 59.5 | 83.6 | 22.2 | 15.1 | 44.3 |
| Stable additions with more experts | | | | | | | | |  |  |
| AR, NF, SF, FI, HO, QU | + MSMARCO | 38.6 | 33.4 | 77.6 | 30.8 | 59.9 | 83.8 | 23.0 | 14.8 | 44.9 |
| AR, NF, SF, FI, HO, QU, MS | + SciDocs | 38.8 | 32.7 | 76.9 | 30.3 | 59.8 | 84.1 | 22.9 | 16.3 | 44.7 |
| AR, NF, SF, FI, HO, QU, MS | + TREC-COVID | 38.4 | 32.7 | 77.3 | 31.4 | 59.9 | 83.9 | 22.7 | 14.6 | 56.2 |

*Table 3:  Performance (nDCG10) across BEIR datasets while sequentially adding more experts to RouterRetriever.*

### 6.3 Analyzing Routing Errors using InstanceOracle

In Figure[6], we hope to understand the gap in performance between RouterRetriever routing and InstanceOracle routing by inspecting which instances are routed to which experts by either method.
First, Figure[6(a)] shows that general-domain datasets like ArguAna and MSMARCO yield experts that can handily tackle instances from other diverse datasets; hence, they receive an even distribution of queries from InstanceOracle.
However, for domain-specific datasets like SciFact or NFCorpus, the best performance is typically achieved by the expert trained specifically on that domain.
Figure[6(b)] then shows that RouterRetriever’s expert selection tends to be much sparser, prioritizing routing instances to their source dataset’s expert.
This explains the similar performance seen in Table[1] between RouterRetriever and DatasetOracle, and motivates future work on more powerful routing mechanisms.
We add a detailed error analysis of our routing technique in Appendix[C.4].

7 Conclusion
------------

In this paper, we introduce RouterRetriever, a retrieval model that leverages a mixture of domain-specific expert embeddings, guided by a routing mechanism to select the most suitable embedding for each query.
This approach is both lightweight and flexible, allowing for the addition or removal of experts without additional training. Our experiments demonstrate that it consistently outperforms single embedding models, showcasing the advantages of integrating domain-specific experts. Additionally, it surpasses various widely used routing techniques in language modeling, emphasizing the significance of effective routing for information retrieval tasks. Our results highlight the crucial role of domain-specific experts in improving retrieval performance across diverse domains.
Yet, there remains much more room for improvement, as indicated by the higher performance of an instance-level oracle router compared to our method.
We hope our work spurs the broader research community to search for more powerful methods for training expert retrievers and combining them with efficient routing techniques.

Acknowledgments
---------------

We thank Nandan Thakur, Orion Weller, Jiyeon Kim, Hanseok Oh, and the Semantic Scholar Research team at Ai2 for helpful discussions and constructive feedback.

References
----------

* Asai et al. (2022)Asai, A.; Schick, T.; Lewis, P.; Chen, X.; Izacard, G.; Riedel, S.; Hajishirzi, H.; and Yih, W.-t. 2022.Task-aware retrieval with instructions.*arXiv preprint arXiv:2211.09260*.
* Belofsky (2023)Belofsky, J. 2023.Token-Level Adaptation of LoRA Adapters for Downstream Task Generalization.In *Proceedings of the 2023 6th Artificial Intelligence and Cloud Computing Conference*, 168–172.
* Bondarenko et al. (2020)Bondarenko, A.; Fröbe, M.; Beloucif, M.; Gienapp, L.; Ajjour, Y.; Panchenko, A.; Biemann, C.; Stein, B.; Wachsmuth, H.; Potthast, M.; and Hagen, M. 2020.Overview of Touché 2020: Argument Retrieval.In *Conference and Labs of the Evaluation Forum*.
* Bonifacio et al. (2022)Bonifacio, L.; Abonizio, H.; Fadaee, M.; and Nogueira, R. 2022.Inpars: Data augmentation for information retrieval using large language models.*arXiv preprint arXiv:2202.05144*.
* Boteva et al. (2016)Boteva, V.; Ghalandari, D. G.; Sokolov, A.; and Riezler, S. 2016.A Full-Text Learning to Rank Dataset for Medical Information Retrieval.In *European Conference on Information Retrieval*.
* Campos et al. (2016)Campos, D. F.; Nguyen, T.; Rosenberg, M.; Song, X.; Gao, J.; Tiwary, S.; Majumder, R.; Deng, L.; and Mitra, B. 2016.MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.*ArXiv*, abs/1611.09268.
* Cohan et al. (2020)Cohan, A.; Feldman, S.; Beltagy, I.; Downey, D.; and Weld, D. S. 2020.SPECTER: Document-level Representation Learning using Citation-informed Transformers.*ArXiv*, abs/2004.07180.
* Diggelmann et al. (2020)Diggelmann, T.; Boyd-Graber, J. L.; Bulian, J.; Ciaramita, M.; and Leippold, M. 2020.CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims.*ArXiv*, abs/2012.00614.
* Fang et al. (2024)Fang, Y.; Ai, Q.; Zhan, J.; Liu, Y.; Wu, X.; and Cao, Z. 2024.Combining Multiple Supervision for Robust Zero-Shot Dense Retrieval.In *AAAI Conference on Artificial Intelligence*.
* Feng et al. (2023)Feng, S.; Shi, W.; Bai, Y.; Balachandran, V.; He, T.; and Tsvetkov, Y. 2023.Knowledge Card: Filling LLMs’ Knowledge Gaps with Plug-in Specialized Language Models.*arXiv preprint arXiv:2305.09955*.
* Gao and Callan (2021)Gao, L.; and Callan, J. 2021.Condenser: a Pre-training Architecture for Dense Retrieval.In Moens, M.-F.; Huang, X.; Specia, L.; and Yih, S. W.-t., eds., *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics.
* Gao, Yao, and Chen (2021)Gao, T.; Yao, X.; and Chen, D. 2021.SimCSE: Simple Contrastive Learning of Sentence Embeddings.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics.
* Hasibi et al. (2017)Hasibi, F.; Nikolaev, F.; Xiong, C.; Balog, K.; Bratsberg, S. E.; Kotov, A.; and Callan, J. 2017.DBpedia-Entity v2: A Test Collection for Entity Search.*Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval*.
* Hu et al. (2021)Hu, J. E.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; and Chen, W. 2021.LoRA: Low-Rank Adaptation of Large Language Models.*ArXiv*, abs/2106.09685.
* Iyer, Dandekar, and Csernai (2017)Iyer, S.; Dandekar, N.; and Csernai, K. 2017.First Quora Dataset Release: Question Pairs.
* Izacard et al. (2021)Izacard, G.; Caron, M.; Hosseini, L.; Riedel, S.; Bojanowski, P.; Joulin, A.; and Grave, E. 2021.Unsupervised dense information retrieval with contrastive learning.*arXiv preprint arXiv:2112.09118*.
* Jang et al. (2023)Jang, J.; Kim, S.; Ye, S.; Kim, D.; Logeswaran, L.; Lee, M.; Lee, K.; and Seo, M. 2023.Exploring the benefits of training expert language models over instruction tuning.In *International Conference on Machine Learning*, 14702–14729. PMLR.
* Jeong et al. (2024)Jeong, S.; Baek, J.; Cho, S.; Hwang, S. J.; and Park, J. C. 2024.Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity.*arXiv preprint arXiv:2403.14403*.
* Kwiatkowski et al. (2019)Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A. P.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.-W.; Dai, A. M.; Uszkoreit, J.; Le, Q. V.; and Petrov, S. 2019.Natural Questions: A Benchmark for Question Answering Research.*Transactions of the Association for Computational Linguistics*, 7: 453–466.
* Lee et al. (2023)Lee, H.; Soldaini, L.; Cohan, A.; Seo, M.; and Lo, K. 2023.Back to Basics: A Simple Recipe for Improving Out-of-Domain Retrieval in Dense Encoders.*arXiv preprint arXiv:2311.09765*.
* Lee, Chang, and Toutanova (2019)Lee, K.; Chang, M.-W.; and Toutanova, K. 2019.Latent Retrieval for Weakly Supervised Open Domain Question Answering.In *ACL 2019*.
* Lin et al. (2023a)Lin, K.; Lo, K.; Gonzalez, J. E.; and Klein, D. 2023a.Decomposing Complex Queries for Tip-of-the-tongue Retrieval.In *Conference on Empirical Methods in Natural Language Processing*.
* Lin et al. (2023b)Lin, S.-C.; Asai, A.; Li, M.; Oğuz, B.; Lin, J. J.; Mehdad, Y.; tau Yih, W.; and Chen, X. 2023b.How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval.*ArXiv*, abs/2302.07452.
* Ma et al. (2020)Ma, J.; Korotkov, I.; Yang, Y.; Hall, K.; and McDonald, R. 2020.Zero-shot neural passage retrieval via domain-targeted synthetic question generation.*arXiv preprint arXiv:2004.14503*.
* Maia et al. (2018)Maia, M.; Handschuh, S.; Freitas, A.; Davis, B.; McDermott, R.; Zarrouk, M.; and Balahur, A. 2018.WWW’18 Open Challenge: Financial Opinion Mining and Question Answering.*Companion Proceedings of the The Web Conference 2018*.
* Mallen et al. (2022)Mallen, A.; Asai, A.; Zhong, V.; Das, R.; Khashabi, D.; and Hajishirzi, H. 2022.When not to trust language models: Investigating effectiveness of parametric and non-parametric memories.*arXiv preprint arXiv:2212.10511*.
* Muqeeth et al. (2024)Muqeeth, M.; Liu, H.; Liu, Y.; and Raffel, C. 2024.Learning to route among specialized experts for zero-shot generalization.*arXiv preprint arXiv:2402.05859*.
* Oh et al. (2024)Oh, H.; Lee, H.; Ye, S.; Shin, H.; Jang, H.; Jun, C.; and Seo, M. 2024.INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models.*arXiv preprint arXiv:2402.14334*.
* Roberts et al. (2020)Roberts, K.; Alam, T.; Bedrick, S.; Demner-Fushman, D.; Lo, K.; Soboroff, I.; Voorhees, E. M.; Wang, L. L.; and Hersh, W. R. 2020.TREC-COVID: rationale and structure of an information retrieval shared task for COVID-19.*Journal of the American Medical Informatics Association : JAMIA*, 27: 1431 – 1436.
* Shen et al. (2024)Shen, S. Z.; Lang, H.; Wang, B.; Kim, Y.; and Sontag, D. 2024.Learning to decode collaboratively with multiple language models.*arXiv preprint arXiv:2403.03870*.
* Su et al. (2022)Su, H.; Shi, W.; Kasai, J.; Wang, Y.; Hu, Y.; Ostendorf, M.; Yih, W.-t.; Smith, N. A.; Zettlemoyer, L.; and Yu, T. 2022.One embedder, any task: Instruction-finetuned text embeddings.*arXiv preprint arXiv:2212.09741*.
* Sukhbaatar et al. (2024)Sukhbaatar, S.; Golovneva, O.; Sharma, V.; Xu, H.; Lin, X. V.; Rozière, B.; Kahn, J.; Li, D.; Yih, W.-t.; Weston, J.; et al. 2024.Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM.*arXiv preprint arXiv:2403.07816*.
* Thakur et al. (2024)Thakur, N.; Bonifacio, L.; Fröbe, M.; Bondarenko, A.; Kamalloo, E.; Potthast, M.; Hagen, M.; and Lin, J. 2024.Systematic Evaluation of Neural Retrieval Models on the Touché 2020 Argument Retrieval Subset of BEIR.
* Thakur et al. (2021)Thakur, N.; Reimers, N.; Rücklé, A.; Srivastava, A.; and Gurevych, I. 2021.Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models.*arXiv preprint arXiv:2104.08663*.
* Thorne et al. (2018)Thorne, J.; Vlachos, A.; Christodoulopoulos, C.; and Mittal, A. 2018.FEVER: a Large-scale Dataset for Fact Extraction and VERification.*ArXiv*, abs/1803.05355.
* Wachsmuth, Syed, and Stein (2018)Wachsmuth, H.; Syed, S.; and Stein, B. 2018.Retrieval of the Best Counterargument without Prior Topic Knowledge.In *Annual Meeting of the Association for Computational Linguistics*.
* Wadden et al. (2020)Wadden, D.; Lo, K.; Wang, L. L.; Lin, S.; van Zuylen, M.; Cohan, A.; and Hajishirzi, H. 2020.Fact or Fiction: Verifying Scientific Claims.*ArXiv*, abs/2004.14974.
* Wang et al. (2021)Wang, K.; Thakur, N.; Reimers, N.; and Gurevych, I. 2021.GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval.*arXiv preprint arXiv:2112.07577*.
* Wang et al. (2023)Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.; and Wei, F. 2023.Improving text embeddings with large language models.*arXiv preprint arXiv:2401.00368*.
* Weller et al. (2024)Weller, O.; Chang, B.; MacAvaney, S.; Lo, K.; Cohan, A.; Van Durme, B.; Lawrie, D.; and Soldaini, L. 2024.FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions.*arXiv preprint arXiv:2403.15246*.
* Xin et al. (2021)Xin, J.; Xiong, C.; Srinivasan, A.; Sharma, A.; Jose, D.; and Bennett, P. N. 2021.Zero-shot dense retrieval with momentum adversarial domain invariant representations.*arXiv preprint arXiv:2110.07581*.
* Yang et al. (2018)Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.; Salakhutdinov, R.; and Manning, C. D. 2018.HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.In *Conference on Empirical Methods in Natural Language Processing*.
* Ye et al. (2022)Ye, S.; Jang, J.; Kim, D.; Jo, Y.; and Seo, M. 2022.Retrieval of soft prompt enhances zero-shot task generalization.*arXiv preprint arXiv:2210.03029*.

Appendix A Experimental Setup
-----------------------------

### A.1 Baselines

#### MSMARCO

This baseline uses a single MSMARCO gate, which is trained on a large-scale, general-domain dataset without any routing techniques applied.

#### Multi-Task

In this approach, we train a single embedding model on all datasets simultaneously in a multi-task manner. We keep the number of training datasets for each label the same, keeping to the one with the minimum value by sampling.

#### Best Individual

This represents the oracle performance when selecting the single best-performing gate for each dataset. For example, if the SciFact gate shows the highest overall performance on the SciDocs evaluation dataset compared to other gates, the performance of the SciFact gate is recorded as the best individual performance for SciDocs.

#### Oracle

This is the oracle performance when selecting the best-performing gate for each individual instance. For example, within the SciDocs dataset, certain instances might achieve the highest performance with the SciFact gate, while others might perform better with the MSMARCO gate. This baseline measures the performance when, for each instance, the gate that yields the best result is selected.

#### ExpertClassifierRouter

This routing technique, inspired from *Shen et al. ([2024])*, uses a binary classifier for each gate. For each instance, the classifier calculates the probability of selecting or not selecting a specific gate. The gate with the highest probability of being selected is chosen.

To construct the training dataset, we use the predicted label ($g_{\text{max}}$) from the Pilot Embedding Library. For each ($x_{i}$, $g_{\text{max}}$) pair, we randomly sample instances where the maximum gate differs, which are used to train the "not choosing the gate" label. The dataset is balanced across labels, with the following number of training instances for each dataset: AR (16,108), FI (1070), SF (1,414), NF (892), HO (4,618), QU (4,326), and MS (4,252). Please note that the training datasets only consist of instances where only a single gate shows maximum performance.
We then train a binary classifier for each gate to predict whether an instance is likely to achieve the highest performance through that gate.

#### ClassificationHeadRouter

This routing technique, inspired from *Muqeeth et al. ([2024])*, uses a classification head where the number of labels corresponds to the number of gates. The gate with the highest predicted probability is selected as the one likely to yield the best performance. To ensure balance, we equalize the number of training instances for each label, matching the dataset with the fewest instances (NFcorpus with 892 instances, other numbers in ExpertClassifierRouter paragraph). AS a result, the total number of training instances is 6,244.

#### DatasetRouter

This routing technique, inspired from *Ye et al. ([2022]); Jang et al. ([2023])*, is the closest baseline to RouterRetriever. It samples 100 training instances from each dataset and when given a query, it retrieves the most relevant instances from these samples. The gate trained on the dataset from which the sample originated is then used.

The key differences between DatasetRouter and RouterRetriever are as follows. (1) RouterRetriever uses the predicted label to map an instance to a gate, while DatasetRouter relies on the original dataset label. For example, if a training instance from MSMARCO performs best with the sciFact gate, RouterRetriever will select the Scifact gate for a similar query, whereas DatasetRouter will select the MSMARCO gate. (2) RouterRetriever incorporates a clustering step, grouping similar instances together and using centroid embeddings, rather than treating each instance individually.

### A.2 Datasets

| Domain | Name | Task | Train (k) | Gen Train (k) | Test | Corpus (k) |
| --- | --- | --- | --- | --- | --- | --- |
| Misc. | ArguAna (AR)(Wachsmuth, Syed, and Stein [2018]) | Argument Retrieval | - | 23 | 1,406 | 8.7 |
| | Touche-2020 (TO)(Bondarenko et al. [2020]) | Argument Retrieval | - | - | 49 | 382.5 |
| MSMARCO (MS)(Campos et al. [2016]) | Passage-Retrieval | 503 | - | 6,980 | 8,842 |
| Wikipedia | NaturalQuestions (NQ)(Kwiatkowski et al. [2019]) | Question Answering | - | - | 3,452 | 2,681 |
| | HotpotQA (HO)(Yang et al. [2018]) | Question Answering | 85 | - | 7,405 | 5,233 |
| DBpedia (DB)(Hasibi et al. [2017]) | Entity-Retrieval | - | - | 400 | 4,636 |
| FEVER (FE)(Thorne et al. [2018]) | Fact Checking | 110 | - | 6,666 | 5,417 |
|  | Climate-FEVER (CL)(Diggelmann et al. [2020]) | Fact Checking | - | - | 1,535 | 5,417 |
| Bio-Medical | TREC-COVID (TR)(Roberts et al. [2020]) | Bio-Medical Retrieval | - | 432 | 50 | 171 |
| | NFCorpus (NF)(Boteva et al. [2016]) | Bio-Medical Retrieval | 2.6 | 10.8 | 323 | 3.6 |
| Scientific | SCIDOCS (SD)(Cohan et al. [2020]) | Citation-Prediction | - | 67 | 1,000 | 25.7 |
| | SciFact (SF)(Wadden et al. [2020]) | Fact Checking | 0.8 | 15.4 | 300 | 5.2 |
| Finance | FIQA-2018 (FI)(Maia et al. [2018]) | Question Answering | 5.5 | 162 | 648 | 57.6 |
| Quora | Quora (QU) | Duplicate-Question Retrieval | - | 200 | 10,000 | 523 |

*Table 4: Data statics of 14 datasets in BEIR benchmark. Units of the numbers of training dataset (Train), generated training dataset (Gen Train), and corpus are in thousands. For most datasets, training datasets are not provided and for some datasets, we failed to download the generated training dataset.*

#### Stats of Training Dataset

Table[4] presents the statistics and details of the datasets in the BEIR benchmark, which we used for training and evaluation. We sampled datasets from Quora to ensure that the number of training instances for AR, HO, NF, SF, FI, and QU matches that of MS.

*Table 5: Examples where the dataset label differs from the predicted label based on the highest-performing gate.*

| Question | Dataset | Max Gate |
| --- | --- | --- |
| APOE4 expression in iPSC-derived neurons results in decreased tau phosphorylation. | SciFact | NFCorpus |
| which mir regulates the autophagy of cells | SciFact | NFCorpus |
| what kind of leader should i be as the chief executive | FiQA-2018 | ArguAna |
| what is casual dining dining | FiQA-2018 | HotpotQA |
| is it better to be a vegan or vegetarian? | ArguAna | SciFact |
| could we ban animal testing | ArguAna | SciFact |
| why do humans eat meat | ArguAna | Quora |

#### Examples of Oracle

Table[5] shows examples of questions where a gate from a different dataset outperforms the gate trained on the dataset to which the question belongs. We observe that questions related to biology often achieve higher performance with the NFCorpus gate, while those involving scientific knowledge tend to favor the SciFact gate, and questions requiring arguments perform better with the ArguAna gate. This pattern suggests that, even within a single dataset, some instances may be more closely aligned with other datasets, likely because the datasets were not labeled or constructed to avoid overlap with existing datasets.

### A.3 Hyperparameters

We trained the Contriever model*(Izacard et al. [2021])* using an asymmetric architecture, where the query encoder encodes the query and the context encoder encodes the context. In our experiments, we fine-tuned only the LoRA (Low-Rank Adaptation) parameters of the query encoder, training approximately 1 million parameters per gate (which accounts for 0.5% of the total model parameters).
For evaluation, we used the NDCG@10 metric, consistent with previous works*(Thakur et al. [2021]; Lee et al. [2023])*, which measures the ranking quality of the top 10 retrieved documents. All results were calculated using the official BEIR evaluation code.
The experiments were conducted on 8 or fewer A6000 GPUs (each with 40GB of memory). We utilized checkpoints from all pretrained models available on Huggingface555<https://huggingface.co/facebook/contriever>. The experiments were performed over various combinations of gates, with all random seeds set to 10.

Appendix B Results
------------------

|  | Misc | | Wiki | Bio | Science | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | HO | NF | SF | QU | FI | Avg |
| MSMARCO | 39.3 | 25.3 | 57.9 | 32.2 | 66.5 | 84.3 | 28.6 | 47.7 |
| Multi-Task | 38.2 | 21.9 | 49.8 | 32.4 | 65.1 | 83.3 | 26.1 | 45.3 |
| RouterRetriever | 40.5 | 21.0 | 61.3 | 32.7 | 68.2 | 82.5 | 30.0 | 48.0 |
| Best Individual | 41.2 | 25.3 | 60.9 | 32.2 | 70.3 | 86.1 | 32.2 | 49.8 |
| Oracle | 48.2 | 33.2 | 68.4 | 39.0 | 76.7 | 90.0 | 38.6 | 56.3 |

*Table 6: Performance of RouterRetriever when context encoder is trainable.*

### B.1 Unfreezing context encoder

In our main experiments, we focus on scenarios where the context encoder is frozen, and only the LoRA of the query encoder is trainable to isolate the impact of routing on the query encoder alone. However, we observe that the overall performance trend remains similar even when the context encoder is not frozen, with the unfrozen models generally achieving higher performance. Table[6] presents the results when the context encoder is frozen. In these experiments, RouterRetriever consistently outperforms the MSMARCO-trained model and the Multi-Task model.

|  | Misc | | Bio | Science | Finance | Avg |
| --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | NF | SF | FI |  |
| MSMARCO | 37.2 | 25.7 | 31.7 | 67.2 | 28.8 | 38.1 |
| Multi-Task | 39.4 | 21.2 | 28.2 | 69.2 | 30.0 | 37.6 |
| RouterRetriever | 40.1 | 22.1 | 32.3 | 76.7 | 30.7 | 40.4 |
| DatasetOracle | 40.2 | 22.4 | 34.4 | 79.8 | 32.2 | 41.8 |
| Oracle | 47.5 | 29.3 | 38.0 | 84.5 | 37.4 | 47.3 |

*Table 7: RouterRetriever performance with four gates: AR, NF, SF, FI. Avg is an average performance over the dataset of all gates and MSMARCO.*

|  | Misc | | Wiki | Bio | Science | Finance | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | HO | NF | SF | FI |  |
| MSMARCO | 37.2 | 25.7 | 57.6 | 31.7 | 67.2 | 28.8 | 41.4 |
| Multi-Task | 37.7 | 22.0 | 58.6 | 31.1 | 69.1 | 28.4 | 41.2 |
| RouterRetriever | 38.5 | 22.3 | 59.2 | 33.0 | 72.2 | 27.9 | 42.2 |
| DatasetOracle | 40.2 | 22.4 | 59.9 | 34.4 | 79.8 | 32.2 | 44.8 |
| Oracle | 47.7 | 31.5 | 65.1 | 38.6 | 84.8 | 38.4 | 51.0 |

*Table 8: RouterRetriever performance with five gates: AR, NF, SF, FI, HO. Avg is an average performance over the dataset of all gates and MSMARCO.*

|  | Misc | | Wiki | Bio | Science | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | HO | NF | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 25.7 | 57.6 | 31.7 | 67.2 | 84.1 | 28.8 | 47.5 |
| Multi-Task | 35.3 | 21.5 | 55.3 | 32.3 | 65.3 | 82.8 | 29.3 | 46.0 |
| RouterRetriever | 39.5 | 22.2 | 59.5 | 33.4 | 76.0 | 83.6 | 30.5 | 49.3 |
| DatasetOracle | 40.2 | 22.6 | 59.9 | 34.4 | 79.8 | 84.5 | 32.2 | 50.5 |
| Oracle | 48.0 | 32.7 | 65.5 | 38.8 | 85.0 | 89.7 | 39.2 | 57.0 |

*Table 9: RouterRetriever performance with six gates: AR, NF, SF, FI, HO, QU. Avg is an average performance over the dataset of all gates and MSMARCO.*

|  | Misc | | Wiki | Bio | Science | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | HO | NF | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 25.7 | 57.6 | 31.7 | 67.2 | 84.1 | 28.8 | 47.5 |
| Multi-Task | 36.9 | 22.4 | 52.1 | 32.9 | 69.4 | 82.0 | 28.9 | 46.4 |
| RouterRetriever | 38.6 | 23.0 | 59.9 | 33.4 | 77.6 | 83.8 | 30.8 | 49.6 |
| Best Individual | 40.2 | 25.6 | 59.9 | 34.4 | 79.8 | 84.5 | 32.2 | 50.9 |
| Oracle | 48.5 | 34.5 | 66.6 | 39.0 | 85.4 | 89.9 | 39.6 | 57.6 |

*Table 10: RouterRetriever performance with seven gates: AR, NF, SF, FI, HO, QU, MS*

### B.2 Detailed numbers by gates

In this section, we show detailed number of performance with different combinations of gates.
Table[7] shows performance with AR, NF, SF, FI as gates.
Table[8] shows performance with AR, HO, NF, SF, FI as gates.
Table[9] shows performance with AR, HO, NF, SF, QU, FI as gates.
Table[10] shows performance with AR, MS, HO, NF, SF, QU, FI as gates.
Figure 4 shows only with three gates, RouterRetriever outperforms the MSMARCO-trained ones thereby in all results, we can see that RouterRetriever outperforms the MSMARCO-trained ones and multi-task baselines.

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 39.5 |
| Multi-Task | 39.4 | 18.9 | 21.2 | 16.1 | 27.0 | 23.5 | 60.0 | 41.6 | 28.2 | 41.6 | 15.0 | 69.2 | 82.2 | 30.0 | 36.7 |
| RouterRetriever | 40.1 | 18.4 | 22.1 | 15.4 | 33.6 | 25.5 | 67.3 | 55.3 | 32.3 | 43.1 | 15.1 | 76.7 | 83.2 | 30.7 | 39.9 |
| DatasetOracle | 40.2 | 18.5 | 22.4 | 15.7 | 31.0 | 26.2 | 68.9 | 54.2 | 34.4 | 44.6 | 15.5 | 79.8 | 83.7 | 32.2 | 40.5 |
| Oracle | 47.5 | 23.8 | 29.3 | 19.7 | 36.5 | 33.5 | 76.7 | 59.4 | 38.0 | 52.5 | 20.0 | 84.5 | 88.1 | 37.4 | 46.2 |

*Table 11: RouterRetriever performance with four gates: AR, NF, SF, FI. Avg is an average performance over all datasets.*

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 39.5 |
| Multi-Task | 37.7 | 17.8 | 22.0 | 15.4 | 33.2 | 26.6 | 65.5 | 58.6 | 31.1 | 41.6 | 15.1 | 69.1 | 82.7 | 28.4 | 39.0 |
| RouterRetriever | 38.5 | 19.8 | 22.3 | 17.1 | 35.4 | 27.8 | 67.4 | 59.2 | 33.0 | 43.6 | 15.9 | 72.2 | 82.7 | 27.9 | 40.2 |
| DatasetOracle | 40.2 | 19.7 | 22.4 | 17.7 | 36.1 | 28.8 | 68.9 | 59.9 | 34.4 | 44.6 | 16.2 | 79.8 | 83.7 | 32.2 | 41.8 |
| Oracle | 47.7 | 25.7 | 31.5 | 21.3 | 40.4 | 37.9 | 79.5 | 65.1 | 38.6 | 53.1 | 20.7 | 84.8 | 88.7 | 38.4 | 48.1 |

*Table 12: RouterRetriever performance with five gates: AR, NF, SF, FI, HO. Avg is an average performance over all datasets.*

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 39.5 |
| Multi-Task | 37.7 | 17.8 | 22.0 | 15.4 | 33.2 | 26.6 | 65.5 | 58.6 | 31.1 | 41.6 | 15.1 | 69.1 | 82.7 | 28.4 | 39.0 |
| RouterRetriever | 38.8 | 17.7 | 22.9 | 15.2 | 31.7 | 27.7 | 67.5 | 59.8 | 32.7 | 44.7 | 16.3 | 76.9 | 84.1 | 30.3 | 40.4 |
| DatasetOracle | 40.2 | 19.7 | 25.6 | 17.7 | 36.1 | 29.3 | 70.8 | 59.9 | 34.4 | 49.7 | 16.2 | 79.8 | 84.5 | 32.2 | 42.6 |
| Oracle | 49.1 | 27.4 | 33.8 | 20.7 | 42.0 | 40.8 | 82.6 | 66.3 | 40.2 | 55.4 | 18.5 | 86.1 | 88.5 | 33.3 | 48.9 |

*Table 13: RouterRetriever performance with eight gates: AR, NF, SF, FI, HO, MS, SD, QU. Avg is the average performance of all datasets.*

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 39.5 |
| Multi-Task | 37.5 | 17.4 | 23.8 | 15.5 | 31.7 | 26.9 | 66.9 | 58.1 | 34.8 | 44.5 | 14.2 | 68.1 | 81.0 | 27.6 | 39.1 |
| RouterRetriever | 38.4 | 17.6 | 22.7 | 15.3 | 32.1 | 27.4 | 67.4 | 59.9 | 32.7 | 56.2 | 14.6 | 77.3 | 83.9 | 31.4 | 41.7 |
| DatasetOracle | 40.2 | 19.7 | 25.6 | 17.7 | 36.1 | 29.3 | 70.8 | 59.9 | 34.4 | 67.3 | 16.2 | 79.8 | 84.5 | 32.2 | 46.9 |
| Oracle | 48.6 | 27.0 | 35.2 | 21.5 | 43.1 | 41.1 | 80.3 | 65.1 | 41.1 | 69.1 | 18.2 | 84.1 | 86.4 | 37.1 | 49.9 |

*Table 14: RouterRetriever performance with eight gates: AR, NF, SF, FI, HO, MS, TR, QU. Avg is the average performance of all datasets.*

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 39.5 |
| Multi-Task | 35.3 | 17.6 | 21.5 | 15.8 | 31.0 | 25.9 | 66.4 | 55.3 | 32.3 | 41.3 | 14.5 | 65.3 | 82.8 | 29.3 | 38.2 |
| RouterRetriever | 39.5 | 17.2 | 22.2 | 16.0 | 32.8 | 27.3 | 68.1 | 59.5 | 33.4 | 44.3 | 15.1 | 76.0 | 83.6 | 30.5 | 40.4 |
| DatasetOracle | 40.2 | 19.7 | 22.6 | 17.7 | 36.1 | 28.8 | 68.9 | 59.9 | 34.4 | 49.7 | 16.2 | 79.8 | 84.5 | 32.2 | 42.2 |
| Oracle | 48.0 | 26.8 | 32.7 | 21.7 | 40.6 | 39.3 | 80.0 | 65.5 | 38.8 | 56.2 | 21.0 | 85.0 | 89.7 | 39.2 | 48.9 |

*Table 15: RouterRetriever performance with six gates: AR, NF, SF, FI, HO, QU. Avg is the average performance of all datasets. The number of total training datasets of RouterRetriever, Multi-Task, and MSMARCO-only are the same.*

|  | Misc | | | Wiki | | | | | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI | Avg |
| MSMARCO | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 | 38.5 |
| Multi-Task | 36.9 | 17.3 | 22.4 | 16.3 | 32.9 | 26.7 | 69.0 | 52.1 | 32.9 | 41.4 | 14.5 | 69.4 | 82.0 | 28.9 | 38.8 |
| RouterRetriever | 38.6 | 17.7 | 23.0 | 15.2 | 33.9 | 27.6 | 69.2 | 59.9 | 33.4 | 44.9 | 14.8 | 77.6 | 83.8 | 30.8 | 40.7 |
| Best Individual | 40.2 | 19.7 | 25.6 | 17.7 | 36.1 | 29.3 | 70.8 | 59.9 | 34.4 | 49.7 | 16.2 | 79.8 | 84.5 | 32.2 | 42.6 |
| Oracle | 48.5 | 27.2 | 34.5 | 22.3 | 41.5 | 40.8 | 81.0 | 66.6 | 39.0 | 56.4 | 21.2 | 85.4 | 89.9 | 39.6 | 49.6 |

*Table 16: RouterRetriever performance with seven gates: AR, NF, SF, FI, HO, QU, MS. Avg is the average performance of all datasets.*

### B.3 Generalization to other datasets

We observe that RouterRetriever demonstrates stable performance not only on datasets with corresponding gates but also on those without them. The performance with different numbers of gates is shown in the following tables: Table[11] (4 gates), Table[12] (5 gates), Table[15] (6 gates), Table[16] (7 gates), and Tables[14] and[13] (8 gates).

When using a similar total number of training datasets (Table[15]), RouterRetriever and the MSMARCO-trained model exhibit comparable generalization performance (both at 31.6). However, RouterRetriever achieves higher performance on datasets that have corresponding gates (47.5 for MSMARCO-only vs. 49.3 for RouterRetriever). As more gates are added, both generalization ability and performance on datasets with corresponding gates tend to improve (Figure[8]).

| Method | Training for Routing | Training Datasets for Routing or Experts | Offline Computation | Storage for Pilot Embedding Library |
| --- | --- | --- | --- | --- |
| Single model on MSMARCO | No | - | - | - |
| Single model with Multi-task | No | All $D$ domains | - | - |
| ClassificationHeadRouter | Yes | All $D$ domains | - | - |
| ExpertClassifierRouter | Yes | Only the new domain | - | - |
| DatasetRouter | No | Only the new domain | $T$ | $D\times T$ |
| RouterRetriever | No | Only the new domain | $D\times T$ | $D\times D$ |

*Table 17: Comparison of routing methods based on training, computation, and storage requirements. $D$ is the number of domains and $T$ is the number of datasets in the new domain. When new domain is added, “Training for Routing” indicates whether the method requires training new router and “Training Datasets for Routing or Experts” indicates number of training datasets required for routing or experts. “Offline computation” shows how much offline computation is required for routing. “Storage for pilot embedding library” indicates how much pilot embeddings are saved in the library for routing.*

### B.4 Efficiency

Table[17] summarizes the efficiency details of RouterRetriever compared to the baselines.
While RouterRetriever requires offline computation to construct the pilot embedding library for routing (as shown in the “Offline Computation” column), it does not necessitate additional training for routing (see “Training for Routing” column). As a result, even though RouterRetriever involves offline inference computation, the overall cost remains significantly lower than the expense of training for routing.

DatasetRouter, which also does not require training for routing, incurs less offline computation for routing than RouterRetriever. Specifically, while DatasetRouter’s computation scales with $T$, the number of datasets in the new domain, RouterRetriever scales with $D\times T$, the number of domains multiplied by number of datasets in the new domain. However, RouterRetriever is much storage-efficient where DatasetRouter need $D\times T$ storage and RouterRetriever requires $D\times D$ storage. Please note that in most cases $T$ is much larger than $D$. For example, if the initial model was trained with three domains and a new domain with 100 training instances is introduced, $D$ becomes 4, and $T$ is 100. In this case, DatasetRouter performs 100 offline computations, whereas RouterRetriever requires 400. However, RouterRetriever is more storage-efficient (as detailed in the “Storage for Pilot Embedding Library” column), since DatasetRouter needs to store 100 new embeddings, while RouterRetriever only stores 4 embeddings for the new domain. Finally, it is crucial to note that RouterRetriever achieves the highest overall performance, as presented in Table[2].

Regarding computation for training experts when adding a new expert, while the training of the expert itself is necessary, all baselines except MSMARCO require this step. However, the efficiency diverges when considering the column “Training Datasets for Routing or Experts.” Both the multi-task and ClassificationHead baselines require retraining from scratch with all datasets, including those from previously trained domains, whenever a new domain is added. In contrast, ExpertClassifierRouter, DatasetRouter, and RouterRetriever only require the training dataset for the new domain to train the corresponding expert.

Appendix C Analysis
-------------------

<img src='x9.png' alt='Refer to caption' title='' width='748' height='500' />

*Figure 7: Average NDCG@10 performance (y-axis) as the number of centroid embeddings from k-means clustering increases. The performance tend to decrease with more pilot embeddings, which suggests that when there are too many pilot embeddings, it tends to distract the performance.*

### C.1 Affect of Number of Pilot Embeddings

We experiment with how the number of pilot embeddings affects performance. In Figure[7], we observe that performance tends to degrade as the number of pilot embeddings increases. We hypothesize that this decline is due to the increased number of pilot embeddings becoming distracting, leading to less effective routing decisions.

| Domain | Training Data | AR | TO | MS | CL | DB | NQ | FE | HO | NF | TR | SD | SF | QU | FI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Misc | AR | 40.2 | 17.0 | 22.4 | 15.7 | 31.0 | 26.2 | 68.9 | 54.2 | 32.8 | 40.3 | 15.5 | 67.8 | 83.7 | 29.6 |
| | MS | 37.2 | 18.3 | 25.7 | 16.0 | 32.7 | 29.3 | 68.8 | 57.6 | 31.7 | 41.2 | 14.6 | 67.2 | 84.1 | 28.8 |
| Wiki | HO | 38.5 | 19.7 | 22.4 | 17.7 | 36.1 | 28.8 | 67.8 | 59.9 | 32.2 | 39.1 | 16.2 | 66.0 | 82.8 | 27.7 |
| Bio | NF | 38.7 | 17.7 | 21.8 | 13.4 | 28.7 | 24.0 | 64.6 | 46.4 | 34.4 | 42.1 | 15.5 | 66.6 | 82.5 | 27.3 |
| | TR | 37.0 | 17.3 | 22.8 | 16.2 | 31.6 | 26.4 | 68.1 | 56.6 | 33.1 | 67.3 | 15.7 | 68.3 | 83.1 | 29.1 |
| Science | SD | 38.9 | 18.2 | 22.8 | 17.0 | 32.0 | 27.3 | 70.0 | 57.2 | 33.2 | 39.6 | 16.3 | 66.7 | 84.3 | 28.4 |
| | SF | 37.9 | 16.5 | 21.8 | 16.0 | 29.4 | 25.5 | 68.1 | 50.2 | 32.3 | 28.8 | 15.1 | 79.8 | 83.6 | 25.1 |
| Quora | QU | 37.5 | 19.4 | 22.6 | 13.9 | 29.0 | 27.4 | 63.8 | 49.1 | 31.1 | 49.7 | 14.3 | 65.7 | 84.5 | 28.3 |
| Finance | FI | 35.1 | 18.5 | 22.1 | 15.4 | 29.5 | 25.2 | 64.6 | 46.9 | 32.3 | 42.7 | 15.0 | 64.5 | 83.4 | 32.2 |

*Table 18: Overall performance when evaluating each gate separately.*

### C.2 Performance of each gates

To analyze the performance trends of each gate, we evaluate them individually without applying any routing techniques in Table[18].
The performance generally shows the highest when the evaluation dataset matches the training dataset of the gate. Additionally, the performance gap between matching and non-matching datasets is larger for domain-specific datasets (NF, TR, SD, SF, QU, FI). In contrast, gates trained on general-domain datasets (AR, MS, HO) tend to perform well across a broader range of datasets.

<img src='x10.png' alt='Refer to caption' title='' width='706' height='480' />

*Figure 8: For each evaluation dataset (y-axis), the rate at which gate the router chooses (x-axis). We could see that the trend generalizes with different combination of gates. Case1 is in order of AR, FI, SF, NF, HO, QU, and MS. Case2 is in reverse order of MS, QU, HO, NF, SF, FI, and AR. Case 3 is in order of SF, NF, HO, QU, AR, MS, and FI.*

### C.3 Impact of Number of Gates

To investigate the impact of the number of gates, we randomly shuffle the gate order and experiment with how adding gates tend to affect performance.
The order of gates added in Figure 4 and Figure 5 is AR, FI, SF, NF, HO, QU, and MS.
We tried various other combinations and could see that the findings are stabilized (Figure[8]): (1) performance tend to increase with more gates added and (2) the improvement rate tends to be higher when adding gates initially and as the number of gates grows, the rate of increase diminishes.

|  | Misc | | Wiki | Bio | | Science | | Quora | Finance |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AR | MS | HO | NF | TR | SD | SF | QU | FI | Avg |
| RouterRetriever | 39.5 | 22.2 | 59.5 | 33.4 | 44.3 | 15.1 | 76.0 | 83.6 | 30.5 | 44.9 |
| RouterRetriever(+ TR) | 38.4 | 22.7 | 59.9 | 33.3 | 56.2 | 14.6 | 77.3 | 83.9 | 31.4 | 46.4 |
| RouterRetriever(+ SD) | 38.8 | 22.9 | 59.8 | 32.7 | 44.7 | 16.1 | 76.9 | 84.1 | 30.3 | 45.1 |

*Table 19: RouterRetriever performance when adding gates within same domain. The performance tend to improve for the dataset, but for the rest the difference tend to be minor.*

### C.4 Routing Mechanism Error Analysis

Figure 7 illustrates the rate at which each router selects a gate, while Figure 6 shows the rate at which each gate tends to deliver high performance for the dataset. The discrepancy between these two heatmaps highlights the gap between RouterRetriever and the oracle performance.
For ArguAna, the maximum gate distribution is evenly spread, and the routing tends to follow this distribution closely.
For Quora, while the maximum gate rate is high overall, the routing often favors the HotpotQA gate in many cases.
For MSMARCO, the gate trained on MSMARCO generally shows high performance, but the routing technique tends to distribute selections across different gates.
For HotpotQA, selecting the HotpotQA gate most frequently results in the highest performance, with MSMARCO being the next best option. The routing technique tends to reflect this pattern.
For SciFact, choosing the SciFact gate is crucial in both cases.
For NFCorpus, selecting the NFCorpus gate is important, yet the routing technique often opts for the ArguAna gate in many instances.
For FiQA-2018, the best performance is achieved by selecting the FiQA-2018 gate, and the routing technique successfully identifies this gate most of the time.

We specifically investigated why NFCorpus often fails to select the NFCorpus gate and instead tends to choose the ArguAna gate. Upon examining the representative embeddings for ArguAna, we found that many of them are confused with ArguAna embeddings that were extracted from the NFCorpus dataset. These instances originally belong to NFCorpus but show the highest performance with the ArguAna gate, leading to their labeling as ArguAna. This suggests that instead of completely removing information about the original dataset, incorporating a weighting factor between the two could further improve performance.
