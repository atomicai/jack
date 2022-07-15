# jack

> Given a set of pre-labeled textual documents $d_1, d_2, \cdots, d_n$ where for every $d_i$ there is a topic $t_j$. The number of topics is usually much lower than the number of documents $k << n$. 


> Given a new incoming document $d_{n+1}$ you want to either assign it to a pre-existing topic, namely, one of $t_1, t_2, \cdots, t_k$ or assign a new topic (aka cluster) $t_{k+1}$. 

---
### Installation

> Before running commands below make sure you have docker up and running.
1.  `make install` (installing the necessary python libs)
2. `make start` (start the docker along with the container mapping named volume)
3. `make run` (fire up the ui and service)

UI is available @ `127.0.0.0:5000`

---
### Workflow
> Before querying and running any ranking pipeline you first need to index your documents $D$. Right now only sparse indexing is supported, namely, the well-known [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm. Simply put, you define the strategy to preprocess your labeled documents to use them later on during the `querying` step. Once the indexing is done, you're ready to respond to the newly incoming documents. Each step is outlined below.

<details>
<summary>Indexing</summary>


| Encoder  | Description  | Notebook |
|:----------|:-------------|:-------------|
| [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) | Under the hood perform the "inverse mapping" for every document. The mapping from each word to a set of documents where specific word $w$ occur.  | TODO |
| [BERT](https://arxiv.org/abs/2004.04906) | Dense semantic encoder. Encode text using pretrained neural network mapping to $\Re^N$. (NOT IMPLEMENTED) | TODO |


</details>
<details>
<summary>Querying</summary>

Given a question $\vec{q}$ you get your $top_k$ documents $d_1, \cdots, d_{top-k}$ that are the most similar to the query $\vec{q}$. How "similar" is defined solely by the encoder you have chosen @ previous step (`indexing`).

| Engine  | Description  | Notebook |
|:----------|:-------------|:-------------|
| [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) | query the indexed documents | TODO |
| [Dense]() | Dense semantic encoder. Encode text using pretrained neural network mapping to $\Re^N$. (NOT IMPLEMENTED) | TODO |

</details>

<details>
<summary>Ranking</summary>

> Once you receive the documents $d_1, \cdots, d_{top-k}$ from the previous step you want to decide whether the given query $\vec{q}$ (aka "newly incoming document") is one of the $\{t_1, \cdots, t_k\}$ or something different, namely, new topic $t_{k+1}$.


| Ranking  | Description  | Notebook |
|:----------|:-------------|:-------------|
| [weak](https://en.wikipedia.org/wiki/Okapi_BM25) | Encode text based on word distribution across all documents | TODO |
| [strict](https://github.com/neuml/tldrstory) | This is (NOT IMPLEMENTED) yet| TODO |

</details>

---

### Evaluation

> The natural question arises: given the two different encoders (`indexing` step) and rankers (`ranking` step) which one is the best?

NOT IMPLEMENTED
