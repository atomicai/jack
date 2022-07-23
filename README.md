# jack

> Given a set of pre-labeled textual documents $d_1, d_2, \cdots, d_n$ where for every $d_i$ there is a topic $t_j$. The number of topics is usually much lower than the number of documents $k << n$.

> Given a new incoming document $d_{n+1}$ you want to either assign it to a pre-existing topic, namely, one of $t_1, t_2, \cdots, t_k$ or assign a new topic (aka cluster) $t_{k+1}$.

---

### Installation

> Before running commands below make sure <u>you have docker</u> up and running.

> On Windows make sure <u>you have "make"</u> on your PATH variable (install it first using e.g. <a href="https://chocolatey.org">choco</a> package manager via `choco install make`)

1.  `make install` (installing the necessary python libs)
2.  `make start` (start the docker along with the container mapping named volume)
3.  `make run` (fire up the ui and service)

UI is available @ `127.0.0.0:5000`

---

### Workflow

> Before querying and running any ranking pipeline you first need to index your documents $D$. Right now only sparse indexing is supported, namely, the well-known [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm. Simply put, you define the strategy to preprocess your labeled documents to use them later on during the `querying` step. Once the indexing is done, you're ready to respond to the newly incoming documents. Each step is outlined below.

<details>
<summary>Indexing</summary>

| Encoder                                          | Description                                                                                                                                      | Notebook                                    |
| :----------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) | Under the hood perform the "inverse mapping" for every document. The mapping from each word to a set of documents where specific word $w$ occur. | [indexing.ipynb](./notebook/indexing.ipynb) |
| [BERT](https://arxiv.org/abs/2004.04906)         | Dense semantic encoder. Encode text using pretrained neural network mapping to $\Re^N$. (NOT IMPLEMENTED)                                        | TODO                                        |

</details>
<details>
<summary>Querying</summary>

> Given a question $\vec{q}$ you get your $top_k$ documents $d_1, \cdots, d_{top-k}$ that are the most similar to the query $\vec{q}$. How "similar" is defined solely by the encoder you have chosen @ previous step (`indexing`).

| Engine                                           | Description                                                                                               | Notebook                                    |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) | query the indexed documents                                                                               | [querying.ipynb](./notebook/querying.ipynb) |
| [Dense]()                                        | Dense semantic encoder. Encode text using pretrained neural network mapping to $\Re^N$. (NOT IMPLEMENTED) | TODO                                        |

</details>

<details>
<summary>Ranking</summary>

> Once you receive the documents $d_1, \cdots, d_{top-k}$ from the previous step you want to decide whether the given query $\vec{q}$ (aka "newly incoming document") is one of the $\{t_1, \cdots, t_k\}$ or something different, namely, new topic $t_{k+1}$.

| Ranking                                                             | Description                                                              | Notebook                                  |
| :------------------------------------------------------------------ | :----------------------------------------------------------------------- | :---------------------------------------- |
| [weak](https://github.com/atomicai/jack/ranking/weak/ranker.py)     | Simply pick the label with highest distribution across all relevant docs | [ranking.ipynb](./notebook/ranking.ipynb) |
| [strict](https://github.com/atomicai/jack/ranking/strict/ranker.py) | This is (NOT IMPLEMENTED) yet                                            | TODO                                      |

</details>

<details>
<summary>DL finetuning</summary>

> This step requires extra framework to be installed for DL experiments - <a href="https://github.com/atomicai/farm">farm</a> which is a wrapper around huggingface models to speed up training and convenient modeling not as pipelines but rather as building blocks. The short description below

---

| Training                                                | Description                                                                                                               | Notebook                                        |
| :------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------- |
| [TEXT classification](https://github.com/atomicai/jack) | The language model for encoding textual data is <a href="https://huggingface.co/cointegrated/rubert-tiny">"tiny" BERT</a> | [finetuning.ipynb](./notebook/finetuning.ipynb) |

</details>

---

### Evaluation

> The natural question arises: given the two different encoders (`indexing` step) and rankers (`ranking` step) which one is the best? The short answer is, it depends on circumstances. The simple evaluation is presented as [evaluating.ipynb](./notebook/evaluating.ipynb) which simlpy <u>picks the highest "recall"</u> while <u>not permitting</u> precision to reach zero on any class.
