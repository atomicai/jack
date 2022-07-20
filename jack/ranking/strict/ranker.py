import collections
from typing import Optional

from jack.engine import bm25
from jack.ranking import ranker
from jack.storing import elastic


class StrictRanker(ranker.BaseRanker):
    def __init__(self, label_list, index: str = "document", theta: float = None, top_k: Optional[int] = None):
        super(StrictRanker, self).__init__(label_list)
        self.engine = bm25.BM25Retriever(elastic.ElasticDocStore(index=index))
        self.top_k = 5 if not top_k else top_k
        self.theta = 1.5 if theta is None else theta

    def rank(self, query: str):
        docs = list(self.engine.retrieve_top_k(query, top_k=self.top_k))[0][0]
        if len(docs) <= 0:
            return 1.0
        response = collections.defaultdict(float)
        labels = [d.meta["label"] for d in docs]
        scores = [d.probability for d in docs]
        for l, s in zip(labels, scores):
            response[l] += s
        ans, _ans = None, 0.0
        for k, v in response.items():
            if v > _ans:
                _ans = v
                ans = k
        return (ans, _ans) if _ans > 1.5 else ("Новый", _ans)
