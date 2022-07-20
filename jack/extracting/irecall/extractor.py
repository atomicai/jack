import collections
import math
from typing import List

import numpy as np

from jack.extracting import extractor
from jack.storing import doc


def idf_recall_score(preds: List[str], labels: List[str], separator=None, mode="hard"):

    answer = ""
    separator = " " if separator is None else separator
    for i, ans in enumerate(labels):
        answer += ans
        if i < len(labels) - 1:
            answer += separator
    answer_tokens = collections.Counter(answer.split(separator))

    ans = []

    for i, _doc in enumerate(preds):
        doc_tokens = collections.Counter(_doc.split(separator))
        # Below the distribution will always be "1" on shared words and "0" on distinct!
        common_tokens = answer_tokens & doc_tokens

        score = sum([1.0 / math.log(1 + common_tokens.get(w, 1)) for w in common_tokens]) / sum(
            [1.0 / math.log(1 + answer_tokens.get(w, 1)) for w in answer_tokens]
        )
        ans.append(score * 1.0 / float(i + 1))

    return np.sum(ans)


class IDFRecallExtractor(extractor.BaseExtractor):
    def __init__(self):
        super(IDFRecallExtractor, self).__init__()

    def extract(self, query: str, docs: List[doc.Document], outline: str = None):
        pass
