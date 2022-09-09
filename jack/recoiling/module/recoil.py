import collections
import pathlib
import string
from dataclasses import asdict, dataclass
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Tuple

import pymorphy2
from icecream import ic
from jack.etc import Document
from jack.recoiling.iface import recoil
from jack.tooling import io, stl


def unify(item: Any, lower_case: bool = False):
    if isinstance(item, Document):
        item = item.text
    elif isinstance(item, Dict):
        item = item["text"]
    else:
        item = str(item)
    item = item.strip()
    item = item.lower() if lower_case else item
    while item.endswith(tuple(string.punctuation)):
        item = item[:-1]
    return item


class Stopper(recoil.IState):

    """
    Class for storing stopwords list along with the method to remove them
    """

    def __init__(self, fpath, lower_case: bool = True):
        super(Stopper, self).__init__()
        self.store = set()
        self.lower_case = lower_case
        with open(str(fpath), encoding="utf-8-sig") as f:
            for w in io.chunkify(f):
                self.store.add(w.strip().lower())

    def __call__(self, x: Iterable[Any]) -> List[Dict[str, str]]:
        response = []

        ptr = stl.NIterator(x)
        while ptr.has_next():
            item = ptr.next()
            item = unify(item, lower_case=self.lower_case)

            if item not in self:
                response.append({"text": item})
        return response

    def __contains__(self, item):
        return str(item).strip() in self.store


@dataclass
class POS:
    noun: ClassVar[str] = "NOUN"
    verb: ClassVar[str] = "VERB"
    advb: ClassVar[str] = "ADVB"
    adjf: ClassVar[str] = "ADJF"

    all: ClassVar[Tuple[str]] = ("NOUN", "VERB", "ADVB", "ADJF")


class Morpher(recoil.IState):

    """
    Class for normalizing each word in a pipeline according to the `tags` (rules) provided
    """

    @dataclass(eq=True, frozen=True)
    class ITag:
        pos: str = None

    def __init__(self, tags: List[str]):
        self.morph = pymorphy2.MorphAnalyzer()
        assert all([_ in POS.all for _ in tags])
        self.itag = set([self.ITag(pos=t) for t in tags])

    def __call__(self, x: Iterable[Any]) -> List[Dict[str, str]]:
        response = []
        ptr = stl.NIterator(x)
        while ptr.has_next():
            word = ptr.next()
            word = unify(word)
            w = self.morph.parse(word)[0]
            ipos = w.tag.pos
            if self.ITag(pos=ipos) in self.itag:
                word = w.normal_form
            response.append(word)
        return response


class Chunker(recoil.IState):

    """
    Class for chunking the textual flow using external api sentenizer.
    The resulting chunks parametrized by `n` tokens at most (e.g. 256) as maximum chunk length
    and `m` tokens at most (e.g. 64) as maximum overlap length. The splitting respect the sentence boundary.
    """

    def __init__(self, n: int = 256, m: int = 64, api: Callable = None):
        self.n = n
        self.m = m
        self.api = api

    def __call__(self, x):
        raise NotImplementedError()


# Построение поисковой системы, 
# отвечающей на запрос в свободной форме
# с управляемой генерацией ответа, основанной на нейронной выдаче неструктурированных текстовых данных.