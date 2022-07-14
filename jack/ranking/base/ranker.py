import abc
from typing import List


class BaseRanker(abc.ABC):
    def __init__(self, label_list: List[str]):
        self.klasses = label_list

    @abc.abstractmethod
    def rank(self, query: str):
        pass
