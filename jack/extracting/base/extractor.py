import abc
from typing import List

from jack.storing import doc


class BaseExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, query: str, docs: List[doc.Document], outline: str = None) -> List[str]:
        pass
