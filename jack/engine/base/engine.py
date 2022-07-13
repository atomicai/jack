import abc
from typing import Dict, List, Optional, Union

from jack.storing import doc, store


class IR(abc.ABC):
    def __init__(
        self,
        store: store.BaseDocStore,
        query_processor=None,
        query_model=None,
    ) -> None:
        self.store = store
        self.query_processor = query_processor
        self.query_model = query_model

    @abc.abstractmethod
    def retrieve_top_k(
        self, query_batch: List[Dict], index: Optional[str] = "document", top_k: Optional[int] = 5, **kwargs
    ) -> List[Union[Dict, doc.Document]]:
        pass
