import abc
from typing import Dict, List, Optional, Union

from jack.etc import Document


class IR(abc.ABC):
    def __init__(
        self,
        store,
        query_processor=None,
        query_model=None,
    ) -> None:
        self.store = store
        self.query_processor = query_processor
        self.query_model = query_model

    @abc.abstractmethod
    def retrieve(
        self, query_batch: Union[List[Dict], Dict], index: Optional[str] = "document", top_k: Optional[int] = 5, **kwargs
    ) -> List[Union[Dict, Document]]:
        pass
