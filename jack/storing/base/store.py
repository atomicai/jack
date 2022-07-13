import abc
import logging
from typing import Dict, List, Optional, Union

import numpy as np

from jack.storing import doc

logger = logging.getLogger(__name__)


try:
    from numba import njit  # pylint: disable=import-error
except (ImportError, ModuleNotFoundError):
    logger.debug("Numba not found, replacing njit() with no-op implementation. Enable it with 'pip install numba'.")

    def njit(f):
        return f


@njit  # (fastmath=True)
def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class BaseDocStore(abc.ABC):
    """
    Base class for implementing Document Stores.
    """

    index: Optional[str]
    label_index: Optional[str]
    similarity: Optional[str]

    @abc.abstractmethod
    def write_documents(
        self,
        documents: Union[List[dict], List[doc.Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.

        :return: None
        """

    @abc.abstractmethod
    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[doc.Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """

    @abc.abstractmethod
    def get_document_count(
        self,
        filters: Optional[Dict[str, List[str]]] = None,
        index: Optional[str] = None,
    ) -> int:
        pass

    @abc.abstractmethod
    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Optional[Dict[str, List[str]]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[doc.Document]:
        pass

    @abc.abstractmethod
    def delete_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
    ):
        pass

    @abc.abstractmethod
    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None, batch_size: int = 10_000) -> List[doc.Document]:
        pass

    def normalize_embedding(self, emb: np.ndarray) -> None:
        """
        Performs L2 normalization of embeddings vector inplace. Input can be a single vector (1D array) or a matrix
        (2D array).
        """
        # Might be extended to other normalizations in future

        # Single vec
        if len(emb.shape) == 1:
            self._normalize_embedding_1D(emb)
        # 2D matrix
        else:
            self._normalize_embedding_2D(emb)

    @staticmethod
    @njit  # (fastmath=True)
    def _normalize_embedding_1D(emb: np.ndarray) -> None:
        norm = np.sqrt(emb.dot(emb))  # faster than np.linalg.norm()
        if norm != 0.0:
            emb /= norm

    @staticmethod
    @njit  # (fastmath=True)
    def _normalize_embedding_2D(emb: np.ndarray) -> None:
        for vec in emb:
            vec = np.ascontiguousarray(vec)
            norm = np.sqrt(vec.dot(vec))
            if norm != 0.0:
                vec /= norm

    def scale_to_unit_interval(self, score: float, similarity: Optional[str]) -> float:
        if similarity == "cosine":
            return (score + 1) / 2
        else:
            return float(expit(score / 100))

    def _handle_duplicate_documents(
        self, documents: List[doc.Document], index: Optional[str] = None, duplicate_documents: Optional[str] = None
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        :param documents: A list of Haystack Document objects.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :return: A list of Haystack Document objects.
        """

        index = index or self.index
        if duplicate_documents in ('skip', 'fail'):
            documents = self._drop_duplicate_documents(documents)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index)
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == 'fail':
                raise DuplicateDocumentError(
                    f"Document with ids '{', '.join(ids_exist_in_db)} already exists" f" in index = '{index}'."
                )

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents
