import uuid
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import dacite
import mmh3
import numpy as np
import pandas as pd


class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        score: Optional[float] = None,
        probability: Optional[float] = None,
        question: Optional[str] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id_hash_keys: Optional[List[str]] = None,
        uuid_type: Optional[str] = None,
    ):
        """

        Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
        into smaller passages. We'll have one Document per passage in this case.

        Each document has a unique ID. This can be supplied by the user or generated automatically.
        It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

        There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

        :param text: Text of the document
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: Retriever's query score for a retrieved document
        :param probability: a pseudo probability by scaling score in the range 0 to 1
        :param question: Question text (e.g. for FAQs where one document usually consists of one question and one answer text).
        :param meta: Meta fields for a document like name, url, or author.
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].
        """

        self.text = text
        self.score = score
        self.probability = probability
        self.question = question
        self.meta = meta or {}
        self.embedding = embedding

        self.id = self._get_id(id_hash_keys, uuid_type=uuid_type) if id is None else str(id)

    def _get_id(self, id_hash_keys, uuid_type=None):
        if uuid_type is None:
            return "{:02x}".format(mmh3.hash128(self.text, signed=False))
        elif uuid_type == "uuid3":
            return str(uuid.uuid3(uuid.NAMESPACE_DNS, self.text))
        elif uuid_type == 'uuid5':
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.text))
        else:
            raise ValueError(f"Choose either \"uuid3\" or \"uuid5\" or None")

    def to_dict(self, field_map={}):
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}, uuid_type=None):
        _doc = deepcopy(dict)
        init_args = [
            "text",
            "id",
            "score",
            "probability",
            "question",
            "meta",
            "embedding",
        ]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        if uuid_type:
            _new_doc["uuid_type"] = uuid_type

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())


class Label:
    def __init__(
        self,
        question: str,
        answer: str,
        is_correct_answer: bool,
        is_correct_document: bool,
        origin: str,
        id: Optional[str] = None,
        document_id: Optional[str] = None,
        offset_start_in_doc: Optional[int] = None,
        no_answer: Optional[bool] = None,
        model_id: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        self.created_at = created_at
        self.updated_at = updated_at
        self.question = question
        self.answer = answer
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.document_id = document_id
        self.offset_start_in_doc = offset_start_in_doc
        self.no_answer = no_answer
        self.model_id = model_id

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    # define __eq__ and __hash__ functions to deduplicate Label Objects
    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, 'question', None) == self.question
            and getattr(other, 'answer', None) == self.answer
            and getattr(other, 'is_correct_answer', None) == self.is_correct_answer
            and getattr(other, 'is_correct_document', None) == self.is_correct_document
            and getattr(other, 'origin', None) == self.origin
            and getattr(other, 'document_id', None) == self.document_id
            and getattr(other, 'offset_start_in_doc', None) == self.offset_start_in_doc
            and getattr(other, 'no_answer', None) == self.no_answer
            and getattr(other, 'model_id', None) == self.model_id
            and getattr(other, 'created_at', None) == self.created_at
            and getattr(other, 'updated_at', None) == self.updated_at
        )

    def __hash__(self):
        return hash(
            self.question
            + self.answer
            + str(self.is_correct_answer)
            + str(self.is_correct_document)
            + str(self.origin)
            + str(self.document_id)
            + str(self.offset_start_in_doc)
            + str(self.no_answer)
            + str(self.model_id)
        )

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


class MultiLabel:
    def __init__(
        self,
        question: str,
        multiple_answers: List[str],
        is_correct_answer: bool,
        is_correct_document: bool,
        origin: str,
        multiple_document_ids: List[Any],
        multiple_offset_start_in_docs: List[Any],
        no_answer: Optional[bool] = None,
        model_id: Optional[int] = None,
    ):
        """
        Object used to aggregate multiple possible answers for the same question

        :param question: the question(or query) for finding answers.
        :param multiple_answers: list of possible answer strings
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param multiple_document_ids: the document_store's IDs for the returned answer documents.
        :param multiple_offset_start_in_docs: the answer start offsets in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        """
        self.question = question
        self.multiple_answers = multiple_answers
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.multiple_document_ids = multiple_document_ids
        self.multiple_offset_start_in_docs = multiple_offset_start_in_docs
        self.no_answer = no_answer
        self.model_id = model_id

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


@dataclass
class Span:
    start: int
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index. 
    For extractive QA: Character where answer starts/ends  
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)
    
    :param start: Position where the span starts
    :param end:  Position where the spand ends
    """


@dataclass
class Answer:
    answer: str
    type: Literal["generative", "extractive", "other"] = "extractive"
    score: Optional[float] = None
    context: Optional[Union[str, pd.DataFrame]] = None
    offsets_in_document: Optional[List[Span]] = None
    offsets_in_context: Optional[List[Span]] = None
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    """
    The fundamental object in Haystack to represent any type of Answers (e.g. extractive QA, generative QA or TableQA).
    For example, it's used within some Nodes like the Reader, but also in the REST API.
    :param answer: The answer string. If there's no possible answer (aka "no_answer" or "is_impossible) this will be an empty string.
    :param type: One of ("generative", "extractive", "other"): Whether this answer comes from an extractive model 
                 (i.e. we can locate an exact answer string in one of the documents) or from a generative model 
                 (i.e. no pointer to a specific document, no offsets ...). 
    :param score: The relevance score of the Answer determined by a model (e.g. Reader or Generator).
                  In the range of [0,1], where 1 means extremely relevant.
    :param context: The related content that was used to create the answer (i.e. a text passage, part of a table, image ...)
    :param offsets_in_document: List of `Span` objects with start and end positions of the answer **in the
                                document** (as stored in the document store).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here) 
    :param offsets_in_context: List of `Span` objects with start and end positions of the answer **in the
                                context** (i.e. the surrounding text/table of a certain window size).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here) 
    :param document_id: ID of the document that the answer was located it (if any)
    :param meta: Dict that can be used to associate any kind of custom meta data with the answer. 
                 In extractive QA, this will carry the meta data of the document where the answer was found.
    """

    def __post_init__(self):
        # In case offsets are passed as dicts rather than Span objects we convert them here
        # For example, this is used when instantiating an object via from_json()
        if self.offsets_in_document is not None:
            self.offsets_in_document = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_document]
        if self.offsets_in_context is not None:
            self.offsets_in_context = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_context]

        if self.meta is None:
            self.meta = {}

    def __lt__(self, other):
        """Enable sorting of Answers by score"""
        return self.score < other.score

    def __str__(self):
        # self.context might be None (therefore not subscriptable)
        if not self.context:
            return f"<Answer: answer='{self.answer}', score={self.score}, context=None>"
        return f"<Answer: answer='{self.answer}', score={self.score}, context='{self.context[:50]}{'...' if len(self.context) > 50 else ''}'>"

    def __repr__(self):
        return f"<Answer {asdict(self)}>"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: dict):
        return dacite.from_dict(data_class=cls, data=dict)

    def to_json(self):
        return simplejson.dumps(self, default=pydantic_encoder)

    @classmethod
    def from_json(cls, data):
        if type(data) == str:
            data = simplejson.loads(data)
        return cls.from_dict(data)
