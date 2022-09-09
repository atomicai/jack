import copy
import logging
import os
import pathlib
import typing
from datetime import datetime
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import simplejson
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from jack.etc import Answer, Document, Label, Span
from jack.logging.module import logger as ai_logger
from jack.recoiling.module import recoil

logger = logging.getLogger(__name__)

stopper = recoil.Stopper(pathlib.Path(os.getcwd()) / "jack" / "configuring" / "stopwords.txt")
# morpher = recoil.Morpher(tags="")
chunker = recoil.Chunker(n=int(os.environ.get("MAX_SEQ_LEN", 192)), m=64)


def _train(model, optimizer, data_silo, epochs, n_gpu, lr_schedule, evaluate_every, tracker, device, prefix=None):
    prefix = "" if prefix is None else str(prefix)
    trainer = Trainer(
        prefix=prefix,
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        log_loss_every=1,
        evaluate_every=evaluate_every,
        tracker=tracker,
        device=device,
    )
    model = trainer.train()

    return model


def _processing(
    pretrained_model_name_or_path,
    do_lower_case: bool,
    label_list: List[str],
    metric: str = None,
    batch_size: int = 4,
    data_dir: pathlib.Path = None,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
    dev_filename: str = "test.csv",
    dev_stratification: bool = True,
    max_seq_len: int = 192,
):
    tokenizer = Tokenizer.load(pretrained_model_name_or_path, do_lower_case=do_lower_case, use_fast=True)

    is_model_name: bool = True
    try:
        path = pathlib.Path(pretrained_model_name_or_path)
        assert path.is_dir()
    except:
        is_model_name = False

    # data_dir = pathlib.Path.home() / "Dataset" if data_dir is None else pathlib.Path(data_dir)
    if is_model_name:
        processor = TextClassificationProcessor.load_from_dir(path)
    else:
        processor = TextClassificationProcessor(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            data_dir=data_dir,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            label_list=label_list,
            metric=metric,
            dev_split=0.0,
            delimiter=",",
            dev_stratification=dev_stratification,
            text_column_name="text",
            label_column_name="label",
        )

    data_silo = DataSilo(processor=processor, max_processes=1, batch_size=batch_size)

    return tokenizer, processor, data_silo


def _modeling(pretrained_model_name_or_path, data_silo, loss_fn: str, label_list: List[str], device, is_training: bool = False):
    language_model = LanguageModel.load(pretrained_model_name_or_path)
    # b) and a prediction head on top that is suited for our task => Text classification
    loss_fn = "crossentropy"
    prediction_head = TextClassificationHead(
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
        num_labels=len(label_list),
        loss_fn=loss_fn,
    )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device,
    )

    return model


def _logging(
    project_name: str,
    experiment_name: str,
    prefix: str = "",
    use_wandb: bool = False,
):
    if use_wandb:
        return ai_logger.WANDBLogger.init_experiment(
            project_name=project_name,
            experiment_name=experiment_name,
            prefix=f"{prefix} ¬ ",
            api=os.environ.get("WANDB_API_KEY"),
            sync_step=False,
        )

    ml_logger = ai_logger.JUSTLogger.init_experiment(experiment_name=experiment_name, project_name=project_name)

    return ml_logger


def _optimizing(model, device, n_batches, n_epochs, use_amp: bool = None, lr: float = 3e-5):
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=lr,
        device=device,
        n_batches=n_batches,
        n_epochs=n_epochs,
        use_amp=use_amp,
    )
    return model, optimizer, lr_schedule


def eval_data_from_json(
    filename: str, max_docs: Union[int, bool] = None, preprocessor: Callable = None, open_domain: bool = False
) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
    :param open_domain: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.
    """
    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r", encoding="utf-8") as file:
        data = simplejson.load(file)
        if "title" not in data["data"][0]:
            logger.warning(f"No title information found for documents in QA file: {filename}")

        for squad_document in data["data"]:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(
                squad_document, preprocessor, open_domain
            )
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)
    if len(problematic_ids) > 0:
        logger.warning(
            f"Could not convert an answer for {len(problematic_ids)} questions.\n"
            f"There were conversion errors for question ids: {problematic_ids}"
        )
    return docs, labels


def eval_data_from_jsonl(
    filename: str,
    batch_size: Optional[int] = None,
    max_docs: Union[int, bool] = None,
    preprocessor: Callable = None,
    open_domain: bool = False,
) -> Generator[Tuple[List[Document], List[Label]], None, None]:
    """
    Read Documents + Labels from a SQuAD-style file in jsonl format, i.e. one document per line.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    This is a generator which will yield one tuple per iteration containing a list
    of batch_size documents and a list with the documents' labels.
    If batch_size is set to None, this method will yield all documents and labels.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
    :param open_domain: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.
    """
    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r", encoding="utf-8-sig") as file:
        for document in file:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            squad_document = simplejson.loads(document)
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(
                squad_document, preprocessor, open_domain
            )
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)

            if batch_size is not None:
                if len(docs) >= batch_size:
                    if len(problematic_ids) > 0:
                        logger.warning(
                            f"Could not convert an answer for {len(problematic_ids)} questions.\n"
                            f"There were conversion errors for question ids: {problematic_ids}"
                        )
                    yield docs, labels
                    docs = []
                    labels = []
                    problematic_ids = []

    yield docs, labels


def squad_json_to_jsonl(squad_file: str, output_file: str):
    """
    Converts a SQuAD-json-file into jsonl format with one document per line.

    :param squad_file: SQuAD-file in json format.
    :param output_file: Name of output file (SQuAD in jsonl format)
    """
    with open(squad_file, encoding="utf-8-sig") as json_file, open(output_file, "w", encoding="utf-8-sig") as jsonl_file:
        squad_json = simplejson.load(json_file)

        for doc in squad_json["data"]:
            simplejson.dump(doc, jsonl_file)
            jsonl_file.write("\n")


def _extract_docs_and_labels_from_dict(document_dict: Dict, preprocessor: Callable = None, open_domain: bool = False):
    """
    Set open_domain to True if you are trying to load open_domain labels (i.e. labels without doc id or start idx)
    """
    docs = []
    labels = []
    problematic_ids = []

    # get all extra fields from document level (e.g. title)
    meta_doc = {k: v for k, v in document_dict.items() if k not in ("paragraphs", "title")}
    for paragraph in document_dict["paragraphs"]:
        ## Create Metadata
        cur_meta = {"name": document_dict.get("title", None)}
        # all other fields from paragraph level
        meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
        cur_meta.update(meta_paragraph)
        # meta from parent document
        cur_meta.update(meta_doc)

        ## Create Document
        cur_full_doc = Document(content=paragraph["context"], meta=cur_meta)
        if preprocessor is not None:
            splits_docs = preprocessor.process(cur_full_doc)
            # we need to pull in _split_id into the document id for unique reference in labels
            splits: List[Document] = []
            offset = 0
            for d in splits_docs:
                id = f"{d.id}-{d.meta['_split_id']}"
                d.meta["_split_offset"] = offset
                offset += len(d.content)
                # offset correction based on splitting method
                if preprocessor.split_by == "word":
                    offset += 1
                elif preprocessor.split_by == "passage":
                    offset += 2
                else:
                    raise NotImplementedError
                mydoc = Document(content=d.content, id=id, meta=d.meta)
                splits.append(mydoc)
        else:
            splits = [cur_full_doc]
        docs.extend(splits)

        ## Assign Labels to corresponding documents
        for qa in paragraph["qas"]:
            if not qa.get("is_impossible", False):
                for answer in qa["answers"]:
                    ans = answer["text"]
                    # TODO The following block of code means that answer_start is never calculated
                    #  and cur_id is always None for open_domain
                    #  This can be rewritten so that this function could try to calculate offsets
                    #  and populate id in open_domain mode
                    if open_domain:
                        # TODO check with Branden why we want to treat open_domain here differently.
                        # Shouldn't this be something configured at eval time only?
                        cur_ans_start = answer.get("answer_start", 0)
                        # cur_id = '0'
                        label = Label(
                            query=qa["question"],
                            answer=Answer(answer=ans, type="extractive", score=0.0),
                            document=None,  # type: ignore
                            is_correct_answer=True,
                            is_correct_document=True,
                            no_answer=qa.get("is_impossible", False),
                            origin="gold-label",
                        )
                        labels.append(label)
                    else:
                        ans_position = cur_full_doc.content[answer["answer_start"] : answer["answer_start"] + len(ans)]
                        if ans != ans_position:
                            # do not use answer
                            problematic_ids.append(qa.get("id", "missing"))
                            break
                        # find corresponding document or split
                        if len(splits) == 1:
                            # cur_id = splits[0].id
                            cur_ans_start = answer["answer_start"]
                            cur_doc = splits[0]
                        else:
                            for s in splits:
                                # If answer start offset is contained in passage we assign the label to that passage
                                if (answer["answer_start"] >= s.meta["_split_offset"]) and (
                                    answer["answer_start"] < (s.meta["_split_offset"] + len(s.content))
                                ):
                                    cur_doc = s
                                    cur_ans_start = answer["answer_start"] - s.meta["_split_offset"]
                                    # If a document is splitting an answer we add the whole answer text to the document
                                    if s.content[cur_ans_start : cur_ans_start + len(ans)] != ans:
                                        s.content = s.content[:cur_ans_start] + ans
                                    break
                        cur_answer = Answer(
                            answer=ans,
                            type="extractive",
                            score=0.0,
                            context=cur_doc.content,
                            offsets_in_document=[Span(start=cur_ans_start, end=cur_ans_start + len(ans))],
                            offsets_in_context=[Span(start=cur_ans_start, end=cur_ans_start + len(ans))],
                            document_id=cur_doc.id,
                        )
                        label = Label(
                            query=qa["question"],
                            answer=cur_answer,
                            document=cur_doc,
                            is_correct_answer=True,
                            is_correct_document=True,
                            no_answer=qa.get("is_impossible", False),
                            origin="gold-label",
                        )
                        labels.append(label)
            else:
                # for no_answer we need to assign each split as not fitting to the question
                for s in splits:
                    label = Label(
                        query=qa["question"],
                        answer=Answer(
                            answer="",
                            type="extractive",
                            score=0.0,
                            offsets_in_document=[Span(start=0, end=0)],
                            offsets_in_context=[Span(start=0, end=0)],
                        ),
                        document=s,
                        is_correct_answer=True,
                        is_correct_document=True,
                        no_answer=qa.get("is_impossible", False),
                        origin="gold-label",
                    )

                    labels.append(label)

    return docs, labels, problematic_ids


def convert_date_to_rfc3339(date: str) -> str:
    """
    Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
    timezone.

    If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
    and UTC as default time zone.

    This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
    and filter_utils.py.
    """
    parsed_datetime = datetime.fromisoformat(date)
    if parsed_datetime.utcoffset() is None:
        converted_date = parsed_datetime.isoformat() + "Z"
    else:
        converted_date = parsed_datetime.isoformat()

    return converted_date
