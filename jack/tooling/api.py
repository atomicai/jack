import logging
import subprocess
import time

import numpy as np

logger = logging.getLogger(__name__)

ES_CONTAINER_NAME = "elastic_7.16.1"


def elastic_query_api(
    query_emb: np.ndarray, top_k: int = 10, embedding_field="embedding", similarity="dot_product", mode: str = "strict"
):
    """
    Generate Elasticsearch query for vector similarity.
    """
    if similarity == "cosine":
        similarity_fn_name = "cosineSimilarity"
    elif similarity == "dot_product":
        similarity_fn_name = "dotProduct"
    else:
        raise Exception("Invalid value for similarity in ElasticDocStore. Either \'cosine\' or \'dot_product\'")

    # To handle scenarios where embeddings may be missing
    script_score_query: dict = {"match_all": {}}
    if mode == "strict":
        script_score_query = {"bool": {"filter": {"bool": {"must": [{"exists": {"field": embedding_field}}]}}}}

    query = {
        "script_score": {
            "query": script_score_query,
            "script": {
                # offset score to ensure a positive range as required by Elasticsearch
                "source": f"{similarity_fn_name}(params.query_vector,'{embedding_field}') + 1000",
                "params": {"query_vector": query_emb.tolist()},
            },
        }
    }
    return query


def sql_query_api():
    raise NotImplementedError()


def _on_es_start(sleep=15, delete_existing=False):
    # Start an Elasticsearch server via Docker

    logger.debug("Starting Elasticsearch ...")
    if delete_existing:
        _ = subprocess.run([f"docker rm --force {ES_CONTAINER_NAME}"], shell=True, stdout=subprocess.DEVNULL)
    status = subprocess.run(
        [
            f'docker start {ES_CONTAINER_NAME} > /dev/null 2>&1 || docker run -d -p 9200:9200 -e "discovery.type=single-node" --name {ES_CONTAINER_NAME} elasticsearch:7.16.1'
        ],
        shell=True,
    )
    if status.returncode:
        logger.warning(
            "Tried to start Elasticsearch through Docker but this failed. "
            "It is likely that there is already an existing Elasticsearch instance running. "
        )
    else:
        time.sleep(sleep)
    return status
