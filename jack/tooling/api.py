import numpy as np


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
