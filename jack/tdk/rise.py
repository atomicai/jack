import json
import logging
import os
import pathlib

import dotenv
from flask import Flask, request, send_from_directory
from jack.engine import bm25
from jack.storing.module import memo

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

dotenv.load_dotenv()

top_k = int(os.environ.get("TOP_K", 5))
index = os.environ.get("INDEX", "document")
store = memo.MemoDocStore(index=index)
engine = bm25.BM25Retriever(store, top_k=top_k)

app = Flask(
    __name__,
    template_folder='build',
    static_folder='build',
    root_path=pathlib.Path(os.getcwd()) / 'jack',
)


@app.route('/', defaults={'path': ''})
@app.route("/<path:path>")
def index(path):
    if path != '' and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route("/search", methods=["POST"])
def search():
    data = request.get_data(parse_form_data=True).decode('utf-8-sig')
    data = json.loads(data)
    query = data["inputValue"]
    response = list(engine.retrieve_top_k(query))[0][0]
    docs = [{"text": d.text, "title": d.meta["label"]} for d in response]
    response = {"query": {"docs": docs}, "passage": {"docs": docs}}

    return response
