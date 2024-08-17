import os
from fastapi import FastAPI
import requests

from haystack import Document, Pipeline
from haystack.utils import Secret
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


# Global Variables
## FastAPI
app = FastAPI()
openai_api_key = os.environ.get("OPENAI_API_KEY")

## Haystack
document_store = InMemoryDocumentStore()
document_embedder = SentenceTransformersDocumentEmbedder()
document_writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)
text_embedder = SentenceTransformersTextEmbedder()
query_retriever = InMemoryEmbeddingRetriever(document_store=document_store)
generator = OpenAIGenerator(model="gpt-3.5-turbo")
template = """
                Given the following information, answer the query
                based on data from Document store.

                Context: 
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}

                Question: {{ query }}?
              """


def get_data():
    url = "https://poetrydb.org/author/"
    author_name = "William Shakespeare"
    data = requests.get(url + author_name)
    data = data.json()

    documents = []
    for doc in data:
        sentence = " ".join(doc["lines"])

        documents.append(
            Document(
                content=sentence,
                meta={
                    "Title": doc["title"],
                    "Author": doc["author"],
                    "Linecount": doc["linecount"],
                },
            )
        )

    return documents


def build_indexing_pipeline():
    # Create a new index pipeline
    indexing_pipeline = Pipeline()

    # Define all the components
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    # Connect the components
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline


def build_query_pipeline():
    # Create a new query pipeline
    query_pipeline = Pipeline()

    # Define all the components
    query_pipeline.add_component("embedder", text_embedder)
    query_pipeline.add_component("retriever", query_retriever)

    query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    query_pipeline.add_component("generator", generator)

    # Connect the components
    query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "generator")

    return query_pipeline


@app.get("/")
def server_status():
    return {"status": "Server is running..."}


@app.post("/index")
def create_document_store():
    documents = get_data()
    indexing_pipeline = build_indexing_pipeline()
    return indexing_pipeline.run({"documents": documents})


@app.get("/list")
def get_documents():
    return {"Documents": document_store.count_documents()}


@app.post("/generate")
def generate_answer(question: str):
    query_pipeline = build_query_pipeline()
    response = query_pipeline.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"query": question},
        }
    )
    return {"response": response}
