import logging
from typing import List

from ..model.custom_document import CustomDocument

from .qdrant_store import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents.base import Document


class QdrantRetriever(VectorStoreRetriever):
    def __init__(self, vector_store: QdrantVectorStore) -> None:
        super().__init__(
            vectorstore=vector_store, search_type="similarity", search_kwargs={"k": 2}
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        # Load full docs based on ids, only add unique docs
        # so if multiple chunks of the same doc are passed the corresponding document should be loaded only once
        docs = CustomDocument.docs_to_custom_docs(docs)
        # documents = LocalStore().get_unique_documents(docs)
        return "\n\n".join(
            [f"Chunk {i}: \n{doc.page_content}" for i, doc in enumerate(docs)]
        )
