import logging
from typing import List

from ..model.custom_document import CustomDocument

from .qdrant_store import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents.base import Document


class QdrantRetriever(VectorStoreRetriever):
    def __init__(self, vector_store: QdrantVectorStore) -> None:
        super().__init__(
            vectorstore=vector_store, search_type="similarity", search_kwargs={"k": 3}
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        # Load full docs based on ids, only add unique docs
        # so if multiple chunks of the same doc are passed the corresponding document should be loaded only once
        docs = CustomDocument.docs_to_custom_docs(docs)
        # documents = LocalStore().get_unique_documents(docs)

        logging.info(
            "Following documents are retrieved: "
            + ",".join([doc.metadata.doc_id for doc in docs])
        )
        logging.info(
            "Following chunks are retrieved: "
            + ",".join([doc.page_content for doc in docs])
        )

        return "\n\n".join(
            [f"Source: {doc.metadata.doc_id}\n\n{doc.page_content}" for doc in docs]
        )
