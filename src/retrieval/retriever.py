from __future__ import annotations
import logging
from typing import Any, List, Optional

from ..model.custom_document import CustomDocument

from .qdrant_store import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableConfig


class QdrantRetriever(VectorStoreRetriever):
    search_type_custom: str = "similarity_with_scores"

    def __init__(
        self, vector_store: QdrantVectorStore, retriever_num_documents: int
    ) -> None:
        super().__init__(
            vectorstore=vector_store,
            search_kwargs={"k": retriever_num_documents},
        )

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        if self.search_type_custom == "similarity_with_scores":
            results = self.vectorstore.similarity_search_with_score(
                input, **self.search_kwargs
            )
            docs_with_scores = []
            for doc, score in results:
                doc.metadata["similarity_score"] = score
                docs_with_scores.append(doc)
            return docs_with_scores
        else:
            return super().invoke(input, config, **kwargs)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        # Load full docs based on ids, only add unique docs
        # so if multiple chunks of the same doc are passed the corresponding document should be loaded only once
        docs = CustomDocument.docs_to_custom_docs(docs)
        # documents = LocalStore().get_unique_documents(docs)
        return "\n\n".join(
            [f"Chunk {i+1}: \n{doc.page_content}" for i, doc in enumerate(docs)]
        )
