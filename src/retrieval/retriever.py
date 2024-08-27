from .qdrant_store import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever


class QdrantRetriever(VectorStoreRetriever):
    def __init__(self, vector_store: QdrantVectorStore) -> None:
        super().__init__(
            vectorstore=vector_store, search_type="similarity", search_kwargs={"k": 3}
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
