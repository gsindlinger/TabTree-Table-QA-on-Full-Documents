from __future__ import annotations
import logging
from typing import Any, List, Optional, Tuple

from ..config.config import Config
from ..model.custom_document import CustomDocument, FullMetadataRetrieval
from ..retrieval.document_preprocessors.table_serializer import TableSerializer


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
            search_kwargs={
                "k": retriever_num_documents,
                "score_threshold": Config.run.retriever_score_threshold,
            },
        )

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from .qdrant_client import QdrantClient
        
        if self.search_type_custom == "similarity_with_scores":
            results_with_scores = self.vectorstore.similarity_search_with_score(
                input, **self.search_kwargs
            )
            results = [result[0] for result in results_with_scores]
            scores = [result[1] for result in results_with_scores]
            
            results = QdrantClient.extend_docs_with_payload(results)
            docs_with_scores = []
            for doc, score in zip(results, scores):
                doc.metadata["similarity_score"] = score
                docs_with_scores.append(doc)
            return docs_with_scores
        else:
            return super().invoke(input, config, **kwargs)

    @staticmethod
    def format_docs(docs: List[Document], table_serializer: TableSerializer | None, max_character: int = 200000,) -> Tuple[str, str]:
        from ..retrieval.document_preprocessors.html_preprocessor import HTMLPreprocessor
        
        """
        Formats retrieved document chunks and related tables separately while enforcing a max character limit.

        Args:
            docs (List[Document]): Retrieved document chunks.
            max_character (int): Maximum character limit for the combined output.

        Returns:
            Tuple[str, str]: Formatted chunk content and related tables (if within limit).
        """
        custom_docs = CustomDocument.docs_to_custom_docs(docs)

        # Format text chunks first
        context_list = []
        current_length = 0

        for i, doc in enumerate(custom_docs):
            chunk_text = f"Chunk {i+1}:\n{doc.page_content}"
            if current_length + len(chunk_text) <= max_character:
                context_list.append(chunk_text)
                current_length += len(chunk_text)
            else:
                break  # Stop adding more chunks if the limit is exceeded

        context_str = "\n\n".join(context_list)

        # Extract and format related tables (if space allows)
        related_tables_list = []
        
        for doc in custom_docs:                
            if isinstance(doc.metadata, FullMetadataRetrieval) and doc.metadata.table_string:
                table_list = []
                if table_serializer:
                    string_splits = HTMLPreprocessor()._split_sentences_and_tables(doc.metadata.table_string)
                    for split_content in string_splits:
                        if split_content.type == "table":
                            table_list.append(split_content.content)
                else:
                    table_list.append(doc.metadata.table_string)
                
        for i, table in enumerate(table_list): 
            if table_serializer:  
                table_serialized = table_serializer.serialize_table_to_str(table)
            else: 
                table_serialized = table
            table_text = f"Table {i+1}:\n{table_serialized}"
            if current_length + len(table_text) <= max_character:
                related_tables_list.append(table_text)
                current_length += len(table_text)
            else:
                break  # Stop adding tables if the character limit is exceeded

        related_tables_str = "\n\n".join(related_tables_list) if related_tables_list else "None"

        return context_str, related_tables_str

