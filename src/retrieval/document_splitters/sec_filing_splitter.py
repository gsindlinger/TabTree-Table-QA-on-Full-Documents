from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language, TextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from .semantic_chunker_custom import (
    SemanticChunkerCustom,
)
from ...retrieval.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from .document_splitter import DocumentSplitter


class SECFilingSplitter(DocumentSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 10,
        )


class SECFilingSplitterHTML(DocumentSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size * 0.05,
        )


class SECFilingSplitterSemantic(DocumentSplitter):
    def __init__(self, embeddings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = SemanticChunkerCustom(
            embeddings=embeddings, max_chunk_length=self.chunk_size
        )
