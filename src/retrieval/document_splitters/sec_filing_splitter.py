from collections import deque
import re
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

from ...model.custom_document import SplitContent


from ..document_preprocessors.table_serializer import TableSerializer

from .semantic_chunker_custom import (
    SemanticChunkerCustom
)
from .document_splitter import DocumentSplitter


class SECFilingSplitter(DocumentSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 10 if self.chunk_size else 0,
        )


class SECFilingSplitterHTML(DocumentSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size * 0.05 if self.chunk_size else 0,
        )


class SECFilingSplitterSemantic(DocumentSplitter):
    def __init__(
        self, embeddings, table_serializer, preprocess_config, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_splitter = SemanticChunkerCustom(
            embeddings=embeddings,
            max_chunk_length=self.chunk_size,
            breakpoint_threshold_amount=95,
            table_serializer=table_serializer,
            preprocess_config=preprocess_config,
        )
        
    def split_text_to_list(self, text: List[SplitContent]) -> List[SplitContent]:
        if isinstance(self.text_splitter, SemanticChunkerCustom):
            return self.text_splitter.split_text_to_list(text)
        else:
            raise ValueError("Text splitter is not a SemanticChunkerCustom instance")
