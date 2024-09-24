import re
from typing import List, Optional
from langchain_experimental.text_splitter import (
    SemanticChunker,
    BreakpointThresholdType,
)
from langchain_core.embeddings import Embeddings


# This regex is the original `split_text` method of the `SemanticChunker` class.
SENTENCE_SPLITTER_REGEX = r"(?<=[.?!])\s+"


class SemanticChunkerCustom(SemanticChunker):
    def __init__(
        self,
        embeddings: Embeddings,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        max_chunk_length: Optional[int] = None,
    ):
        super().__init__(
            embeddings=embeddings,
            add_start_index=add_start_index,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
        )
        self.max_chunk_length = max_chunk_length

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        chunks = super().split_text(text)

        if not self.max_chunk_length:
            return chunks

        # Modify chunk creation with max_chunk_length check
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_length:
                final_chunks.extend(self.split_chunk_by_length(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def split_chunk_by_length(self, chunk: str) -> List[str]:
        # Splitting the chunk into sentences
        sentences = re.split(SENTENCE_SPLITTER_REGEX, chunk)
        new_chunks = []
        current_chunk = []

        # Check no sentence is longer than the max_chunk_length
        longer_sentence_length = max(len(sentence) for sentence in sentences)
        if longer_sentence_length > self.max_chunk_length:
            raise ValueError(
                f"Got a sentence longer than `max_chunk_length`: {longer_sentence_length}"
            )

        for sentence in sentences:
            # Check if adding the next sentence exceeds the max_chunk_length
            if len(" ".join(current_chunk + [sentence])) <= self.max_chunk_length:
                current_chunk.append(sentence)
            else:
                # If current_chunk is not empty, save it as a new chunk
                if current_chunk:
                    new_chunks.append(" ".join(current_chunk))
                # Start a new chunk with the current sentence
                current_chunk = [sentence]

        # Add the last chunk if it exists
        if current_chunk:
            new_chunks.append(" ".join(current_chunk))

        return new_chunks
