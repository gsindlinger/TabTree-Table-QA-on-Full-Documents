import re
from typing import Dict, List, Optional, Tuple
from langchain_experimental.text_splitter import (
    SemanticChunker,
    BreakpointThresholdType,
)
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter

from ...model.custom_document import SplitContent


SENTENCE_SPLITTER_REGEX = (
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+|\n\n\(.+?\)\n\n"
)
TABLE_REGEX = r"(<table>.*?</table>)"


class SemanticChunkerCustom(SemanticChunker, TextSplitter):
    def __init__(
        self,
        embeddings: Embeddings,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        max_chunk_length: Optional[int] = None,
        table_separator: str = TABLE_REGEX,
        ignore_tables_for_embeddings: bool = False,
    ):
        self.sentence_split_regex = SENTENCE_SPLITTER_REGEX
        self.max_chunk_length = max_chunk_length

        super().__init__(
            embeddings=embeddings,
            add_start_index=add_start_index,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
            sentence_split_regex=self.sentence_split_regex,
        )

        # introduce a regexp for treating tables differently
        self.table_separator = table_separator
        # if embeddings should be generated without considering table input data then set this to True
        self.ignore_tables_for_embeddings = ignore_tables_for_embeddings

    def split_text(
        self,
        text: str | List[SplitContent],
    ) -> List[str]:
        if self.ignore_tables_for_embeddings:
            if not isinstance(text, list):
                raise ValueError(
                    "The input text should be a list of SplitContent objects. Please set `ignore_tables_for_embeddings` to True."
                )
            chunks = self.split_text_without_tables(text)
        else:
            if not isinstance(text, str):
                raise ValueError(
                    "The input text should be a string. Please set `ignore_tables_for_embeddings` to False."
                )
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
        sentences = re.split(self.sentence_split_regex, chunk, flags=re.DOTALL)
        new_chunks = []
        current_chunk = []

        # Check no sentence is longer than the max_chunk_length
        longer_sentence_length = max(len(sentence) for sentence in sentences)
        for sentence in sentences:
            if self.max_chunk_length and len(sentence) > self.max_chunk_length:
                raise ValueError(
                    f"Got a sentence longer than `max_chunk_length`: {len(sentence)}"
                )
        if self.max_chunk_length and longer_sentence_length > self.max_chunk_length:
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

    def split_text_without_tables(
        self,
        content_list: List[SplitContent],
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        # List to hold sentences and tables with their type and content
        # Filter out only the sentences for distance calculations
        sentences = [item.content for item in content_list if item.type == "text"]

        # Calculate distances for sentences
        distances, sentences = self._calculate_sentence_distances(sentences)

        # If there's only one sentence, return it directly
        if len(sentences) == 1:
            return [sentences[0]["content"]], content_list

        # Calculate breakpoints for chunking
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through breakpoints to slice the sentences and tables
        for index in indices_above_thresh:
            end_index = index
            # Find first and last items in the content list
            first_item = next(
                item
                for item in content_list
                if item.content == sentences[start_index]["sentence"]
                and not item.visited
            )
            last_item = next(
                item
                for item in content_list
                if item.content == sentences[end_index]["sentence"] and not item.visited
            )

            # Mark the items as visited
            for i in range(first_item.position, last_item.position + 1):
                content_list[i].visited = True

            group = content_list[first_item.position : last_item.position + 1]
            combined_text = " ".join(
                item.content for item in group if item.content.strip() != ""
            )
            chunks.append(combined_text)
            start_index = index + 1

        # Handle remaining content after the last breakpoint
        if start_index < len(sentences):
            first_item = next(
                item
                for item in content_list
                if item.content == sentences[start_index]["sentence"]
                and not item.visited
            )
            group = content_list[first_item.position :]
            combined_text = " ".join(item.content for item in group)
            chunks.append(combined_text)
        return chunks
