import json
import logging
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
import tiktoken

from ..config.config import Config
from ..retrieval.document_splitters.document_splitter import DocumentSplitter
from ..retrieval.document_loaders.document_loader import DocumentLoader
from .full_document_storage.local_store import LocalStore
from .qdrant_store import QdrantVectorStore


class IndexingService(BaseModel):
    qdrant_store: QdrantVectorStore

    class Config:
        arbitrary_types_allowed = True

    def create_index(self):
        self.qdrant_store.client.create_index(
            collection_name=self.qdrant_store.collection_name
        )

    def embed_documents(self):
        # Load & Preprocess documents
        document_loader = DocumentLoader.from_config()
        documents = [
            document_loader.load_single_document(Config.sec_filings.data_path_single)
        ]

        # documents = document_loader.load_documents(
        #     num_of_documents=10, preprocess_mode="default"
        # )

        # Store full documents
        initial_number_of_documents = LocalStore().store_full_documents(
            documents=documents
        )

        # Split documents
        document_splitter = DocumentSplitter.from_config(
            embeddings=self.qdrant_store.embeddings
        )

        documents = document_splitter.split_documents(documents)

        # Embed documents
        vectors = self.qdrant_store.embeddings.embed_documents(
            texts=[doc.page_content for doc in documents]
        )
        logging.info(
            f"Generated {len(vectors)} for {initial_number_of_documents} documents"
        )

        # Check if collection exists, else delete it
        if self.qdrant_store.client.is_populated(
            collection_name=self.qdrant_store.collection_name, accept_empty=False
        ):
            self.qdrant_store.client.delete_collection(
                collection_name=self.qdrant_store.collection_name
            )
            logging.info(
                f"Deleted existing collection {self.qdrant_store.collection_name}"
            )

        # Store documents using Qdrant client
        self.qdrant_store.client.add_documents(
            documents=documents,
            collection_name=self.qdrant_store.collection_name,
            vectors=vectors,
        )

    def analyze_documents(self):
        """
        Analyze the documents which are used for the application.
        Contains the following approach:
            - Count tokens based on tiktoken counts of cl100k_base
        """
        document_loader = DocumentLoader.from_config()

        token_counts: List[List[int]] = []

        for preprocess_mode in ["none", "remove-attributes", "remove-invisible", "all"]:
            loaded_documents = document_loader.load_documents(
                preprocess_mode=preprocess_mode, num_of_documents=10
            )
            num_tokens = []
            num_characters = []
            for document in loaded_documents:
                num_tokens.append(
                    IndexingService.num_tokens_from_string(document.page_content)
                )
                num_characters.append(len(document.page_content))

            token_counts.append(
                {
                    "preprocess_mode": preprocess_mode,
                    "num_characters": num_characters,
                    "num_tokens": num_tokens,
                    "avg": sum(num_tokens) / len(num_tokens),
                    "min": min(num_tokens),
                    "max": max(num_tokens),
                    "std": np.std(num_tokens),
                }
            )
            logging.info("Analysed documents for preprocess_mode: " + preprocess_mode)

        # Store token counts at ".data/analysis/token_counts.json" locally
        token_counts_path = "./data/analysis/token_counts/"
        # Ensure the directory exists
        os.makedirs(token_counts_path, exist_ok=True)
        with open(f"{token_counts_path}token_counts.json", "w") as file:
            json.dump(token_counts, file, indent=4)

        avg_token_counts = [entry["avg"] for entry in token_counts]
        std_token_counts = [entry["std"] for entry in token_counts]
        preprocess_modes = [entry["preprocess_mode"] for entry in token_counts]

        # Create bar plot with error bars
        plt.bar(
            preprocess_modes,
            avg_token_counts,
            yerr=std_token_counts,
            align="center",
            alpha=0.5,
            ecolor="black",
            capsize=10,
        )

        # Add dashed lines and text values on top of the bars
        for i, avg in enumerate(avg_token_counts):
            rounded_avg = round(avg)
            plt.text(
                i,
                avg + std_token_counts[i],
                f"{rounded_avg:,}",
                ha="center",
                va="bottom",
                color="black",
                fontstyle="italic",
            )
            plt.axhline(y=avg, color="gray", linestyle="--", linewidth=0.75)

        # Labels and title
        plt.ylabel("Average number of tokens")
        plt.xlabel("Preprocess mode")
        plt.yscale("log")
        plt.title("Average number of tokens per document")

        # Save the plot
        plt.savefig(f"{token_counts_path}avg_token_counts.png")

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
