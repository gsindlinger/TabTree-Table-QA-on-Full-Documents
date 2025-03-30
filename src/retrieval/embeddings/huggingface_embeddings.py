import logging
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import requests
from .custom_embeddings import CustomEmbeddings
from ...config import Config

class HuggingFaceEmbeddings(HuggingFaceInferenceAPIEmbeddings, CustomEmbeddings):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self._default_kwargs() | kwargs))
        self.model_name = "BGE-M3"

    @staticmethod
    def _default_kwargs() -> dict:
        api_key = Config.env_variables.HUGGINGFACE_API_KEY
        api_url = Config.huggingface.endpoint_url
        
        return {
            "api_url": api_url,
            "api_key": api_key,
            "model_kwargs": {"normalize_embeddings": True},
            "show_progress": True,
        }

    def get_model_name(self) -> str:
        return self.model_name  # Since we're using an API, return the endpoint URL
    
    
    def embed_query(self, text: str) -> List[float]:
        text = self.get_detailed_instruct(text)
        return self.embed_documents([text])[0]
    
    def get_detailed_instruct(self, question: str, task_description: Optional[str] = None) -> str:
        if not task_description:
            task_description = 'You are an AI model that is tasked retrieving relevant given question. Please retrieve relevant documents based on the given question.'
        return f'Instruct: {task_description}\nQuestion: {question}'
    
    @staticmethod
    def batch_list(input_list: List[str], batch_size: int) -> List[List[str]]:
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Get the embeddings for a list of texts in batches
        Args:
            texts (List[str]): A list of texts to get embeddings for.
            batch_size (int): Number of texts to send per request batch.
        Returns:
            List[List[float]]: Embedded texts as a list of embedding vectors.
        """
        all_embeddings = []
        batches = self.batch_list(texts, batch_size)
        total_batches = len(batches)
        
        logging.info(f"Embedding {len(texts)} documents in {total_batches} batch(es)...")

        for idx, batch in enumerate(batches, start=1):
            try:
                logging.debug(f"Sending batch {idx}/{total_batches} with {len(batch)} items.")
                response = requests.post(
                    self._api_url,
                    headers=self._headers,
                    json={
                        "inputs": batch,
                        "options": {"wait_for_model": True, "use_cache": True},
                    },
                )
                response.raise_for_status()
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)
                logging.info(f"Batch {idx}/{total_batches} processed successfully.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to process batch {idx}/{total_batches}: {e}")
                raise

        logging.info("All batches processed.")
        return all_embeddings


        