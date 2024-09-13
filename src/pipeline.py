from __future__ import annotations
from typing import List
from pydantic.v1 import BaseModel

from .model.custom_document import CustomDocument
from .retrieval.retriever import QdrantRetriever
from .retrieval.qdrant_store import QdrantVectorStore
from .generation.huggingface_llm import HuggingFaceLLM
from .generation.ollama_llm import OllamaLLM
from .config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.base import BaseLanguageModel


class Pipeline(BaseModel):
    template: str
    prompt: PromptTemplate
    output_parser: StrOutputParser
    llm: BaseLanguageModel
    llm_chain: RunnableSequence
    retriever: QdrantRetriever

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, vector_store: QdrantVectorStore) -> Pipeline:
        retriever = QdrantRetriever(vector_store)
        template = Config.pipeline.template
        llm = (
            OllamaLLM()
            if Config.pipeline.llm_implementation == "ollama"
            else HuggingFaceLLM()
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=Config.pipeline.template,
        )
        output_parser = StrOutputParser()
        llm_chain = (
            {
                "context": retriever | retriever.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | output_parser
        )
        return cls(
            retriever=retriever,
            template=template,
            llm=llm,
            prompt=prompt,
            output_parser=output_parser,
            llm_chain=llm_chain,
        )

    def retrieve(self, question: str = "What is love?") -> List[CustomDocument]:
        docs = self.retriever.invoke(question)
        return CustomDocument.docs_to_custom_docs(docs)

    def invoke(self, question: str = "What is love?") -> str:
        return self.llm_chain.invoke(question)

    def stream(self, question: str = "What is love?") -> str:
        return self.llm_chain.stream(question)
