from __future__ import annotations
from pydantic.v1 import BaseModel

from .retrieval.retriever import QdrantRetriever
from .retrieval.qdrant_store import QdrantVectorStore
from .generation.huggingface_llm import HuggingFaceLLM
from .generation.ollama_llm import OllamaLLM
from .config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough


class Pipeline(BaseModel):
    template: str
    prompt: PromptTemplate
    output_parser: StrOutputParser
    llm: BaseChatModel
    llm_chain: RunnableSequence
    retriever: QdrantRetriever

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, vector_store: QdrantVectorStore) -> Pipeline:
        retriever = QdrantRetriever(vector_store)
        docs = retriever.invoke("What is love?")
        template = Config.pipeline.template
        llm = (
            OllamaLLM()
            if Config.pipeline.llm_implementation == "ollama"
            else HuggingFaceLLM()
        )
        prompt = PromptTemplate(
            input_variables=["content", "question"],
            template=Config.pipeline.template,
        )
        output_parser = StrOutputParser()
        llm_chain = (
            {
                "content": retriever | retriever.format_docs,
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

    def invoke(self, context: str = "", question: str = "What is love?") -> str:
        return self.llm_chain.invoke({"context": context, "question": question})
