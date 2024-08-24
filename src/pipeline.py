from typing import Any
from pydantic.v1 import BaseModel

from .generation.huggingface_llm import HuggingFaceLLM
from .generation.ollama_llm import OllamaLLM
from .config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.language_models.chat_models import BaseChatModel


class Pipeline(BaseModel):
    template: str
    prompt: PromptTemplate
    output_parser: StrOutputParser
    llm: BaseChatModel
    llm_chain: RunnableSequence

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls):
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
        llm_chain = prompt | llm | output_parser
        return cls(
            template=template,
            llm=llm,
            prompt=prompt,
            output_parser=output_parser,
            llm_chain=llm_chain,
        )

    def invoke(self, context: str = "", question: str = "What is love?") -> str:
        return self.llm_chain.invoke({"context": context, "question": question})
