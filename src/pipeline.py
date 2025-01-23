from __future__ import annotations
from typing import Iterator, List, Literal, Tuple
from pydantic import BaseModel

from .retrieval.document_preprocessors.table_parser.custom_table import (
    CustomTableWithHeaderOptional,
)
from .retrieval.document_preprocessors.table_serializer import ExtendedTable
from .generation.abstract_llm import LLM
from .model.custom_document import CustomDocument
from .retrieval.retriever import QdrantRetriever
from .retrieval.qdrant_store import QdrantVectorStore
from .config import Config
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence, RunnableSerializable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.base import BaseLanguageModel
from langchain.retrievers.multi_query import MultiQueryRetriever


class Pipeline(BaseModel):
    template: str
    prompt: PromptTemplate
    output_parser: StrOutputParser
    llm: BaseLanguageModel
    llm_chain: RunnableSequence
    retriever: MultiQueryRetriever | QdrantRetriever

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(
        cls, vector_store: QdrantVectorStore, retriever_num_documents: int
    ) -> Pipeline:
        retriever = QdrantRetriever(
            vector_store, retriever_num_documents=retriever_num_documents
        )
        template = Config.text_generation.prompt_template
        llm = LLM.from_config()
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        output_parser = StrOutputParser()

        # retriever = MultiQueryRetriever.from_llm(retriever=retriever_base, llm=llm)
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

    def stream(self, question: str = "What is love?") -> Iterator:
        return self.llm_chain.stream(question)


class TableHeaderRowsPipeline(BaseModel):
    template: str
    prompt: PromptTemplate
    output_parser: StrOutputParser
    llm: BaseLanguageModel
    llm_chain: RunnableSerializable

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(
        cls,
    ) -> TableHeaderRowsPipeline:
        template = Config.tabtree.prompt_template
        llm = LLM.from_tabtree_config()
        prompt = PromptTemplate(
            input_variables=["table"],
            template=template,
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

    def predict_headers(
        self, custom_table: CustomTableWithHeaderOptional
    ) -> Tuple[int, int]:
        """Get the row and column headers of the table.
        First item of return tuple refers to column header rows and second item refers to row label columns.
        """
        column_header_rows = self.predict_headers_single(
            full_table=custom_table.raw_table, table_df=custom_table, mode="row"
        )
        row_label_columns = self.predict_headers_single(
            full_table=custom_table.raw_table, table_df=custom_table, mode="column"
        )
        return column_header_rows, row_label_columns

    def predict_headers_single(
        self,
        full_table: str,
        table_df: CustomTableWithHeaderOptional | None,
        mode: Literal["row", "column"] = "row",
    ) -> int:
        index = -1
        while True:
            if not table_df:
                break

            previous_items = []
            next_items = []
            second_previous_items = []
            second_next_items = []

            match mode:
                case "row":
                    items = table_df.get_row(index + 1)
                    if index > 0:
                        second_previous_items = table_df.get_row(index - 1)
                    if index > -1:
                        previous_items = table_df.get_row(index)
                    if index + 3 < table_df.rows:
                        next_items = table_df.get_row(index + 2)
                    if index + 4 < table_df.rows:
                        second_next_items = table_df.get_row(index + 3)
                case "column":
                    items = table_df.get_column(index + 1)
                    if index > 0:
                        second_previous_items = table_df.get_column(index - 1)
                    if index > -1:
                        previous_items = table_df.get_column(index)
                    if index + 3 < table_df.columns:
                        next_items = table_df.get_column(index + 2)
                    if index + 4 < table_df.columns:
                        second_next_items = table_df.get_column(index + 3)

            row_description = Config.tabtree.header_description
            column_description = Config.tabtree.label_description
            negative_description = Config.tabtree.negative_description

            if not self.predict_headers_single_llm(
                table=full_table,
                line=CustomTableWithHeaderOptional.print_line_with_span(items, mode),
                mode=mode,
                mode_header_name="header" if mode == "row" else "label",
                mode_description=(
                    row_description if mode == "row" else column_description
                ),
                negative_description=negative_description,
                previous_items=CustomTableWithHeaderOptional.print_line_with_span(
                    previous_items, mode
                ),
                next_items=CustomTableWithHeaderOptional.print_line_with_span(
                    next_items, mode
                ),
                second_previous_items=CustomTableWithHeaderOptional.print_line_with_span(
                    second_previous_items, mode
                ),
                second_next_items=CustomTableWithHeaderOptional.print_line_with_span(
                    second_next_items, mode
                ),
                index=index + 1,
            ):
                break

            index += 1

        if index == -1:
            response = -1
        else:
            response = index

        return response

    def predict_headers_single_llm(
        self,
        table: str,
        line: str,
        mode: str,
        mode_header_name: str,
        mode_description: str,
        negative_description: str,
        previous_items: str,
        next_items: str,
        second_previous_items: str,
        second_next_items: str,
        index: int,
    ) -> bool:
        if previous_items != "[]":
            previous_index = f"The previous {mode} looks like this: {previous_items}"
        else:
            previous_index = f"The provided {mode} is the first {mode}. Please consider this when answering the question."

        if next_items != "[]":
            next_index = f"The next {mode} looks like this: {next_items}"
        else:
            next_index = f"The provided {mode}  is the last {mode}. Please consider this when answering the question."

        if second_previous_items != "[]":
            second_previous_index = (
                f"The second previous {mode} looks like this: {second_previous_items}"
            )
        else:
            second_previous_index = ""

        if second_next_items != "[]":
            second_next_index = (
                f"The second next {mode} looks like this: {second_next_items}"
            )
        else:
            second_next_index = ""

        response = self.llm_chain.invoke(
            input={
                "table": table,
                "line": line,
                "mode": mode,
                "mode_header_name": mode_header_name,
                "mode_description": mode_description,
                "negative_description": negative_description,
                "previous_index": previous_index,
                "next_index": next_index,
                "second_previous_index": second_previous_index,
                "second_next_index": second_next_index,
                "index": index,
            }
        )

        # check whether response contains yes or no
        if "yes" in response.lower():
            print("Yes")
            return True
        elif "no" in response.lower():
            print("No")
            return False
        else:
            raise ValueError(f"Response does not contain yes or no: {response}")
