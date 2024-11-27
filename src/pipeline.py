from __future__ import annotations
from typing import Iterator, List, Literal, Tuple
from pydantic import BaseModel

from .retrieval.document_preprocessors.table_serializer import CustomTable
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
        template = Config.tabgraph.prompt_template
        llm = LLM.from_tabgraph_config()
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

    def predict_headers(self, custom_table: CustomTable) -> Tuple[List[int], List[int]]:
        rows = self.predict_headers_single(
            full_table=custom_table.raw_table, table_df=custom_table, mode="row"
        )
        columns = self.predict_headers_single(
            full_table=custom_table.raw_table, table_df=custom_table, mode="column"
        )
        return rows, columns

    def predict_headers_single(
        self,
        full_table: str,
        table_df: CustomTable | None,
        mode: Literal["row", "column"] = "row",
    ):
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
                    items = table_df.df.iloc[index + 1].tolist()
                    if index > 0:
                        second_previous_items = table_df.df.iloc[index - 1].tolist()
                    if index > -1:
                        previous_items = table_df.df.iloc[index].tolist()
                    if index + 3 < len(table_df.df):
                        next_items = table_df.df.iloc[index + 2].tolist()
                    if index + 4 < len(table_df.df):
                        second_next_items = table_df.df.iloc[index + 3].tolist()
                case "column":
                    items = table_df.df.iloc[:, index + 1].tolist()
                    if index > 0:
                        second_previous_items = table_df.df.iloc[:, index - 1].tolist()
                    if index > -1:
                        previous_items = table_df.df.iloc[:, index].tolist()
                    if index + 3 < len(table_df.df.columns):
                        next_items = table_df.df.iloc[:, index + 2].tolist()
                    if index + 4 < len(table_df.df.columns):
                        second_next_items = table_df.df.iloc[:, index + 3].tolist()

            row_description = Config.tabgraph.header_description
            column_description = Config.tabgraph.label_description
            negative_description = Config.tabgraph.negative_description

            if not self.predict_headers_single_llm(
                table=full_table,
                line=str(items),
                mode=mode,
                mode_header_name="header" if mode == "row" else "label",
                mode_description=(
                    row_description if mode == "row" else column_description
                ),
                negative_description=negative_description,
                previous_items=str(previous_items),
                next_items=str(next_items),
                second_previous_items=str(second_previous_items),
                second_next_items=str(second_next_items),
                index=index + 1,
            ):
                break

            index += 1

        if index == -1:
            response = []
        else:
            response = list(range(index + 1))

        return response

    # def get_table_header_rows_columns(
    #     self,
    #     table: str,
    # ) -> tuple[list[int], list[int]]:
    #     response = self._get_table_header_rows_columns(table)

    #     # Search for pattern in response
    #     # Columns: [<column_index_1>, <column_index_1>, ...]
    #     # Rows: [<row_index_1>, <row_index_2>, ...]
    #     list_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    #     column_pattern = rf"Columns: {list_pattern}"
    #     row_pattern = rf"Rows: {list_pattern}"

    #     if column_match := re.search(column_pattern, response):
    #         column_indices = list(map(int, column_match.group(1).split(", ")))
    #     if row_match := re.search(row_pattern, response):
    #         row_indices = list(map(int, row_match.group(1).split(", ")))

    #     if len(column_indices) == 0 and len(row_indices) == 0:
    #         logging.info(f"Found no column or row headers in response: {response}")
    #     return row_indices, column_indices

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
            next_index = f"The provided {mode}  is the last {mode}. Pleas consider this when answering the question."

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
