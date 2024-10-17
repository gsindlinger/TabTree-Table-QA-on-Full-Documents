import os
from typing import List, Optional
import pandas as pd

from ..retrieval.document_loaders.wiki_table_questions_loader import (
    WikiTableQuestionsLoader,
)

from .evaluation_document import EvaluationDocument
from ..config.config import Config
from .evaluator import Evaluator


class WikiTableQuestionsEvaluator(Evaluator):

    def get_evaluation_docs(self) -> List[EvaluationDocument]:
        if Config.wiki_table_questions.is_single_evaluation:
            questions = WikiTableQuestionsLoader.load_unique_questions(
                Config.wiki_table_questions.single_document_id
            )
        else:
            questions = WikiTableQuestionsLoader.load_questions()

        if self.evaluation_num_documents:
            questions = questions[: self.evaluation_num_documents]
        return questions
