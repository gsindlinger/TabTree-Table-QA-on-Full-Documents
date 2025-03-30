from typing import List
from pydantic import BaseModel
import nltk
nltk.download('punkt_tab')


class SentenceSplitter(BaseModel):
    def __init__(self):
        super().__init__()
        
    def _split_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        splitter = SentenceSplitter()
        return splitter._split_sentences(text)

