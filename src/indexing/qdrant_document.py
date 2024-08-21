from pydantic import BaseModel
from typing import List, Dict, Optional


class QDrantDocument(BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None
    id: Optional[str] = None


class QdrantDocumentCollection(BaseModel):
    documents: List[QDrantDocument]

    def texts(self):
        return [doc.text for doc in self.documents]

    def metadata(self):
        return [doc.metadata for doc in self.documents]

    def ids(self):
        return [doc.id for doc in self.documents]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

    def add_single(self, document: QDrantDocument):
        self.documents.append(document)

    def add_multiple(self, documents: List[QDrantDocument]):
        self.documents.extend(documents)
