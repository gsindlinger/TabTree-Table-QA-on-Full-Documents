import glob
import logging
import os
from typing import List, Optional

from bs4 import BeautifulSoup
from pydantic import Field

from .document_loader import DocumentLoader
from ...config.config import Config
from ...model.custom_document import CustomDocument, FullMetadata


class SECFilingLoader(DocumentLoader):
    folder_path: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder_path = Config.data.path_local

    def load_documents(
        self, preprocess_mode: str = "default", num_of_documents: Optional[int] = None
    ) -> List[CustomDocument]:
        """Loads documents (in this case from local storage) and calls preprocessing method.

        Args:
            preprocess_mode (str, optional): _description_. Defaults to "default".
                                                            Options are "none", "remove_attributes", "remove_xbrl", "all", "default". "All" and "default" are equivalent.
            num_of_documents (Optional[int], optional): _description_. Defaults to None.

        Returns:
            List[CustomDocument]: _description_
        """

        documents = []

        count = 0
        for file_path in glob.glob(self.folder_path + "*"):
            if num_of_documents and count == num_of_documents:
                break
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()
                filing = CustomDocument(
                    page_content=html_content,
                    metadata=FullMetadata(doc_id=os.path.basename(file_path)),
                )

                # Preprocess document
                filing = self.preprocess_document(
                    document=filing, preprocess_mode=preprocess_mode
                )

                # Append to documents
                documents.append(filing)
                logging.info(f"Loaded document {filing.metadata.doc_id}")
                count += 1
        return documents

    @staticmethod
    def preprocess_document(
        document: CustomDocument, preprocess_mode: str = "default"
    ) -> CustomDocument:
        if preprocess_mode == "none":
            return document

        remove_attributes = False
        remove_xbrl = False

        if preprocess_mode in ["remove_attributes", "all", "default"]:
            remove_attributes = True
        if preprocess_mode in ["remove_xbrl", "all", "default"]:
            remove_xbrl = True

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(document.page_content, "html.parser")

        # Define xblr prefixes to remove
        xbrl_prefixes = ["ix:", "xbrli:", "xbrldi:", "iso4217:"]

        # Iterate through all tags in the parsed HTML
        for tag in soup.find_all(True):
            # Remove all attributes for each tag
            if remove_attributes:
                tag.attrs = {}

            # Check if the tag name starts with any of the xbrl prefixes and remove the tag but keep the text inside
            if (
                remove_xbrl
                and tag.name
                and any(tag.name.startswith(prefix) for prefix in xbrl_prefixes)
            ):
                tag.unwrap()
                continue

            # Remove all tags that do not have any text or content
            if not tag.text.strip() and not tag.contents:
                tag.decompose()

        # Convert back to string
        document.page_content = str(soup)
        return document
