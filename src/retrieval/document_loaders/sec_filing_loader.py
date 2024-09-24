import glob
import logging
import os
import re
from typing import List, Optional, Tuple
import unicodedata

from bs4 import BeautifulSoup
from pydantic import Field

from .document_loader import DocumentLoader
from ...config.config import Config
from ...model.custom_document import CustomDocument, FullMetadata


class SECFilingLoader(DocumentLoader):
    folder_path: str = Field(default_factory=lambda: Config.sec_filings.data_path)
    preprocess_mode: List[str] = Field(
        default_factory=lambda: Config.sec_filings.preprocess_mode
    )

    def load_single_document(
        self,
        file_path: Optional[str] = None,
        preprocess_mode: Optional[List[str]] = None,
    ) -> CustomDocument:
        """Loads a single document (in this case from local storage) and calls preprocessing method."""

        if not file_path:
            file_path = glob.glob(self.folder_path + "*")[0]

        if not preprocess_mode:
            preprocess_mode = self.preprocess_mode

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
            logging.info(f"Loaded document {filing.metadata.doc_id}")
            return filing

    def load_documents(
        self,
        preprocess_mode: Optional[str] = None,
        num_of_documents: Optional[int] = None,
    ) -> List[CustomDocument]:
        """Loads documents (in this case from local storage) and calls preprocessing method.

        Args:
            preprocess_mode (str, optional): _description_. Defaults to "default".
                                                            Options are "none", "remove-attributes", "remove_xbrl", "all", "default". "All" and "default" are equivalent.
            num_of_documents (Optional[int], optional): _description_. Defaults to None.

        Returns:
            List[CustomDocument]: _description_
        """

        if not preprocess_mode:
            preprocess_mode = self.preprocess_mode

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
            count += 1
        return documents

    @staticmethod
    def preprocess_document(
        document: CustomDocument, preprocess_mode: Optional[List[str]] = None
    ) -> CustomDocument:

        if not preprocess_mode or "none" in preprocess_mode:
            return document

        soup = BeautifulSoup(document.page_content, "html.parser")

        if "remove-invisible" in preprocess_mode:
            soup = SECFilingLoader.remove_invisible_elements(soup)

        if "remove-xbrl" in preprocess_mode:
            soup = SECFilingLoader.handle_xbrl_elements(soup)

        if "remove-attributes" in preprocess_mode:
            soup = SECFilingLoader.remove_attributes(soup)

        if "remove-spans-a" in preprocess_mode:
            soup = SECFilingLoader.unwrap_spans_a(soup)

        if "only-text-except-tables" in preprocess_mode:
            soup = SECFilingLoader.get_only_text_except_tables(
                soup, keep_hr_tags="split-into-reduced-sections" in preprocess_mode
            )

        soup = SECFilingLoader.remove_whitespace(soup)
        soup = SECFilingLoader.remove_empty_tags(
            soup, keep_hr_tags="split-into-reduced-sections" in preprocess_mode
        )

        if "split-into-reduced-sections" in preprocess_mode:
            soup, splitted_sections = SECFilingLoader.split_into_reduced_sections(soup)
            # document.splitted_content = splitted_sections

        # attention: potentially overrides the splitted content variable
        if "split-into-table-and-text" in preprocess_mode:
            if document.splitted_content:
                splitted_sections = SECFilingLoader.split_into_table_and_text(
                    document.splitted_content
                )
            else:
                splitted_sections = SECFilingLoader.split_into_table_and_text(soup)
            document.splitted_content = splitted_sections

        document.page_content = str(soup)
        return document

    @staticmethod
    def split_into_table_and_text_list(
        sections: List[str],
    ) -> List[str]:
        table_pattern = re.compile(
            r"(<table.*?>.*?</table>)", re.DOTALL | re.IGNORECASE
        )
        splitted_content = [
            split for section in sections for split in table_pattern.split(section)
        ]
        return table_pattern.split(splitted_content)

    @staticmethod
    def split_into_table_and_text(
        soup: BeautifulSoup,
    ) -> List[str]:
        table_pattern = re.compile(
            r"(<table.*?>.*?</table>)", re.DOTALL | re.IGNORECASE
        )
        splitted_content = table_pattern.split(str(soup.body))
        return table_pattern.split(str(soup.body))

    @staticmethod
    def split_into_reduced_sections(
        soup: BeautifulSoup,
    ) -> Tuple[BeautifulSoup, List[str]]:
        """Split the document into sections based on <hr> tags, which indicate different pages in the document. Keep only the sections that contain a <table> tag."""

        hr_pattern = re.compile(r"<hr\s*[^>]*>", re.IGNORECASE)
        sections = hr_pattern.split(str(soup))

        modified_sections = []
        # Check each section for the presence of a <table>
        for i, section in enumerate(sections):
            replacement_str = "\n\n(...)\n\n"
            if i == 0 or "<table>" in section or i == len(sections) - 1:
                modified_sections.append(section.strip())
            else:
                if modified_sections[-1] != replacement_str:
                    modified_sections.append(replacement_str)

        # Reconstruct the HTML content
        splitted_content = modified_sections
        soup = BeautifulSoup("".join(modified_sections), "html.parser")
        return soup, splitted_content

    @staticmethod
    def remove_empty_tags(
        soup: BeautifulSoup, keep_hr_tags: bool = False
    ) -> BeautifulSoup:
        """Remove tags with no content. Also removes self-closing tags like br tags. If keep_hr_tags is True, hr tags are kept."""
        for tag in soup.find_all(True):
            if keep_hr_tags and tag.name and tag.name == "hr":
                continue
            elif not tag.text.strip():
                tag.decompose()
        return soup

    @staticmethod
    def remove_whitespace(soup: BeautifulSoup) -> BeautifulSoup:
        """Remove all whitespace from the document. Uses unicodedata 'NFKD' to normalize the text."""
        for string in list(soup.strings):
            cleaned_string = unicodedata.normalize("NFKD", str(string))
            string.replace_with(cleaned_string)
        return soup

    @staticmethod
    def handle_xbrl_elements(soup: BeautifulSoup) -> BeautifulSoup:
        """Unwrap or remove XBRL elements from the document."""
        xbrl_prefixes = ["ix:", "xbrli:", "xbrldi:", "iso4217:"]
        xbrl_prefixes_to_delete = ["ix:hidden"]

        for tag in soup.find_all(True):
            if tag.name:
                if any(
                    tag.name.startswith(prefix) for prefix in xbrl_prefixes_to_delete
                ):
                    tag.decompose()
                elif any(tag.name.startswith(prefix) for prefix in xbrl_prefixes):
                    tag.unwrap()
        return soup

    @staticmethod
    def remove_invisible_elements(soup: BeautifulSoup) -> BeautifulSoup:
        """Remove invisible elements from the document."""
        for tag in soup.find_all(True):
            if tag.attrs and "style" in tag.attrs:
                if "display:none" in tag.attrs.get("style").replace(" ", "").lower():
                    tag.decompose()
        return soup

    @staticmethod
    def remove_attributes(soup: BeautifulSoup) -> BeautifulSoup:
        """Remove all html attributes from the document."""
        for tag in soup.find_all(True):
            tag.attrs = {}
        return soup

    @staticmethod
    def unwrap_spans_a(soup: BeautifulSoup) -> BeautifulSoup:
        """Unwrap span and a tags from the document."""
        for tag in soup.find_all(["span", "a"]):
            tag.unwrap()
        return soup

    @staticmethod
    def get_only_text_except_tables(
        soup: BeautifulSoup, keep_hr_tags: bool = False
    ) -> BeautifulSoup:
        """Remove all tags except for text and tables from the document. If keep_hr_tags is True, hr tags are kept."""

        def clean_tag(tag):
            if tag.name == "hr" and keep_hr_tags:
                return
            for child in tag.find_all(recursive=False):
                clean_tag(child)
            if tag.name not in ["table", "td", "th", "tr"]:
                # if tag is div then add an empty space to separate the text
                if tag.name == "div":
                    tag.append(" ")
                # remove the tag and keep the text only
                tag.unwrap()

        body = soup.find("body")
        if body:
            clean_tag(body)
        return soup
