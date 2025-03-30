import logging
import re
from typing import List, Optional
import unicodedata
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .sentence_splitter import SentenceSplitter

from ..document_preprocessors.preprocess_config import PreprocessConfig
from ..document_splitters.semantic_chunker_custom import (
    TABLE_REGEX,
    SENTENCE_SPLITTER_REGEX,
)
from ...model.custom_document import CustomDocument, SplitContent
from .document_preprocessor import DocumentPreprocessor


class HTMLPreprocessor(DocumentPreprocessor):
    def preprocess_document(self, document: CustomDocument) -> CustomDocument:
        if (
            not self.preprocess_config
            or "none" in self.preprocess_config.preprocess_mode
        ):
            return document

        document.original_content = self._preprocess_document(document)
        document.page_content = self._get_body_of_html(document.original_content)
        document.splitted_content = self._split_sentences_and_tables(
            document.page_content
        )

        # Serialize tables in the document if table_serialization is enabled
        if self.table_serializer:
            document.page_content, document.splitted_content = (
                self.table_serializer.serialize_tables_in_document(document)
            )
        return document

    def _get_body_of_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        if isinstance(body, NavigableString):
            return str(body)
        else:
            return body.prettify() if body else html

    def _split_sentences_and_tables(
        self,
        text: str,
        table_regex: str = TABLE_REGEX,
        sentence_split_regex: str = SENTENCE_SPLITTER_REGEX,
    ) -> List[SplitContent]:
        content_list = []
        parts = re.split(pattern=table_regex, string=text, flags=re.DOTALL)
        index_counter = 0
        for part in parts:
            part = part
            if part.startswith("<table>") and part.endswith("</table>"):
                content_list.append(
                    SplitContent(
                        type="table",
                        content=part,
                        position=index_counter,
                        visited=False,
                        original_content=part,
                    )
                )
                index_counter += 1
                
                # generate table summary and store it together with table headers
            else:
                # sentences = re.split(sentence_split_regex, part, flags=re.DOTALL)
                sentences = SentenceSplitter.split_sentences(part)
                for sentence in sentences:
                    content_list.append(
                        SplitContent(
                            type="text",
                            content=sentence,
                            position=index_counter,
                            visited=False,
                            original_content=sentence,
                        )
                    )
                    index_counter += 1
        return content_list

    def _preprocess_document(self, document: CustomDocument) -> str:
        if not self.preprocess_config:
            return document.page_content

        # For simplicity, I introduced a "basic" mode that includes the most common preprocessing steps
        if "basic" in self.preprocess_config.preprocess_mode:
            self.preprocess_config.preprocess_mode.extend(
                [
                    "remove-invisible",
                    "remove-images",
                    "remove-xbrl",
                    "remove-attributes",
                    "unwrap-irrelevant",
                    "unwrap-divs",
                    "replace-br",
                    "normalize-whitespace",
                ]
            )

        soup = BeautifulSoup(document.page_content, "html.parser")
        soup = soup.find("html") or soup

        if "remove-invisible" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: remove-invisible")
            soup = HTMLPreprocessor.remove_invisible_elements(soup)

        if "remove-images" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: remove-images")
            soup = HTMLPreprocessor.remove_images(soup)

        if "remove-xbrl" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: remove-xbrl")
            soup = HTMLPreprocessor.handle_xbrl_elements(soup)

        if "remove-attributes" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: remove-attributes")
            soup = HTMLPreprocessor.remove_attributes(
                soup, preprocess_config=self.preprocess_config
            )

        if "unwrap-irrelevant" in self.preprocess_config.preprocess_mode:
            logging.info(
                "Preprocessing: unwrap-irrelevant, e.g. 'span', 'a', 'b', 'strong', ..."
            )
            soup = HTMLPreprocessor.unwrap_spans_a(soup)

        if "unwrap-divs" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: unwrap-divs")
            soup = HTMLPreprocessor.unwrap_divs(soup)

        if "replace-br" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: replace-br")
            soup = HTMLPreprocessor.replace_br(soup)

        if "delete-hr" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: delete-hr")
            if not self.preprocess_config.reduced_sections:
                soup = HTMLPreprocessor.delete_hr(soup)

        if "only-text-except-tables" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: only-text-except-tables")
            soup = HTMLPreprocessor.get_only_text_except_tables(
                soup,
                keep_hr_tags=(
                    self.preprocess_config.reduced_sections
                    if self.preprocess_config.reduced_sections
                    else False
                ),
            )

        if "normalize-whitespace" in self.preprocess_config.preprocess_mode:
            logging.info("Preprocessing: normalize-whitespace")
            soup = HTMLPreprocessor.normalize_whitespace(soup)

        if self.preprocess_config.reduced_sections:
            logging.info("Preprocessing: split-into-reduced-sections")
            soup = HTMLPreprocessor.split_into_reduced_sections(soup)

        if isinstance(soup, NavigableString):
            return str(soup)
        else:
            return HTMLPreprocessor.custom_prettify(soup)

    @staticmethod
    def delete_hr(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        if not isinstance(soup, NavigableString):
            for tag in soup.find_all("hr"):
                tag.decompose()
        return soup

    @staticmethod
    def unwrap_divs(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Unwrap div tags from the document."""
        if isinstance(soup, NavigableString):
            return soup
        for tag in soup.find_all("div"):
            tag.unwrap()
        return soup

    @staticmethod
    def remove_images(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Remove image tags from the document."""
        if isinstance(soup, NavigableString):
            return soup
        for tag in soup.find_all("img"):
            tag.decompose()
        return soup

    @staticmethod
    def replace_br(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Replace br tags with newlines."""
        if isinstance(soup, NavigableString):
            return soup
        for br in soup.find_all("br"):
            br.replace_with(" ")
        return soup

    @staticmethod
    def split_into_reduced_sections(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Split the document into sections based on <hr> tags, which indicate different pages in the document. Keep only the sections that contain a <table> tag."""

        sections = re.split(r"<hr\s*[^>]*>", str(soup), flags=re.DOTALL)

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
        soup = BeautifulSoup("".join(modified_sections), "html.parser")
        return soup

    @staticmethod
    def normalize_whitespace(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Remove all whitespace from the document. Uses unicodedata 'NFC' to normalize the text."""
        for string in list(soup.strings):
            cleaned_string = unicodedata.normalize("NFC", str(string))
            cleaned_string = cleaned_string.replace("\u00A0", " ")

            # Replace multiple whitespace into single whitespace
            cleaned_string = re.sub(r"\s+", " ", cleaned_string)

            # Remove zero-width space characters
            cleaned_string = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned_string)
            cleaned_string = "".join(ch for ch in cleaned_string if ch.isprintable())
            string.replace_with(cleaned_string)  # type: ignore
        return soup

    @staticmethod
    def handle_xbrl_elements(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Unwrap or remove XBRL elements from the document."""
        xbrl_prefixes = ["ix:", "xbrli:", "xbrldi:", "iso4217:"]
        xbrl_prefixes_to_delete = ["ix:hidden"]

        if isinstance(soup, NavigableString):
            return soup
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
    def remove_invisible_elements(
        soup: Tag | BeautifulSoup | NavigableString,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Remove invisible elements from the document."""
        if isinstance(soup, NavigableString):
            return soup
        for tag in soup.find_all(True):
            if tag.attrs and "style" in tag.attrs:
                if "display:none" in tag.attrs.get("style").replace(" ", "").lower():
                    tag.decompose()

        # remove html comments
        comments = soup.findAll(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        return soup

    @staticmethod
    def remove_attributes(
        soup: BeautifulSoup | NavigableString | Tag,
        preprocess_config: Optional[PreprocessConfig] = None,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Remove all html attributes from the document."""
        if isinstance(soup, NavigableString):
            return soup

        # if tag is either colspan or rowspan, keep the attribute else remove it
        for tag in soup.find_all(True):
            for attr in list(tag.attrs.keys()):
                if preprocess_config and preprocess_config.consider_colspans_rowspans:
                    if attr not in ["colspan", "rowspan"]:
                        del tag[attr]
                else:
                    del tag[attr]
        return soup

    @staticmethod
    def unwrap_spans_a(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> BeautifulSoup | NavigableString | Tag:
        """Unwrap span and a tags from the document."""
        unwrap_tags = [
            "span",
            "a",
            "b",
            "strong",
            "i",
            "em",
            "u",
            "small",
            "mark",
            "abbr",
            "acronym",
            "del",
            "ins",
            "code",
        ]
        if isinstance(soup, NavigableString):
            return soup
        for tag in soup.find_all(unwrap_tags):
            tag.unwrap()

        # replace br tags with newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")
        return soup

    @staticmethod
    def get_only_text_except_tables(
        soup: BeautifulSoup | NavigableString | Tag, keep_hr_tags: bool = False
    ) -> BeautifulSoup | NavigableString | Tag:
        """Remove all tags except for text and tables from the document. If keep_hr_tags is True, hr tags are kept."""

        def clean_tag(tag):
            if tag.name == "hr" and keep_hr_tags:
                return
            for child in tag.find_all(recursive=False):
                clean_tag(child)
            if tag.name not in ["table", "thead", "tbody", "tr", "th", "td"]:
                # remove the tag and keep the text only
                tag.unwrap()

        body = soup.find("body")
        if body:
            clean_tag(body)
        return soup

    @staticmethod
    def custom_prettify(
        soup: BeautifulSoup | NavigableString | Tag,
    ) -> str:
        """Custom prettify function that merges consecutive NavigableString objects."""
        if isinstance(soup, NavigableString):
            return soup
        soup = HTMLPreprocessor.merge_navigable_strings(soup)
        return soup.prettify()

    @staticmethod
    def merge_navigable_strings(
        element: Tag,
    ) -> Tag:
        new_children = []
        buffer = ""

        for child in element.children:
            if isinstance(child, NavigableString):
                # check if last character of buffer is a whitespace or first character of child is a whitespace
                # based on that ensure that exactly one whitespace is present
                if buffer and buffer[-1] != " " and child.string and child.string[0] != " ":  # type: ignore
                    buffer += " "
                if (
                    buffer
                    and buffer[-1] == " "
                    and child.string  # type: ignore
                    and child.string[0] == " "  # type: ignore
                ):
                    buffer = buffer[:-1]
                buffer += str(child)
            else:
                if buffer:
                    new_children.append(NavigableString(buffer))
                    buffer = ""

                new_children.append(child)

        if buffer:
            new_children.append(NavigableString(buffer))

        element.clear()
        for new_child in new_children:
            element.append(new_child)

        # Recursively merge navigable strings for child elements
        for child in element.children:
            if not isinstance(child, NavigableString):
                HTMLPreprocessor.merge_navigable_strings(child)  # type: ignore

        return element
