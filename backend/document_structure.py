from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentElement:
    element_type: str
    text: str
    level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


HEADING_PATTERNS = [
    re.compile(r"^(#{1,6})\s+(.+)$"),
    re.compile(r"^(\d+(?:\.\d+){0,5})[.)]?\s+(.{1,120})$"),
]
LIST_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+).+")
IMAGE_PATTERN = re.compile(r"!\[(?P<caption>[^\]]*)\]\((?P<src>[^)]+)\)|\[(?:image|图片|图像)[:：]?\s*(?P<label>[^\]]*)\]", re.I)


class DocumentStructureParser:
    """Parse plain loader text into multimodal-ready structural elements."""

    def parse(self, text: str) -> list[DocumentElement]:
        text = (text or "").replace("\r\n", "\n").strip()
        if not text:
            return []
        try:
            elements = self._parse_structured_text(text)
        except Exception:
            return []
        return elements or []

    def _parse_structured_text(self, text: str) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        paragraph_lines: list[str] = []
        list_lines: list[str] = []
        table_lines: list[str] = []
        code_lines: list[str] = []
        in_code = False
        code_language = ""

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if paragraph_lines:
                body = "\n".join(paragraph_lines).strip()
                if body:
                    elements.append(DocumentElement("paragraph", body))
                paragraph_lines = []

        def flush_list() -> None:
            nonlocal list_lines
            if list_lines:
                elements.append(DocumentElement("list", "\n".join(list_lines).strip()))
                list_lines = []

        def flush_table() -> None:
            nonlocal table_lines
            if table_lines:
                elements.append(DocumentElement("table", self._table_to_text(table_lines)))
                table_lines = []

        def flush_code() -> None:
            nonlocal code_lines, code_language
            if code_lines:
                elements.append(
                    DocumentElement(
                        "code",
                        "\n".join(code_lines).strip("\n"),
                        metadata={"language": code_language},
                    )
                )
                code_lines = []
                code_language = ""

        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code:
                    flush_code()
                    in_code = False
                else:
                    flush_paragraph()
                    flush_list()
                    flush_table()
                    in_code = True
                    code_language = stripped.strip("`").strip()
                continue

            if in_code:
                code_lines.append(line)
                continue

            image_match = IMAGE_PATTERN.search(stripped)
            if image_match:
                flush_paragraph()
                flush_list()
                flush_table()
                caption = (image_match.group("caption") or image_match.group("label") or "").strip()
                elements.append(
                    DocumentElement(
                        "image",
                        caption,
                        metadata={
                            "caption": caption,
                            "source": image_match.groupdict().get("src"),
                            "multimodal_embedding": False,
                        },
                    )
                )
                continue

            heading = self._parse_heading(stripped)
            if heading:
                flush_paragraph()
                flush_list()
                flush_table()
                level, title = heading
                elements.append(DocumentElement("heading", title, level=level))
                continue

            if self._looks_like_table_row(stripped):
                flush_paragraph()
                flush_list()
                table_lines.append(stripped)
                continue

            if LIST_PATTERN.match(stripped):
                flush_paragraph()
                flush_table()
                list_lines.append(stripped)
                continue

            if not stripped:
                flush_paragraph()
                flush_list()
                flush_table()
                continue

            flush_list()
            flush_table()
            paragraph_lines.append(stripped)

        flush_code()
        flush_paragraph()
        flush_list()
        flush_table()
        self._attach_context(elements)
        return elements

    @staticmethod
    def _parse_heading(line: str) -> tuple[int, str] | None:
        if not line:
            return None
        markdown = HEADING_PATTERNS[0].match(line)
        if markdown:
            return len(markdown.group(1)), markdown.group(2).strip()
        numbered = HEADING_PATTERNS[1].match(line)
        if numbered and len(line) <= 140:
            return numbered.group(1).count(".") + 1, numbered.group(2).strip()
        if len(line) <= 80 and line.isupper() and any(ch.isalpha() for ch in line):
            return 1, line
        return None

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        if not line:
            return False
        return line.count("|") >= 2 or line.count("\t") >= 2

    @staticmethod
    def _table_to_text(lines: list[str]) -> str:
        rows = []
        for line in lines:
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            cells = [cell.strip() for cell in re.split(r"\||\t", line) if cell.strip()]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _attach_context(elements: list[DocumentElement]) -> None:
        for index, element in enumerate(elements):
            if element.element_type != "image":
                continue
            before = elements[index - 1].text if index > 0 else ""
            after = elements[index + 1].text if index + 1 < len(elements) else ""
            element.metadata["surrounding_text"] = " ".join([before[-240:], after[:240]]).strip()


class SemanticChunker:
    """Chunk structural elements by approximate token budgets without breaking code."""

    def __init__(
        self,
        coarse_min_tokens: int = 2000,
        coarse_max_tokens: int = 3000,
        semantic_min_tokens: int = 512,
        semantic_max_tokens: int = 1024,
    ):
        self.coarse_min_tokens = coarse_min_tokens
        self.coarse_max_tokens = coarse_max_tokens
        self.semantic_min_tokens = semantic_min_tokens
        self.semantic_max_tokens = semantic_max_tokens

    def chunk(self, elements: list[DocumentElement]) -> list[dict[str, Any]]:
        if not elements:
            return []
        coarse_blocks = self._coarse_blocks(elements)
        chunks: list[dict[str, Any]] = []
        for block_index, block in enumerate(coarse_blocks):
            chunks.extend(self._semantic_chunks(block, block_index))
        return chunks

    def fallback_chunks(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(1500, self.semantic_max_tokens * 2),
            chunk_overlap=160,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "。", "；", ";", " ", ""],
        )
        docs = splitter.create_documents([text], [metadata])
        return [
            {
                "text": doc.page_content.strip(),
                "elements": [],
                "metadata": {**metadata, "parser_fallback": True},
            }
            for doc in docs
            if doc.page_content.strip()
        ]

    def _coarse_blocks(self, elements: list[DocumentElement]) -> list[list[DocumentElement]]:
        blocks: list[list[DocumentElement]] = []
        current: list[DocumentElement] = []
        current_tokens = 0
        for element in elements:
            tokens = estimate_tokens(element.text)
            starts_section = element.element_type == "heading" and current_tokens >= self.coarse_min_tokens
            exceeds = current and current_tokens + tokens > self.coarse_max_tokens
            if starts_section or exceeds:
                blocks.append(current)
                current = []
                current_tokens = 0
            current.append(element)
            current_tokens += tokens
        if current:
            blocks.append(current)
        return blocks

    def _semantic_chunks(self, block: list[DocumentElement], block_index: int) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_parts: list[str] = []
        current_elements: list[dict[str, Any]] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_parts, current_elements, current_tokens
            text = "\n\n".join(part for part in current_parts if part.strip()).strip()
            if text:
                chunks.append(
                    {
                        "text": text,
                        "elements": current_elements,
                        "metadata": {
                            "coarse_block_index": block_index,
                            "semantic_chunking": True,
                            "token_estimate": current_tokens,
                            "parser_fallback": False,
                        },
                    }
                )
            current_parts = []
            current_elements = []
            current_tokens = 0

        for element in block:
            pieces = [element.text] if element.element_type == "code" else split_semantic_units(element.text)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                tokens = estimate_tokens(piece)
                if current_parts and current_tokens + tokens > self.semantic_max_tokens:
                    flush()
                current_parts.append(piece)
                current_elements.append(
                    {
                        "type": element.element_type,
                        "level": element.level,
                        **element.metadata,
                    }
                )
                current_tokens += tokens
        flush()
        return chunks


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = max(0, len(text) - chinese_chars) // 4
    return max(1, chinese_chars + latin_words + other)


def split_semantic_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= 1024:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[。！？.!?；;])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units or ([text.strip()] if text.strip() else [])
