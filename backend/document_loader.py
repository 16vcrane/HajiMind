"""文档加载和分片服务"""
import os
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from document_structure import DocumentStructureParser, SemanticChunker


class DocumentLoader:
    """文档加载和分片服务"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 保留原有参数以兼容外部调用；默认启用三层滑动窗口分块。
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._structure_parser = DocumentStructureParser()
        self._semantic_chunker = SemanticChunker()

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        if not text:
            return []

        structured_chunks = self._split_structured_page_to_three_levels(
            text=text,
            base_doc=base_doc,
            page_global_chunk_idx=page_global_chunk_idx,
        )
        if structured_chunks:
            return structured_chunks

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            if "code" in common_metadata["structure_types"]:
                level_2_docs = [{"page_content": level_1_text}]
            else:
                level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                if "code" in common_metadata["structure_types"]:
                    level_3_docs = [{"page_content": level_2_text}]
                else:
                    level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def _split_structured_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        elements = self._structure_parser.parse(text)
        semantic_chunks = self._semantic_chunker.chunk(elements)
        if not semantic_chunks:
            semantic_chunks = self._semantic_chunker.fallback_chunks(text, base_doc)
        if not semantic_chunks:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        for level_1_counter, semantic_chunk in enumerate(semantic_chunks):
            level_1_text = semantic_chunk["text"].strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            common_metadata = self._chunk_structure_metadata(semantic_chunk)
            root_chunks.append(
                {
                    **base_doc,
                    **common_metadata,
                    "text": level_1_text,
                    "chunk_id": level_1_id,
                    "parent_chunk_id": "",
                    "root_chunk_id": level_1_id,
                    "chunk_level": 1,
                    "chunk_idx": page_global_chunk_idx,
                }
            )
            page_global_chunk_idx += 1

            if "code" in common_metadata["structure_types"]:
                level_2_docs = [{"page_content": level_1_text}]
            else:
                level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_counter, level_2_doc in enumerate(level_2_docs):
                level_2_text = (
                    getattr(level_2_doc, "page_content", None)
                    or level_2_doc.get("page_content", "")
                ).strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(
                    filename,
                    page_number,
                    2,
                    level_1_counter * 1000 + level_2_counter,
                )
                root_chunks.append(
                    {
                        **base_doc,
                        **common_metadata,
                        "text": level_2_text,
                        "chunk_id": level_2_id,
                        "parent_chunk_id": level_1_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 2,
                        "chunk_idx": page_global_chunk_idx,
                    }
                )
                page_global_chunk_idx += 1

                if "code" in common_metadata["structure_types"]:
                    level_3_docs = [{"page_content": level_2_text}]
                else:
                    level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_counter, level_3_doc in enumerate(level_3_docs):
                    level_3_text = (
                        getattr(level_3_doc, "page_content", None)
                        or level_3_doc.get("page_content", "")
                    ).strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(
                        filename,
                        page_number,
                        3,
                        level_1_counter * 1000000 + level_2_counter * 1000 + level_3_counter,
                    )
                    root_chunks.append(
                        {
                            **base_doc,
                            **common_metadata,
                            "text": level_3_text,
                            "chunk_id": level_3_id,
                            "parent_chunk_id": level_2_id,
                            "root_chunk_id": level_1_id,
                            "chunk_level": 3,
                            "chunk_idx": page_global_chunk_idx,
                        }
                    )
                    page_global_chunk_idx += 1

        return root_chunks

    @staticmethod
    def _chunk_structure_metadata(semantic_chunk: Dict) -> Dict:
        elements = semantic_chunk.get("elements") or []
        element_types = sorted({item.get("type") for item in elements if item.get("type")})
        image_elements = [item for item in elements if item.get("type") == "image"]
        return {
            "structure_types": element_types,
            "primary_structure_type": element_types[0] if element_types else "paragraph",
            "semantic_chunking": bool(semantic_chunk.get("metadata", {}).get("semantic_chunking")),
            "parser_fallback": bool(semantic_chunk.get("metadata", {}).get("parser_fallback")),
            "token_estimate": int(semantic_chunk.get("metadata", {}).get("token_estimate") or 0),
            "image_metadata": [
                {
                    "caption": item.get("caption", ""),
                    "source": item.get("source"),
                    "surrounding_text": item.get("surrounding_text", ""),
                    "multimodal_embedding": False,
                }
                for item in image_elements
            ],
        }

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并分片
        :param file_path: 文件路径
        :param filename: 文件名
        :return: 分片后的文档列表
        """
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": doc.metadata.get("page", 0),
                }
                page_chunks = self._split_page_to_three_levels(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹加载所有文档并分片
        :param folder_path: 文件夹路径
        :return: 所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
