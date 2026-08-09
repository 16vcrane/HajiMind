"""PostgreSQL-backed parent chunk store for Auto-merging."""
from datetime import datetime
from typing import List

from sqlalchemy import delete, select

from database import db_session
from models import ParentChunk


class ParentChunkStore:
    """Store L1/L2 parent chunks in PostgreSQL."""

    def upsert_documents(self, docs: List[dict]) -> int:
        if not docs:
            return 0

        count = 0
        with db_session() as db:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                existing = db.execute(
                    select(ParentChunk).where(ParentChunk.chunk_id == chunk_id)
                ).scalar_one_or_none()
                values = {
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **values))
                count += 1
        return count

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []
        with db_session() as db:
            rows = db.execute(
                select(ParentChunk).where(ParentChunk.chunk_id.in_(chunk_ids))
            ).scalars().all()
            row_map = {row.chunk_id: row for row in rows}
            return [self._to_dict(row_map[item]) for item in chunk_ids if item in row_map]

    def delete_by_filename(self, filename: str) -> int:
        if not filename:
            return 0
        with db_session() as db:
            result = db.execute(delete(ParentChunk).where(ParentChunk.filename == filename))
            return result.rowcount or 0

    @staticmethod
    def _to_dict(row: ParentChunk) -> dict:
        return {
            "text": row.text,
            "filename": row.filename,
            "file_type": row.file_type,
            "file_path": row.file_path,
            "page_number": row.page_number,
            "chunk_id": row.chunk_id,
            "parent_chunk_id": row.parent_chunk_id,
            "root_chunk_id": row.root_chunk_id,
            "chunk_level": row.chunk_level,
            "chunk_idx": row.chunk_idx,
        }
