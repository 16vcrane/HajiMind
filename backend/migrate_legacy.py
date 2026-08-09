"""One-time migration from the old JSON stores to PostgreSQL.

Usage:
    python backend/migrate_legacy.py --default-password "change-me-now"
"""
import argparse
import json
from pathlib import Path

from sqlalchemy import select

from auth import hash_password
from database import db_session, init_db
from models import ChatMessage, ChatSession, ParentChunk, User


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def migrate_history(default_password: str) -> tuple[int, int]:
    data = load_json(DATA_DIR / "customer_service_history.json", {})
    users_created = 0
    messages_created = 0

    with db_session() as db:
        for legacy_user_id, sessions in data.items():
            username = f"legacy_{legacy_user_id}"
            user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
            if not user:
                user = User(username=username, password_hash=hash_password(default_password), role="user")
                db.add(user)
                db.flush()
                users_created += 1

            for session_id, session_data in (sessions or {}).items():
                session = db.execute(
                    select(ChatSession).where(
                        ChatSession.user_id == user.id,
                        ChatSession.session_id == session_id,
                    )
                ).scalar_one_or_none()
                if not session:
                    session = ChatSession(
                        user_id=user.id,
                        session_id=session_id,
                        metadata_json=session_data.get("metadata", {}),
                    )
                    db.add(session)
                    db.flush()

                if session.messages:
                    continue
                for item in session_data.get("messages", []):
                    db.add(ChatMessage(
                        session_pk=session.id,
                        user_id=user.id,
                        type=item.get("type", "unknown"),
                        content=item.get("content", ""),
                        rag_trace=item.get("rag_trace"),
                    ))
                    messages_created += 1

    return users_created, messages_created


def migrate_parent_chunks() -> int:
    data = load_json(DATA_DIR / "parent_chunks.json", {})
    migrated = 0
    with db_session() as db:
        for chunk_id, item in (data or {}).items():
            existing = db.execute(
                select(ParentChunk).where(ParentChunk.chunk_id == chunk_id)
            ).scalar_one_or_none()
            if existing:
                continue
            db.add(ParentChunk(
                chunk_id=chunk_id,
                text=item.get("text", ""),
                filename=item.get("filename", ""),
                file_type=item.get("file_type", ""),
                file_path=item.get("file_path", ""),
                page_number=int(item.get("page_number", 0) or 0),
                parent_chunk_id=item.get("parent_chunk_id", ""),
                root_chunk_id=item.get("root_chunk_id", ""),
                chunk_level=int(item.get("chunk_level", 0) or 0),
                chunk_idx=int(item.get("chunk_idx", 0) or 0),
            ))
            migrated += 1
    return migrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-password", required=True)
    args = parser.parse_args()
    if len(args.default_password) < 8:
        raise SystemExit("default password must contain at least 8 characters")

    init_db()
    users, messages = migrate_history(args.default_password)
    parents = migrate_parent_chunks()
    print(f"migrated users={users}, messages={messages}, parent_chunks={parents}")


if __name__ == "__main__":
    main()
