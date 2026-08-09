import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent import chat_with_agent, chat_with_agent_stream, storage
from auth import (
    create_access_token,
    get_current_user,
    hash_password,
    is_first_user,
    require_admin,
    verify_password,
)
from database import get_db
from document_loader import DocumentLoader
from embedding import EmbeddingService
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from models import ChatSession, User
from parent_chunk_store import ParentChunkStore
from schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    LoginRequest,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
    TokenResponse,
    UserInfo,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
embedding_service = EmbeddingService()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

router = APIRouter()


@router.post("/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = request.username.strip()
    if len(username) < 3 or len(username) > 80:
        raise HTTPException(status_code=400, detail="用户名长度必须为 3-80 个字符")
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="密码至少需要 8 个字符")
    if db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="用户名已存在")

    user = User(
        username=username,
        password_hash=hash_password(request.password),
        role="admin" if is_first_user(db) else "user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(
        access_token=create_access_token(user),
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(
        select(User).where(User.username == request.username.strip())
    ).scalar_one_or_none()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="用户已停用")
    return TokenResponse(
        access_token=create_access_token(user),
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


@router.get("/auth/me", response_model=UserInfo)
async def me(current_user: User = Depends(get_current_user)):
    return UserInfo(user_id=current_user.id, username=current_user.username, role=current_user.role)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    try:
        sessions = storage.list_sessions(current_user.id)
        return SessionListResponse(sessions=[SessionInfo(**item) for item in sessions])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取会话失败: {exc}")


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = db.execute(
        select(ChatSession).where(
            ChatSession.user_id == current_user.id,
            ChatSession.session_id == session_id,
        )
    ).scalar_one_or_none()
    if not session:
        return SessionMessagesResponse(messages=[])
    return SessionMessagesResponse(
        messages=[
            MessageInfo(
                type=msg.type,
                content=msg.content,
                timestamp=msg.timestamp.isoformat(),
                rag_trace=msg.rag_trace,
            )
            for msg in session.messages
        ]
    )


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        deleted = storage.delete_session(current_user.id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _model_error(exc: Exception) -> HTTPException:
    message = str(exc)
    match = re.search(r"Error code:\s*(\d{3})", message)
    if match:
        return HTTPException(status_code=int(match.group(1)), detail=message)
    return HTTPException(status_code=500, detail=message)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        return ChatResponse(**chat_with_agent(request.message, current_user.id, request.session_id))
    except Exception as exc:
        raise _model_error(exc)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(
                request.message,
                current_user.id,
                request.session_id,
            ):
                yield chunk
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    try:
        milvus_manager.init_collection()
        results = milvus_manager.query(output_fields=["filename", "file_type"], limit=10000)
        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": item.get("file_type", ""),
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1
        return DocumentListResponse(documents=[DocumentInfo(**item) for item in file_stats.values()])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {exc}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
            raise HTTPException(status_code=400, detail="仅支持 PDF 和 Word 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection()
        milvus_manager.delete(f'filename == "{filename}"')
        parent_chunk_store.delete_by_filename(filename)

        file_path = UPLOAD_DIR / Path(filename).name
        with open(file_path, "wb") as target:
            target.write(await file.read())

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {exc}")
        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)
        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，父级分块 {len(parent_docs)} 个（存入 PostgreSQL）",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {exc}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    try:
        milvus_manager.init_collection()
        result = milvus_manager.delete(f'filename == "{filename}"')
        parent_chunk_store.delete_by_filename(filename)
        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {exc}")
