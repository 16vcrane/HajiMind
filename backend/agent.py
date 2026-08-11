import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from sqlalchemy import select, desc

from auth import get_current_user
from cache import cache_delete, cache_get_json, cache_set_json
from database import db_session
from models import ChatMessage, ChatSession, User
from tools import (
    get_agent_tools,
    get_last_rag_context,
    reset_tool_call_guards,
    set_rag_step_queue,
)

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class ConversationStorage:
    """PostgreSQL-backed conversation storage with optional Redis cache."""

    def __init__(self):
        self.session_cache_ttl = int(os.getenv("SESSION_CACHE_TTL", "300"))

    def _session_cache_key(self, user_id: int, session_id: str) -> str:
        return f"hajimind:sessions:{user_id}:{session_id}"

    def _list_cache_key(self, user_id: int) -> str:
        return f"hajimind:sessions:list:{user_id}"

    def _serialize_message(self, msg: ChatMessage | HumanMessage | AIMessage | SystemMessage) -> dict:
        rag_trace = None
        if isinstance(msg, ChatMessage):
            rag_trace = msg.rag_trace
        return {
            "type": msg.type if hasattr(msg, "type") else "unknown",
            "content": msg.content,
            "timestamp": getattr(msg, "timestamp", datetime.utcnow()).isoformat() if not isinstance(msg, (HumanMessage, AIMessage, SystemMessage)) else datetime.utcnow().isoformat(),
            "rag_trace": rag_trace,
        }

    def _load_session_model(self, db, user_id: int, session_id: str):
        stmt = select(ChatSession).where(
            ChatSession.user_id == user_id,
            ChatSession.session_id == session_id,
        )
        return db.execute(stmt).scalar_one_or_none()

    def save(self, user_id: int, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        metadata = metadata or {}
        extra_message_data = extra_message_data or []

        with db_session() as db:
            user = db.get(User, user_id)
            if not user:
                raise HTTPException(status_code=404, detail="用户不存在")

            session = self._load_session_model(db, user_id, session_id)
            if not session:
                session = ChatSession(
                    user_id=user_id,
                    session_id=session_id,
                    metadata_json=metadata,
                    updated_at=datetime.utcnow(),
                )
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata
                session.updated_at = datetime.utcnow()
                db.query(ChatMessage).filter(ChatMessage.session_pk == session.id).delete()

            payload = []
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
                payload.append(ChatMessage(
                    session_pk=session.id,
                    user_id=user_id,
                    type=msg.type,
                    content=msg.content,
                    rag_trace=rag_trace,
                    timestamp=datetime.utcnow(),
                ))

            db.add_all(payload)

        cache_delete(self._session_cache_key(user_id, session_id), self._list_cache_key(user_id))

    def load(self, user_id: int, session_id: str) -> list:
        cache_key = self._session_cache_key(user_id, session_id)
        cached = cache_get_json(cache_key)
        if cached:
            return [HumanMessage(content=item["content"]) if item["type"] == "human"
                    else AIMessage(content=item["content"]) if item["type"] == "ai"
                    else SystemMessage(content=item["content"])
                    for item in cached]

        with db_session() as db:
            session = self._load_session_model(db, user_id, session_id)
            if not session:
                return []
            messages = []
            serialized = []
            for msg in session.messages:
                if msg.type == "human":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    messages.append(AIMessage(content=msg.content))
                elif msg.type == "system":
                    messages.append(SystemMessage(content=msg.content))
                serialized.append(self._serialize_message(msg))

            cache_set_json(cache_key, serialized, ttl=self.session_cache_ttl)
            return messages

    def list_sessions(self, user_id: int) -> list:
        cache_key = self._list_cache_key(user_id)
        cached = cache_get_json(cache_key)
        if cached is not None:
            return cached

        with db_session() as db:
            stmt = (
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(desc(ChatSession.updated_at))
            )
            sessions = db.execute(stmt).scalars().all()
            result = [
                {
                    "session_id": session.session_id,
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": len(session.messages),
                }
                for session in sessions
            ]
            cache_set_json(cache_key, result, ttl=self.session_cache_ttl)
            return result

    def delete_session(self, user_id: int, session_id: str) -> bool:
        with db_session() as db:
            session = self._load_session_model(db, user_id, session_id)
            if not session:
                return False
            db.delete(session)
        cache_delete(self._session_cache_key(user_id, session_id), self._list_cache_key(user_id))
        return True


def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=get_agent_tools(),
        system_prompt=(
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        ),
    )
    return agent, model


agent, model = create_agent_instance()
storage = ConversationStorage()


def summarize_old_messages(model, messages: list) -> str:
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


def _response_to_text(result) -> str:
    if isinstance(result, dict):
        if "output" in result:
            return result["output"]
        if "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            return getattr(msg, "content", str(msg))
        return str(result)
    if hasattr(result, "content"):
        return result.content
    return str(result)


def chat_with_agent(user_text: str, user_id: int, session_id: str):
    messages = storage.load(user_id, session_id)
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    result = agent.invoke({"messages": messages}, config={"recursion_limit": 8})
    response_content = _response_to_text(result)
    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {"response": response_content, "rag_trace": rag_trace}


async def chat_with_agent_stream(user_text: str, user_id: int, session_id: str):
    messages = storage.load(user_id, session_id)
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    output_queue = asyncio.Queue()

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    full_response = ""

    async def _agent_worker():
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")
                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            await output_queue.put(None)

    agent_task = asyncio.create_task(_agent_worker())

    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
