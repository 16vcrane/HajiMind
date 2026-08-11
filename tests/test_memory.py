import sys
import time
import unittest
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "backend")

import agent
from memory import MemoryPolicyError, MemoryService, MemoryType, extract_memory_candidate
from models import User, UserMemory


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        User.__table__.create(self.engine)
        UserMemory.__table__.create(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, future=True)

        @contextmanager
        def session_scope():
            db = self.session_factory()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        self.service = MemoryService(session_scope=session_scope, max_retrieved=6)

    def tearDown(self):
        self.engine.dispose()

    def test_preference_extraction_creates_long_lived_memory(self):
        candidate, reason = extract_memory_candidate("以后请始终使用中文回答。")

        self.assertIsNone(reason)
        self.assertEqual(candidate.memory_type, MemoryType.PREFERENCE)
        self.assertIn("以后请始终使用中文回答", candidate.content)

        extracted = self.service.extract_and_store(1, candidate.content, "session-1")
        self.assertEqual(len(extracted.memory_ids), 1)
        self.assertEqual(self.service.retrieve(1)[0]["memory_type"], MemoryType.PREFERENCE.value)

    def test_temporary_and_sensitive_text_is_not_extracted(self):
        temporary, temporary_reason = extract_memory_candidate("提醒我明天提交作业。")
        sensitive, sensitive_reason = extract_memory_candidate("我的 API key 是 sk-secret-token-value")

        self.assertIsNone(temporary)
        self.assertEqual(temporary_reason, "temporary_or_task_information")
        self.assertIsNone(sensitive)
        self.assertEqual(sensitive_reason, "sensitive_information")

    def test_crud_is_scoped_to_the_current_user(self):
        created = self.service.create(
            1,
            content="我偏好简洁的中文回答。",
            memory_type=MemoryType.PREFERENCE,
        )
        self.service.create(
            2,
            content="我偏好英文回答。",
            memory_type=MemoryType.PREFERENCE,
        )

        self.assertEqual([item["id"] for item in self.service.retrieve(1)], [created["id"]])
        self.assertEqual(self.service.retrieve(2)[0]["user_id"], 2)
        self.assertIsNone(self.service.update(2, created["id"], content="越权修改"))

        updated = self.service.update(1, created["id"], content="我偏好结构化中文回答。")
        self.assertIn("结构化中文", updated["content"])
        self.assertFalse(self.service.delete(2, created["id"]))
        self.assertTrue(self.service.delete(1, created["id"]))
        self.assertEqual(self.service.retrieve(1), [])

    def test_memory_service_never_persists_sensitive_or_temporary_records(self):
        with self.assertRaises(MemoryPolicyError):
            self.service.create(
                1,
                content="API_KEY=super-secret",
                memory_type=MemoryType.LONG_TERM,
            )
        with self.assertRaises(MemoryPolicyError):
            self.service.create(
                1,
                content="提醒我明天完成报告",
                memory_type=MemoryType.LONG_TERM,
            )


class AgentMemoryTraceTests(unittest.TestCase):
    def test_memory_trace_is_included_in_agent_trace(self):
        memory_trace = {
            "working_memory_used": True,
            "conversation_message_count": 4,
            "retrieved_memory_ids": [11],
            "extracted_memory_ids": [12],
            "skipped_reasons": [],
        }
        trace = agent._build_agent_trace(
            route_context={"route_target": "llm", "memory_trace": memory_trace},
            legacy_trace={"tool_used": False, "tool_name": "LLM"},
            request_id="request-1",
            session_id="session-1",
            user_id=1,
            started_at=time.perf_counter(),
            final_status="success",
        )

        self.assertEqual(trace["agent_trace"]["memory"]["retrieved_memory_ids"], [11])
        self.assertEqual(trace["memory_trace"]["extracted_memory_ids"], [12])


if __name__ == "__main__":
    unittest.main()
