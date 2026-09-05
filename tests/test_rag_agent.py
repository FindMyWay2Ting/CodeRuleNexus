"""文档 RAG 自适应检索状态测试。"""

import unittest

from app.adaptive_retrieval import AdaptiveRetrievalState


class AdaptiveRetrievalStateTests(unittest.TestCase):
    def test_replan_only_after_execution_reports_no_evidence(self) -> None:
        state = AdaptiveRetrievalState("如何配置消息队列？")
        first = state.create()
        state.complete(first, 0, "没有命中")
        second = state.replan()
        self.assertIsNotNone(second)
        self.assertNotEqual(first.query, second.query)
        self.assertEqual("pending", second.status)

    def test_replan_is_bounded_and_deduplicated(self) -> None:
        state = AdaptiveRetrievalState("Kafka client configuration in production")
        state.create()
        self.assertIsNotNone(state.replan())
        self.assertIsNotNone(state.replan())
        self.assertIsNone(state.replan())


if __name__ == "__main__":
    unittest.main()
