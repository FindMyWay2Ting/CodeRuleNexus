"""Agent Loop 状态、进展和停止规则测试。"""

import unittest

from app.agent_loop import AgentLoopState


class AgentLoopStateTests(unittest.TestCase):
    def test_default_round_budget_is_ten(self) -> None:
        self.assertEqual(10, AgentLoopState().max_rounds)

    def test_full_file_inventory_is_only_admitted_once(self) -> None:
        state = AgentLoopState()
        self.assertEqual((True, None), state.admit("list_project_files", {}))
        state.record("list_project_files", {}, {"items": [{"path": "compose.yml"}], "coverage": {"total": 1}})
        self.assertEqual((False, "project_files_already_inspected"), state.admit("list_project_files", {}))

    def test_same_query_is_blocked_but_changed_query_is_allowed(self) -> None:
        state = AgentLoopState()
        self.assertEqual((True, None), state.admit("find_config_facts", {"query": "kafka"}))
        state.record("find_config_facts", {"query": "kafka"}, {"items": [], "count": 0})
        self.assertEqual((False, "same_query_repeated"), state.admit("find_config_facts", {"query": "kafka"}))
        self.assertEqual((True, None), state.admit("find_config_facts", {"query": "compose"}))

    def test_two_empty_results_stop_for_no_progress(self) -> None:
        state = AgentLoopState()
        self.assertEqual((True, None), state.admit("search_symbols", {"query": "Kafka"}))
        state.record("search_symbols", {"query": "Kafka"}, {"items": []})
        self.assertTrue(state.evaluate_round(0, executed_calls=1).should_continue)
        self.assertEqual((True, None), state.admit("search_symbols", {"query": "KafkaProducer"}))
        state.record("search_symbols", {"query": "KafkaProducer"}, {"items": []})
        transition = state.evaluate_round(1, executed_calls=1)
        self.assertFalse(transition.should_continue)
        self.assertEqual("no_new_evidence", transition.reason)

    def test_two_empty_tools_in_one_round_count_as_one_empty_round(self) -> None:
        state = AgentLoopState()
        state.admit("search_symbols", {"query": "Kafka"})
        state.record("search_symbols", {"query": "Kafka"}, {"items": []})
        state.admit("find_config_facts", {"query": "Kafka"})
        state.record("find_config_facts", {"query": "Kafka"}, {"items": []})

        first_round = state.evaluate_round(0, executed_calls=2)

        self.assertTrue(first_round.should_continue)
        self.assertEqual(1, first_round.unchanged_rounds)
        state.admit("search_symbols", {"query": "KafkaProducer"})
        state.record("search_symbols", {"query": "KafkaProducer"}, {"items": []})
        second_round = state.evaluate_round(1, executed_calls=1)
        self.assertEqual("no_new_evidence", second_round.reason)

    def test_any_new_evidence_resets_round_no_progress(self) -> None:
        state = AgentLoopState(unchanged_rounds=1)
        state.admit("search_symbols", {"query": "missing"})
        state.record("search_symbols", {"query": "missing"}, {"items": []})
        state.admit("find_config_facts", {"query": "compose"})
        state.record("find_config_facts", {"query": "compose"}, {"items": [{"path": "compose.yaml"}]})

        transition = state.evaluate_round(1, executed_calls=2)

        self.assertTrue(transition.should_continue)
        self.assertEqual(0, transition.unchanged_rounds)

    def test_changed_legal_call_can_progress_when_another_call_is_blocked(self) -> None:
        state = AgentLoopState()
        state.admit("find_config_facts", {"query": "kafka"})
        state.record("find_config_facts", {"query": "kafka"}, {"items": [], "count": 0})
        allowed, reason = state.admit("find_config_facts", {"query": "kafka"})
        self.assertFalse(allowed)
        self.assertEqual("same_query_repeated", reason)
        self.assertEqual((True, None), state.admit("find_config_facts", {"query": "compose"}))
        state.record("find_config_facts", {"query": "compose"}, {"items": [{"path": "compose.yml"}]})

        transition = state.evaluate_round(1, executed_calls=1, blocked_reasons=[reason])

        self.assertTrue(transition.should_continue)
        self.assertEqual(0, transition.unchanged_rounds)

    def test_explicit_agent_finish_sets_terminal_transition(self) -> None:
        state = AgentLoopState()
        transition = state.finish("agent_completed", round_index=2)
        self.assertFalse(transition.should_continue)
        self.assertEqual("agent_completed", transition.reason)
        self.assertEqual(3, transition.completed_rounds)

    def test_round_budget_is_evaluated_by_engine(self) -> None:
        state = AgentLoopState(max_rounds=2, max_tool_calls=10)
        state.admit("search_symbols", {"query": "one"})
        state.record("search_symbols", {"query": "one"}, {"items": [{"name": "one"}]})
        self.assertTrue(state.evaluate_round(0, executed_calls=1).should_continue)
        state.admit("search_symbols", {"query": "two"})
        state.record("search_symbols", {"query": "two"}, {"items": [{"name": "two"}]})
        transition = state.evaluate_round(1, executed_calls=1)
        self.assertEqual("round_budget_exhausted", transition.reason)

    def test_tool_budget_is_evaluated_by_engine(self) -> None:
        state = AgentLoopState(max_rounds=10, max_tool_calls=1)
        state.admit("search_symbols", {"query": "one"})
        state.record("search_symbols", {"query": "one"}, {"items": [{"name": "one"}]})
        transition = state.evaluate_round(0, executed_calls=1)
        self.assertEqual("tool_budget_exhausted", transition.reason)


if __name__ == "__main__":
    unittest.main()
