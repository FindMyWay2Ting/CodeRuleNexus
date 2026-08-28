"""Code Wiki 事实扫描器的最小回归测试。"""

import tempfile
import unittest
from pathlib import Path

from app.code_wiki import scan_project


class CodeWikiScannerTests(unittest.TestCase):
    def test_python_symbols_and_calls_keep_source_function(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.py").write_text(
                "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
                encoding="utf-8",
            )

            scan = scan_project(str(root))
            file_fact = scan["files"][0]
            symbol_names = {symbol.qualified_name for symbol in file_fact.symbols}
            calls = [relation for relation in file_fact.relations if relation.relation_type == "calls"]

            self.assertIn("helper", symbol_names)
            self.assertIn("run", symbol_names)
            self.assertTrue(any(
                relation.source_symbol_key == "service.py::run" and relation.target_ref == "helper"
                for relation in calls
            ))

    def test_docker_requires_real_deployment_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 普通源码中出现 Dockerfile/docker-compose 字样不能作为部署组件证据。
            (root / "notes.py").write_text(
                "MESSAGE = 'Dockerfile and docker-compose are only examples'\n",
                encoding="utf-8",
            )
            without_file = scan_project(str(root))
            self.assertNotIn("Docker", {item["name"] for item in without_file["components"]})

            (root / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
            with_file = scan_project(str(root))
            self.assertIn("Docker", {item["name"] for item in with_file["components"]})

    def test_go_tree_sitter_extracts_functions_and_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.go").write_text(
                "package main\n\ntype Service struct{}\n\nfunc (s *Service) Run() {}\n\nfunc main() {}\n",
                encoding="utf-8",
            )

            scan = scan_project(str(root))
            symbols = scan["files"][0].symbols
            self.assertIn("Service", {symbol.name for symbol in symbols})
            self.assertIn("Run", {symbol.name for symbol in symbols})
            self.assertIn("main", {symbol.name for symbol in symbols})


if __name__ == "__main__":
    unittest.main()
