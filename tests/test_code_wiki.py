"""Code Wiki 事实扫描器的最小回归测试。"""

import asyncio
from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi import UploadFile

from app.code_wiki import (
    _scip_indexer_status,
    managed_local_repository_path,
    normalize_github_url,
    normalize_uploaded_path,
    scan_project,
)
from app.main import import_local_code_project


class CodeWikiScannerTests(unittest.TestCase):
    def test_browser_folder_upload_preserves_tree_and_updates_snapshot(self) -> None:
        def fake_persist(scan: dict) -> dict:
            return {
                "project_id": scan["project_id"],
                "project_name": scan["project_name"],
                "commit_hash": scan["commit_hash"],
                "file_count": len(scan["files"]),
                "symbol_count": sum(len(item.symbols) for item in scan["files"]),
                "relation_count": sum(len(item.relations) for item in scan["files"]),
                "component_count": len(scan["components"]),
                "scip_indexers": scan["scip_indexers"],
                "scip": scan["scip"],
            }

        def upload(content: bytes) -> UploadFile:
            return UploadFile(filename="demo-project/src/main.py", file=BytesIO(content))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("app.main.DEFAULT_REPOSITORY_ROOT", root),
                patch("app.main.persist_scan", side_effect=fake_persist),
            ):
                first = asyncio.run(import_local_code_project([upload(b"def run():\n    return 1\n")]))
                second = asyncio.run(import_local_code_project([upload(b"def run():\n    return 2\n")]))

            target = managed_local_repository_path("demo-project", root)
            self.assertEqual("uploaded", first["import_action"])
            self.assertEqual("updated", second["import_action"])
            self.assertNotEqual(first["commit_hash"], second["commit_hash"])
            self.assertEqual("def run():\n    return 2\n", (target / "src" / "main.py").read_text(encoding="utf-8"))

    def test_local_directory_upload_paths_are_confined(self) -> None:
        project, relative = normalize_uploaded_path("payment-service/app/main.py")
        self.assertEqual("payment-service", project)
        self.assertEqual("app/main.py", relative.as_posix())
        for invalid_path in ("main.py", "../main.py", "project/../../secret.txt", "C:/secret.txt"):
            with self.subTest(path=invalid_path), self.assertRaises(ValueError):
                normalize_uploaded_path(invalid_path)

    def test_local_upload_target_is_stable_for_same_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = managed_local_repository_path("支付服务", root)
            second = managed_local_repository_path("支付服务", root)
            another = managed_local_repository_path("订单服务", root)
            self.assertEqual(first, second)
            self.assertNotEqual(first, another)

    def test_github_url_is_normalized_without_accepting_file_pages(self) -> None:
        canonical, owner, repository = normalize_github_url("https://github.com/openai/openai-python/")
        self.assertEqual("https://github.com/openai/openai-python.git", canonical)
        self.assertEqual(("openai", "openai-python"), (owner, repository))

        invalid_urls = [
            "git@github.com:openai/openai-python.git",
            "https://gitlab.com/openai/openai-python",
            "https://github.com/openai/openai-python/tree/main",
            "https://user:token@github.com/openai/openai-python",
        ]
        for invalid_url in invalid_urls:
            with self.subTest(url=invalid_url), self.assertRaises(ValueError):
                normalize_github_url(invalid_url)

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

    def test_generated_bindings_are_not_scanned_as_business_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / "schema_pb2.py").write_text("def generated():\n    return 2\n", encoding="utf-8")
            (root / ".codex_tmp").mkdir()
            (root / ".codex_tmp" / "browser.js").write_text("function noise(){}", encoding="utf-8")

            scan = scan_project(str(root))

            self.assertEqual(["service.py"], [item.path for item in scan["files"]])

    def test_same_line_duplicate_symbols_are_collapsed_to_database_fact_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bundle.js").write_text("function same(){} function same(){}", encoding="utf-8")

            scan = scan_project(str(root))
            symbols = [item for item in scan["files"][0].symbols if item.kind != "file"]

            self.assertEqual(1, len(symbols))

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

    def test_go_scip_index_is_generated_and_merged(self) -> None:
        """真实执行 scip-go，防止“CLI 已安装”被误当成“索引链路已接通”。"""
        if _scip_indexer_status()["go"]["status"] != "ready":
            self.skipTest("scip-go is not available")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "go.mod").write_text("module example.com/wiki-test\n\ngo 1.20\n", encoding="utf-8")
            (root / "main.go").write_text(
                "package main\n\nfunc helper() int { return 1 }\n\nfunc main() { _ = helper() }\n",
                encoding="utf-8",
            )

            scan = scan_project(str(root))
            report = scan["scip"]["go"]
            file_fact = next(item for item in scan["files"] if item.path == "main.go")

            self.assertEqual("succeeded", report["status"])
            self.assertEqual(1, report["summary"]["succeeded"])
            self.assertGreater(report["modules"][0]["definitions"], 0)
            self.assertEqual(64, len(report["modules"][0]["index_sha256"]))
            self.assertTrue(any(symbol.metadata.get("parser") == "scip" for symbol in file_fact.symbols))
            self.assertTrue(any(
                relation.relation_type == "references"
                and relation.evidence.get("parser") == "scip"
                and "helper" in (relation.target_ref or "")
                and relation.target_symbol_key == "main.go::helper"
                for relation in file_fact.relations
            ))


if __name__ == "__main__":
    unittest.main()
