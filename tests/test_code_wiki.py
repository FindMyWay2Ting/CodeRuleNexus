"""Code Wiki 事实扫描器的最小回归测试。"""

import asyncio
import hashlib
from io import BytesIO
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

from fastapi import UploadFile

from app.code_wiki import (
    _architecture_source_symbol,
    _scip_indexer_status,
    github_repository_key,
    managed_local_repository_path,
    import_and_scan_github_repository,
    normalize_github_url,
    normalize_uploaded_path,
    read_code_inventory_source,
    scan_project,
)
from app.code_agent import CitationRegistry, execute_code_agent_tool
from app.access_control import AccessScope
from app.main import import_local_code_project


class CodeWikiScannerTests(unittest.TestCase):
    def test_github_repository_identity_is_case_insensitive(self) -> None:
        self.assertEqual(
            github_repository_key("OpenAI", "OpenAI-Python"),
            github_repository_key("openai", "openai-python"),
        )
    def test_non_git_version_changes_when_only_yaml_changes(self) -> None:
        """配置文件也是 Agent 证据，修改 YAML 必须产生新的内容版本。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "compose.yaml"
            config.write_text("services:\n  api:\n    image: demo:v1\n", encoding="utf-8")
            first = scan_project(str(root))["commit_hash"]
            config.write_text("services:\n  api:\n    image: demo:v2\n", encoding="utf-8")
            second = scan_project(str(root))["commit_hash"]
            self.assertNotEqual(first, second)

    def test_non_git_version_changes_when_config_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "compose.yaml"
            config.write_text("services: {}\n", encoding="utf-8")
            first = scan_project(str(root))["commit_hash"]
            config.unlink()
            second = scan_project(str(root))["commit_hash"]
            self.assertNotEqual(first, second)

    def test_non_git_version_includes_extensionless_dockerfile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dockerfile = root / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\n", encoding="utf-8")
            first = scan_project(str(root))["commit_hash"]
            dockerfile.write_text("FROM python:3.12\n", encoding="utf-8")
            second = scan_project(str(root))["commit_hash"]
            self.assertNotEqual(first, second)

    def test_inventory_source_rejects_changed_snapshot_content(self) -> None:
        """配置原文与扫描时哈希不一致时，不能继续作为 Agent 证据。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "compose.yaml"
            original = "services:\n  api:\n    image: demo:v1\n"
            config.write_text(original, encoding="utf-8")
            fake_connection = MagicMock()
            fake_connection.execute.return_value.fetchone.return_value = (
                str(root), "commit-a", {
                    "file_inventory": [{
                        "path": "compose.yaml",
                        "content_hash": hashlib.sha256(original.encode()).hexdigest(),
                    }],
                },
            )
            fake_context = MagicMock()
            fake_context.__enter__.return_value = fake_connection
            config.write_text("services:\n  api:\n    image: demo:v2\n", encoding="utf-8")
            with patch("app.code_wiki.connection", return_value=fake_context):
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    read_code_inventory_source(
                        "project-1", "compose.yaml", commit_hash="commit-a",
                    )

    def test_github_import_promotes_immutable_commit_snapshot(self) -> None:
        """远程仓库先在 staging 扫描，成功后才以 Commit 名晋升。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_connection = MagicMock()
            fake_connection.execute.return_value.fetchone.return_value = None
            fake_context = MagicMock()
            fake_context.__enter__.return_value = fake_connection

            def fake_clone(command, timeout=120):
                repository = Path(command[-1])
                (repository / ".git").mkdir(parents=True)
                (repository / "main.py").write_text("print('ok')\n", encoding="utf-8")
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            scan = {
                "project_id": "temporary", "project_name": "temporary",
                "root_path": "temporary", "commit_hash": "abc123",
                "commit_source": "git", "files": [], "components": [],
                "architecture_facts": [], "architecture_links": [],
            }
            with (
                patch("app.code_wiki.DEFAULT_REPOSITORY_ROOT", root),
                patch("app.code_wiki.repository_import_lock") as import_lock,
                patch("app.code_wiki.connection", return_value=fake_context),
                patch("app.code_wiki._run_git", side_effect=fake_clone),
                patch("app.code_wiki.scan_project", return_value=scan),
                patch("app.code_wiki.persist_scan", return_value={"project_id": "stable", "commit_hash": "abc123"}) as persist,
            ):
                import_lock.return_value.__enter__.return_value = None
                result = import_and_scan_github_repository(
                    "https://github.com/example/demo", "workspace-001", "user-001",
                )

            persisted_scan = persist.call_args.args[0]
            snapshot_path = Path(persisted_scan["root_path"])
            self.assertEqual("abc123", snapshot_path.name)
            self.assertEqual("example__demo", snapshot_path.parent.name)
            self.assertTrue((snapshot_path / "main.py").is_file())
            self.assertEqual("cloned", result["import_action"])
            self.assertFalse(any(path.name.startswith(".example__demo-") for path in root.iterdir()))

    def test_github_import_failure_keeps_previous_repository(self) -> None:
        """新版本落库失败只能回收新快照，不能误删旧仓库目录。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_repository = root / "example__demo"
            old_repository.mkdir()
            (old_repository / "keep.txt").write_text("old", encoding="utf-8")
            fake_connection = MagicMock()
            fake_connection.execute.return_value.fetchone.return_value = (1,)
            fake_context = MagicMock()
            fake_context.__enter__.return_value = fake_connection

            def fake_clone(command, timeout=120):
                repository = Path(command[-1])
                (repository / ".git").mkdir(parents=True)
                (repository / "main.py").write_text("new", encoding="utf-8")
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            scan = {
                "project_id": "temporary", "project_name": "temporary",
                "root_path": "temporary", "commit_hash": "new123",
                "commit_source": "git", "files": [], "components": [],
                "architecture_facts": [], "architecture_links": [],
            }
            with (
                patch("app.code_wiki.DEFAULT_REPOSITORY_ROOT", root),
                patch("app.code_wiki.repository_import_lock") as import_lock,
                patch("app.code_wiki.connection", return_value=fake_context),
                patch("app.code_wiki._run_git", side_effect=fake_clone),
                patch("app.code_wiki.scan_project", return_value=scan),
                patch("app.code_wiki.persist_scan", side_effect=RuntimeError("database failed")),
            ):
                import_lock.return_value.__enter__.return_value = None
                with self.assertRaises(RuntimeError):
                    import_and_scan_github_repository(
                        "https://github.com/example/demo", "workspace-001", "user-001",
                    )

            self.assertTrue((old_repository / "keep.txt").is_file())
            self.assertFalse((root / ".snapshots" / "example__demo" / "new123").exists())

    def test_code_agent_tools_keep_project_scope_and_source_budget(self) -> None:
        citations = CitationRegistry()
        with patch("app.code_agent.read_code_source") as read_source:
            read_source.return_value = {
                "path": "app/main.py", "start_line": 10, "end_line": 129,
                "stale": False, "numbered_content": "10: def run(): pass",
            }
            result = execute_code_agent_tool(
                "project-1", "read_source",
                {"path": "app/main.py", "start_line": 10, "end_line": 9999},
                citations,
                expected_commit="commit-a",
            )

        read_source.assert_called_once_with("project-1", "app/main.py", 10, 129, "commit-a")
        self.assertEqual("[C1]", result["citation"])
        self.assertEqual(1, len(citations.items))

    def test_code_agent_requires_non_empty_commit(self) -> None:
        """Agent 工具不能省略 Commit，也不能用空值回退到项目最新版本。"""
        citations = CitationRegistry()
        with self.assertRaises(TypeError):
            execute_code_agent_tool("project-1", "list_project_files", {}, citations)
        for empty_commit in ("", "   "):
            with self.assertRaisesRegex(ValueError, "expected_commit is required"):
                execute_code_agent_tool(
                    "project-1", "list_project_files", {}, citations,
                    expected_commit=empty_commit,
                )

    def test_code_agent_rejects_symbol_from_another_project(self) -> None:
        citations = CitationRegistry()
        with patch("app.code_agent.get_code_symbol", return_value={"project_id": "project-2"}):
            result = execute_code_agent_tool(
                "project-1", "get_symbol", {"symbol_id": "symbol-1"}, citations,
                expected_commit="commit-a",
            )
        self.assertIn("error", result)
        self.assertEqual([], citations.items)

    def test_code_agent_rejects_symbol_from_another_commit(self) -> None:
        """一次 Agent 请求不能在扫描更新后悄悄读取另一个 Commit 的符号。"""
        citations = CitationRegistry()
        symbol = {
            "project_id": "project-1", "commit_hash": "commit-b",
            "path": "app/main.py", "start_line": 1, "qualified_name": "main.run",
            "symbol_id": "symbol-1", "name": "run", "symbol_kind": "function",
        }
        with patch("app.code_agent.get_code_symbol", return_value=symbol):
            result = execute_code_agent_tool(
                "project-1", "get_symbol", {"symbol_id": "symbol-1"}, citations,
                expected_commit="commit-a",
            )
        self.assertIn("error", result)
        self.assertEqual([], citations.items)

    def test_modules_entrypoints_apis_and_jobs_keep_symbol_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "app" / "__init__.py").write_text("", encoding="utf-8")
            (root / "app" / "main.py").write_text(
                "from fastapi import FastAPI\n\n"
                "app = FastAPI()\n\n"
                "@app.get('/orders/{order_id}')\n"
                "async def get_order(order_id: str):\n"
                "    return {'id': order_id}\n\n"
                "@app.on_event('startup')\n"
                "async def initialize():\n"
                "    return None\n\n"
                "@worker.task\n"
                "def rebuild_index():\n"
                "    return None\n\n"
                "if __name__ == '__main__':\n"
                "    print('start')\n",
                encoding="utf-8",
            )
            (root / "api.proto").write_text(
                "service Payment {\n  rpc Charge (ChargeRequest) returns (ChargeReply);\n}\n",
                encoding="utf-8",
            )

            scan = scan_project(str(root))
            facts = scan["architecture_facts"]
            keys = {(item.fact_type, item.name, item.value) for item in facts}
            route = next(item for item in facts if item.fact_type == "http_api")
            route_symbol = _architecture_source_symbol(route, scan["files"])

            self.assertIn(("module", "app", "python_package"), keys)
            self.assertIn(("entrypoint", "app", "FastAPI application"), keys)
            self.assertIn(("http_api", "get_order", "GET /orders/{order_id}"), keys)
            self.assertIn(("lifecycle_hook", "initialize", "startup"), keys)
            self.assertIn(("background_job", "rebuild_index", "worker.task"), keys)
            self.assertIn(("rpc_service", "Payment", "grpc"), keys)
            self.assertIn(("rpc_method", "Charge", "ChargeRequest -> ChargeReply"), keys)
            self.assertEqual("get_order", route_symbol.name if route_symbol else None)

    def test_architecture_summary_ignores_test_fixtures_and_generated_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_kafka.py").write_text("from kafka import KafkaProducer\n", encoding="utf-8")
            (root / "service_pb2_grpc.py").write_text("import grpc\nclass PaymentStub: pass\n", encoding="utf-8")
            (root / "package-lock.json").write_text('{"funding":{"url":"https://example.com"}}', encoding="utf-8")
            (root / "client.go").write_text("package client\nfunc NewPaymentClient() {}\n", encoding="utf-8")

            scan = scan_project(str(root))

            # Go package 是合法模块边界，但测试夹具、生成绑定、锁文件和普通构造器不能产生基础设施事实。
            self.assertEqual(
                [("module", "client", "go_package:client")],
                [(item.fact_type, item.name, item.value) for item in scan["architecture_facts"]],
            )

    def test_architecture_facts_capture_resources_and_redact_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(
                "KAFKA_TOPIC: order-created\n"
                "KAFKA_GROUP: order-worker\n"
                "KAFKA_PASSWORD: kafka-secret\n"
                "MONGODB_DATABASE: commerce\n"
                "MONGODB_COLLECTION: orders\n"
                "MONGODB_URI: mongodb://reader:mongo-secret@mongo.internal:27017/commerce\n"
                "PAYMENT_SERVICE_URL: https://payment.internal/api\n"
                "GRPC_PAYMENT_TARGET: payment.internal:50051\n"
                "kafka:\n"
                "  topics:\n"
                "    audit: audit-created\n"
                "mongodb:\n"
                "  collections:\n"
                "    archive: audit_events\n"
                "services:\n"
                "  settlement:\n"
                "    base_url: https://settlement.internal/api\n",
                encoding="utf-8",
            )
            (root / "service.py").write_text(
                "from kafka import KafkaProducer\n"
                "from pymongo import MongoClient\n"
                "from payment_pb2_grpc import PaymentStub\n\n"
                "def publish(producer, db, channel):\n"
                "    producer.send('order-created')\n"
                "    collection = db['orders']\n"
                "    stub = PaymentStub(channel)\n"
                "    reply = stub.Charge({})\n"
                "    return collection, reply\n",
                encoding="utf-8",
            )
            (root / "checkout.go").write_text(
                "package checkout\n\n"
                "import \"google.golang.org/grpc\"\n\n"
                "func setup() { svc.paymentClient = pb.NewPaymentServiceClient(conn) }\n\n"
                "func charge() { svc.paymentClient.Charge(ctx) }\n",
                encoding="utf-8",
            )

            scan = scan_project(str(root))
            facts = scan["architecture_facts"]
            links = scan["architecture_links"]
            fact_keys = {(item.fact_type, item.name) for item in facts}
            link_types = {item.relation_type for item in links}
            serialized = "\n".join(f"{item.name} {item.value} {item.evidence}" for item in facts)

            self.assertIn(("component", "Kafka"), fact_keys)
            self.assertIn(("component", "MongoDB"), fact_keys)
            self.assertIn(("kafka_topic_config", "order-created"), fact_keys)
            self.assertIn(("kafka_topic_config", "audit-created"), fact_keys)
            self.assertIn(("kafka_topic_producer", "order-created"), fact_keys)
            self.assertIn(("kafka_consumer_group", "order-worker"), fact_keys)
            self.assertIn(("mongo_database", "commerce"), fact_keys)
            self.assertIn(("mongo_collection", "orders"), fact_keys)
            self.assertIn(("mongo_collection", "audit_events"), fact_keys)
            self.assertIn(("downstream_http", "payment-service"), fact_keys)
            self.assertIn(("downstream_http", "settlement"), fact_keys)
            self.assertIn(("downstream_grpc", "Payment"), fact_keys)
            self.assertIn(("client_initialization", "Payment"), fact_keys)
            self.assertIn(("downstream_call", "Payment.Charge"), fact_keys)
            self.assertTrue(any(
                item.fact_type == "downstream_call" and item.name == "PaymentService.Charge"
                and item.path == "checkout.go"
                for item in facts
            ))
            self.assertIn("configures_client", link_types)
            self.assertIn("client_invokes", link_types)
            self.assertIn("configures_usage", link_types)
            self.assertNotIn("kafka-secret", serialized)
            self.assertNotIn("mongo-secret", serialized)
            self.assertIn("[REDACTED]", serialized)

    def test_browser_folder_upload_preserves_tree_and_updates_snapshot(self) -> None:
        def fake_persist(scan: dict) -> dict:
            return {
                "project_id": scan["project_id"],
                "project_name": scan["project_name"],
                "commit_hash": scan["commit_hash"],
                "root_path": scan["root_path"],
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
                patch("app.main.request_scope", return_value=AccessScope(
                    "user-1", "workspace-1", "测试空间", None, frozenset(), "owner",
                )),
                patch("app.main.current_principal", return_value=SimpleNamespace(display_name="测试用户")),
            ):
                first = asyncio.run(import_local_code_project([upload(b"def run():\n    return 1\n")]))
                second = asyncio.run(import_local_code_project([upload(b"def run():\n    return 2\n")]))

            first_snapshot = Path(first["root_path"])
            second_snapshot = Path(second["root_path"])
            self.assertEqual("uploaded", first["import_action"])
            self.assertEqual("updated", second["import_action"])
            self.assertNotEqual(first["commit_hash"], second["commit_hash"])
            self.assertEqual("def run():\n    return 1\n", (first_snapshot / "src" / "main.py").read_text(encoding="utf-8"))
            self.assertEqual("def run():\n    return 2\n", (second_snapshot / "src" / "main.py").read_text(encoding="utf-8"))

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

    def test_file_inventory_preserves_config_assets_outside_code_symbols(self) -> None:
        """文件资产清单必须发现 YAML 和无扩展名部署文件，避免 Agent 只看到源码。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "otel-collector-config.yaml").write_text("receivers:\n  otlp:\n    protocols:\n      grpc:\n", encoding="utf-8")
            (root / "docker-compose.yml").write_text("services:\n  collector:\n    image: otel/opentelemetry-collector\n", encoding="utf-8")
            (root / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")

            inventory = {item["path"]: item for item in scan_project(str(root))["file_inventory"]}

            self.assertEqual(".yaml", inventory["otel-collector-config.yaml"]["extension"])
            self.assertEqual("possible_observability_config", inventory["otel-collector-config.yaml"]["detected_format"])
            self.assertEqual("docker_compose", inventory["docker-compose.yml"]["detected_format"])
            self.assertEqual("parsed", inventory["main.py"]["parser_status"])


if __name__ == "__main__":
    unittest.main()
