"""通用 YAML、Compose 和 Helm 配置解析器测试。"""

import unittest

from app.config_parsers import parse_compose, parse_helm, parse_yaml


class ConfigParserTests(unittest.TestCase):
    def test_yaml_keeps_nested_key_and_line(self) -> None:
        facts = parse_yaml("config/application.yaml", "server:\n  port: 8080\n  token: abc\n")
        port = next(item for item in facts if item["key_path"] == "server.port")
        token = next(item for item in facts if item["key_path"] == "server.token")
        self.assertEqual(2, port["line"])
        self.assertEqual("8080", port["value"])
        self.assertEqual("[REDACTED]", token["value"])

    def test_compose_emits_services_and_dependencies(self) -> None:
        facts = parse_compose("docker-compose.yml", "services:\n  api:\n    image: example/api:1\n    depends_on:\n      - kafka\n")
        self.assertTrue(any(item["fact_type"] == "service" and item["value"] == "api" for item in facts))
        self.assertTrue(any(item["fact_type"] == "dependency" and item["value"] == "kafka" for item in facts))
        self.assertTrue(any(item["fact_type"] == "compose_image" for item in facts))

    def test_helm_keeps_template_reference_as_partial_fact(self) -> None:
        facts = parse_helm("templates/deployment.yaml", "kind: Deployment\nspec:\n  image: {{ .Values.image.repository }}\n")
        self.assertTrue(any(item["fact_type"] == "helm_template" and item["value"] == "Deployment" for item in facts))
        reference = next(item for item in facts if item["key_path"] == "helm.Values")
        self.assertEqual("partial", reference["parse_status"])
        self.assertEqual("image.repository", reference["value"])


if __name__ == "__main__":
    unittest.main()
