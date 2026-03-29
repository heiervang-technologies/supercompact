"""Tests for lib/eval/entity_coverage.py — entity extraction and coverage metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.entity_coverage import extract_entities, ENTITY_TYPES


# ---------------------------------------------------------------------------
# Entity type weights
# ---------------------------------------------------------------------------

class TestEntityTypeWeights:
    def test_all_expected_types_present(self):
        expected = {"file_path", "port", "http_status", "exception", "function",
                    "class_name", "url", "package", "command", "env_var"}
        assert expected.issubset(set(ENTITY_TYPES.keys()))

    def test_weights_are_positive_floats(self):
        for key, weight in ENTITY_TYPES.items():
            assert isinstance(weight, (int, float)), f"{key} weight not numeric"
            assert weight > 0, f"{key} weight not positive"

    def test_error_and_path_are_high_weight(self):
        # file_path and error/exception should be among the highest weights
        assert ENTITY_TYPES.get("file_path", 0) >= 0.5
        assert ENTITY_TYPES.get("exception", 0) >= 0.5


# ---------------------------------------------------------------------------
# extract_entities — file paths
# ---------------------------------------------------------------------------

class TestExtractFilePaths:
    def test_absolute_path_detected(self):
        entities = extract_entities("/home/user/project/src/main.py")
        paths = {v for t, v in entities.all_entities() if t == "file_path"}
        assert any("main.py" in p for p in paths)

    def test_relative_path_detected(self):
        entities = extract_entities("See ./lib/parser.py for details")
        paths = {v for t, v in entities.all_entities() if t == "file_path"}
        assert any("parser.py" in p for p in paths)

    def test_url_not_confused_with_path(self):
        entities = extract_entities("Visit https://example.com/path/to/resource")
        paths = {v for t, v in entities.all_entities() if t == "file_path"}
        # URL paths should not show up as file_path entities
        assert not any("example.com" in p for p in paths)


# ---------------------------------------------------------------------------
# extract_entities — exceptions
# ---------------------------------------------------------------------------

class TestExtractExceptions:
    def test_valueerror_detected(self):
        entities = extract_entities("Traceback: ValueError: invalid literal")
        excepts = {v for t, v in entities.all_entities() if t == "exception"}
        # Entity values are normalised to lowercase
        assert "valueerror" in excepts

    def test_module_not_found_detected(self):
        entities = extract_entities("ModuleNotFoundError: No module named 'foo'")
        excepts = {v for t, v in entities.all_entities() if t == "exception"}
        assert "modulenotfounderror" in excepts

    def test_plain_word_not_exception(self):
        entities = extract_entities("This is a normal sentence without errors.")
        excepts = {v for t, v in entities.all_entities() if t == "exception"}
        assert not excepts


# ---------------------------------------------------------------------------
# extract_entities — URLs
# ---------------------------------------------------------------------------

class TestExtractURLs:
    def test_https_url_detected(self):
        entities = extract_entities("See https://github.com/org/repo for details")
        urls = {v for t, v in entities.all_entities() if t == "url"}
        assert any("github.com" in u for u in urls)

    def test_http_url_detected(self):
        entities = extract_entities("Endpoint: http://localhost:8080/api/v1")
        urls = {v for t, v in entities.all_entities() if t == "url"}
        assert any("localhost" in u for u in urls)


# ---------------------------------------------------------------------------
# extract_entities — ports
# ---------------------------------------------------------------------------

class TestExtractPorts:
    def test_colon_port_detected(self):
        entities = extract_entities("Server running on :8080")
        ports = {v for t, v in entities.all_entities() if t == "port"}
        assert "8080" in ports

    def test_port_keyword_detected(self):
        entities = extract_entities("Listening on port 3000")
        ports = {v for t, v in entities.all_entities() if t == "port"}
        assert "3000" in ports


# ---------------------------------------------------------------------------
# extract_entities — packages
# ---------------------------------------------------------------------------

class TestExtractPackages:
    def test_pip_install_detected(self):
        entities = extract_entities("Run: pip install requests")
        pkgs = {v for t, v in entities.all_entities() if t == "package"}
        assert "requests" in pkgs

    def test_npm_install_detected(self):
        entities = extract_entities("npm install express")
        pkgs = {v for t, v in entities.all_entities() if t == "package"}
        assert "express" in pkgs


# ---------------------------------------------------------------------------
# extract_entities — all_entities and coverage
# ---------------------------------------------------------------------------

class TestAllEntities:
    def test_all_entities_returns_type_value_pairs(self):
        entities = extract_entities("ValueError at /src/app.py:42")
        pairs = entities.all_entities()
        assert isinstance(pairs, (set, frozenset, list))
        for item in pairs:
            assert len(item) == 2
            assert isinstance(item[0], str)  # entity type
            assert isinstance(item[1], str)  # entity value

    def test_empty_text_returns_no_entities(self):
        entities = extract_entities("")
        assert len(entities.all_entities()) == 0

    def test_repeated_entity_not_duplicated(self):
        # Same entity mentioned twice should not double-count
        entities = extract_entities("ValueError here. Also ValueError there.")
        # Entity values are normalised to lowercase
        excepts = [v for t, v in entities.all_entities() if t == "exception"]
        assert excepts.count("valueerror") == 1
