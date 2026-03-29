"""Tests for lib/eval/entity_coverage.py — EntitySet, extract_entities."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.entity_coverage import EntitySet, extract_entities


# ---------------------------------------------------------------------------
# EntitySet
# ---------------------------------------------------------------------------

class TestEntitySet:
    def test_empty_total_count_zero(self):
        es = EntitySet()
        assert es.total_count == 0

    def test_total_count_sums_all_types(self):
        es = EntitySet(entities={
            "file_path": {"/foo/bar", "/baz/qux"},
            "url": {"https://example.com"},
        })
        assert es.total_count == 3

    def test_all_entities_returns_type_value_pairs(self):
        es = EntitySet(entities={"file_path": {"/foo/bar"}})
        pairs = es.all_entities()
        assert ("file_path", "/foo/bar") in pairs

    def test_all_entities_empty_is_empty_set(self):
        es = EntitySet()
        assert es.all_entities() == set()

    def test_all_entities_multiple_types(self):
        es = EntitySet(entities={
            "url": {"https://a.com"},
            "port": {"8080"},
        })
        pairs = es.all_entities()
        assert ("url", "https://a.com") in pairs
        assert ("port", "8080") in pairs


# ---------------------------------------------------------------------------
# extract_entities — URLs
# ---------------------------------------------------------------------------

class TestExtractEntitiesUrls:
    def test_http_url_extracted(self):
        es = extract_entities("See https://example.com for details")
        assert any(v == "https://example.com" for v in es.entities.get("url", set()))

    def test_https_url_extracted(self):
        es = extract_entities("Connect to https://api.github.com/repos")
        assert "url" in es.entities

    def test_url_path_not_duplicated_as_file_path(self):
        es = extract_entities("Visit https://github.com/foo/bar")
        # /foo/bar from the URL should NOT be counted as a separate file_path
        file_paths = es.entities.get("file_path", set())
        assert not any("github.com" in p for p in file_paths)


# ---------------------------------------------------------------------------
# extract_entities — file paths
# ---------------------------------------------------------------------------

class TestExtractEntitiesFilePaths:
    def test_absolute_path_extracted(self):
        es = extract_entities("Edit /home/user/project/main.py")
        assert "file_path" in es.entities
        paths = es.entities["file_path"]
        assert any("main.py" in p for p in paths)

    def test_relative_path_extracted(self):
        es = extract_entities("See ./src/utils.py for reference")
        assert "file_path" in es.entities

    def test_non_path_string_not_extracted(self):
        es = extract_entities("hello world no paths here")
        assert "file_path" not in es.entities


# ---------------------------------------------------------------------------
# extract_entities — exceptions
# ---------------------------------------------------------------------------

class TestExtractEntitiesExceptions:
    def test_value_error_extracted(self):
        es = extract_entities("Raised ValueError: invalid literal")
        assert "exception" in es.entities
        assert "valueerror" in es.entities["exception"]

    def test_module_not_found_error(self):
        es = extract_entities("ModuleNotFoundError: No module named 'foo'")
        assert "exception" in es.entities

    def test_type_error_extracted(self):
        es = extract_entities("TypeError: unsupported operand")
        assert "exception" in es.entities


# ---------------------------------------------------------------------------
# extract_entities — ports
# ---------------------------------------------------------------------------

class TestExtractEntitiesPorts:
    def test_port_8080_extracted(self):
        es = extract_entities("Server running on port 8080")
        assert "port" in es.entities
        assert "8080" in es.entities["port"]

    def test_port_80_extracted(self):
        es = extract_entities("HTTP on port 80")
        assert "port" in es.entities

    def test_low_port_not_extracted(self):
        """Port numbers below 80 are too common as false positives."""
        es = extract_entities("step 10, index 42")
        # 42 should not be extracted as a port
        ports = es.entities.get("port", set())
        assert "42" not in ports


# ---------------------------------------------------------------------------
# extract_entities — functions
# ---------------------------------------------------------------------------

class TestExtractEntitiesFunctions:
    def test_function_call_extracted(self):
        es = extract_entities("Call process_data() to transform")
        assert "function" in es.entities

    def test_short_common_functions_skipped(self):
        es = extract_entities("print('hello'), len(items), str(x)")
        funcs = es.entities.get("function", set())
        assert "print" not in funcs
        assert "len" not in funcs

    def test_dotted_method_extracted(self):
        es = extract_entities("Use parser.extract_text() for content")
        assert "function" in es.entities


# ---------------------------------------------------------------------------
# extract_entities — class names
# ---------------------------------------------------------------------------

class TestExtractEntitiesClassNames:
    def test_camel_case_class_extracted(self):
        es = extract_entities("Use SuffixAutomaton for string matching")
        assert "class_name" in es.entities
        classes = es.entities["class_name"]
        assert any("automaton" in c for c in classes)

    def test_single_word_not_extracted(self):
        es = extract_entities("Python is great")
        classes = es.entities.get("class_name", set())
        assert "python" not in classes


# ---------------------------------------------------------------------------
# extract_entities — packages
# ---------------------------------------------------------------------------

class TestExtractEntitiesPackages:
    def test_pip_install_package(self):
        es = extract_entities("Run pip install requests to install")
        assert "package" in es.entities
        assert "requests" in es.entities["package"]

    def test_npm_install_package(self):
        es = extract_entities("npm install express")
        assert "package" in es.entities


# ---------------------------------------------------------------------------
# extract_entities — general
# ---------------------------------------------------------------------------

class TestExtractEntitiesGeneral:
    def test_empty_text_returns_empty(self):
        es = extract_entities("")
        assert es.total_count == 0

    def test_returns_entity_set_instance(self):
        assert isinstance(extract_entities("hello"), EntitySet)

    def test_multiple_entity_types_in_one_text(self):
        text = "TypeError in /home/user/app.py at port 8080"
        es = extract_entities(text)
        assert es.total_count > 0
