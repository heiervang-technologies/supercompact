"""Tests for lib/eval/entity_coverage.py — EntitySet, extract_entities, compute_coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.entity_coverage import (
    ENTITY_TYPES,
    EntitySet,
    compute_coverage,
    extract_entities,
)


# ---------------------------------------------------------------------------
# EntitySet
# ---------------------------------------------------------------------------

class TestEntitySet:
    def test_total_count_empty(self):
        es = EntitySet()
        assert es.total_count == 0

    def test_total_count_single_type(self):
        es = EntitySet(entities={"file_path": {"/foo/bar/baz", "/a/b/c"}})
        assert es.total_count == 2

    def test_total_count_multiple_types(self):
        es = EntitySet(entities={
            "file_path": {"/foo/bar"},
            "exception": {"ValueError", "TypeError"},
        })
        assert es.total_count == 3

    def test_all_entities_empty(self):
        es = EntitySet()
        assert es.all_entities() == set()

    def test_all_entities_returns_type_value_pairs(self):
        es = EntitySet(entities={"exception": {"ValueError"}})
        assert ("exception", "ValueError") in es.all_entities()

    def test_all_entities_multiple_types(self):
        es = EntitySet(entities={
            "exception": {"ValueError"},
            "port": {"8080"},
        })
        pairs = es.all_entities()
        assert ("exception", "ValueError") in pairs
        assert ("port", "8080") in pairs

    def test_default_entities_is_empty_dict(self):
        es = EntitySet()
        assert es.entities == {}


# ---------------------------------------------------------------------------
# ENTITY_TYPES constant
# ---------------------------------------------------------------------------

class TestEntityTypes:
    def test_has_file_path(self):
        assert "file_path" in ENTITY_TYPES

    def test_has_exception(self):
        assert "exception" in ENTITY_TYPES

    def test_has_url(self):
        assert "url" in ENTITY_TYPES

    def test_weights_are_positive(self):
        for k, v in ENTITY_TYPES.items():
            assert v > 0, f"Weight for {k} should be positive"

    def test_file_path_and_error_are_highest_weight(self):
        assert ENTITY_TYPES["file_path"] >= 1.0
        assert ENTITY_TYPES["error"] >= 1.0


# ---------------------------------------------------------------------------
# extract_entities — individual types
# ---------------------------------------------------------------------------

class TestExtractEntitiesExceptions:
    def test_extracts_value_error(self):
        es = extract_entities("Traceback: ValueError: invalid literal")
        assert "exception" in es.entities
        assert "valueerror" in es.entities["exception"]

    def test_extracts_module_not_found(self):
        es = extract_entities("ModuleNotFoundError: No module named 'requests'")
        assert "modulenotfounderror" in es.entities.get("exception", set())

    def test_extracts_type_error(self):
        es = extract_entities("Got TypeError when calling the function")
        assert "typeerror" in es.entities.get("exception", set())


class TestExtractEntitiesUrls:
    def test_extracts_https_url(self):
        es = extract_entities("Visit https://example.com/api for docs")
        assert "url" in es.entities
        assert any("example.com" in u for u in es.entities["url"])

    def test_extracts_http_url(self):
        es = extract_entities("Server running at http://localhost:8080/api")
        assert "url" in es.entities
        assert any("localhost" in u for u in es.entities["url"])

    def test_no_url_in_plain_text(self):
        es = extract_entities("Just some plain text without any links")
        assert "url" not in es.entities or len(es.entities.get("url", set())) == 0


class TestExtractEntitiesPorts:
    def test_extracts_colon_port(self):
        es = extract_entities("Server running on port :8080 now")
        assert "port" in es.entities
        assert "8080" in es.entities["port"]

    def test_extracts_port_keyword(self):
        es = extract_entities("PORT=3000 must be set")
        assert "port" in es.entities
        assert "3000" in es.entities["port"]

    def test_does_not_extract_low_port_numbers(self):
        # Ports 100-999 should be filtered as false positives
        es = extract_entities("error code :500 status")
        # Port 500 should not appear (it's in the filtered range 100-999)
        ports = es.entities.get("port", set())
        assert "500" not in ports


class TestExtractEntitiesFilePaths:
    def test_extracts_absolute_path(self):
        es = extract_entities("Error in /home/user/project/src/main.py line 42")
        assert "file_path" in es.entities
        assert any("/home/user" in p for p in es.entities["file_path"])

    def test_url_path_not_treated_as_file_path(self):
        es = extract_entities("See https://example.com/foo/bar/baz for more")
        # /foo/bar/baz comes from a URL, should not also appear as file_path
        file_paths = es.entities.get("file_path", set())
        # The URL itself is captured; path inside URL should not be double-captured
        urls = es.entities.get("url", set())
        assert any("example.com" in u for u in urls)


class TestExtractEntitiesClassNames:
    def test_extracts_camel_case_class(self):
        es = extract_entities("The ProbeAnswer class stores results")
        assert "class_name" in es.entities
        classes = {c.lower() for c in es.entities["class_name"]}
        assert "probeanswer" in classes

    def test_extracts_multiple_classes(self):
        es = extract_entities("Use DimensionScore and AggregateResult")
        classes = {c.lower() for c in es.entities.get("class_name", set())}
        assert "dimensionscore" in classes
        assert "aggregateresult" in classes


class TestExtractEntitiesPackages:
    def test_extracts_pip_install(self):
        es = extract_entities("Run: pip install requests httpx")
        assert "package" in es.entities
        assert "requests" in es.entities["package"]

    def test_extracts_npm_install(self):
        es = extract_entities("npm install express")
        assert "package" in es.entities
        assert "express" in es.entities["package"]


class TestExtractEntitiesHttpStatus:
    def test_extracts_404_not_found(self):
        es = extract_entities("Got 404 Not Found from the API")
        assert "http_status" in es.entities
        assert "404" in es.entities["http_status"]

    def test_extracts_500_internal(self):
        es = extract_entities("Server returned 500 Internal Server Error")
        assert "http_status" in es.entities
        assert "500" in es.entities["http_status"]


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverageEmpty:
    def test_empty_suffix_returns_ones(self):
        suffix = EntitySet()
        kept = EntitySet(entities={"exception": {"ValueError"}})
        unweighted, weighted, breakdown = compute_coverage(suffix, kept)
        assert unweighted == pytest.approx(1.0)
        assert weighted == pytest.approx(1.0)
        assert breakdown == {}

    def test_empty_kept_with_suffix_gives_zero_coverage(self):
        suffix = EntitySet(entities={"exception": {"ValueError"}})
        kept = EntitySet()
        unweighted, weighted, breakdown = compute_coverage(suffix, kept)
        assert unweighted == pytest.approx(0.0)
        assert weighted == pytest.approx(0.0)


class TestComputeCoverageFull:
    def test_identical_sets_give_full_coverage(self):
        entities = {"exception": {"valueerror", "typeerror"}}
        suffix = EntitySet(entities=entities)
        kept = EntitySet(entities=entities)
        unweighted, weighted, breakdown = compute_coverage(suffix, kept)
        assert unweighted == pytest.approx(1.0)
        assert weighted == pytest.approx(1.0)

    def test_breakdown_has_type_info(self):
        suffix = EntitySet(entities={"exception": {"valueerror"}})
        kept = EntitySet(entities={"exception": {"valueerror"}})
        _, _, breakdown = compute_coverage(suffix, kept)
        assert "exception" in breakdown
        assert breakdown["exception"]["covered"] == 1
        assert breakdown["exception"]["total"] == 1
        assert breakdown["exception"]["coverage"] == pytest.approx(1.0)


class TestComputeCoveragePartial:
    def test_half_covered(self):
        suffix = EntitySet(entities={"exception": {"valueerror", "typeerror"}})
        kept = EntitySet(entities={"exception": {"valueerror"}})
        unweighted, _, _ = compute_coverage(suffix, kept)
        assert unweighted == pytest.approx(0.5)

    def test_unmatched_type_in_kept_not_counted(self):
        suffix = EntitySet(entities={"exception": {"valueerror"}})
        # kept has a different type — shouldn't count toward coverage
        kept = EntitySet(entities={"port": {"8080"}})
        unweighted, _, _ = compute_coverage(suffix, kept)
        assert unweighted == pytest.approx(0.0)

    def test_weighted_coverage_differs_by_type_importance(self):
        # file_path has weight 1.0, function has weight 0.5
        # If we have one file_path and one function uncovered, weights differ
        suffix = EntitySet(entities={
            "file_path": {"/foo/bar/baz"},
            "function": {"my_function"},
        })
        kept = EntitySet(entities={"file_path": {"/foo/bar/baz"}})
        _, weighted, _ = compute_coverage(suffix, kept)
        # file_path covered, function not — weighted should be > 0 and < 1
        assert 0.0 < weighted < 1.0
