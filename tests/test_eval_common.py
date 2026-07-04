from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from agent_from_scratch.eval_common import (
    ModelConfig,
    atomic_write_json,
    config_hash,
    json_safe,
    normalize_tools_config,
    trace_document,
    validate_prompt_template,
)


def test_model_config_is_strict_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        ModelConfig.model_validate({"model": "test", "unknown": True})
    with pytest.raises(ValueError, match="int_type"):
        ModelConfig.model_validate({"model": "test", "max_tokens": "10"})


def test_validate_prompt_template_checks_allowed_and_required_fields() -> None:
    assert (
        validate_prompt_template(
            "{question} {attachment}",
            allowed_fields={"question", "attachment"},
            required_fields={"question"},
        )
        == "{question} {attachment}"
    )
    with pytest.raises(ValueError, match="unknown placeholder"):
        validate_prompt_template(
            "{unknown}",
            allowed_fields={"question"},
            required_fields={"question"},
        )
    with pytest.raises(ValueError, match=r"must contain \{question\}"):
        validate_prompt_template(
            "missing",
            allowed_fields={"question"},
            required_fields={"question"},
        )


def test_atomic_write_json_creates_parents_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.json"

    atomic_write_json(path, {"old": True})
    atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert list(path.parent.iterdir()) == [path]


def test_trace_document_hashes_config_and_redacts_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLLM_API_KEY", "secret-value")
    resolved = {"model": {"model": "test"}}

    document = trace_document(
        resolved=resolved,
        run={"started_at": "now"},
        summary={"total_tasks": 0},
        entries=[],
    )

    assert document["configuration"]["sha256"] == config_hash(resolved)
    assert document["configuration"]["credential"] == {
        "source": "LLLM_API_KEY",
        "present": True,
    }
    assert "secret-value" not in json.dumps(document)


def test_json_safe_and_tool_normalization() -> None:
    @dataclass
    class Value:
        path: Path

    assert json_safe(Value(Path("/tmp/example"))) == {"path": "/tmp/example"}
    assert normalize_tools_config(["compute"]) == {"enabled": ["compute"]}
    mapping = {"enabled": ["python"]}
    assert normalize_tools_config(mapping) is mapping
