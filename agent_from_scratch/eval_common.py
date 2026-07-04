"""Shared configuration and trace utilities for evaluation applications."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import string
import tempfile
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)

from .agent_llm import LlmClient
from .tool_common import Tool
from .tool_compute import compute_tool
from .tool_python import python_tool


class StrictModel(BaseModel):
    """Pydantic base model that rejects unknown keys and coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ModelConfig(StrictModel):
    """LiteLLM model and generation settings shared by evaluations."""

    model: str
    base_url: str | None = None
    max_tokens: PositiveInt = 1024
    temperature: Annotated[float, Field(ge=0.0)] = 0.0
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    top_k: PositiveInt | None = None
    enable_thinking: bool | None = None
    extra_body: dict[str, object] = Field(default_factory=dict)
    timeout: Annotated[float, Field(gt=0.0)] = 3600.0

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


RuntimeToolName = Literal["compute", "python"]


class ToolsConfig(StrictModel):
    """Runtime tools enabled for an evaluation agent."""

    enabled: list[RuntimeToolName] = Field(
        default_factory=lambda: cast(list[RuntimeToolName], [])
    )

    @field_validator("enabled")
    @classmethod
    def unique_tools(cls, value: list[RuntimeToolName]) -> list[RuntimeToolName]:
        if len(value) != len(set(value)):
            raise ValueError("tool names must be unique")
        return value


TOOL_REGISTRY: Mapping[RuntimeToolName, Callable[[], Tool]] = {
    "compute": compute_tool,
    "python": python_tool,
}


def normalize_tools_config(value: object) -> object:
    """Accept either a tool list or the normalized ``enabled`` mapping."""
    if isinstance(value, list):
        return {"enabled": cast(list[object], value)}
    return value


def build_tools(config: ToolsConfig) -> list[Tool]:
    """Build runtime tools in configured order."""
    return [TOOL_REGISTRY[name]() for name in config.enabled]


def build_llm(config: ModelConfig, api_key: str | None) -> LlmClient:
    """Build a LiteLLM client from shared model configuration."""
    return LlmClient(
        config.model,
        base_url=config.base_url,
        api_key=api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        enable_thinking=config.enable_thinking,
        extra_body=config.extra_body,
        timeout=config.timeout,
    )


def validate_prompt_template(
    value: str,
    *,
    allowed_fields: set[str],
    required_fields: set[str],
) -> str:
    """Validate simple named placeholders for an evaluation prompt."""
    template_fields: set[str] = set()
    try:
        for _, field_name, format_spec, conversion in string.Formatter().parse(value):
            if field_name is None:
                continue
            if field_name not in allowed_fields:
                raise ValueError(
                    f"unknown placeholder {{{field_name}}}; "
                    f"supported placeholders: {', '.join(sorted(allowed_fields))}"
                )
            if format_spec or conversion:
                raise ValueError(
                    "format specifications and conversions are unsupported"
                )
            template_fields.add(field_name)
    except ValueError as error:
        raise ValueError(f"invalid prompt template: {error}") from error
    missing = sorted(required_fields - template_fields)
    if missing:
        placeholders = ", ".join(f"{{{field}}}" for field in missing)
        raise ValueError(f"prompt template must contain {placeholders}")
    return value


def json_safe(value: object) -> object:
    """Recursively convert common evaluation values into JSON-safe values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return {str(key): json_safe(item) for key, item in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return [json_safe(item) for item in cast(Sequence[Any], value)]
    return str(value)


def atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    """Atomically replace a formatted JSON document and fsync its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def environment_credential(name: str = "LLLM_API_KEY") -> str | None:
    """Read a non-empty credential from the named environment variable."""
    return os.environ.get(name) or None


def resolved_config(config: BaseModel) -> dict[str, object]:
    """Serialize a validated configuration using resolved field names."""
    return cast(dict[str, object], config.model_dump(mode="json"))


def config_hash(config: Mapping[str, object]) -> str:
    """Hash a resolved configuration using canonical JSON."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def trace_document(
    *,
    resolved: Mapping[str, object],
    run: Mapping[str, object],
    summary: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    credential_name: str = "LLLM_API_KEY",
) -> dict[str, object]:
    """Build the common trace envelope used by evaluation applications."""
    return {
        "schema_version": 1,
        "configuration": {
            "resolved": dict(resolved),
            "sha256": config_hash(resolved),
            "credential": {
                "source": credential_name,
                "present": environment_credential(credential_name) is not None,
            },
        },
        "run": dict(run),
        "summary": dict(summary),
        "entries": list(entries),
    }
