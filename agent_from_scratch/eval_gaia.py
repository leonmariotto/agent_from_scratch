"""
GAIA benchmark evaluation helpers for agent-style systems.

GAIA evaluates general AI assistants on real-world questions that may require
reasoning, browsing, file handling, and tool use.
Gaia is used in this project to evaluate agent harness improvment.
The caller must provide an agent_evaluate function that take a GaiaTask in parameter.
Then caller can dispose of the GaiaTask parameter.

Gaia level-1 with Qwen3-06B got ~ 1/10 without any tool.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import random
import re
import string
import tempfile
import time
from typing import Annotated, Any, Literal, cast

import click
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    field_validator,
)

from .agent import Agent, AgentMode
from .agent_llm import LlmClient
from .agent_context import AgentResult, ExecutionContext
from .container_env import ContainerEnv, DEFAULT_CONTAINER_ENV_IMAGE
from .tool_common import Tool
from .tool_compute import compute_tool
from .tool_python import python_tool
from .yaml_parser import YamlParser, YamlParserError

load_dataset = cast(
    Callable[..., Any], importlib.import_module("datasets").load_dataset
)
snapshot_download = cast(
    Callable[..., str], importlib.import_module("huggingface_hub").snapshot_download
)

GAIA_DATASET_ID = "gaia-benchmark/GAIA"
GaiaSplit = Literal["validation", "test"]
GaiaLevel = Literal[1, 2, 3]
GaiaToolName = Literal[
    "access to academic journal websites",
    "access to excel files",
    "audio capability",
    "audio processing software",
    "calculator",
    "calculator (or ability to count)",
    "color recognition",
    "file interface",
    "image recognition",
    "image recognition/ocr",
    "markdown",
    "pdf viewer",
    "powerpoint viewer",
    "python",
    "rubik's cube model",
    "search engine",
    "speech-to-text audio processing tool",
    "speech-to-text tool",
    "spreadsheet",
    "text editor",
    "video parsing",
    "video processing software",
    "video recognition tools",
    "web browser",
    "wikipedia",
    "word document access",
    "word reversal tool / script",
]
DatasetRow = Mapping[str, Any]


@dataclass(frozen=True)
class GaiaTask:
    """One GAIA task passed to an agent_evaluate function implementation."""

    task_id: str
    question: str
    level: int
    file_path: Path | None
    file_name: str | None
    metadata: Mapping[str, Any]
    expected_answer: str | None


@dataclass(frozen=True)
class GaiaResult:
    """Result for one attempted GAIA task."""

    task_id: str
    question: str
    level: int
    file_path: Path | None
    file_name: str | None
    prediction: str
    expected_answer: str | None
    correct: bool | None
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class GaiaEvaluationResult:
    """Aggregate GAIA evaluation metrics and per-task results."""

    total_tasks: int
    scored_tasks: int
    correct_tasks: int
    overall_accuracy: float | None
    per_level_accuracy: dict[int, float]
    results: tuple[GaiaResult, ...]


GaiaAgentOutput = str | AgentResult
GaiaAgent = Callable[[GaiaTask], GaiaAgentOutput]


def load_gaia_tasks(
    *,
    split: GaiaSplit = "validation",
    level: GaiaLevel | None = None,
    limit: int | None = None,
    offset: int = 0,
    data_dir: str | Path | None = None,
    allowed_tools: Sequence[GaiaToolName] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> list[GaiaTask]:
    """
    Load GAIA tasks from Hugging Face or an existing dataset snapshot.

    Args:
        split: ``"validation"`` for locally scored development data or
            ``"test"`` for leaderboard-style prediction export.
        level: Optional GAIA level filter.  ``None`` loads all levels.
        limit: Optional maximum number of rows to return.
        offset: Number of eligible rows to skip after filtering and shuffling.
        data_dir: Optional local GAIA snapshot path.  When omitted, the dataset
            is resolved with ``huggingface_hub.snapshot_download``.
        allowed_tools: Optional normalized tool names.  When provided, only
            rows whose required tools are a subset of this list are loaded.
            Rows with no required tools are always included.
        shuffle: Whether to shuffle eligible rows before applying ``limit``.
        shuffle_seed: Seed used for row shuffling when ``shuffle`` is true.

    Returns:
        A list of normalized :class:`GaiaTask` objects.  Attachment paths are
        absolute paths when the row contains a non-empty ``file_path``.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    config_name = _gaia_config_name(level)
    root = _resolve_gaia_data_dir(data_dir)
    logger.info(
        "Loading GAIA dataset split={} config={} data_dir={}",
        split,
        config_name,
        root,
    )
    dataset = load_dataset(str(root), config_name, split=split)
    rows = [cast(DatasetRow, row) for row in dataset]
    if allowed_tools is not None:
        rows = _filter_rows_by_allowed_tools(rows, allowed_tools)
    if shuffle:
        random.Random(shuffle_seed).shuffle(rows)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    tasks = [_row_to_task(row, root) for row in rows]
    logger.info("Loaded {} GAIA tasks", len(tasks))
    return tasks


def evaluate_gaia_agent(
    agent_evaluate: GaiaAgent,
    *,
    split: GaiaSplit = "validation",
    level: GaiaLevel | None = None,
    limit: int | None = None,
    offset: int = 0,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    trace_output_path: str | Path | None = None,
    allowed_tools: Sequence[GaiaToolName] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> GaiaEvaluationResult:
    """
    Evaluate an agent_evaluate callable on GAIA tasks.

    The supplied agent_evaluate receives one :class:`GaiaTask` and must return the final
    answer as a string.  Exceptions are captured as row-level failures so a long
    evaluation can continue.  Rows with hidden or missing expected answers, such
    as GAIA test rows, are included in the results with ``correct=None``.

    When ``trace_output_path`` is provided, a JSON analysis artifact is written
    containing every normalized GAIA task, its scored result, and the complete
    agent execution context when the callable returns :class:`AgentResult`.
    """
    tasks = load_gaia_tasks(
        split=split,
        level=level,
        limit=limit,
        offset=offset,
        data_dir=data_dir,
        allowed_tools=allowed_tools,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    results: list[GaiaResult] = []
    trace_entries: list[dict[str, object]] = []

    for index, task in enumerate(tasks, start=1):
        logger.info(
            "Evaluating GAIA task {}/{} task_id={} level={}",
            index,
            len(tasks),
            task.task_id,
            task.level,
        )
        started = time.perf_counter()
        prediction = ""
        error: str | None = None
        agent_context: ExecutionContext | None = None
        agent_status: str | None = None
        try:
            agent_output = agent_evaluate(task)
            prediction, agent_context, agent_status = _coerce_agent_output(agent_output)
        except Exception as exc:
            error = str(exc)
            logger.exception("GAIA task {} failed", task.task_id)
        elapsed = time.perf_counter() - started

        logger.info(
            "GAIA TASK {} REPORT:\nquestion=[{}]\nprediction=[{}]\nexpected_answer=[{}]\n"
            "expected_tools=[{}]\nexpected_time=[{}]\ncurrent_time={:.3f}s\n",
            index,
            task.question,
            prediction,
            task.expected_answer,
            task.metadata["Tools"].replace("\n", ", "),
            task.metadata["How long did this take?"],
            elapsed,
        )

        correct = _score_prediction(prediction, task.expected_answer, error)
        logger.info(
            "GAIA task {} finished correct={} elapsed={:.3f}s",
            task.task_id,
            correct,
            elapsed,
        )
        result = GaiaResult(
            task_id=task.task_id,
            question=task.question,
            level=task.level,
            file_path=task.file_path,
            file_name=task.file_name,
            prediction=prediction,
            expected_answer=task.expected_answer,
            correct=correct,
            elapsed_seconds=elapsed,
            error=error,
        )
        results.append(result)
        trace_entries.append(
            _trace_entry_to_json(
                task,
                result,
                agent_context=agent_context,
                agent_status=agent_status,
            )
        )

    evaluation = _build_evaluation_result(results)
    logger.info(
        "GAIA evaluation complete total_tasks={} scored_tasks={} correct_tasks={} "
        "overall_accuracy={} per_level_accuracy={}",
        evaluation.total_tasks,
        evaluation.scored_tasks,
        evaluation.correct_tasks,
        evaluation.overall_accuracy,
        evaluation.per_level_accuracy,
    )
    if output_path is not None:
        write_gaia_results(evaluation.results, output_path)
    if trace_output_path is not None:
        write_gaia_trace(
            evaluation,
            trace_entries,
            trace_output_path,
            run_metadata={
                "dataset_id": GAIA_DATASET_ID,
                "split": split,
                "level": level,
                "limit": limit,
                "offset": offset,
                "data_dir": str(data_dir) if data_dir is not None else None,
                "allowed_tools": list(allowed_tools or []),
                "shuffle": shuffle,
                "shuffle_seed": shuffle_seed,
                "trace_enabled": True,
                "generated_at_unix": time.time(),
            },
        )
    return evaluation


def write_gaia_results(
    results: tuple[GaiaResult, ...] | list[GaiaResult],
    output_path: str | Path,
) -> None:
    """Write complete GAIA row results as JSONL."""
    path = Path(output_path)
    logger.info("Writing {} GAIA result rows to {}", len(results), path)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_result_to_json(result), sort_keys=True) + "\n")


def write_gaia_trace(
    evaluation: GaiaEvaluationResult,
    entries: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    run_metadata: Mapping[str, object] | None = None,
) -> None:
    """Write a full GAIA analysis trace as one JSON document."""
    path = Path(output_path)
    logger.info("Writing {} GAIA trace entries to {}", len(entries), path)
    document = {
        "schema_version": 1,
        "run": dict(run_metadata or {}),
        "summary": _evaluation_summary_to_json(evaluation),
        "entries": list(entries),
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_gaia_predictions(
    results: GaiaEvaluationResult | tuple[GaiaResult, ...] | list[GaiaResult],
    output_path: str | Path,
) -> None:
    """
    Write leaderboard-style GAIA predictions as JSONL.

    Only ``task_id`` and ``answer`` are exported, so this format can be used for
    GAIA test predictions without leaking validation answers.
    """
    row_results = (
        results.results if isinstance(results, GaiaEvaluationResult) else results
    )
    path = Path(output_path)
    logger.info("Writing {} GAIA predictions to {}", len(row_results), path)
    with path.open("w", encoding="utf-8") as handle:
        for result in row_results:
            row = {"task_id": result.task_id, "answer": result.prediction}
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_gaia_answer(text: str) -> str:
    """
    Normalize a GAIA answer for pragmatic local exact-match scoring.

    This mirrors the intent of GAIA's short-answer scoring, but it is not
    guaranteed to be byte-for-byte identical to the leaderboard scorer.
    """
    normalized = text.strip()
    normalized = re.sub(
        r"^\s*final\s+answer\s*:\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    return normalized


def score_gaia_answer(prediction: str, expected: str) -> bool:
    """
    Score a GAIA prediction against a public expected answer.

    The scorer accepts normalized exact matches and cases where the normalized
    expected answer is contained in a longer normalized model response.
    """
    normalized_prediction = normalize_gaia_answer(prediction)
    normalized_expected = normalize_gaia_answer(expected)
    if not normalized_expected:
        return False
    return (
        normalized_prediction == normalized_expected
        or normalized_expected in normalized_prediction
    )


def _gaia_config_name(level: GaiaLevel | None) -> str:
    if level is None:
        return "2023_all"
    if level not in {1, 2, 3}:
        raise ValueError("level must be one of 1, 2, 3, or None")
    return f"2023_level{level}"


def _resolve_gaia_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    downloaded = snapshot_download(repo_id=GAIA_DATASET_ID, repo_type="dataset")
    return Path(downloaded).expanduser().resolve()


def _filter_rows_by_allowed_tools(
    rows: Sequence[DatasetRow],
    allowed_tools: Sequence[GaiaToolName],
) -> list[DatasetRow]:
    allowed = set(allowed_tools)
    return [row for row in rows if _row_required_tools(row).issubset(allowed)]


def _row_required_tools(row: DatasetRow) -> set[GaiaToolName]:
    metadata = row.get("Annotator Metadata")
    if not isinstance(metadata, Mapping):
        return set()
    metadata = cast(Mapping[str, Any], metadata)
    tools_value = metadata.get("Tools")
    if not isinstance(tools_value, str):
        return set()

    tools: set[GaiaToolName] = set()
    for line in tools_value.splitlines():
        normalized = _normalize_gaia_tool_name(line)
        if normalized is not None:
            tools.add(normalized)
    return tools


def _normalize_gaia_tool_name(raw_tool: str) -> GaiaToolName | None:
    tool = raw_tool.strip()
    tool = re.sub(r"^\s*\d+\s*[.)-]\s*", "", tool)
    tool = re.sub(r"\s+", " ", tool).strip(" .;:")
    tool = re.sub(r"^(an?|the)\s+", "", tool, flags=re.IGNORECASE)
    if not tool or tool.lower() in {"none", "no tools required"}:
        return None

    normalized = tool.lower()
    aliases: dict[str, GaiaToolName] = {
        "browser": "web browser",
        "web browsing": "web browser",
        "google search": "search engine",
        "search": "search engine",
        "pdf access": "pdf viewer",
        "pdf reader": "pdf viewer",
        "excel": "spreadsheet",
        "spreadsheet software": "spreadsheet",
        "image recognition tool": "image recognition",
        "image recognition tools": "image recognition",
    }
    normalized = aliases.get(normalized, cast(GaiaToolName, normalized))
    return normalized


def _row_to_task(row: DatasetRow, root: Path) -> GaiaTask:
    file_path_value = _optional_str(row.get("file_path"))
    expected_answer = _optional_str(row.get("Final answer"))
    metadata = row.get("Annotator Metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return GaiaTask(
        task_id=_required_str(row, "task_id"),
        question=_required_str(row, "Question"),
        level=int(row["Level"]),
        file_path=(root / file_path_value).resolve() if file_path_value else None,
        file_name=_optional_str(row.get("file_name")),
        metadata=cast(Mapping[str, Any], metadata),
        expected_answer=expected_answer,
    )


def _required_str(row: DatasetRow, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"GAIA row missing non-empty {key!r}")
    return value


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _score_prediction(
    prediction: str,
    expected_answer: str | None,
    error: str | None,
) -> bool | None:
    if expected_answer is None:
        return None
    if error is not None:
        return False
    return score_gaia_answer(prediction, expected_answer)


def score_prediction(
    prediction: str,
    expected_answer: str | None,
    error: str | None,
) -> bool | None:
    """Score a prediction while accounting for hidden answers and task errors."""
    return _score_prediction(prediction, expected_answer, error)


def _coerce_agent_output(
    agent_output: GaiaAgentOutput,
) -> tuple[str, ExecutionContext | None, str | None]:
    if isinstance(agent_output, AgentResult):
        output = agent_output.output
        prediction = output if isinstance(output, str) else str(output)
        return prediction, agent_output.context, agent_output.status
    return agent_output, None, None


def _build_evaluation_result(results: list[GaiaResult]) -> GaiaEvaluationResult:
    """Build aggregate metrics for a set of GAIA row results."""
    scored = [result for result in results if result.correct is not None]
    correct_count = sum(1 for result in scored if result.correct)
    per_level_accuracy: dict[int, float] = {}
    for level in sorted({result.level for result in scored}):
        level_results = [result for result in scored if result.level == level]
        if level_results:
            per_level_accuracy[level] = sum(
                1 for result in level_results if result.correct
            ) / len(level_results)

    return GaiaEvaluationResult(
        total_tasks=len(results),
        scored_tasks=len(scored),
        correct_tasks=correct_count,
        overall_accuracy=correct_count / len(scored) if scored else None,
        per_level_accuracy=per_level_accuracy,
        results=tuple(results),
    )


def _evaluation_summary_to_json(evaluation: GaiaEvaluationResult) -> dict[str, object]:
    return {
        "total_tasks": evaluation.total_tasks,
        "scored_tasks": evaluation.scored_tasks,
        "correct_tasks": evaluation.correct_tasks,
        "overall_accuracy": evaluation.overall_accuracy,
        "per_level_accuracy": evaluation.per_level_accuracy,
    }


def _task_to_json(task: GaiaTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "question": task.question,
        "level": task.level,
        "file_path": str(task.file_path) if task.file_path is not None else None,
        "file_name": task.file_name,
        "metadata": _json_safe(task.metadata),
        "expected_answer": task.expected_answer,
    }


def _result_to_json(result: GaiaResult) -> dict[str, object]:
    normalized_prediction = normalize_gaia_answer(result.prediction)
    normalized_expected = (
        normalize_gaia_answer(result.expected_answer)
        if result.expected_answer is not None
        else None
    )
    return {
        "task_id": result.task_id,
        "question": result.question,
        "level": result.level,
        "file_path": str(result.file_path) if result.file_path is not None else None,
        "file_name": result.file_name,
        "prediction": result.prediction,
        "normalized_prediction": normalized_prediction,
        "expected_answer": result.expected_answer,
        "normalized_expected_answer": normalized_expected,
        "correct": result.correct,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
    }


def _trace_entry_to_json(
    task: GaiaTask,
    result: GaiaResult,
    *,
    agent_context: ExecutionContext | None,
    agent_status: str | None,
) -> dict[str, object]:
    return {
        "task": _task_to_json(task),
        "result": _result_to_json(result),
        "agent": {
            "status": agent_status,
            "context": (
                _execution_context_to_json(agent_context)
                if agent_context is not None
                else None
            ),
        },
    }


def trace_entry_to_json(
    task: GaiaTask,
    result: GaiaResult,
    *,
    agent_context: ExecutionContext | None,
    agent_status: str | None,
) -> dict[str, object]:
    """Serialize one task, result, and complete agent context for a trace."""
    return _trace_entry_to_json(
        task,
        result,
        agent_context=agent_context,
        agent_status=agent_status,
    )


def _execution_context_to_json(context: ExecutionContext) -> dict[str, object]:
    return cast(dict[str, object], _json_safe(context))


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        value = cast(Mapping[str, Any], value)
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        value = cast(Sequence[Any], value)
        return [_json_safe(item) for item in value]
    return str(value)


DEFAULT_PROMPT_TEMPLATE = """Answer this GAIA benchmark question.
Use the available tools when useful. Return only the final answer in the form:
FINAL ANSWER: <answer>

Question: {question}
{attachment}"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvaluationConfig(StrictModel):
    """Reserved run-level settings."""

    name: str = "gaia"


class ModelConfig(StrictModel):
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


class AgentConfig(StrictModel):
    mode: AgentMode = "dummy"
    instruction: str = ""
    max_steps: NonNegativeInt = Field(
        default=8,
        validation_alias=AliasChoices("max_steps", "maximum_steps"),
    )
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    @field_validator("prompt_template")
    @classmethod
    def valid_prompt_template(cls, value: str) -> str:
        allowed = {
            "question",
            "attachment",
            "attachment_path",
            "file_name",
            "task_id",
            "level",
        }
        template_fields: set[str] = set()
        try:
            for _, field_name, format_spec, conversion in string.Formatter().parse(
                value
            ):
                if field_name is None:
                    continue
                if field_name not in allowed:
                    raise ValueError(
                        f"unknown placeholder {{{field_name}}}; "
                        f"supported placeholders: {', '.join(sorted(allowed))}"
                    )
                if format_spec or conversion:
                    raise ValueError(
                        "format specifications and conversions are unsupported"
                    )
                template_fields.add(field_name)
        except ValueError as error:
            raise ValueError(f"invalid prompt template: {error}") from error
        if "question" not in template_fields:
            raise ValueError("prompt template must contain {question}")
        return value


RuntimeToolName = Literal["compute", "python"]


class ToolsConfig(StrictModel):
    enabled: list[RuntimeToolName] = Field(
        default_factory=lambda: cast(list[RuntimeToolName], [])
    )

    @field_validator("enabled")
    @classmethod
    def unique_tools(cls, value: list[RuntimeToolName]) -> list[RuntimeToolName]:
        if len(value) != len(set(value)):
            raise ValueError("tool names must be unique")
        return value


class ContainerConfig(StrictModel):
    image: str = DEFAULT_CONTAINER_ENV_IMAGE
    network: bool = Field(
        default=False,
        validation_alias=AliasChoices("network", "network_access"),
    )
    auto_remove: bool = True

    @field_validator("image")
    @classmethod
    def image_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class DatasetConfig(StrictModel):
    split: Literal["validation", "test"] = "validation"
    entry_count: PositiveInt
    entry_offset: NonNegativeInt = 0
    tool_filter: list[GaiaToolName] | None = None
    level: GaiaLevel | None = None
    data_dir: Path | None = None
    shuffle: bool = False
    shuffle_seed: int = 0

    @field_validator("data_dir", mode="before")
    @classmethod
    def path_from_yaml(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value)
        return value


class AppConfig(StrictModel):
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    dataset: DatasetConfig

    @field_validator("tools", mode="before")
    @classmethod
    def normalize_tools(cls, value: object) -> object:
        if isinstance(value, list):
            return {"enabled": cast(list[object], value)}
        return value


TOOL_REGISTRY: Mapping[RuntimeToolName, Callable[[], Tool]] = {
    "compute": compute_tool,
    "python": python_tool,
}


def load_config(path: Path) -> AppConfig:
    """Parse, validate, and resolve a configuration file."""
    raw = YamlParser().parse(path)
    config = AppConfig.model_validate(raw)
    if config.dataset.data_dir is not None:
        resolved = (path.resolve().parent / config.dataset.data_dir).resolve()
        config = config.model_copy(
            update={"dataset": config.dataset.model_copy(update={"data_dir": resolved})}
        )
    return config


def build_tools(config: ToolsConfig) -> list[Tool]:
    """Build tools in the configured order."""
    return [TOOL_REGISTRY[name]() for name in config.enabled]


def build_llm(config: ModelConfig, api_key: str | None) -> LlmClient:
    """Build the configured LiteLLM client."""
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


def _prompt_for_task(template: str, task: GaiaTask, attachment_path: str | None) -> str:
    attachment = (
        f"Attached file path: {attachment_path}" if attachment_path is not None else ""
    )
    return template.format(
        question=task.question,
        attachment=attachment,
        attachment_path=attachment_path or "",
        file_name=task.file_name or "",
        task_id=task.task_id,
        level=task.level,
    )


def _container_attachment(task: GaiaTask) -> tuple[list[Path], str | None]:
    if task.file_path is None:
        return [], None
    return [task.file_path.parent], f"/tmp/0/{task.file_path.name}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_config(config: AppConfig) -> dict[str, object]:
    return cast(dict[str, object], config.model_dump(mode="json"))


def _config_hash(config: Mapping[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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


def _trace_document(
    *,
    resolved: Mapping[str, object],
    run: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    evaluation: GaiaEvaluationResult,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "configuration": {
            "resolved": dict(resolved),
            "sha256": _config_hash(resolved),
            "credential": {
                "source": "LLLM_API_KEY",
                "present": bool(os.environ.get("LLLM_API_KEY")),
            },
        },
        "run": dict(run),
        "summary": {
            "total_tasks": evaluation.total_tasks,
            "scored_tasks": evaluation.scored_tasks,
            "correct_tasks": evaluation.correct_tasks,
            "overall_accuracy": evaluation.overall_accuracy,
            "per_level_accuracy": evaluation.per_level_accuracy,
        },
        "entries": list(entries),
    }


def run_evaluation(config: AppConfig, trace_path: Path) -> GaiaEvaluationResult:
    """Execute a configured run and checkpoint its trace after each task."""
    resolved = _resolved_config(config)
    llm = build_llm(config.model, os.environ.get("LLLM_API_KEY") or None)
    agent = Agent(
        llm,
        build_tools(config.tools),
        instruction=config.agent.instruction,
        max_step=config.agent.max_steps,
        agent_mode=config.agent.mode,
    )
    dataset = config.dataset
    tasks = load_gaia_tasks(
        split=dataset.split,
        level=dataset.level,
        limit=dataset.entry_count,
        offset=dataset.entry_offset,
        data_dir=dataset.data_dir,
        allowed_tools=dataset.tool_filter,
        shuffle=dataset.shuffle,
        shuffle_seed=dataset.shuffle_seed,
    )
    run: dict[str, object] = {
        "dataset_id": GAIA_DATASET_ID,
        "started_at": _now(),
        "finished_at": None,
        "selected_task_ids": [task.task_id for task in tasks],
    }
    results: list[GaiaResult] = []
    entries: list[dict[str, object]] = []

    for task in tasks:
        started = time.perf_counter()
        prediction = ""
        error: str | None = None
        context: ExecutionContext | None = None
        status: str | None = None
        container: ContainerEnv | None = None
        try:
            mounts, attachment_path = _container_attachment(task)
            if "python" in config.tools.enabled:
                container = ContainerEnv(
                    image=config.container.image,
                    auto_remove=config.container.auto_remove,
                )
                container.start(mounts, network=config.container.network)
            prompt = _prompt_for_task(
                config.agent.prompt_template, task, attachment_path
            )
            output = agent.run(prompt, container_env=container, trace_enabled=True)
            prediction = (
                output.output if isinstance(output.output, str) else str(output.output)
            )
            context = output.context
            status = output.status
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "error"
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception as exc:
                    close_error = f"{type(exc).__name__}: {exc}"
                    error = (
                        f"{error}; container close failed: {close_error}"
                        if error is not None
                        else f"container close failed: {close_error}"
                    )
                    status = "error"
        elapsed = time.perf_counter() - started
        result = GaiaResult(
            task_id=task.task_id,
            question=task.question,
            level=task.level,
            file_path=task.file_path,
            file_name=task.file_name,
            prediction=prediction,
            expected_answer=task.expected_answer,
            correct=score_prediction(prediction, task.expected_answer, error),
            elapsed_seconds=elapsed,
            error=error,
        )
        results.append(result)
        entries.append(
            trace_entry_to_json(
                task, result, agent_context=context, agent_status=status
            )
        )
        partial = _build_evaluation_result(results)
        _atomic_write_json(
            trace_path,
            _trace_document(
                resolved=resolved,
                run=run,
                entries=entries,
                evaluation=partial,
            ),
        )

    evaluation = _build_evaluation_result(results)
    run["finished_at"] = _now()
    _atomic_write_json(
        trace_path,
        _trace_document(
            resolved=resolved,
            run=run,
            entries=entries,
            evaluation=evaluation,
        ),
    )
    return evaluation


HELP = """Run a real GAIA evaluation from a strict YAML configuration.

The YAML sections are evaluation, model, agent, tools, container, and dataset.
Dataset selection order is split/level, tool_filter, seeded shuffle, entry_offset,
then entry_count. Runtime tools currently supported are compute and python. Python
starts a fresh container per entry. Custom endpoint credentials are read only from
LLLM_API_KEY.

TRACE is overwritten atomically, its parent directories are created, and it is
checkpointed after every completed entry. No other result file is produced.

Example:

\b
  evaluation:
    name: gaia-smoke
  model:
    model: lllm
    base_url: http://127.0.0.1:8000/v1
    max_tokens: 1024
  agent:
    mode: dummy
    max_steps: 8
    prompt_template: |
      Answer the question and return FINAL ANSWER: <answer>.
      Question: {question}
      {attachment}
  tools:
    enabled: [compute, python]
  container:
    image: python:3.13-slim
    network: false
    auto_remove: true
  dataset:
    split: validation
    level: 1
    data_dir: ./gaia
    tool_filter: [calculator, python]
    shuffle: false
    shuffle_seed: 0
    entry_offset: 0
    entry_count: 1
"""


@click.command(
    "evaluate_gaia",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=HELP,
)
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, readable=True),
    help="YAML evaluation configuration.",
)
@click.option(
    "-t",
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="JSON trace destination (overwritten).",
)
def main(config_path: Path, trace_path: Path) -> None:
    """Run the configuration-driven GAIA evaluation CLI."""
    try:
        config = load_config(config_path)
    except (YamlParserError, ValidationError, ValueError) as error:
        raise click.UsageError(f"invalid configuration: {error}") from error
    try:
        evaluation = run_evaluation(config, trace_path)
    except Exception as error:
        raise click.ClickException(f"evaluation failed: {error}") from error
    click.echo(
        f"Completed {evaluation.total_tasks} tasks; "
        f"correct={evaluation.correct_tasks}/{evaluation.scored_tasks}"
    )


if __name__ == "__main__":
    main()
