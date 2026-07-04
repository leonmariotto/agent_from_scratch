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
import importlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Literal, cast

from loguru import logger
from pydantic import BaseModel

from .agent_context import AgentResult, ExecutionContext

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


def _coerce_agent_output(
    agent_output: GaiaAgentOutput,
) -> tuple[str, ExecutionContext | None, str | None]:
    if isinstance(agent_output, AgentResult):
        output = agent_output.output
        prediction = output if isinstance(output, str) else str(output)
        return prediction, agent_output.context, agent_output.status
    return agent_output, None, None


def _build_evaluation_result(results: list[GaiaResult]) -> GaiaEvaluationResult:
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
