"""
SWE-bench evaluation helpers for repository-editing agents.

This module does not use the official SWE-bench evaluation harness.  It loads
SWE-bench-compatible rows, prepares an isolated Docker-backed repository
workspace for each task, invokes a caller-provided agent, applies the task test
patch, runs tests, and records JSONL results.

The containerization layer is required for running SWE-bench tests on produced
code, so tool containerization alone is not enough. Use ContainerizedAgent to
isolate tool execution.

DockerSwebenchRunner clones the repository into a mounted volume:
"{host_root}/repo" -> "/workspace/repo". The agent runs in this Python process
against the host path and must modify the repository directly.
"{host_root}/repo" need to be available to ContainerizedAgent.

WIP: to be updated when I'll have a real agent capable of doing SWE-bench tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
import time
from typing import Any, Protocol, cast

import click
from loguru import logger
from pydantic import (
    AliasChoices,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    field_validator,
)

from .agent import Agent, AgentMode
from .agent_context import ExecutionContext
from .container_env import ContainerEnv
from .eval_common import (
    ModelConfig,
    StrictModel,
    ToolsConfig,
    atomic_write_json as _swebench_atomic_write,
    build_llm,
    build_tools,
    environment_credential,
    json_safe as _swebench_json_safe,
    normalize_tools_config,
    resolved_config,
    trace_document,
    utc_now,
    validate_prompt_template,
)
from .yaml_parser import YamlParser, YamlParserError

load_dataset = cast(
    Callable[..., Any], importlib.import_module("datasets").load_dataset
)

DEFAULT_SWEBENCH_DATASET_ID = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SWEBENCH_SPLIT = "test"
DEFAULT_DOCKER_IMAGE = "python:3.12"
DEFAULT_TEST_COMMAND = ("python", "-m", "pytest")
DatasetRow = Mapping[str, Any]


@dataclass(frozen=True)
class SwebenchTask:
    """One SWE-bench task passed to an agent implementation."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str | None
    created_at: str | None
    version: str | None
    environment_setup_commit: str | None
    patch: str | None
    test_patch: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CommandResult:
    """Captured process result for a command run in the task workspace."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class SwebenchResult:
    """Result for one attempted SWE-bench task."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    agent_patch: str
    resolved: bool
    elapsed_seconds: float
    test_result: CommandResult | None
    artifact_dir: Path | None
    error: str | None = None


@dataclass(frozen=True)
class SwebenchEvaluationResult:
    """Aggregate SWE-bench metrics and per-task results."""

    total_tasks: int
    resolved_tasks: int
    failed_tasks: int
    error_tasks: int
    resolved_rate: float | None
    results: tuple[SwebenchResult, ...]


SwebenchAgent = Callable[[SwebenchTask, Path], None]


class SwebenchRunner(Protocol):
    """Runner contract used by the evaluator."""

    def run(self, task: SwebenchTask, agent: SwebenchAgent) -> SwebenchResult: ...


def load_swebench_tasks(
    *,
    dataset_name: str = DEFAULT_SWEBENCH_DATASET_ID,
    split: str = DEFAULT_SWEBENCH_SPLIT,
    limit: int | None = None,
    offset: int = 0,
    instance_ids: Sequence[str] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> list[SwebenchTask]:
    """Load and normalize SWE-bench tasks from Hugging Face datasets."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    logger.info("Loading SWE-bench dataset={} split={}", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    rows = [cast(DatasetRow, row) for row in dataset]
    if instance_ids is not None:
        selected_ids = set(instance_ids)
        rows = [row for row in rows if row.get("instance_id") in selected_ids]
    if shuffle:
        random.Random(shuffle_seed).shuffle(rows)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    tasks = [_row_to_task(row) for row in rows]
    logger.info("Loaded {} SWE-bench tasks", len(tasks))
    return tasks


def evaluate_swebench_agent(
    agent: SwebenchAgent,
    *,
    dataset_name: str = DEFAULT_SWEBENCH_DATASET_ID,
    split: str = DEFAULT_SWEBENCH_SPLIT,
    limit: int | None = None,
    offset: int = 0,
    instance_ids: Sequence[str] | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
    docker_image: str | None = DEFAULT_DOCKER_IMAGE,
    test_command: Sequence[str] | None = DEFAULT_TEST_COMMAND,
    timeout_seconds: int = 1800,
    output_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    runner: SwebenchRunner | None = None,
) -> SwebenchEvaluationResult:
    """
    Evaluate an agent callable on SWE-bench tasks.

    The supplied agent receives a :class:`SwebenchTask` and the prepared
    repository workspace path.  It must mutate the workspace directly.
    """
    tasks = load_swebench_tasks(
        dataset_name=dataset_name,
        split=split,
        limit=limit,
        offset=offset,
        instance_ids=instance_ids,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    task_runner = runner
    if task_runner is None:
        if docker_image is None:
            raise ValueError("docker_image must be provided when runner is omitted")
        task_runner = DockerSwebenchRunner(
            docker_image=docker_image,
            test_command=test_command,
            timeout_seconds=timeout_seconds,
            artifacts_dir=artifacts_dir,
        )

    results: list[SwebenchResult] = []
    for index, task in enumerate(tasks, start=1):
        logger.info(
            "Evaluating SWE-bench task {}/{} instance_id={}",
            index,
            len(tasks),
            task.instance_id,
        )
        try:
            result = task_runner.run(task, agent)
        except Exception as exc:
            logger.exception("SWE-bench task {} failed", task.instance_id)
            result = SwebenchResult(
                instance_id=task.instance_id,
                repo=task.repo,
                base_commit=task.base_commit,
                problem_statement=task.problem_statement,
                agent_patch="",
                resolved=False,
                elapsed_seconds=0.0,
                test_result=None,
                artifact_dir=None,
                error=str(exc),
            )
        logger.info(
            "SWE-bench task {} finished resolved={} elapsed={:.3f}s",
            task.instance_id,
            result.resolved,
            result.elapsed_seconds,
        )
        results.append(result)

    evaluation = _build_evaluation_result(results)
    if output_path is not None:
        write_swebench_results(evaluation.results, output_path)
    return evaluation


def write_swebench_results(
    results: tuple[SwebenchResult, ...] | list[SwebenchResult],
    output_path: str | Path,
) -> None:
    """Write complete SWE-bench row results as JSONL."""
    logger.info("Writing {} SWE-bench result rows to {}", len(results), output_path)
    _write_jsonl_rows((_result_to_json(result) for result in results), output_path)


def export_swebench_predictions(
    results: (
        SwebenchEvaluationResult | tuple[SwebenchResult, ...] | list[SwebenchResult]
    ),
    output_path: str | Path,
    *,
    model_name_or_path: str = "LLLM-agent",
) -> None:
    """Write SWE-bench-compatible prediction rows as JSONL."""
    row_results = (
        results.results if isinstance(results, SwebenchEvaluationResult) else results
    )
    rows = (
        {
            "instance_id": result.instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": result.agent_patch,
        }
        for result in row_results
    )
    logger.info(
        "Writing {} SWE-bench predictions to {}",
        len(row_results),
        output_path,
    )
    _write_jsonl_rows(rows, output_path)


class DockerSwebenchRunner:
    """Run a SWE-bench task in a Docker-mounted temporary workspace."""

    def __init__(
        self,
        *,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        test_command: Sequence[str] | None = DEFAULT_TEST_COMMAND,
        timeout_seconds: int = 1800,
        artifacts_dir: str | Path | None = None,
        auto_remove: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.docker_image = docker_image
        self.test_command = tuple(test_command) if test_command is not None else None
        self.timeout_seconds = timeout_seconds
        self.auto_remove = auto_remove
        self.artifacts_dir = (
            Path(artifacts_dir).expanduser().resolve()
            if artifacts_dir is not None
            else None
        )

    def run(self, task: SwebenchTask, agent: SwebenchAgent) -> SwebenchResult:
        started = time.perf_counter()
        test_result: CommandResult | None = None
        error: str | None = None
        agent_patch = ""
        artifact_dir: Path | None = None

        with tempfile.TemporaryDirectory(
            prefix="lllm-swebench-",
            ignore_cleanup_errors=True,
        ) as directory:
            host_root = Path(directory).resolve()
            repo_path = host_root / "repo"
            try:
                self._docker_shell(
                    host_root,
                    f"git clone {shlex.quote(_repo_url(task.repo))} repo",
                )
                self._docker_shell(
                    host_root,
                    f"git -C repo checkout {shlex.quote(task.base_commit)}",
                )
                agent(task, repo_path)
                agent_patch = self._git_diff(host_root)

                if task.test_patch.strip():
                    self._apply_patch(host_root, task.test_patch, "test.patch")
                else:
                    error = "SWE-bench task missing test_patch"

                if error is None:
                    test_result = self._run_tests(host_root)
            except Exception as exc:
                error = str(exc)

            if self.artifacts_dir is not None:
                artifact_dir = self._write_artifacts(
                    task,
                    agent_patch=agent_patch,
                    test_result=test_result,
                    error=error,
                )
            self._restore_host_ownership(host_root)

        return SwebenchResult(
            instance_id=task.instance_id,
            repo=task.repo,
            base_commit=task.base_commit,
            problem_statement=task.problem_statement,
            agent_patch=agent_patch,
            resolved=error is None
            and test_result is not None
            and test_result.returncode == 0,
            elapsed_seconds=time.perf_counter() - started,
            test_result=test_result,
            artifact_dir=artifact_dir,
            error=error,
        )

    def _run_tests(self, host_root: Path) -> CommandResult:
        if self.test_command is None:
            raise ValueError("test_command must be configured to score SWE-bench tasks")
        return self._docker_exec(
            host_root,
            self.test_command,
            workdir="/workspace/repo",
        )

    def _git_diff(self, host_root: Path) -> str:
        result = self._docker_exec(
            host_root,
            ("git", "-C", "repo", "diff", "--binary"),
            workdir="/workspace",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git diff failed")
        return result.stdout

    def _apply_patch(self, host_root: Path, patch: str, filename: str) -> None:
        patch_path = host_root / filename
        patch_path.write_text(patch, encoding="utf-8")
        result = self._docker_exec(
            host_root,
            ("git", "-C", "repo", "apply", f"/workspace/{filename}"),
            workdir="/workspace",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"failed to apply {filename}")

    def _docker_shell(self, host_root: Path, command: str) -> CommandResult:
        return self._docker_exec(
            host_root, ("sh", "-lc", command), workdir="/workspace"
        )

    def _docker_exec(
        self,
        host_root: Path,
        command: Sequence[str],
        *,
        workdir: str,
    ) -> CommandResult:
        docker_options = ("--rm",) if self.auto_remove else ()
        docker_command = (
            "docker",
            "run",
            *docker_options,
            "-v",
            f"{host_root}:/workspace",
            "-w",
            workdir,
            self.docker_image,
            *command,
        )
        started = time.perf_counter()
        process = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        return CommandResult(
            command=tuple(docker_command),
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _restore_host_ownership(self, host_root: Path) -> None:
        """Best-effort fix for root-owned files created by Docker bind mounts."""
        uid = os.getuid()
        gid = os.getgid()
        try:
            result = self._docker_exec(
                host_root,
                ("chown", "-R", f"{uid}:{gid}", "/workspace"),
                workdir="/workspace",
            )
        except Exception as exc:
            logger.warning(
                "Could not restore SWE-bench temp directory ownership: {}",
                exc,
            )
            return
        if result.returncode != 0:
            logger.warning(
                "Could not restore SWE-bench temp directory ownership: {}",
                result.stderr.strip() or result.stdout.strip(),
            )

    def _write_artifacts(
        self,
        task: SwebenchTask,
        *,
        agent_patch: str,
        test_result: CommandResult | None,
        error: str | None,
    ) -> Path:
        if self.artifacts_dir is None:
            raise RuntimeError("artifacts_dir is not configured")
        artifact_dir = self.artifacts_dir / _safe_path_name(task.instance_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "agent.patch").write_text(agent_patch, encoding="utf-8")
        (artifact_dir / "test.patch").write_text(task.test_patch, encoding="utf-8")
        if test_result is not None:
            (artifact_dir / "test.stdout").write_text(
                test_result.stdout, encoding="utf-8"
            )
            (artifact_dir / "test.stderr").write_text(
                test_result.stderr, encoding="utf-8"
            )
        if error is not None:
            (artifact_dir / "error.txt").write_text(error, encoding="utf-8")
        return artifact_dir


def _row_to_task(row: DatasetRow) -> SwebenchTask:
    metadata = {str(key): value for key, value in row.items()}
    return SwebenchTask(
        instance_id=_required_str(row, "instance_id"),
        repo=_required_str(row, "repo"),
        base_commit=_required_str(row, "base_commit"),
        problem_statement=_required_str(row, "problem_statement"),
        hints_text=_optional_str(row.get("hints_text")),
        created_at=_optional_str(row.get("created_at")),
        version=_optional_str(row.get("version")),
        environment_setup_commit=_optional_str(row.get("environment_setup_commit")),
        patch=_optional_str(row.get("patch")),
        test_patch=_required_str(row, "test_patch"),
        metadata=metadata,
    )


def _required_str(row: DatasetRow, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"SWE-bench row missing non-empty {key!r}")
    return value


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _build_evaluation_result(results: list[SwebenchResult]) -> SwebenchEvaluationResult:
    resolved_count = sum(1 for result in results if result.resolved)
    error_count = sum(1 for result in results if result.error is not None)
    failed_count = len(results) - resolved_count - error_count
    return SwebenchEvaluationResult(
        total_tasks=len(results),
        resolved_tasks=resolved_count,
        failed_tasks=failed_count,
        error_tasks=error_count,
        resolved_rate=resolved_count / len(results) if results else None,
        results=tuple(results),
    )


def _repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _safe_path_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value
    )


def _write_jsonl_rows(
    rows: Sequence[Mapping[str, object]] | Any,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _result_to_json(result: SwebenchResult) -> dict[str, object]:
    test_result = None
    if result.test_result is not None:
        test_result = {
            "command": list(result.test_result.command),
            "returncode": result.test_result.returncode,
            "stdout": result.test_result.stdout,
            "stderr": result.test_result.stderr,
            "elapsed_seconds": result.test_result.elapsed_seconds,
        }
    return {
        "instance_id": result.instance_id,
        "repo": result.repo,
        "base_commit": result.base_commit,
        "problem_statement": result.problem_statement,
        "agent_patch": result.agent_patch,
        "resolved": result.resolved,
        "elapsed_seconds": result.elapsed_seconds,
        "test_result": test_result,
        "artifact_dir": (
            str(result.artifact_dir) if result.artifact_dir is not None else None
        ),
        "error": result.error,
    }


DEFAULT_SWEBENCH_PROMPT_TEMPLATE = """Fix this SWE-bench issue in the repository.
Use the Python tool to inspect and edit files under {repo_path}. Do not only
describe the fix: modify the repository. Return a concise summary when done.

Repository: {repo}
Instance: {instance_id}
Problem:
{problem_statement}
{hints}"""


class SwebenchEvaluationConfig(StrictModel):
    name: str = "swebench"


class SwebenchAgentConfig(StrictModel):
    mode: AgentMode = "dummy"
    instruction: str = ""
    max_steps: NonNegativeInt = Field(
        default=12,
        validation_alias=AliasChoices("max_steps", "maximum_steps"),
    )
    prompt_template: str = DEFAULT_SWEBENCH_PROMPT_TEMPLATE

    @field_validator("prompt_template")
    @classmethod
    def valid_prompt_template(cls, value: str) -> str:
        return validate_prompt_template(
            value,
            allowed_fields={
                "instance_id",
                "repo",
                "base_commit",
                "problem_statement",
                "hints",
                "repo_path",
            },
            required_fields={"problem_statement", "repo_path"},
        )


class SwebenchContainerConfig(StrictModel):
    image: str = DEFAULT_DOCKER_IMAGE
    network: bool = True
    auto_remove: bool = True
    test_command: list[str] = Field(default_factory=lambda: list(DEFAULT_TEST_COMMAND))
    timeout_seconds: PositiveInt = 1800
    artifacts_dir: Path | None = None

    @field_validator("image")
    @classmethod
    def image_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("test_command", mode="before")
    @classmethod
    def parse_test_command(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return shlex.split(value)
            except ValueError as error:
                raise ValueError(f"invalid shell-style command: {error}") from error
        return value

    @field_validator("test_command")
    @classmethod
    def test_command_not_empty(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("must contain non-empty command arguments")
        return value

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def path_from_yaml(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value


class SwebenchDatasetConfig(StrictModel):
    dataset_name: str = DEFAULT_SWEBENCH_DATASET_ID
    split: str = DEFAULT_SWEBENCH_SPLIT
    entry_count: PositiveInt
    entry_offset: NonNegativeInt = 0
    instance_ids: list[str] | None = None
    shuffle: bool = False
    shuffle_seed: int = 0

    @field_validator("dataset_name", "split")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("instance_ids", mode="before")
    @classmethod
    def scalar_instance_id(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("instance_ids")
    @classmethod
    def unique_instance_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            if any(not item.strip() for item in value):
                raise ValueError("instance IDs must not be empty")
            if len(value) != len(set(value)):
                raise ValueError("instance IDs must be unique")
        return value


class SwebenchAppConfig(StrictModel):
    evaluation: SwebenchEvaluationConfig = Field(
        default_factory=SwebenchEvaluationConfig
    )
    model: ModelConfig
    agent: SwebenchAgentConfig = Field(default_factory=SwebenchAgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    container: SwebenchContainerConfig = Field(default_factory=SwebenchContainerConfig)
    dataset: SwebenchDatasetConfig

    @field_validator("tools", mode="before")
    @classmethod
    def normalize_tools(cls, value: object) -> object:
        return normalize_tools_config(value)


def load_swebench_config(path: Path) -> SwebenchAppConfig:
    """Parse, validate, and resolve a SWE-bench application configuration."""
    config = SwebenchAppConfig.model_validate(YamlParser().parse(path))
    artifacts_dir = config.container.artifacts_dir
    if artifacts_dir is not None:
        resolved = (path.resolve().parent / artifacts_dir).resolve()
        config = config.model_copy(
            update={
                "container": config.container.model_copy(
                    update={"artifacts_dir": resolved}
                )
            }
        )
    return config


def _swebench_prompt(
    template: str,
    task: SwebenchTask,
    repo_path: str,
) -> str:
    hints = f"Hints:\n{task.hints_text}" if task.hints_text else ""
    return template.format(
        instance_id=task.instance_id,
        repo=task.repo,
        base_commit=task.base_commit,
        problem_statement=task.problem_statement,
        hints=hints,
        repo_path=repo_path,
    )


def _swebench_task_to_json(task: SwebenchTask) -> dict[str, object]:
    return cast(dict[str, object], _swebench_json_safe(task))


def _swebench_evaluation_summary(
    evaluation: SwebenchEvaluationResult,
) -> dict[str, object]:
    return {
        "total_tasks": evaluation.total_tasks,
        "resolved_tasks": evaluation.resolved_tasks,
        "failed_tasks": evaluation.failed_tasks,
        "error_tasks": evaluation.error_tasks,
        "resolved_rate": evaluation.resolved_rate,
    }


def _swebench_trace_document(
    *,
    resolved_config: Mapping[str, object],
    run: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    evaluation: SwebenchEvaluationResult,
) -> dict[str, object]:
    return trace_document(
        resolved=resolved_config,
        run=run,
        summary=_swebench_evaluation_summary(evaluation),
        entries=entries,
    )


def run_swebench_evaluation(
    config: SwebenchAppConfig,
    trace_path: Path,
) -> SwebenchEvaluationResult:
    """Run SWE-bench and atomically checkpoint the requested trace."""
    resolved = resolved_config(config)
    agent = Agent(
        build_llm(config.model, environment_credential()),
        build_tools(config.tools),
        instruction=config.agent.instruction,
        max_step=config.agent.max_steps,
        agent_mode=config.agent.mode,
    )
    tasks = load_swebench_tasks(
        dataset_name=config.dataset.dataset_name,
        split=config.dataset.split,
        limit=config.dataset.entry_count,
        offset=config.dataset.entry_offset,
        instance_ids=config.dataset.instance_ids,
        shuffle=config.dataset.shuffle,
        shuffle_seed=config.dataset.shuffle_seed,
    )
    runner = DockerSwebenchRunner(
        docker_image=config.container.image,
        test_command=config.container.test_command,
        timeout_seconds=config.container.timeout_seconds,
        artifacts_dir=config.container.artifacts_dir,
        auto_remove=config.container.auto_remove,
    )
    run: dict[str, object] = {
        "dataset_id": config.dataset.dataset_name,
        "started_at": utc_now(),
        "finished_at": None,
        "selected_instance_ids": [task.instance_id for task in tasks],
    }
    results: list[SwebenchResult] = []
    entries: list[dict[str, object]] = []

    for task in tasks:
        context: ExecutionContext | None = None
        agent_status: str | None = None

        def evaluate_task(current_task: SwebenchTask, repo_path: Path) -> None:
            nonlocal context, agent_status
            container: ContainerEnv | None = None
            try:
                if "python" in config.tools.enabled:
                    container = ContainerEnv(
                        image=config.container.image,
                        auto_remove=config.container.auto_remove,
                    )
                    container.start(
                        [repo_path],
                        network=config.container.network,
                    )
                prompt = _swebench_prompt(
                    config.agent.prompt_template,
                    current_task,
                    "/tmp/0",
                )
                output = agent.run(
                    prompt,
                    container_env=container,
                    trace_enabled=True,
                )
                context = output.context
                agent_status = output.status
                if output.status == "error":
                    agent_error = output.context.state.get("agent_error")
                    raise RuntimeError(
                        str(agent_error)
                        if agent_error is not None
                        else "agent execution failed"
                    )
            except Exception:
                agent_status = "error"
                raise
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        agent_status = "error"
                        raise

        try:
            result = runner.run(task, evaluate_task)
        except Exception as exc:
            result = SwebenchResult(
                instance_id=task.instance_id,
                repo=task.repo,
                base_commit=task.base_commit,
                problem_statement=task.problem_statement,
                agent_patch="",
                resolved=False,
                elapsed_seconds=0.0,
                test_result=None,
                artifact_dir=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            agent_status = "error"
        results.append(result)
        entries.append(
            {
                "task": _swebench_task_to_json(task),
                "result": _result_to_json(result),
                "agent": {
                    "status": agent_status,
                    "context": (
                        _swebench_json_safe(context) if context is not None else None
                    ),
                },
            }
        )
        partial = _build_evaluation_result(results)
        _swebench_atomic_write(
            trace_path,
            _swebench_trace_document(
                resolved_config=resolved,
                run=run,
                entries=entries,
                evaluation=partial,
            ),
        )

    evaluation = _build_evaluation_result(results)
    run["finished_at"] = utc_now()
    _swebench_atomic_write(
        trace_path,
        _swebench_trace_document(
            resolved_config=resolved,
            run=run,
            entries=entries,
            evaluation=evaluation,
        ),
    )
    return evaluation


SWEBENCH_HELP = """Run a real SWE-bench evaluation from strict YAML.

The YAML sections configure the model, agent, runtime tools, Docker container,
and dataset. Selection order is dataset/split, instance_ids, seeded shuffle,
entry_offset, then entry_count. Runtime tools are compute and python. Python
receives only the current task repository mounted at /tmp/0.

Credentials for custom endpoints are read only from LLLM_API_KEY. The JSON trace
is overwritten atomically, parent directories are created, and a checkpoint is
written after every completed instance. Incorrect patches do not fail the run.

Example:

\b
  evaluation:
    name: swebench-smoke
  model:
    model: lllm
    base_url: http://127.0.0.1:8000/v1
    max_tokens: 2048
  agent:
    mode: dummy
    max_steps: 12
    prompt_template: |
      Fix this issue in {repo_path}.
      Problem: {problem_statement}
  tools:
    enabled: [compute, python]
  container:
    image: python:3.12
    network: true
    auto_remove: true
    test_command: python -m pytest
    timeout_seconds: 1800
  dataset:
    dataset_name: princeton-nlp/SWE-bench_Lite
    split: test
    instance_ids: django__django-11099
    shuffle: false
    shuffle_seed: 0
    entry_offset: 0
    entry_count: 1
"""


@click.command(
    "evaluate_swebench",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=SWEBENCH_HELP,
)
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, readable=True),
    help="YAML SWE-bench evaluation configuration.",
)
@click.option(
    "-t",
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="JSON trace destination (overwritten).",
)
def swebench_main(config_path: Path, trace_path: Path) -> None:
    """Run the configuration-driven SWE-bench evaluation CLI."""
    try:
        config = load_swebench_config(config_path)
    except (YamlParserError, ValidationError, ValueError) as error:
        raise click.UsageError(f"invalid configuration: {error}") from error
    try:
        evaluation = run_swebench_evaluation(config, trace_path)
    except Exception as error:
        raise click.ClickException(f"evaluation failed: {error}") from error
    click.echo(
        f"Completed {evaluation.total_tasks} tasks; "
        f"resolved={evaluation.resolved_tasks}/{evaluation.total_tasks}"
    )


if __name__ == "__main__":
    swebench_main()
