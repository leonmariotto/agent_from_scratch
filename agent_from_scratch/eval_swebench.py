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
import shlex
import subprocess
import tempfile
import time
from typing import Any, Protocol, cast

from loguru import logger

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
    instance_ids: Sequence[str] | None = None,
) -> list[SwebenchTask]:
    """Load and normalize SWE-bench tasks from Hugging Face datasets."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    logger.info("Loading SWE-bench dataset={} split={}", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    if instance_ids is not None:
        selected_ids = set(instance_ids)

        def has_selected_id(row: object) -> bool:
            return cast(DatasetRow, row)["instance_id"] in selected_ids

        dataset = dataset.filter(has_selected_id)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    tasks = [_row_to_task(cast(DatasetRow, row)) for row in dataset]
    logger.info("Loaded {} SWE-bench tasks", len(tasks))
    return tasks


def evaluate_swebench_agent(
    agent: SwebenchAgent,
    *,
    dataset_name: str = DEFAULT_SWEBENCH_DATASET_ID,
    split: str = DEFAULT_SWEBENCH_SPLIT,
    limit: int | None = None,
    instance_ids: Sequence[str] | None = None,
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
        instance_ids=instance_ids,
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
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.docker_image = docker_image
        self.test_command = tuple(test_command) if test_command is not None else None
        self.timeout_seconds = timeout_seconds
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
        docker_command = (
            "docker",
            "run",
            "--rm",
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
