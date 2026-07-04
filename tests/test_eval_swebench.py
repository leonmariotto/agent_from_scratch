from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest

from agent_from_scratch import eval_swebench
from agent_from_scratch.agent_context import AgentResult, ExecutionContext
from agent_from_scratch.eval_swebench import (
    CommandResult,
    SwebenchAgent,
    SwebenchResult,
    SwebenchTask,
    evaluate_swebench_agent,
    export_swebench_predictions,
    load_swebench_tasks,
)


class TinyDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)

    def select(self, selected: range) -> "TinyDataset":
        return TinyDataset([self.rows[index] for index in selected])

    def filter(self, predicate: Any) -> "TinyDataset":
        return TinyDataset([row for row in self.rows if predicate(row)])


class FakeRunner:
    def __init__(self, results: list[SwebenchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[SwebenchTask, SwebenchAgent]] = []

    def run(self, task: SwebenchTask, agent: SwebenchAgent) -> SwebenchResult:
        self.calls.append((task, agent))
        if self.results:
            return self.results.pop(0)
        return SwebenchResult(
            instance_id=task.instance_id,
            repo=task.repo,
            base_commit=task.base_commit,
            problem_statement=task.problem_statement,
            agent_patch="diff --git a/file.py b/file.py\n",
            resolved=True,
            elapsed_seconds=1.0,
            test_result=CommandResult(
                command=("python", "-m", "pytest"),
                returncode=0,
                stdout="passed",
                stderr="",
                elapsed_seconds=0.5,
            ),
            artifact_dir=None,
            error=None,
        )


def _row(
    instance_id: str,
    *,
    repo: str = "owner/project",
    base_commit: str = "abc123",
    problem_statement: str = "Fix the bug.",
    hints_text: object = "",
    created_at: object = "2024-01-01",
    version: object = "1.0",
    environment_setup_commit: object = "",
    patch: object = "gold patch",
    test_patch: object = "test patch",
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "created_at": created_at,
        "version": version,
        "environment_setup_commit": environment_setup_commit,
        "patch": patch,
        "test_patch": test_patch,
    }


def _patch_dataset(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> TinyDataset:
        calls.append({"args": args, "kwargs": kwargs})
        return TinyDataset(rows)

    monkeypatch.setattr(eval_swebench, "load_dataset", fake_load_dataset)
    return calls


def test_load_swebench_tasks_uses_default_dataset_and_maps_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_dataset(monkeypatch, [_row("task-1")])

    tasks = load_swebench_tasks()

    assert calls[0]["args"] == ("princeton-nlp/SWE-bench_Lite",)
    assert calls[0]["kwargs"] == {"split": "test"}
    assert tasks == [
        SwebenchTask(
            instance_id="task-1",
            repo="owner/project",
            base_commit="abc123",
            problem_statement="Fix the bug.",
            hints_text=None,
            created_at="2024-01-01",
            version="1.0",
            environment_setup_commit=None,
            patch="gold patch",
            test_patch="test patch",
            metadata=_row("task-1"),
        )
    ]


def test_load_swebench_tasks_applies_instance_filter_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1"), _row("task-2"), _row("task-3")])

    tasks = load_swebench_tasks(instance_ids=["task-1", "task-3"], limit=1)

    assert [task.instance_id for task in tasks] == ["task-1"]


def test_load_swebench_tasks_orders_filter_shuffle_offset_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(
        monkeypatch,
        [_row("skip"), _row("task-1"), _row("task-2"), _row("task-3")],
    )

    tasks = load_swebench_tasks(
        instance_ids=["task-1", "task-2", "task-3"],
        shuffle=True,
        shuffle_seed=0,
        offset=1,
        limit=1,
    )

    assert [task.instance_id for task in tasks] == ["task-3"]


def test_load_swebench_tasks_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        load_swebench_tasks(limit=-1)


def test_load_swebench_tasks_rejects_missing_required_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("task-1")
    row["test_patch"] = ""
    _patch_dataset(monkeypatch, [row])

    with pytest.raises(ValueError, match="test_patch"):
        load_swebench_tasks()


def test_evaluate_swebench_agent_aggregates_runner_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1"), _row("task-2")])
    failing_result = SwebenchResult(
        instance_id="task-2",
        repo="owner/project",
        base_commit="abc123",
        problem_statement="Fix the bug.",
        agent_patch="",
        resolved=False,
        elapsed_seconds=2.0,
        test_result=CommandResult(
            command=("python", "-m", "pytest"),
            returncode=1,
            stdout="",
            stderr="failed",
            elapsed_seconds=0.5,
        ),
        artifact_dir=None,
        error=None,
    )
    runner = FakeRunner(
        results=[
            FakeRunner().run(load_swebench_tasks(limit=1)[0], lambda t, p: None),
            failing_result,
        ]
    )

    evaluation = evaluate_swebench_agent(lambda task, path: None, runner=runner)

    assert len(runner.calls) == 2
    assert evaluation.total_tasks == 2
    assert evaluation.resolved_tasks == 1
    assert evaluation.failed_tasks == 1
    assert evaluation.error_tasks == 0
    assert evaluation.resolved_rate == 0.5


def test_evaluate_swebench_agent_captures_runner_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1")])

    class ExplodingRunner:
        def run(self, task: SwebenchTask, agent: SwebenchAgent) -> SwebenchResult:
            raise RuntimeError("container failed")

    evaluation = evaluate_swebench_agent(
        lambda task, path: None,
        runner=ExplodingRunner(),
    )

    assert evaluation.total_tasks == 1
    assert evaluation.resolved_tasks == 0
    assert evaluation.error_tasks == 1
    assert evaluation.results[0].error == "container failed"


def test_evaluate_swebench_agent_writes_full_jsonl_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1")])
    output_path = tmp_path / "results.jsonl"

    evaluate_swebench_agent(
        lambda task, path: None,
        runner=FakeRunner(),
        output_path=output_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["instance_id"] == "task-1"
    assert rows[0]["resolved"] is True
    assert rows[0]["test_result"]["returncode"] == 0


def test_export_swebench_predictions_writes_prediction_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1")])
    evaluation = evaluate_swebench_agent(lambda task, path: None, runner=FakeRunner())
    output_path = tmp_path / "predictions.jsonl"

    export_swebench_predictions(
        evaluation,
        output_path,
        model_name_or_path="local-agent",
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "instance_id": "task-1",
            "model_name_or_path": "local-agent",
            "model_patch": "diff --git a/file.py b/file.py\n",
        }
    ]


def test_docker_runner_uses_workspace_diff_as_agent_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_exec(
        runner: eval_swebench.DockerSwebenchRunner,
        host_root: Path,
        command: tuple[str, ...],
        *,
        workdir: str,
    ) -> CommandResult:
        commands.append(command)
        if command == ("git", "-C", "repo", "diff", "--binary"):
            return CommandResult(command, 0, "workspace patch", "", 0.1)
        return CommandResult(command, 0, "ok", "", 0.1)

    monkeypatch.setattr(eval_swebench.DockerSwebenchRunner, "_docker_exec", fake_exec)
    runner = eval_swebench.DockerSwebenchRunner(artifacts_dir=tmp_path)
    task = load_swebench_tasks.__globals__["_row_to_task"](_row("task-1"))

    result = runner.run(task, lambda task, path: None)

    assert result.agent_patch == "workspace patch"
    assert result.resolved is True
    assert (tmp_path / "task-1" / "agent.patch").read_text() == "workspace patch"
    assert ("git", "-C", "repo", "apply", "/workspace/agent.patch") not in commands


def test_docker_runner_counts_test_failure_as_unresolved_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_exec(
        runner: eval_swebench.DockerSwebenchRunner,
        host_root: Path,
        command: tuple[str, ...],
        *,
        workdir: str,
    ) -> CommandResult:
        if command == ("git", "-C", "repo", "diff", "--binary"):
            return CommandResult(command, 0, "workspace patch", "", 0.1)
        if command == ("python", "-m", "pytest"):
            return CommandResult(command, 1, "", "failed", 0.1)
        return CommandResult(command, 0, "ok", "", 0.1)

    monkeypatch.setattr(eval_swebench.DockerSwebenchRunner, "_docker_exec", fake_exec)
    runner = eval_swebench.DockerSwebenchRunner()
    task = load_swebench_tasks.__globals__["_row_to_task"](_row("task-1"))

    result = runner.run(task, lambda task, path: None)

    assert result.resolved is False
    assert result.error is None
    assert result.test_result is not None
    assert result.test_result.returncode == 1


def test_docker_runner_ownership_restore_failure_does_not_fail_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_exec(
        runner: eval_swebench.DockerSwebenchRunner,
        host_root: Path,
        command: tuple[str, ...],
        *,
        workdir: str,
    ) -> CommandResult:
        if command == ("git", "-C", "repo", "diff", "--binary"):
            return CommandResult(command, 0, "workspace patch", "", 0.1)
        if command[0] == "chown":
            return CommandResult(command, 1, "", "permission denied", 0.1)
        return CommandResult(command, 0, "ok", "", 0.1)

    monkeypatch.setattr(eval_swebench.DockerSwebenchRunner, "_docker_exec", fake_exec)
    runner = eval_swebench.DockerSwebenchRunner()
    task = load_swebench_tasks.__globals__["_row_to_task"](_row("task-1"))

    result = runner.run(task, lambda task, path: None)

    assert result.resolved is True
    assert result.error is None


def _application_config(tmp_path: Path) -> eval_swebench.SwebenchAppConfig:
    return eval_swebench.SwebenchAppConfig.model_validate(
        {
            "model": {"model": "test", "base_url": "http://localhost/v1"},
            "agent": {
                "prompt_template": (
                    "Fix {problem_statement} in the repository at {repo_path}"
                )
            },
            "tools": {"enabled": ["compute", "python"]},
            "container": {"artifacts_dir": tmp_path / "artifacts"},
            "dataset": {"entry_count": 1},
        }
    )


def test_swebench_application_config_help_and_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "evaluation.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
model:
  model: test
container:
  artifacts_dir: ../artifacts
  test_command: python -m pytest -q
dataset:
  entry_count: 1
  instance_ids: task-1
""",
        encoding="utf-8",
    )

    config = eval_swebench.load_swebench_config(config_path)
    assert config.container.artifacts_dir == (tmp_path / "artifacts").resolve()
    assert config.container.test_command == ["python", "-m", "pytest", "-q"]
    assert config.dataset.instance_ids == ["task-1"]
    with pytest.raises(ValueError, match="extra_forbidden"):
        eval_swebench.SwebenchAppConfig.model_validate(
            {
                "model": {"model": "test"},
                "dataset": {"entry_count": 1, "unknown": True},
            }
        )
    with pytest.raises(ValueError, match=r"must contain \{repo_path\}"):
        eval_swebench.SwebenchAgentConfig(prompt_template="{problem_statement}")

    help_result = CliRunner().invoke(eval_swebench.swebench_main, ["--help"])
    normalized_help = " ".join(help_result.output.split())
    assert help_result.exit_code == 0
    assert "instance_ids, seeded shuffle, entry_offset" in normalized_help
    assert "LLLM_API_KEY" in normalized_help
    assert "compute and python" in normalized_help
    assert "checkpoint" in normalized_help

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("model:\n  model: test\nunknown: true\n", encoding="utf-8")
    bad_result = CliRunner().invoke(
        eval_swebench.swebench_main,
        ["--config", str(bad_path), "--trace", str(tmp_path / "trace.json")],
    )
    assert bad_result.exit_code == 2
    assert "invalid configuration" in bad_result.output


def test_swebench_application_container_prompt_trace_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = load_swebench_tasks.__globals__["_row_to_task"](_row("task-1"))
    monkeypatch.setattr(eval_swebench, "load_swebench_tasks", lambda **kwargs: [task])
    prompts: list[str] = []
    containers: list[Any] = []

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, prompt: str, **kwargs: object) -> AgentResult:
            prompts.append(prompt)
            context = ExecutionContext()
            context.add_user_message(prompt)
            return AgentResult(output="done", context=context)

    class FakeContainer:
        def __init__(self, **kwargs: object) -> None:
            self.started: tuple[list[Path], bool] | None = None
            self.closed = False
            containers.append(self)

        def start(self, mounts: list[Path], network: bool = False) -> None:
            self.started = (mounts, network)

        def close(self) -> None:
            self.closed = True

    class FakeDockerRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(
            self,
            current_task: SwebenchTask,
            agent: SwebenchAgent,
        ) -> SwebenchResult:
            repo_path = tmp_path / "repo"
            repo_path.mkdir()
            agent(current_task, repo_path)
            return SwebenchResult(
                current_task.instance_id,
                current_task.repo,
                current_task.base_commit,
                current_task.problem_statement,
                "agent patch",
                True,
                1.0,
                CommandResult(("pytest",), 0, "passed", "", 0.5),
                None,
                None,
            )

    monkeypatch.setattr(eval_swebench, "Agent", FakeAgent)
    monkeypatch.setattr(eval_swebench, "ContainerEnv", FakeContainer)
    monkeypatch.setattr(eval_swebench, "DockerSwebenchRunner", FakeDockerRunner)
    monkeypatch.setenv("LLLM_API_KEY", "do-not-write-me")
    writes = 0
    real_write = eval_swebench._swebench_atomic_write

    def counting_write(path: Path, document: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        real_write(path, document)

    monkeypatch.setattr(eval_swebench, "_swebench_atomic_write", counting_write)
    trace_path = tmp_path / "nested" / "trace.json"
    evaluation = eval_swebench.run_swebench_evaluation(
        _application_config(tmp_path),
        trace_path,
    )

    assert evaluation.resolved_tasks == 1
    assert containers[0].started == ([tmp_path / "repo"], True)
    assert containers[0].closed is True
    assert "/tmp/0" in prompts[0]
    assert str(tmp_path / "repo") not in prompts[0]
    assert writes == 2
    trace_text = trace_path.read_text(encoding="utf-8")
    trace = json.loads(trace_text)
    assert "do-not-write-me" not in trace_text
    assert trace["run"]["selected_instance_ids"] == ["task-1"]
    assert trace["entries"][0]["result"]["agent_patch"] == "agent patch"
    assert trace["entries"][0]["agent"]["context"] is not None
