import subprocess

import pytest
from loguru import logger

from ..LLLM.tool_common import Tool
from ..LLLM.tool_python import execute_python, python_tool


class FakeContainerEnv:
    def __init__(
        self,
        result: subprocess.CompletedProcess[str] | BaseException | None = None,
    ) -> None:
        self.result = result or subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=0,
            stdout="",
            stderr="",
        )
        self.calls: list[dict[str, object]] = []

    def exec(
        self,
        command: list[str],
        *,
        input: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append({"command": command, "input": input, "timeout": timeout})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_python_tool_returns_registered_tool() -> None:
    tool = python_tool()

    assert isinstance(tool, Tool)
    assert tool.schema["type"] == "function"
    function = tool.schema["function"]
    assert isinstance(function, dict)
    assert function["name"] == "python"


def test_execute_python_runs_raw_python_code() -> None:
    env = FakeContainerEnv(
        subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=0,
            stdout="6\n9.0\n",
            stderr="",
        )
    )

    output = execute_python(
        {
            "code": (
                "import math\n"
                "values = [1, 2, 3]\n"
                "print(sum(values))\n"
                "print(math.sqrt(81))"
            )
        },
        env,  # type: ignore[arg-type]
    )

    assert output == "Exit code: 0\nstdout:\n6\n9.0"
    assert env.calls == [
        {
            "command": [
                "python",
                "-c",
                (
                    "import math\n"
                    "values = [1, 2, 3]\n"
                    "print(sum(values))\n"
                    "print(math.sqrt(81))"
                ),
            ],
            "input": None,
            "timeout": 10,
        }
    ]


def test_execute_python_returns_tracebacks_without_raising() -> None:
    env = FakeContainerEnv(
        subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=1,
            stdout="before\n",
            stderr="Traceback...\nZeroDivisionError\n",
        )
    )

    output = execute_python({"code": "print('before')\n1 / 0"}, env)  # type: ignore[arg-type]

    assert "Exit code: 1" in output
    assert "stdout:\nbefore" in output
    assert "stderr:" in output
    assert "ZeroDivisionError" in output


def test_execute_python_reports_empty_output() -> None:
    env = FakeContainerEnv()
    assert execute_python({"code": "x = 1"}, env) == "Exit code: 0\nstdout: <empty>"  # type: ignore[arg-type]


def test_execute_python_requires_container_env() -> None:
    with pytest.raises(ValueError, match="requires container_env"):
        execute_python({"code": "print(1)"})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"code": 1},
        {"code": ""},
        {"code": "   "},
        {"code": "print(1)", "timeout_seconds": True},
        {"code": "print(1)", "timeout_seconds": 0},
        {"code": "print(1)", "timeout_seconds": 31},
    ],
)
def test_execute_python_validates_arguments(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        execute_python(arguments)


def test_execute_python_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    env = FakeContainerEnv(TimeoutError("container command timed out after 2s"))

    with pytest.raises(ValueError, match="timed out"):
        execute_python({"code": "while True: pass", "timeout_seconds": 2}, env)  # type: ignore[arg-type]


def test_execute_python_reports_container_execution_failure() -> None:
    env = FakeContainerEnv(RuntimeError("container env has not been started"))

    with pytest.raises(ValueError, match="container env has not been started"):
        execute_python({"code": "print(1)"}, env)  # type: ignore[arg-type]


def test_execute_python_truncates_large_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    env = FakeContainerEnv(
        subprocess.CompletedProcess(
            args=["uv", "run", "python", "-c", "..."],
            returncode=0,
            stdout="x" * 13000,
            stderr="",
        )
    )

    output = execute_python({"code": "print('large')"}, env)  # type: ignore[arg-type]

    assert output.endswith("[truncated]")
    assert len(output) < 12100


def test_execute_python_logs_execution_summary() -> None:
    logs: list[str] = []
    sink_id = logger.add(lambda message: logs.append(str(message)), level="INFO")

    try:
        env = FakeContainerEnv(
            subprocess.CompletedProcess(
                args=["python", "-c", "..."],
                returncode=0,
                stdout="123\n",
                stderr="",
            )
        )
        execute_python({"code": "print(123)"}, env)  # type: ignore[arg-type]
    finally:
        logger.remove(sink_id)

    text = "".join(logs)
    assert "Python tool execution started" in text
    assert "Python tool execution completed with returncode=0" in text
