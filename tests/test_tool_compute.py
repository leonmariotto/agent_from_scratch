import subprocess
from collections.abc import Callable

import pytest

from ..LLLM.tool_common import Tool
from ..LLLM.tool_compute import compute_tool, execute_compute


def test_compute_tool_returns_registered_bc_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run(stdout="4"),
    )

    tool = compute_tool()

    assert isinstance(tool, Tool)
    assert tool.schema["type"] == "function"
    function = tool.schema["function"]
    assert isinstance(function, dict)
    assert function["name"] == "compute"
    assert "bc syntax" in str(function["description"])
    assert tool.execute({"expression": "2 + 2"}) == "4"


def test_execute_compute_forwards_expression_to_bc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="3973\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert execute_compute({"expression": "137 * 29"}) == "3973"
    assert calls == [
        {
            "command": ["bc", "-l"],
            "input": "137 * 29\n",
            "text": True,
            "capture_output": True,
            "timeout": 5,
            "check": False,
        }
    ]


def test_execute_compute_rewrites_pi_for_bc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(input)
        return subprocess.CompletedProcess(command, 0, stdout="1548.4512\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert execute_compute({"expression": "scale=4; 4*pi*11.1^2"}) == "1548.4512"
    assert calls == ["scale=4; 4*4*a(1)*11.1^2\n"]


def test_execute_compute_rewrites_unicode_pi_for_bc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(input)
        return subprocess.CompletedProcess(command, 0, stdout="1548.4512\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert execute_compute({"expression": "scale=4; 4*π*11.1^2"}) == "1548.4512"
    assert calls == ["scale=4; 4*4*a(1)*11.1^2\n"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"expression": 12},
        {"expression": ""},
        {"expression": "   "},
    ],
)
def test_execute_compute_validates_expression_argument(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="expression"):
        execute_compute(arguments)


def test_execute_compute_reports_missing_bc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_bc(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_bc)

    with pytest.raises(ValueError, match="bc executable was not found"):
        execute_compute({"expression": "2 + 2"})


def test_execute_compute_reports_bc_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run(returncode=1, stderr="syntax error"),
    )

    with pytest.raises(ValueError, match="bc error: syntax error"):
        execute_compute({"expression": "2 +"})


def test_execute_compute_reports_bc_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(ValueError, match="bc execution timed out"):
        execute_compute({"expression": "while (1) 1"})


def _fake_run(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return fake_run
