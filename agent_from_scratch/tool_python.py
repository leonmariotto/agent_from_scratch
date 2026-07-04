"""
Raw Python interpreter tool for model tool-use loops.

This tool intentionally exposes normal Python syntax rather than a restricted
calculator expression language.  It runs code in a caller-provided container
environment so exceptions, prints, imports, and statements behave like a
regular Python script while filesystem/network effects stay isolated by the
container setup.
"""

from __future__ import annotations

import copy

from loguru import logger

from .container_env import ContainerEnv
from .tool_common import Tool

_DEFAULT_TIMEOUT_SECONDS = 10
_MAX_TIMEOUT_SECONDS = 30
_MAX_OUTPUT_CHARS = 12000

PYTHON_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run raw Python code in an interpreter. Use this for calculations, "
            "data manipulation, quick scripts, loops, imports, and checking "
            "Python behavior. Print values you want to see."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute as a script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout from 1 to 30 seconds.",
                },
            },
            "required": ["code"],
        },
    },
}


def python_tool() -> Tool:
    """Return the ready-to-register ``python`` tool."""
    return Tool(schema=copy.deepcopy(PYTHON_TOOL_SCHEMA), execute=execute_python)


def execute_python(
    arguments: dict[str, object],
    container_env: ContainerEnv | None = None,
) -> str:
    """Execute raw Python code with a ``{"code": "..."}`` argument."""
    code = arguments.get("code")
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    if not code.strip():
        raise ValueError("code must not be empty")
    if container_env is None:
        raise ValueError("python tool requires container_env")
    timeout_seconds = _optional_timeout(arguments)
    logger.info(
        "Python tool execution started with {} code chars and timeout={}s",
        len(code),
        timeout_seconds,
    )

    try:
        completed = container_env.exec(
            ["python", "-c", code],
            timeout=timeout_seconds,
        )
    except TimeoutError as error:
        logger.info("Python tool execution timed out after {}s", timeout_seconds)
        raise ValueError(
            f"python execution timed out after {timeout_seconds}s"
        ) from error
    except RuntimeError as error:
        logger.info("Python tool execution failed: {}", error)
        raise ValueError(f"python execution failed: {error}") from error

    logger.info(
        "Python tool execution completed with returncode={}, stdout_chars={}, stderr_chars={}",
        completed.returncode,
        len(completed.stdout),
        len(completed.stderr),
    )
    return _format_result(completed.returncode, completed.stdout, completed.stderr)


def _optional_timeout(arguments: dict[str, object]) -> int:
    value = arguments.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("timeout_seconds must be an integer")
    if value < 1 or value > _MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds must be between 1 and 30")
    return value


def _format_result(returncode: int, stdout: str, stderr: str) -> str:
    parts = [f"Exit code: {returncode}"]
    if stdout:
        parts.append(f"stdout:\n{_truncate(stdout)}")
    if stderr:
        parts.append(f"stderr:\n{_truncate(stderr)}")
    if len(parts) == 1:
        parts.append("stdout: <empty>")
    return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text.rstrip()
    logger.info(
        "Python tool output truncated from {} to {} chars",
        len(text),
        _MAX_OUTPUT_CHARS,
    )
    marker = "\n[truncated]"
    return text[: _MAX_OUTPUT_CHARS - len(marker)].rstrip() + marker
