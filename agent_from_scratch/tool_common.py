from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol
from dataclasses import dataclass
from collections.abc import Callable
from pydantic import BaseModel

if TYPE_CHECKING:
    from .agent_context import ContentItem
    from .agent_context import AgentToolResult
    from .container_env import ContainerEnv


class ToolExecutor(Protocol):
    def __call__(
        self,
        arguments: dict[str, object],
        container_env: "ContainerEnv | None" = None,
    ) -> str: ...


@dataclass(frozen=True)
class ToolContextPolicy:
    """
    Optional request-only compaction hooks for one tool.

    Hooks receive the raw context item and return the item to put in the next
    LLM request. They must not mutate the stored execution context.
    """

    compact_call: Callable[[ToolCall], ContentItem] | None = None
    compact_answer: Callable[[AgentToolResult], ContentItem] | None = None


@dataclass(frozen=True)
class Tool:
    """A function schema exposed to the model and its local implementation."""

    schema: dict[str, object]
    execute: ToolExecutor
    context_policy: ToolContextPolicy | None = None


def execute_tool(
    tool: Tool,
    arguments: dict[str, object],
    container_env: "ContainerEnv | None" = None,
) -> str:
    """Execute a tool with the current optional container environment."""
    return tool.execute(arguments, container_env)


class ToolCall(BaseModel):
    """A parsed assistant request to call one tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = ""
    name: str
    arguments: dict[str, object]

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            if len(args) != 2 or "name" in data or "arguments" in data:
                raise TypeError(
                    "ToolCall accepts either ToolCall(name, arguments) or keyword "
                    "arguments"
                )
            data["name"] = args[0]
            data["arguments"] = args[1]
        super().__init__(**data)

    def __str__(self) -> str:
        return f"[{self.name}, {self.arguments}]"
