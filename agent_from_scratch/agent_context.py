"""
Agent execution context central storage, and its internal types.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal
from loguru import logger

from .tool_common import ToolCall


class Message(BaseModel):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str

    def __str__(self) -> str:
        return f"[{self.role}: {self.content}]"


class AgentToolResult(BaseModel):
    """
    Result from tool execution.
    Built by agent before feeding the ExecutionContext.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list[Any]


def _empty_int_list() -> list[int]:
    return []


class UpdateTaskState(BaseModel):
    """
    Structured update operations for task-state lists.
    todos is a list of string. Push todos can push a list of string, pop_todos can remove
    several string of the list according to their index.
    same for facts list.
    """

    push_todos: list[str] = Field(default_factory=list)
    pop_todos: list[int] = Field(default_factory=_empty_int_list)
    push_facts: list[str] = Field(default_factory=list)
    pop_facts: list[int] = Field(default_factory=_empty_int_list)


class TaskState(BaseModel):
    """
    Durable task state injected into agent context.
    We use it to build a system message at the start of the context.
    """

    original_request: str
    todos: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    def apply_update(self, update: UpdateTaskState) -> None:
        """Apply push/pop operations to todos and facts."""
        self._push_values("todos", update.push_todos)
        self._pop_values("todos", update.pop_todos)
        self._push_values("facts", update.push_facts)
        self._pop_values("facts", update.pop_facts)

    def _push_values(
        self, target: Literal["todos", "facts"], values: list[str]
    ) -> None:
        items = self.todos if target == "todos" else self.facts
        for value in values:
            clean_value = value.strip()
            if not clean_value:
                logger.info("task_state push skipped empty target={}", target)
                continue
            items.append(clean_value)
            logger.info("task_state push target={} value={!r}", target, clean_value)

    def _pop_values(
        self, target: Literal["todos", "facts"], indexes: list[int]
    ) -> None:
        items = self.todos if target == "todos" else self.facts
        for index in sorted(set(indexes), reverse=True):
            if 0 <= index < len(items):
                value = items.pop(index)
                logger.info(
                    "task_state pop target={} index={} value={!r}",
                    target,
                    index,
                    value,
                )
            else:
                logger.info(
                    "task_state pop ignored target={} invalid_index={}",
                    target,
                    index,
                )


class AgentStructuredResponse(BaseModel):
    """
    Structured assistant response carrying visible answer and state updates.
    That's the formated output that is forced at generation.
    """

    answer: str
    task_state_update: UpdateTaskState | None = None


ContentItem = Message | ToolCall | AgentToolResult


def _empty_content() -> list[ContentItem]:
    return []


def _empty_events() -> list["Event"]:
    return []


def _empty_state() -> dict[str, Any]:
    return {}


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: Sequence[ContentItem] = Field(default_factory=_empty_content)
    metadata: dict[str, Any] = Field(default_factory=_empty_state)


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=_empty_events)
    current_step: int = 0
    state: dict[str, Any] = field(default_factory=_empty_state)
    final_result: str | BaseModel | None = None
    task_state: TaskState | None = None

    def add_event(self, event: Event) -> None:
        """Append an event to the execution history."""
        self.events.append(event)

    def add_user_message(self, content: str) -> Message:
        """Record a user message and return the stored item."""
        message = Message(role="user", content=content)
        self.add_event(
            Event(
                execution_id=self.execution_id,
                author="user",
                content=[message],
            )
        )
        return message

    def items(self) -> list[ContentItem]:
        """Return all content items in event order."""
        return [item for event in self.events for item in event.content]

    def messages(self) -> list[Message]:
        """Return all text messages in event order."""
        return [item for item in self.items() if isinstance(item, Message)]

    def increment_step(self) -> None:
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""

    output: Any  # str | BaseModel
    context: ExecutionContext
    status: str = "complete"  # "complete" | "pending" | "error"
