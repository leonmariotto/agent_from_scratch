from loguru import logger

from ..LLLM.agent_context import (
    Event,
    ExecutionContext,
    Message,
    AgentToolResult,
    TaskState,
    UpdateTaskState,
)
from ..LLLM.tool_common import ToolCall


def test_execution_context_records_and_flattens_events() -> None:
    context = ExecutionContext()

    user_message = context.add_user_message("hello")
    agent_items = [
        Message(role="assistant", content="checking"),
        ToolCall(tool_call_id="call_0_0", name="lookup", arguments={"q": "x"}),
        AgentToolResult(
            tool_call_id="call_0_0",
            name="lookup",
            status="success",
            content=["found"],
        ),
    ]
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="assistant",
            content=agent_items,
        )
    )
    context.final_result = "done"

    assert user_message == Message(role="user", content="hello")
    assert [event.author for event in context.events] == ["user", "assistant"]
    assert context.items() == [user_message, *agent_items]
    assert context.messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="checking"),
    ]
    assert context.final_result == "done"


def test_task_state_applies_push_pop_operations_and_logs() -> None:
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        task_state = TaskState(
            original_request="build it",
            todos=["old todo", "keep todo", "last todo"],
            facts=["old fact", "keep fact"],
        )

        task_state.apply_update(
            UpdateTaskState(
                push_todos=["new todo", "  "],
                pop_todos=[2, 0, 99],
                push_facts=["new fact"],
                pop_facts=[0],
            )
        )
    finally:
        logger.remove(sink_id)

    assert task_state == TaskState(
        original_request="build it",
        todos=["keep todo", "new todo"],
        facts=["keep fact", "new fact"],
    )
    joined_logs = "\n".join(logs)
    assert "task_state push target=todos value='new todo'" in joined_logs
    assert "task_state push skipped empty target=todos" in joined_logs
    assert "task_state pop target=todos index=2 value='last todo'" in joined_logs
    assert "task_state pop target=todos index=0 value='old todo'" in joined_logs
    assert "task_state pop ignored target=todos invalid_index=99" in joined_logs
    assert "task_state push target=facts value='new fact'" in joined_logs
    assert "task_state pop target=facts index=0 value='old fact'" in joined_logs
