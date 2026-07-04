from collections.abc import Sequence
from dataclasses import dataclass
import json

import pytest
from pydantic import BaseModel

from agent_from_scratch import agent_llm as agent_llm_module
from agent_from_scratch.agent import (
    Agent,
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V2,
    SUMMARY_PROMPT,
    SUM_KEEP_RECENTS,
    SUMMARIZE_TOKEN_THRESHOLD,
)
from agent_from_scratch.agent_context import (
    AgentStructuredResponse,
    ExecutionContext,
    Event,
    Message,
    AgentToolResult,
    TaskState,
)
from agent_from_scratch.agent_llm import ChatMessage, LlmClient
from agent_from_scratch.tool_common import Tool, ToolCall, ToolContextPolicy


@dataclass(frozen=True)
class AssistantOutput:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class FakeCompletion:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def model_dump(self) -> dict[str, object]:
        return self.response


class FakeGenerator(str):
    def __new__(
        cls,
        outputs: Sequence[AssistantOutput | Exception],
        *,
        token_count: int = 0,
    ) -> "FakeGenerator":
        del token_count
        return super().__new__(cls, f"fake-{id(outputs)}")

    def __init__(
        self,
        outputs: Sequence[AssistantOutput | Exception],
        *,
        token_count: int = 0,
    ) -> None:
        self.outputs = list(outputs)
        self.messages: list[list[ChatMessage]] = []
        self.tool_schemas: list[list[dict[str, object]]] = []
        self.response_formats: list[type[BaseModel] | None] = []
        self.token_count = token_count
        self.model = str(self)
        _FAKE_MODELS[self.model] = self

    def complete(self, **kwargs: object) -> FakeCompletion:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        typed_messages = [
            dict(message) for message in messages if isinstance(message, dict)
        ]
        tools = kwargs.get("tools")
        response_format = kwargs.get("response_format")
        self.messages.append(typed_messages)
        self.tool_schemas.append(list(tools) if isinstance(tools, list) else [])
        self.response_formats.append(
            response_format if isinstance(response_format, type) else None
        )
        output = self.outputs[len(self.messages) - 1]
        if isinstance(output, Exception):
            raise output
        if response_format is AgentStructuredResponse and not output.content.startswith(
            ("{", "<think>")
        ):
            output = AssistantOutput(
                json.dumps(
                    {
                        "answer": output.content,
                    }
                ),
                output.tool_calls,
            )
        tool_calls = [
            {
                "id": tool_call.tool_call_id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for index, tool_call in enumerate(output.tool_calls)
        ]
        return FakeCompletion(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output.content or None,
                            "tool_calls": tool_calls or None,
                        },
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )


_FAKE_MODELS: dict[str, FakeGenerator] = {}


@pytest.fixture(autouse=True)
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    def completion(**kwargs: object) -> FakeCompletion:
        model = kwargs["model"]
        assert isinstance(model, str)
        return _FAKE_MODELS[model].complete(**kwargs)

    def token_counter(**kwargs: object) -> int:
        model = kwargs["model"]
        assert isinstance(model, str)
        return _FAKE_MODELS[model].token_count

    monkeypatch.setattr(agent_llm_module.litellm, "completion", completion)
    monkeypatch.setattr(agent_llm_module.litellm, "token_counter", token_counter)


def tool(name: str, result: str | Exception) -> Tool:
    def execute(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
        if isinstance(result, Exception):
            raise result
        return f"{result}:{arguments.get('q', '')}"

    return Tool(
        schema={
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        },
        execute=execute,
    )


def long_message_context() -> ExecutionContext:
    context = ExecutionContext()
    for index in range(SUM_KEEP_RECENTS + 3):
        message = (
            Message(role="user", content=f"item {index}")
            if index % 2 == 0
            else Message(role="assistant", content=f"item {index}")
        )
        context.add_event(
            Event(
                execution_id=context.execution_id,
                author=message.role,
                content=[message],
            )
        )
    return context


def assert_task_state_message(message: ChatMessage, original_request: str) -> None:
    assert message["role"] == "system"
    assert f"original_request: {original_request}" in str(message["content"])


def test_agent_task_state_message_omits_empty_todos_and_facts() -> None:
    context = ExecutionContext()
    generator = FakeGenerator([AssistantOutput("finished")])
    agent = Agent(LlmClient(generator), [])

    agent.run("question", context=context)

    content = str(generator.messages[0][1]["content"])
    assert content == "Task state:\noriginal_request: question"
    assert "todos:" not in content
    assert "facts:" not in content
    assert "<empty>" not in content


def test_agent_task_state_message_includes_only_non_empty_lists() -> None:
    context = ExecutionContext(
        task_state=TaskState(
            original_request="root",
            facts=["repo uses uv"],
        )
    )
    generator = FakeGenerator([AssistantOutput("finished")])
    agent = Agent(LlmClient(generator), [])

    agent.run("continue", context=context)

    content = str(generator.messages[0][1]["content"])
    assert content == "Task state:\noriginal_request: root\nfacts:\n- repo uses uv"
    assert "todos:" not in content
    assert "<empty>" not in content


def test_agent_run_returns_simple_answer_and_updates_context() -> None:
    context = ExecutionContext()
    generator = FakeGenerator([AssistantOutput("finished")])
    llm = LlmClient(generator, max_tokens=11, temperature=0.2)
    agent = Agent(llm, [], instruction="be brief")

    result = agent.run("question", context=context)

    assert result.output == "finished"
    assert result.context is context
    assert context.final_result == "finished"
    assert context.task_state == TaskState(original_request="question")
    assert context.messages() == [
        Message(role="assistant", content="finished"),
    ]
    assert generator.messages[0][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[0][1] == {"role": "system", "content": "be brief"}
    assert_task_state_message(generator.messages[0][2], "question")
    assert generator.response_formats == [None]
    assert context.events[0].metadata == {}


def test_agent_preserves_llm_error_context_instead_of_returning_final_answer() -> None:
    context = ExecutionContext()
    agent = Agent(LlmClient(FakeGenerator([ValueError("bad tool call")])), [])

    result = agent.run("question", context=context, trace_enabled=True)

    assert result.status == "error"
    assert result.output is None
    assert context.final_result is None
    assert context.state["agent_error"] == "bad tool call"
    assert context.events[-1].content == []
    assert context.events[-1].metadata["llm"]["error"]["message"] == "bad tool call"


def test_agent_trace_enabled_records_llm_and_tool_metadata() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput(
                "checking",
                (ToolCall(name="lookup", arguments={"q": "x"}),),
            ),
            AssistantOutput("done"),
        ]
    )
    context = ExecutionContext()
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    result = agent.run("question", context=context, trace_enabled=True)

    assert result.output == "done"
    assert (
        context.events[0].metadata["llm"]["completion"]["choices"][0]["message"][
            "content"
        ]
        == "checking"
    )
    assert context.events[1].metadata["tools"][0]["name"] == "lookup"
    assert context.events[1].metadata["tools"][0]["arguments"] == {"q": "x"}
    assert context.events[1].metadata["tools"][0]["status"] == "success"
    assert (
        context.events[2].metadata["llm"]["completion"]["choices"][0]["message"][
            "content"
        ]
        == "done"
    )


def test_agent_structured_mode_requests_structured_output() -> None:
    context = ExecutionContext()
    generator = FakeGenerator([AssistantOutput("finished")])
    agent = Agent(LlmClient(generator), [], agent_mode="structured")

    result = agent.run("question", context=context)

    assert result.output == "finished"
    assert generator.messages[0][0] == {"role": "system", "content": SYSTEM_PROMPT_V2}
    assert generator.response_formats == [AgentStructuredResponse]


def test_agent_rejects_unknown_agent_mode() -> None:
    generator = FakeGenerator([AssistantOutput("finished")])

    with pytest.raises(ValueError, match="unsupported agent_mode"):
        Agent(LlmClient(generator), [], agent_mode="unknown")  # type: ignore[arg-type]


def test_agent_run_executes_successful_tool_round_then_final_answer() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput(
                "checking",
                (ToolCall(name="lookup", arguments={"q": "x"}),),
            ),
            AssistantOutput("answer"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    result = agent.run("question")

    assert result.output == "answer"

    assert generator.messages[1][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert_task_state_message(generator.messages[1][1], "question")
    assert generator.messages[1][2:] == [
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Tool result: found:x",
            "tool_call_id": "call_0",
        },
    ]


def test_agent_run_forwards_container_env_to_tools() -> None:
    class FakeContainerEnv:
        pass

    env = FakeContainerEnv()
    seen_envs: list[object] = []

    def execute(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        seen_envs.append(container_env)
        return f"ok:{arguments['q']}"

    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )
    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=execute,
            )
        ],
    )

    result = agent.run("go", container_env=env)  # type: ignore[arg-type]

    assert result.output == "done"
    assert seen_envs == [env]
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": "Tool result: ok:x",
        "tool_call_id": "call_0",
    }


def test_agent_does_not_manage_container_env_lifecycle() -> None:
    class FakeContainerEnv:
        def __init__(self) -> None:
            self.started = False
            self.closed = False

        def start(self, *_args: object, **_kwargs: object) -> None:
            self.started = True

        def close(self) -> None:
            self.closed = True

    env = FakeContainerEnv()
    generator = FakeGenerator([AssistantOutput("done")])
    agent = Agent(LlmClient(generator), [])

    result = agent.run("go", container_env=env)  # type: ignore[arg-type]

    assert result.output == "done"
    assert env.started is False
    assert env.closed is False


@pytest.mark.parametrize(
    ("first_call", "registered_tools", "expected_tool_message"),
    [
        (
            ToolCall(name="missing", arguments={}),
            [],
            "Tool error: unknown tool 'missing'",
        ),
        (
            ToolCall(name="explode", arguments={}),
            [tool("explode", RuntimeError("failure"))],
            "Tool error: 'explode' failed: failure",
        ),
    ],
)
def test_agent_feeds_unknown_tool_and_exceptions_back_for_recovery(
    first_call: ToolCall,
    registered_tools: list[Tool],
    expected_tool_message: str,
) -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (first_call,)),
            AssistantOutput("recovered"),
        ]
    )
    agent = Agent(LlmClient(generator), registered_tools)

    result = agent.run("go")

    assert result.output == "recovered"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": expected_tool_message,
        "tool_call_id": "call_0",
    }


def test_agent_returns_without_final_result_when_tool_round_limit_is_exhausted() -> (
    None
):
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="again", arguments={}),)),
            AssistantOutput("", (ToolCall(name="again", arguments={}),)),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("again", "loop")], max_step=1)

    result = agent.run("loop")

    assert result.output is None
    assert result.context.final_result is None
    assert result.context.current_step == 1


def test_agent_context_records_tool_events_and_final_result() -> None:
    context = ExecutionContext()
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    result = agent.run("go", context=context)

    assert result.output == "done"
    assert result.context is context

    assert context.items() == [
        ToolCall(tool_call_id="call_0", name="lookup", arguments={"q": "x"}),
        AgentToolResult(
            tool_call_id="call_0",
            name="lookup",
            status="success",
            content=["found:x"],
        ),
        Message(role="assistant", content="done"),
    ]
    assert context.final_result == "done"


def test_agent_reuses_task_state_and_records_later_user_messages() -> None:
    context = ExecutionContext()
    generator = FakeGenerator(
        [
            AssistantOutput(
                json.dumps(
                    {
                        "answer": "started",
                        "task_state_update": {
                            "push_todos": ["inspect repo"],
                            "pop_todos": [],
                            "push_facts": ["repo uses uv"],
                            "pop_facts": [],
                        },
                    }
                )
            ),
            AssistantOutput(
                json.dumps(
                    {
                        "answer": "continued",
                        "task_state_update": {
                            "push_todos": [],
                            "pop_todos": [0],
                            "push_facts": ["tests are targeted"],
                            "pop_facts": [],
                        },
                    }
                )
            ),
        ]
    )
    agent = Agent(LlmClient(generator), [], agent_mode="structured")

    first = agent.run("initial task", context=context)
    second = agent.run("continue", context=context)

    assert first.output == "started"
    assert second.output == "continued"
    assert context.task_state == TaskState(
        original_request="initial task",
        todos=[],
        facts=["repo uses uv", "tests are targeted"],
    )
    assert context.messages() == [
        Message(role="assistant", content="started"),
        Message(role="user", content="continue"),
        Message(role="assistant", content="continued"),
    ]
    assert_task_state_message(generator.messages[1][1], "initial task")
    assert "repo uses uv" in str(generator.messages[1][1]["content"])


def test_agent_keeps_latest_tool_answer_raw_in_llm_request() -> None:
    raw_answer = "raw answer with lots of detail"
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_answer(result: AgentToolResult) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content=["compact answer"],
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _, container_env=None: raw_answer,
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": f"Tool result: {raw_answer}",
        "tool_call_id": "call_0",
    }
    assert result.context.items()[1] == AgentToolResult(
        tool_call_id="call_0",
        name="lookup",
        status="success",
        content=[raw_answer],
    )


def test_agent_does_not_summarize_short_history() -> None:
    context = ExecutionContext()
    for index in range(SUM_KEEP_RECENTS + 1):
        context.add_user_message(f"item {index}")
    generator = FakeGenerator([AssistantOutput("done")])
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 1
    assert generator.messages[0][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[0][1:] == [
        {"role": "user", "content": f"item {index}"}
        for index in range(SUM_KEEP_RECENTS + 1)
    ]


def test_agent_does_not_summarize_long_history_below_token_threshold() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [AssistantOutput("done")],
        token_count=SUMMARIZE_TOKEN_THRESHOLD,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 1
    assert generator.messages[0][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[0][1:] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]


def test_agent_summarizes_middle_history_only_in_llm_request() -> None:
    context = long_message_context()
    raw_items = context.items()
    generator = FakeGenerator(
        [
            AssistantOutput("summary text"),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.tool_schemas == [[], []]
    assert generator.messages[0] == [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
    ]
    assert generator.messages[1][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[1][1:] == [
        {
            "role": "system",
            "content": "Conversation summary so far:\nsummary text",
        },
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.items() == [*raw_items, Message(role="assistant", content="done")]


def test_agent_summarization_preserves_task_state_and_summarizes_history_after_it() -> (
    None
):
    context = long_message_context()
    context.task_state = TaskState(original_request="root task")
    generator = FakeGenerator(
        [
            AssistantOutput("summary text"),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert generator.messages[0] == [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
    ]
    assert generator.messages[1][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert_task_state_message(generator.messages[1][1], "root task")
    assert generator.messages[1][2:] == [
        {
            "role": "system",
            "content": "Conversation summary so far:\nsummary text",
        },
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]


def test_agent_falls_back_to_unsummarized_history_on_summary_error() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [
            ValueError("summary failed"),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.messages[1][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[1][1:] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.messages()[-1] == Message(role="assistant", content="done")


def test_agent_falls_back_to_unsummarized_history_without_assistant_summary() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="noop", arguments={}),)),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.messages[1][0] == {"role": "system", "content": SYSTEM_PROMPT_V1}
    assert generator.messages[1][1:] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.messages()[-1] == Message(role="assistant", content="done")


def test_agent_compacts_previous_tool_answer_only_in_llm_request() -> None:
    context = ExecutionContext()
    context.add_user_message("go")
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="agent",
            content=[
                ToolCall(
                    tool_call_id="call_0",
                    name="lookup",
                    arguments={"q": "x"},
                )
            ],
        )
    )
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="tool",
            content=[
                AgentToolResult(
                    tool_call_id="call_0",
                    name="lookup",
                    status="success",
                    content=["raw answer with lots of detail"],
                )
            ],
        )
    )
    context.add_user_message("continue")
    generator = FakeGenerator([AssistantOutput("done")])

    def compact_answer(result: AgentToolResult) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content=["compact answer"],
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _, container_env=None: "unused",
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    agent.step(context)

    assert generator.messages[0][-2] == {
        "role": "tool",
        "content": "Tool result: compact answer",
        "tool_call_id": "call_0",
    }
    assert generator.messages[0][-1] == {"role": "user", "content": "continue"}
    assert context.items()[2] == AgentToolResult(
        tool_call_id="call_0",
        name="lookup",
        status="success",
        content=["raw answer with lots of detail"],
    )


def test_agent_compacts_tool_call_only_in_llm_request() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "raw"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_call(call: ToolCall) -> ToolCall:
        return ToolCall(
            tool_call_id=call.tool_call_id,
            name=call.name,
            arguments={"q": "compact"},
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _, container_env=None: "answer",
                context_policy=ToolContextPolicy(compact_call=compact_call),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assistant_message = generator.messages[1][2]
    assert assistant_message["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"q":"compact"}',
            },
        }
    ]
    assert result.context.items()[0] == ToolCall(
        tool_call_id="call_0",
        name="lookup",
        arguments={"q": "raw"},
    )


def test_agent_uses_raw_item_when_context_policy_fails() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_answer(_: AgentToolResult) -> AgentToolResult:
        raise RuntimeError("bad policy")

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _, container_env=None: "raw answer",
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": "Tool result: raw answer",
        "tool_call_id": "call_0",
    }
