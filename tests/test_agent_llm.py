from typing import Any

import pytest
from pydantic import BaseModel

from agent_from_scratch import agent_llm as agent_llm_module
from agent_from_scratch.agent_context import AgentToolResult, Message
from agent_from_scratch.agent_llm import LlmClient, LlmRequest, build_messages
from agent_from_scratch.tool_common import ToolCall


class FakeCompletion:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def model_dump(self) -> dict[str, object]:
        return self.response


class FakeErrorResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


class FakeUpstreamError(Exception):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__("LLLM returned status 422")
        self.response = FakeErrorResponse(payload)


def completion_response(
    *,
    content: str | None = "answer",
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str = "stop",
) -> FakeCompletion:
    return FakeCompletion(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }
    )


def install_completion(
    monkeypatch: pytest.MonkeyPatch,
    result: FakeCompletion | Exception,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> FakeCompletion:
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agent_llm_module.litellm, "completion", fake_completion)
    return calls


def test_build_messages_converts_context_items_and_preserves_ids() -> None:
    request = LlmRequest(
        content=[
            Message(role="system", content="be useful"),
            Message(role="user", content="question"),
            Message(role="assistant", content="checking"),
            ToolCall(tool_call_id="call_2_0", name="lookup", arguments={"q": "x"}),
            AgentToolResult(
                tool_call_id="call_2_0",
                name="lookup",
                status="success",
                content=["found"],
            ),
        ],
    )

    assert build_messages(request) == [
        {"role": "system", "content": "be useful"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call_2_0",
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
            "content": "Tool result: found",
            "tool_call_id": "call_2_0",
        },
    ]


def test_llm_client_calls_custom_openai_endpoint_and_maps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_completion(
        monkeypatch,
        completion_response(
            content="I will check",
            tool_calls=[
                {
                    "id": "server_call",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        ),
    )
    schema: dict[str, object] = {"type": "function"}
    client = LlmClient(
        "local",
        base_url="http://localhost:8000/v1",
        max_tokens=9,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        enable_thinking=False,
        extra_body={"custom": "value"},
    )

    response = client.complete(
        LlmRequest(
            content=[Message(role="user", content="question")],
            tool_schemas=[schema],
        )
    )

    assert response.error_message is None
    assert response.raw_completion == "I will check"
    assert response.usage_metadata == {
        "prompt_tokens": 3,
        "generated_tokens": 4,
        "finish_reason": "tool_calls",
    }
    assert response.content == [
        Message(role="assistant", content="I will check"),
        ToolCall(
            tool_call_id="server_call",
            name="lookup",
            arguments={"q": "x"},
        ),
    ]
    assert calls == [
        {
            "model": "openai/local",
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 9,
            "temperature": 0.2,
            "timeout": 3600.0,
            "max_retries": 0,
            "base_url": "http://localhost:8000/v1",
            "api_key": "not-needed",
            "top_p": 0.9,
            "tools": [schema],
            "extra_body": {
                "custom": "value",
                "top_k": 20,
                "enable_thinking": False,
            },
        }
    ]


def test_llm_client_preserves_native_litellm_model_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_completion(monkeypatch, completion_response())

    _ = LlmClient("anthropic/claude", api_key="secret").complete(LlmRequest())

    assert calls[0]["model"] == "anthropic/claude"
    assert calls[0]["api_key"] == "secret"
    assert "base_url" not in calls[0]
    assert "extra_body" not in calls[0]


def test_llm_client_complete_returns_trace_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_completion(monkeypatch, completion_response())

    response = LlmClient(
        "local",
        base_url="http://localhost:8000/v1",
        api_key="secret",
    ).complete(
        LlmRequest(
            content=[Message(role="user", content="question")],
            trace_enabled=True,
        )
    )

    assert response.error_message is None
    assert response.trace is not None
    assert response.trace["request"]["messages"] == [
        {"role": "user", "content": "question"}
    ]
    assert response.trace["completion"]["choices"][0]["message"]["content"] == "answer"
    assert "api_key" not in response.trace["client_config"]


def test_llm_client_returns_litellm_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_completion(monkeypatch, ValueError("invalid"))

    response = LlmClient("openai/test").complete(LlmRequest(trace_enabled=True))

    assert response.error_message == "invalid"
    assert response.trace is not None
    assert response.trace["error"] == {
        "type": "ValueError",
        "message": "invalid",
    }


def test_llm_client_preserves_upstream_completion_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_error = {
        "type": "ValueError",
        "message": "bad tool call",
        "raw_completion": "<tool_call>bad</tool_call>",
    }
    server_trace = {
        "raw_completion": "<tool_call>bad</tool_call>",
        "parse_error": parse_error,
    }
    _ = install_completion(
        monkeypatch,
        FakeUpstreamError(
            {
                "detail": {
                    "type": "completion_parse_error",
                    "parse_error": parse_error,
                    "trace": server_trace,
                }
            }
        ),
    )

    response = LlmClient("openai/test").complete(LlmRequest(trace_enabled=True))

    assert response.error_message == "LLLM returned status 422"
    assert response.trace is not None
    assert response.trace["parse_error"] == parse_error
    assert response.trace["server_trace"] == server_trace


def test_llm_client_counts_tokens_with_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_token_counter(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 17

    monkeypatch.setattr(
        agent_llm_module.litellm,
        "token_counter",
        fake_token_counter,
    )
    schema: dict[str, object] = {"type": "function"}
    request = LlmRequest(
        content=[Message(role="user", content="question")],
        tool_schemas=[schema],
    )

    count = LlmClient("local", base_url="http://localhost/v1").count_tokens(request)

    assert count == 17
    assert calls == [
        {
            "model": "openai/local",
            "messages": [{"role": "user", "content": "question"}],
            "tools": [schema],
        }
    ]


class StructuredProbe(BaseModel):
    answer: str
    count: int


@pytest.mark.parametrize(
    "content",
    [
        '{"answer":"ok","count":2}',
        '<think>hidden</think>\n{"answer":"ok","count":2}',
    ],
)
def test_llm_client_forwards_and_parses_response_format(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    calls = install_completion(monkeypatch, completion_response(content=content))

    response = LlmClient("openai/test").complete(
        LlmRequest(response_format=StructuredProbe)
    )

    assert response.error_message is None
    assert response.parsed == StructuredProbe(answer="ok", count=2)
    assert calls[0]["response_format"] is StructuredProbe


def test_llm_client_rejects_invalid_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_completion(
        monkeypatch,
        completion_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "not-json"},
                }
            ],
        ),
    )

    response = LlmClient("openai/test").complete(LlmRequest())

    assert response.error_message is not None
    assert "Expecting value" in response.error_message
