"""LLM communication layer for the context-aware agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import litellm
from pydantic import BaseModel, ValidationError

from .agent_context import AgentToolResult, ContentItem, Message
from .tool_common import ToolCall

ChatMessage = dict[str, object]


def _empty_content() -> list[ContentItem]:
    return []


def _empty_tool_schemas() -> list[dict[str, object]]:
    return []


def _empty_usage_metadata() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class LlmRequest:
    """One assistant-turn request."""

    content: list[ContentItem] = field(default_factory=_empty_content)
    tool_schemas: list[dict[str, object]] = field(default_factory=_empty_tool_schemas)
    response_format: type[BaseModel] | None = None
    trace_enabled: bool = False


@dataclass(frozen=True)
class LlmResponse:
    """One assistant-turn response."""

    content: list[ContentItem] = field(default_factory=_empty_content)
    raw_completion: str = ""
    usage_metadata: dict[str, object] = field(default_factory=_empty_usage_metadata)
    error_message: str | None = None
    parsed: BaseModel | None = None
    trace: dict[str, object] | None = None


def build_messages(request: LlmRequest) -> list[ChatMessage]:
    """Convert agent history to OpenAI-compatible chat messages."""
    messages: list[ChatMessage] = []
    fallback_call_index = 0
    last_tool_call_id = ""

    for item in request.content:
        if isinstance(item, Message):
            messages.append({"role": item.role, "content": item.content})
            continue

        if isinstance(item, ToolCall):
            call_id = item.tool_call_id or f"call_{fallback_call_index}"
            fallback_call_index += 1
            last_tool_call_id = call_id
            tool_call: dict[str, object] = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(
                        item.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            if messages and messages[-1].get("role") == "assistant":
                calls = messages[-1].setdefault("tool_calls", [])
                cast(list[dict[str, object]], calls).append(tool_call)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call],
                    }
                )
            continue

        messages.append(
            {
                "role": "tool",
                "content": _format_tool_result(item),
                "tool_call_id": item.tool_call_id or last_tool_call_id,
            }
        )
    return messages


class LlmClient:
    """Synchronous LiteLLM adapter used by the agent."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = None,
        extra_body: dict[str, object] | None = None,
        timeout: float = 3600.0,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.base_url = base_url
        self.api_key = (
            api_key if api_key is not None else ("not-needed" if base_url else None)
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self._litellm_model = (
            model
            if base_url is None or model.startswith("openai/")
            else f"openai/{model}"
        )

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate and parse one assistant turn."""
        messages = build_messages(request)
        request_trace = self._request_trace(messages, request)
        kwargs: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if request.tool_schemas:
            kwargs["tools"] = request.tool_schemas
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.enable_thinking is not None:
            extra_body["enable_thinking"] = self.enable_thinking
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            completion_call = cast(
                Callable[..., object],
                getattr(litellm, "completion"),
            )
            completion = completion_call(**kwargs)
            model_dump = getattr(completion, "model_dump", None)
            if not callable(model_dump):
                raise TypeError("LiteLLM returned a non-completion response")
            response = cast(
                dict[str, Any],
                cast(Callable[[], object], model_dump)(),
            )
            return self._parse_response(request, request_trace, response)
        except Exception as error:
            trace = None
            if request.trace_enabled:
                trace = {
                    **request_trace,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                parse_error_detail = _upstream_parse_error_detail(error)
                if parse_error_detail is not None:
                    parse_error = parse_error_detail.get("parse_error")
                    if isinstance(parse_error, Mapping):
                        typed_parse_error = cast(Mapping[object, object], parse_error)
                        trace["parse_error"] = {
                            str(key): value for key, value in typed_parse_error.items()
                        }
                    server_trace = parse_error_detail.get("trace")
                    if isinstance(server_trace, Mapping):
                        typed_server_trace = cast(Mapping[object, object], server_trace)
                        trace["server_trace"] = {
                            str(key): value for key, value in typed_server_trace.items()
                        }
            return LlmResponse(error_message=str(error), trace=trace)

    def count_tokens(self, request: LlmRequest) -> int:
        """Estimate request tokens with LiteLLM's model-aware token counter."""
        token_counter = cast(
            Callable[..., object],
            getattr(litellm, "token_counter"),
        )
        count = token_counter(
            model=self._litellm_model,
            messages=cast(Any, build_messages(request)),
            tools=cast(Any, request.tool_schemas or None),
        )
        if not isinstance(count, int):
            raise TypeError("LiteLLM token counter returned a non-integer value")
        return count

    def _parse_response(
        self,
        request: LlmRequest,
        request_trace: dict[str, object],
        response: dict[str, Any],
    ) -> LlmResponse:
        choices = cast(list[dict[str, Any]], response.get("choices") or [])
        if not choices:
            raise ValueError("LiteLLM response contains no choices")
        choice = choices[0]
        message = cast(dict[str, Any], choice.get("message") or {})
        raw_completion = message.get("content")
        if not isinstance(raw_completion, str):
            raw_completion = ""

        content: list[ContentItem] = []
        if raw_completion:
            content.append(Message(role="assistant", content=raw_completion))

        for index, raw_call in enumerate(message.get("tool_calls") or []):
            call = cast(dict[str, Any], raw_call)
            function = cast(dict[str, Any], call.get("function") or {})
            name = function.get("name")
            arguments_json = function.get("arguments", "{}")
            if not isinstance(name, str) or not name:
                raise ValueError("LiteLLM response contains a tool call without a name")
            if not isinstance(arguments_json, str):
                raise ValueError(f"tool call {name!r} arguments are not JSON text")
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError(f"tool call {name!r} arguments must be a JSON object")
            call_id = call.get("id")
            content.append(
                ToolCall(
                    tool_call_id=(
                        call_id
                        if isinstance(call_id, str) and call_id
                        else f"call_{index}"
                    ),
                    name=name,
                    arguments=cast(dict[str, object], arguments),
                )
            )

        usage = cast(dict[str, Any], response.get("usage") or {})
        usage_metadata: dict[str, object] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "generated_tokens": usage.get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
        }
        parsed: BaseModel | None = None
        if request.response_format is not None:
            payload = _structured_json_payload(raw_completion, raw_completion)
            try:
                parsed = request.response_format.model_validate_json(payload)
            except ValidationError as error:
                return LlmResponse(
                    content=content,
                    raw_completion=raw_completion,
                    usage_metadata=usage_metadata,
                    error_message=str(error),
                    trace=(
                        self._response_trace(
                            request_trace,
                            response,
                            content,
                            parsed=None,
                            error=error,
                        )
                        if request.trace_enabled
                        else None
                    ),
                )

        return LlmResponse(
            content=content,
            raw_completion=raw_completion,
            usage_metadata=usage_metadata,
            parsed=parsed,
            trace=(
                self._response_trace(
                    request_trace,
                    response,
                    content,
                    parsed=parsed,
                    error=None,
                )
                if request.trace_enabled
                else None
            ),
        )

    def _request_trace(
        self,
        messages: list[ChatMessage],
        request: LlmRequest,
    ) -> dict[str, object]:
        return {
            "request": {
                "messages": messages,
                "tool_schemas": request.tool_schemas,
                "response_format": (
                    request.response_format.__name__
                    if request.response_format is not None
                    else None
                ),
            },
            "client_config": {
                "model": self.model,
                "base_url": self.base_url,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "enable_thinking": self.enable_thinking,
                "extra_body": self.extra_body,
                "timeout": self.timeout,
            },
        }

    @staticmethod
    def _response_trace(
        request_trace: dict[str, object],
        completion: dict[str, Any],
        content: list[ContentItem],
        *,
        parsed: BaseModel | None,
        error: Exception | None,
    ) -> dict[str, object]:
        trace: dict[str, object] = {
            **request_trace,
            "completion": completion,
            "parsed_content": [item.model_dump(mode="json") for item in content],
            "parsed_structured_response": (
                parsed.model_dump(mode="json") if parsed is not None else None
            ),
        }
        if error is not None:
            trace["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        return trace


def _format_tool_result(result: AgentToolResult) -> str:
    prefix_by_status: dict[Literal["success", "error"], str] = {
        "success": "Tool result",
        "error": "Tool error",
    }
    content = "\n".join(str(item) for item in result.content)
    if content:
        return f"{prefix_by_status[result.status]}: {content}"
    return f"{prefix_by_status[result.status]}:"


def _upstream_parse_error_detail(error: Exception) -> Mapping[str, object] | None:
    """Recover LLLM's structured parse error from a LiteLLM HTTP exception."""
    candidates: list[object] = [getattr(error, "body", None)]
    response = getattr(error, "response", None)
    if response is not None:
        candidates.append(response)
        response_json = getattr(response, "json", None)
        if callable(response_json):
            try:
                candidates.append(cast(Callable[[], object], response_json)())
            except Exception:
                pass
    candidates.extend(error.args)
    for candidate in candidates:
        detail = _find_parse_error_detail(candidate)
        if detail is not None:
            return detail
    return None


def _find_parse_error_detail(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        parse_error = mapping.get("parse_error")
        if isinstance(parse_error, Mapping):
            return {str(key): item for key, item in mapping.items()}
        for item in mapping.values():
            found = _find_parse_error_detail(item)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for item in cast(Sequence[object], value):
            found = _find_parse_error_detail(item)
            if found is not None:
                return found
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _find_parse_error_detail(decoded)
    return None


def _structured_json_payload(content: str, raw_completion: str) -> str:
    """Return JSON content from plain or optional-think structured output."""
    for candidate in (content, raw_completion):
        payload = _strip_optional_think(candidate).strip()
        if payload:
            return payload
    return ""


def _strip_optional_think(text: str) -> str:
    close_tag = "</think>"
    close_index = text.find(close_tag)
    if text.lstrip().startswith("<think>") and close_index != -1:
        return text[close_index + len(close_tag) :]
    return text
