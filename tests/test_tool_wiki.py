from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import requests
import torch

from ..LLLM.agent_context import AgentToolResult
from ..LLLM.tool_common import Tool
from ..LLLM.tool_wiki import wiki_tools


class FakeEmbedder:
    def embed(self, text: str) -> torch.Tensor:
        lowered = text.lower()
        return torch.tensor(
            [
                1.0 if any(word in lowered for word in ("market", "capital")) else 0.0,
                1.0 if any(word in lowered for word in ("coffee", "café")) else 0.0,
                1.0 if any(word in lowered for word in ("python", "language")) else 0.0,
            ]
        )

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.embed(text) for text in texts])


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        url: str = "https://en.wikipedia.org/w/api.php",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers: Mapping[str, str] = {"content-type": "application/json"}
        self.url = url

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self) -> None:
        pass


def _tools_by_name() -> dict[str, Tool]:
    return {
        cast_name(tool): tool
        for tool in wiki_tools(FakeEmbedder())
    }


def cast_name(tool: Tool) -> str:
    function = tool.schema["function"]
    assert isinstance(function, dict)
    name = function["name"]
    assert isinstance(name, str)
    return name


def _page_payload(title: str, extract: str) -> dict[str, object]:
    return {"query": {"pages": {"1": {"title": title, "extract": extract}}}}


def test_wiki_tools_return_three_registered_tools() -> None:
    tools = wiki_tools(FakeEmbedder())

    assert [cast_name(tool) for tool in tools] == [
        "find_wiki_page",
        "search_in_wiki_page",
        "read_wiki_page",
    ]
    for tool in tools:
        assert isinstance(tool, Tool)
        assert tool.schema["type"] == "function"
        assert tool.context_policy is not None
        assert tool.context_policy.compact_answer is not None
        function = tool.schema["function"]
        assert isinstance(function, dict)
        parameters = function["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        assert "url" not in properties


def test_wiki_context_policy_compacts_long_success_answer() -> None:
    tool = _tools_by_name()["read_wiki_page"]
    assert tool.context_policy is not None
    assert tool.context_policy.compact_answer is not None
    raw_content = (
        "URL: https://en.wikipedia.org/wiki/Long\n"
        "Title: Long\n\n"
        f"{'x' * 3000}\n"
        "[read_wiki_page guard: page was too long]"
    )
    result = AgentToolResult(
        tool_call_id="call_0",
        name="read_wiki_page",
        status="success",
        content=[raw_content],
    )

    compacted = tool.context_policy.compact_answer(result)

    assert isinstance(compacted, AgentToolResult)
    assert compacted.tool_call_id == "call_0"
    assert compacted.name == "read_wiki_page"
    assert compacted.status == "success"
    content = compacted.content[0]
    assert isinstance(content, str)
    assert "URL: https://en.wikipedia.org/wiki/Long" in content
    assert "Title: Long" in content
    assert "[wiki answer compacted from" in content
    assert "[read_wiki_page guard:" in content
    assert len(content) < len(raw_content)


def test_wiki_context_policy_leaves_short_and_error_answers() -> None:
    tool = _tools_by_name()["read_wiki_page"]
    assert tool.context_policy is not None
    assert tool.context_policy.compact_answer is not None
    short = AgentToolResult(
        tool_call_id="call_0",
        name="read_wiki_page",
        status="success",
        content=["URL: https://en.wikipedia.org/wiki/Short\nTitle: Short\n\nbrief"],
    )
    error = AgentToolResult(
        tool_call_id="call_1",
        name="read_wiki_page",
        status="error",
        content=["wiki failed"],
    )

    assert tool.context_policy.compact_answer(short) == short
    assert tool.context_policy.compact_answer(error) == error


def test_find_wiki_page_parses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "query": {
            "search": [
                {
                    "title": "CAC 40",
                    "snippet": "French &lt;span class='searchmatch'&gt;index&lt;/span&gt;.",
                },
                {"title": "CAC Next 20", "snippet": "Another result."},
            ]
        }
    }
    calls: list[dict[str, object]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        calls.append({"args": args, "kwargs": kwargs})
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["find_wiki_page"].execute({"query": "CAC 40"})

    assert calls[0]["args"] == ("https://en.wikipedia.org/w/api.php",)
    kwargs = calls[0]["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["params"]["list"] == "search"
    assert kwargs["params"]["srsearch"] == "CAC 40"
    assert kwargs["params"]["srlimit"] == "5"
    assert "1. CAC 40" in output
    assert "URL: https://en.wikipedia.org/wiki/CAC_40" in output
    assert "2. CAC Next 20" in output
    assert "Next call read_wiki_page" in output


def test_find_wiki_page_uses_requested_wiki_and_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "search": [
                {"title": "Paris", "snippet": "Ville."},
                {"title": "Paris Saint-Germain FC", "snippet": "Club."},
            ]
        }
    }
    captured_params: list[dict[str, str]] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_params.append(kwargs["params"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["find_wiki_page"].execute(
        {
            "query": "Paris",
            "wiki": "https://fr.wikipedia.org",
            "max_results": 1,
        }
    )

    assert captured_params[0]["srlimit"] == "1"
    assert "URL: https://fr.wikipedia.org/wiki/Paris" in output
    assert "2. Paris Saint-Germain FC" not in output


def test_find_wiki_page_reports_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse({"query": {"search": []}}),
    )

    output = _tools_by_name()["find_wiki_page"].execute({"query": "nothing"})

    assert output == "No wiki results found for: nothing"


def test_read_wiki_page_reads_title(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _page_payload(
        "CAC 40",
        "The CAC 40 is a benchmark French stock market index.",
    )

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        assert kwargs["params"]["prop"] == "extracts"
        assert kwargs["params"]["titles"] == "CAC 40"
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["read_wiki_page"].execute({"title": "CAC 40"})

    assert "URL: https://en.wikipedia.org/wiki/CAC_40" in output
    assert "Title: CAC 40" in output
    assert "benchmark French stock market index" in output


def test_read_wiki_page_reads_url_from_title(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _page_payload("Café", "A café is a type of restaurant.")
    captured_titles: list[str] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_titles.append(kwargs["params"]["titles"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["read_wiki_page"].execute(
        {"title": "https://fr.wikipedia.org/wiki/Caf%C3%A9"}
    )

    assert captured_titles == ["Café"]
    assert "URL: https://fr.wikipedia.org/wiki/Caf%C3%A9" in output
    assert "A café is a type of restaurant." in output


def test_read_wiki_page_applies_high_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _page_payload("Long", "x" * 14000)
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    output = _tools_by_name()["read_wiki_page"].execute({"title": "Long"})

    assert len(output) <= 13000
    assert "[read_wiki_page guard:" in output
    assert "Use search_in_wiki_page" in output


def test_search_in_wiki_page_uses_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    extract = (
        "The cafe serves coffee and pastries. " * 20
        + "The CAC 40 market capitalization appears in financial summaries. "
        + "Python is a programming language. " * 20
    )
    payload = _page_payload("CAC 40", extract)
    captured_params: list[dict[str, str]] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_params.append(kwargs["params"])
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["search_in_wiki_page"].execute(
        {"title": "CAC 40", "query": "market capital", "top_k": 1}
    )

    assert captured_params[0]["prop"] == "extracts"
    assert captured_params[0]["titles"] == "CAC 40"
    assert "URL: https://en.wikipedia.org/wiki/CAC_40" in output
    assert "Title: CAC 40" in output
    assert "Query: market capital" in output
    assert "Result 1 (score=" in output
    assert "market capitalization" in output


def test_search_in_wiki_page_reads_url_from_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _page_payload("Café", "A café may serve coffee. The café is small.")
    captured_titles: list[str] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_titles.append(kwargs["params"]["titles"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = _tools_by_name()["search_in_wiki_page"].execute(
        {
            "title": "https://fr.wikipedia.org/wiki/Caf%C3%A9",
            "query": "coffee",
            "top_k": 1,
        }
    )

    assert captured_titles == ["Café"]
    assert "URL: https://fr.wikipedia.org/wiki/Caf%C3%A9" in output
    assert "coffee" in output


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("find_wiki_page", {}),
        ("find_wiki_page", {"query": ""}),
        ("find_wiki_page", {"query": "x", "wiki": "https://example.com"}),
        ("find_wiki_page", {"query": "x", "max_results": 0}),
        ("find_wiki_page", {"query": "x", "max_results": 11}),
        ("find_wiki_page", {"query": "x", "max_results": True}),
        ("read_wiki_page", {}),
        ("read_wiki_page", {"title": ""}),
        ("read_wiki_page", {"title": "file:///tmp/a"}),
        ("read_wiki_page", {"title": "https://example.com/wiki/CAC_40"}),
        ("read_wiki_page", {"title": "https://en.wikipedia.org/notwiki/CAC_40"}),
        ("search_in_wiki_page", {}),
        ("search_in_wiki_page", {"query": "market"}),
        ("search_in_wiki_page", {"title": "CAC 40", "query": ""}),
        (
            "search_in_wiki_page",
            {"title": "CAC 40", "query": "market", "top_k": 0},
        ),
        (
            "search_in_wiki_page",
            {"title": "CAC 40", "query": "market", "top_k": 11},
        ),
        (
            "search_in_wiki_page",
            {"title": "CAC 40", "query": "market", "top_k": False},
        ),
    ],
)
def test_wiki_tools_validate_arguments(
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _tools_by_name()[tool_name].execute(arguments)


def test_wiki_tools_report_request_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*_: Any, **__: Any) -> FakeResponse:
        raise requests.RequestException("timeout")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(ValueError, match="wiki request failed"):
        _tools_by_name()["find_wiki_page"].execute({"query": "llm"})


def test_wiki_tools_report_http_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse("missing", status_code=404),
    )

    with pytest.raises(ValueError, match="HTTP 404"):
        _tools_by_name()["read_wiki_page"].execute({"title": "Missing"})


def test_wiki_tools_report_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse({"error": {"info": "bad title"}}),
    )

    with pytest.raises(ValueError, match="bad title"):
        _tools_by_name()["read_wiki_page"].execute({"title": "Bad"})


def test_wiki_tools_report_missing_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    with pytest.raises(ValueError, match="wiki page not found"):
        _tools_by_name()["read_wiki_page"].execute({"title": "Missing"})
