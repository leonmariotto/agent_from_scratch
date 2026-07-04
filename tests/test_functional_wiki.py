from collections.abc import Generator as YieldGenerator
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from ..LLLM.agent import Agent
from ..LLLM.agent_llm import LlmClient
from ..LLLM.fetch import fetch_embedding_model_ir
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.sentence_transformer import SentenceTransformerEmbedder
from ..LLLM.tool_common import Tool
from ..LLLM.tool_wiki import wiki_tools
from ..LLLM.vector_db import DEFAULT_EMBEDDING_MODEL
from .http_inference_server import serve_generator

pytestmark = pytest.mark.slow

QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


class FakeEmbedder:
    def embed(self, text: str) -> torch.Tensor:
        del text
        return torch.ones(3)

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.ones((len(texts), 3))


def _tool_by_name(name: str) -> Tool:
    return _tool_by_name_from_tools(name, wiki_tools(FakeEmbedder()))


def _tool_by_name_from_tools(name: str, tools: Sequence[Tool]) -> Tool:
    for tool in tools:
        function = tool.schema["function"]
        assert isinstance(function, dict)
        if function["name"] == name:
            return tool
    raise AssertionError(f"missing wiki tool {name}")


@pytest.fixture(scope="module")
def qwen3_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.fixture(scope="module")
def qwen3_server_url(
    qwen3_generator: Generator,
) -> YieldGenerator[str, None, None]:
    with serve_generator(qwen3_generator) as base_url:
        yield base_url


@pytest.fixture(scope="module")
def wiki_embedder() -> SentenceTransformerEmbedder:
    ir = fetch_embedding_model_ir(DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformerEmbedder.from_ir(ir)


def wiki_agent(
    base_url: str,
    tools: list[Tool],
    *,
    max_step: int = 3,
) -> Agent:
    return Agent(
        LlmClient(
            "lllm",
            base_url=base_url,
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
        tools,
        max_step=max_step,
    )


def tool_call_prompt(tool_name: str, arguments_json: str, follow_up: str) -> str:
    return (
        f"Call exactly one tool now. Use this exact tool name: {tool_name}. "
        "Your first assistant message must contain only this XML block, with "
        "valid JSON inside it:\n"
        "<tool_call>\n"
        "{\n"
        f'  "name": "{tool_name}",\n'
        f'  "arguments": {arguments_json}\n'
        "}\n"
        "</tool_call>\n"
        f"After the tool result, {follow_up}"
    )


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_find_wiki_page(
    qwen3_server_url: str,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = _tool_by_name("find_wiki_page")

    def record_wiki(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
        calls.append(arguments)
        return (
            "1. Frobnicate\n"
            "URL: https://en.wikipedia.org/wiki/Frobnicate\n"
            "Snippet: The controlled wiki tool response says frobnicate."
        )

    agent = wiki_agent(
        qwen3_server_url,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
    )

    response = agent.run(
        tool_call_prompt(
            "find_wiki_page",
            '{"query": "frobnicate"}',
            "answer with the URL from the result and the word frobnicate.",
        )
    )

    assert calls
    query = calls[0].get("query")
    assert isinstance(query, str)
    assert "frobnicate" in query.lower()
    assert isinstance(response.output, str)
    assert "https://en.wikipedia.org/wiki/Frobnicate" in response.output
    assert "frobnicate" in response.output


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_read_wiki_page(
    qwen3_server_url: str,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = _tool_by_name("read_wiki_page")

    def record_wiki(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
        calls.append(arguments)
        return (
            "URL: https://en.wikipedia.org/wiki/CAC_40\n"
            "Title: CAC 40\n\n"
            "The controlled wiki page extract says calisson."
        )

    agent = wiki_agent(
        qwen3_server_url,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
    )

    response = agent.run(
        tool_call_prompt(
            "read_wiki_page",
            '{"title": "https://en.wikipedia.org/wiki/CAC_40"}',
            "answer with the page title and the word calisson.",
        )
    )

    assert calls
    assert calls[0].get("title") == "https://en.wikipedia.org/wiki/CAC_40"
    assert isinstance(response.output, str)
    assert "CAC 40" in response.output
    assert "calisson" in response.output


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_search_in_wiki_page(
    qwen3_server_url: str,
    wiki_embedder: SentenceTransformerEmbedder,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = _tool_by_name_from_tools(
        "search_in_wiki_page",
        wiki_tools(wiki_embedder),
    )

    def record_wiki(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
        calls.append(arguments)
        return (
            "URL: https://en.wikipedia.org/wiki/CAC_40\n"
            "Title: CAC 40\n"
            "Best matching passages:\n"
            "1. Score: 0.93\n"
            "The controlled vector search result says castagnade."
        )

    agent = wiki_agent(
        qwen3_server_url,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
    )

    response = agent.run(
        tool_call_prompt(
            "search_in_wiki_page",
            (
                '{"title": "https://en.wikipedia.org/wiki/CAC_40", '
                '"query": "controlled vector search"}'
            ),
            "answer with the page title and the word castagnade.",
        )
    )

    assert calls
    assert calls[0].get("title") == "https://en.wikipedia.org/wiki/CAC_40"
    query = calls[0].get("query")
    assert isinstance(query, str)
    assert "controlled vector search" in query.lower()
    assert isinstance(response.output, str)
    assert "CAC 40" in response.output
    assert "castagnade" in response.output


@pytest.mark.slow
@pytest.mark.parametrize(
    ("prompt", "expected_in_response"),
    [
        pytest.param(
            (
                "What's the CAC40 latest market cap ? I believe that this "
                "information is present in Wikipedia. Use the split Wikipedia "
                "tools step by step: first find_wiki_page if you need the "
                "page, then search_in_wiki_page for the specific fact, and "
                "read_wiki_page only if the search result is insufficient. "
                "Keep trying until you have an answer or the tools clearly fail."
            ),
            None,
            id="cac40-market-cap",
        ),
        pytest.param(
            (
                "According to Wikipedia, which city hosted the 2024 Summer "
                "Olympics? Use the split Wikipedia tools step by step: first "
                "find_wiki_page if you need the page, then "
                "search_in_wiki_page for the specific fact, and "
                "read_wiki_page only if the search result is insufficient. "
                "Keep trying until you have an answer or the tools clearly fail."
            ),
            "Paris",
            id="2024-summer-olympics-host-city",
        ),
    ],
)
def test_functional_qwen3_with_thinking_calls_wiki_autoload(
    qwen3_server_url: str,
    wiki_embedder: SentenceTransformerEmbedder,
    prompt: str,
    expected_in_response: str | None,
) -> None:
    agent = wiki_agent(
        qwen3_server_url,
        list(wiki_tools(wiki_embedder)),
        max_step=8,
    )

    response = agent.run(prompt)

    assert isinstance(response.output, str)
    if expected_in_response is not None:
        assert expected_in_response in response.output
