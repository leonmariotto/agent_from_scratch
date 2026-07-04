from collections.abc import Generator as YieldGenerator
from pathlib import Path

import pytest

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.agent import Agent
from ..LLLM.agent_llm import LlmClient
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.tool_common import Tool
from ..LLLM.tool_compute import compute_tool, execute_compute
from .http_inference_server import serve_generator

pytestmark = pytest.mark.slow

QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


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


def compute_agent(base_url: str, tools: list[Tool]) -> Agent:
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
        max_step=3,
    )


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_compute_tool(
    qwen3_server_url: str,
) -> None:
    calls: list[dict[str, object]] = []
    results: list[str] = []
    base_tool = compute_tool()

    def record_compute(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
        calls.append(arguments)
        result = execute_compute(arguments)
        results.append(result)
        return result

    agent = compute_agent(
        qwen3_server_url,
        [Tool(schema=base_tool.schema, execute=record_compute)],
    )

    response = agent.run(
        "Use the compute tool exactly once to calculate 137 * 29. "
        "The compute tool expression must use bc syntax. "
        "First reply only with a valid <tool_call></tool_call> block. "
        "After the tool response, answer with the numeric result and "
        "no extra explanation."
    )

    assert calls
    assert results == ["3973"]
    assert isinstance(response.output, str)
    assert "3973" in response.output


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_compute_tool_autoload(
    qwen3_server_url: str,
) -> None:
    agent = compute_agent(
        qwen3_server_url,
        [compute_tool()],
    )

    response = agent.run(
        "I want to measure the surface of an orb that have 22.2 "
        "centimeter of diameter. What's the surface of the orb in "
        "square centimeter ? Use bc syntax when calling the compute "
        "tool."
    )

    assert response.output is not None
