from collections.abc import Generator as YieldGenerator
from pathlib import Path
from typing import Any, cast

import pytest

from ..LLLM.agent import Agent
from ..LLLM.agent_llm import LlmClient
from ..LLLM.container_env import ContainerEnv
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.tool_common import Tool
from ..LLLM.tool_python import execute_python, python_tool
from .http_inference_server import serve_generator

from loguru import logger

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


def python_agent(base_url: str, tools: list[Tool]) -> Agent:
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
def test_functional_qwen3_calls_python_tool_in_container(
    qwen3_server_url: str,
) -> None:
    docker_client = _require_docker_client()
    base_tool = python_tool()
    calls: list[dict[str, object]] = []
    results: list[str] = []

    def record_python(
        arguments: dict[str, object],
        container_env: ContainerEnv | None,
    ) -> str:
        calls.append(arguments)
        result = execute_python(arguments, container_env)
        results.append(result)
        return result

    agent = python_agent(
        qwen3_server_url,
        [Tool(schema=base_tool.schema, execute=record_python)],
    )
    container_env = ContainerEnv(client=docker_client)
    container_env.start([], network=False)
    try:
        response = agent.run(
            "/no_think\n"
            "Call the python tool exactly once with this exact code: "
            "\"print(sum(i * i for i in range(1, 6)))\". First reply only "
            "with a valid <tool_call></tool_call> block. After the tool "
            "response, answer with the stdout number exactly and no extra text.",
            container_env=container_env,
        )
    finally:
        container_env.close()

    assert calls
    assert calls[0].get("code") == "print(sum(i * i for i in range(1, 6)))"
    assert results == ["Exit code: 0\nstdout:\n55"]
    assert isinstance(response.output, str)
    assert "55" in response.output


def _require_docker_client() -> Any:
    try:
        import docker

        client = cast(Any, docker.from_env())
        client.ping()
        client.images.pull("python:3.13-slim")
    except Exception as error:
        logger.error(f"{error}")
        pytest.skip(f"Docker is required for this functional test: {error}")
    return client
