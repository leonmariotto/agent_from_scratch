"""
Run GAIA dataset tests.
Warning: a valid HF token must be set in environemnt var HF_TOKEN.
"""

import math
from pathlib import Path

import pytest
from loguru import logger

from ..Lagents.agent import Agent
from ..Lagents.agent_context import AgentResult
from ..Lagents.agent_llm import LlmClient
from ..Lagents.eval_gaia import GaiaTask, GaiaToolName, evaluate_gaia_agent
from ..Lagents.tool_compute import compute_tool
#from ..Lagents.tool_common import Tool
#from ..Lagents.tool_wiki import wiki_tools
#from ..Lagents.sentence_transformer import SentenceTransformerEmbedder
#from ..Lagents.vector_db import DEFAULT_EMBEDDING_MODEL

# DEBUG
import litellm 

pytestmark = pytest.mark.slow


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"
QWEN3_4B_REPO_ID = "Qwen/Qwen3-4B"
GAIA_COMPUTE_WIKIPEDIA_TOOLS: tuple[GaiaToolName, ...] = (
    "calculator",
    "calculator (or ability to count)",
#    "wikipedia",
)
GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH = Path(
#    "gaia_qwen3_06b_compute_wikipedia_trace.json"
    "gaia_qwen3_06b_compute_trace.json"
)

qwen3_06b_server_url = "http://localhost:8000"

# TODO find a solution for embedding model in this repo.
# def _wiki_tools() -> list[Tool]:
#     ir = fetch_embedding_model_ir(DEFAULT_EMBEDDING_MODEL)
#     embedder = SentenceTransformerEmbedder.from_ir(ir)
#     return list(wiki_tools(embedder))

@pytest.fixture(scope="module")
def qwen3_06b_gaia_agent() -> Agent:
    return Agent(
        LlmClient(
            "lllm",
            base_url=qwen3_06b_server_url,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
        [compute_tool()],
        agent_mode="dummy",
    )

@pytest.mark.slow
def test_functional_qwen3_06b_gaia_compute_wikipedia_validation(
    qwen3_06b_gaia_agent: Agent,
) -> None:
    def agent(task: GaiaTask) -> AgentResult:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else ""
        )
        result = qwen3_06b_gaia_agent.run(
            "Answer this GAIA benchmark question. You are running in dummy "
            "agent mode, so return a plain final answer string. Use the "
            "compute tool for arithmetic, counting, unit conversion, or exact "
            "calculation. Use Wikipedia tools step by step for encyclopedia "
            "facts: find_wiki_page to locate a page, search_in_wiki_page to "
            "look for a specific fact inside a known page, and read_wiki_page "
            "only when a whole page is needed. Keep using tools until you have "
            "the answer or the available tools clearly cannot answer. Return "
            "only the final answer using this exact format: "
            "FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}",
            trace_enabled=True,
        )
        if not isinstance(result.output, str):
            raise AssertionError(
                f"agent did not return a final string: {result.output!r}"
            )
        return result

    litellm._turn_on_debug()
    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        allowed_tools=GAIA_COMPUTE_WIKIPEDIA_TOOLS,
        trace_output_path=GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH,
    )
    logger.info(
        "evaluation={} trace_output_path={}",
        evaluation,
        GAIA_COMPUTE_WIKIPEDIA_TRACE_PATH,
    )

    assert evaluation.total_tasks > 0
    assert evaluation.scored_tasks == evaluation.total_tasks
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0
    assert all(result.error is None for result in evaluation.results)


