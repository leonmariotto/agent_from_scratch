"""
Run a real SWE-bench smoke test with a bare Qwen3 model.

This intentionally does not use an agent harness.  The goal is only to verify
that the SWE-bench evaluator can drive a real model-backed callable end to end.
Qwen3 0.6B without repository tools is not expected to resolve the task.
"""

from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.slow

from ..LLLM.eval_swebench import SwebenchTask, evaluate_swebench_agent
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_06b_swebench_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.mark.slow
def test_functional_qwen3_06b_runs_real_swebench_lite_smoke(
    qwen3_06b_swebench_generator: Generator,
    tmp_path: Path,
) -> None:
    tokenizer = cast(Qwen3Tokenizer, qwen3_06b_swebench_generator.tokenizer)

    def agent(task: SwebenchTask, repo_path: Path) -> None:
        prompt = (
            "You are given one SWE-bench task. Inspect the task and briefly "
            "describe what files you would investigate first.\n\n"
            f"Repository: {task.repo}\n"
            f"Checkout path: {repo_path}\n"
            f"Base commit: {task.base_commit}\n\n"
            f"Problem statement:\n{task.problem_statement}\n"
        )
        prompt_tokens = tokenizer.encode_instruct_prompt(
            prompt,
            enable_thinking=False,
        )
        qwen3_06b_swebench_generator.generate_from_tokens(
            prompt_tokens,
            max_generated_token=512,
            temperature=0.0,
            include_prompt=False,
        )

    evaluation = evaluate_swebench_agent(
        agent,
        limit=1,
        timeout_seconds=600,
        artifacts_dir=tmp_path,
    )

    assert evaluation.total_tasks == 1
    assert evaluation.resolved_tasks in {0, 1}
    assert evaluation.failed_tasks + evaluation.error_tasks + evaluation.resolved_tasks == 1
    assert evaluation.resolved_rate is not None
    assert 0.0 <= evaluation.resolved_rate <= 1.0

    result = evaluation.results[0]
    assert result.instance_id
    assert result.repo
    assert result.base_commit
    assert result.problem_statement
    assert result.elapsed_seconds >= 0.0
    assert result.artifact_dir is not None
    assert result.artifact_dir.exists()
