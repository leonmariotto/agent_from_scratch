# Leon's agent from scratch

Agent implementation using LiteLLM backend.

Containerisation is integrated, so no execution check is done.

Based on:
- *Build an AI agent from scratch* book by Younghee Song & Jungjun Hur.

## LiteLLLM

The agent always connects through LiteLLM. To use the local server:

```python
from LLLM.agent import Agent
from LLLM.agent_llm import LlmClient

llm = LlmClient(
    "lllm",
    base_url="http://127.0.0.1:8000/v1",
    max_tokens=1024,
    top_k=20,
)
agent = Agent(llm, [])
```

For an authenticated OpenAI-compatible server, pass its model name, `base_url`,
and `api_key`. Without `base_url`, native LiteLLM model identifiers such as
`anthropic/<model>` are passed through unchanged. Provider-specific request
fields can be supplied with `extra_body`.

## Development note.

### Context management

- an agent need enough **effective context management**
- a reasonably large cache_length
- truncation of noisy old messages
- summaries of old history
- retrieval from files/vector DB/search
- selective tool output inclusion
- compact state like “current goal”, “known facts”, “open issues”
- keeping the current task and constraints near the end of the prompt
- "thinking" block not preserved.
- tool executor can have state and memory.

- system prompt must:
    - define boundaries, which request is accepted, which is rejected.
    - output format and style
    - clarify knowledge/capacity limits
    - "When the user's intent is clear, execute immediately without confirmation.
        Only when intent is unclear, ask minimal questions to clarify"
    - "use tool proactively, without asking permission"
    - Clearly define when to use tools: use trigger pattern.
    - Define when not use tool : "Do not search for timeless information,
        fundamental concepts, definitions, or well- established technical facts."
    - Provide concrete examples.


Different tools response/requests must have differents compaction.
Each tool can declare a ToolContextPolicy class.

History summarization should use layers:
    - recent_message: exact last N turn
    - session summary: compressed older conversation
    - task_state: current goals, decision and constraint
    - facts_memory: durable facts discovered.
    - open_thread: unresolved questions.

Distinguish memory:
    - Instruction memory: stable behavior rules
    - Task memory, current goal, plan, constraint, decision, TODOs.
    - Conversation memory: user/assistant turns, summarizable.
    - Evidence memory: RAG snippet, search result, citation.
    - Tool memory: calls, output
Each category have different compaction rules.

Context invalidation is important, when files changes, or something happen. This
can be done using some context items metadata.

Deduplication.

So there is a general context management part that handle which messages are pinned, 
which can be summarized, etc
And there is a per-tool context management that handle tool call tool response in context.
Some LLM call can be done to extract information of tool result in order to summarize them.
Don't do callback, I don't like this.

task_state should be made from
    - deterministic parsing of all events.
    - LLM produced task_state_patch, validated deterministically.

