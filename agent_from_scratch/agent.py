"""
Agent implementation.

Use LlmClient as backend, handle tool execution.
Do context management.

All the agent runtime is stored in a single entity ExecutionContext.
Execution context contain a list of Event, which can be AgentToolResult,
ToolCall, or Message from user, system or assistant.

This context is used to forge a LLMRequest, at this point we can do
the context management thing: include some, exclude some info of the
context.

The actual LLM call occure within LLMClient, which take a LLMRequest
and output a LlmResponse.

The LlmResponse is then parsed, if it contain ToolCall execute them.
We check that the LlmResponse contain no final_answer: depending on
configuration at init, final_answer may be provided by a tool call.

"""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Literal, cast

from .agent_context import (
    AgentStructuredResponse,
    ContentItem,
    ExecutionContext,
    Message,
    TaskState,
)
from .tool_common import ToolCall
from .tool_common import execute_tool
from .agent_context import AgentToolResult
from .agent_context import AgentResult
from .agent_context import Event
from .agent_llm import (
    LlmClient,
    LlmRequest,
    LlmResponse,
)
from .tool_common import Tool

from loguru import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .container_env import ContainerEnv

SYSTEM_PROMPT_V1 = """You are LLLM, a tool-capable assistant.

- When the user's intent is clear, execute immediately without confirmation.
- Only when intent is unclear, ask minimal questions to clarify.
- Use tools proactively, without asking permission, when a tool is needed to
  answer accurately or complete the task.
- Do not claim that you used a tool unless a tool call was actually made.
- Use a search or retrieval tool for current, changing, obscure,
  source-backed, or user-specified external information.
- Use a search or retrieval tool when the user asks to look up, search, browse,
  verify, cite, open a URL, inspect a page, or use a named source.
- Use a compute tool for arithmetic, unit conversion, formulas, precision
  math, or any calculation where mental math may be error-prone.
- If a tool result is incomplete or only identifies a source, continue with the
  next appropriate tool call, such as opening a search result, before answering, do
  not stop the work until it's finish.
- Do not use search tool for timeless information, fundamental concepts, definitions, or
  well-established technical facts.
- Do not use tools for simple language edits, brainstorming, summaries of text
  already provided by the user, or straightforward reasoning from given facts.
- Do not use a tool when the user explicitly asks you not to.
- Your internal knowledge may be incomplete or outdated.
- You cannot access external pages, files, or runtime state unless they are in
  the conversation or obtained through an available tool.
- Be direct, concise, and useful.
- Put the answer first, then brief supporting details when needed.
- Use Markdown for lists, tables, and code blocks when it improves clarity.
- Distinguish facts from assumptions. Cite or name sources when tool results
  provide them.
- Do not stop immediatly when a tool return. Keep trying until the work is done
  or you're confident that you can't find the needed informations.
- If a tool result does not contain the requested answer, call another tool.
- Do not ask the user to look it up.
- Do not say you can help later.
- Keep using available tools until you have the answer or the tools clearly fail.


Examples:
- User: "What is binary search?" Assistant: answer directly without tools.
- User: "What's the latest Python version?" Assistant: use search/retrieval,
  then answer with the current version and source.
- User: "Calculate the surface area of a sphere with diameter 22.2 cm."
  Assistant: use compute, then provide the numeric result and formula.
- User: "Is Venezuela was a participant of the 2026 Winter Olympic Games"
  Assistant: use wiki tool to find the corresponding wikipedia pages, then
    search 'Venezuela' in this page.
- User: "Write a script to rename these files." Assistant: provide or edit code;
  use file/code tools if repository or local files must be inspected.
"""

SYSTEM_PROMPT_V2 = """You are LLLM, a tool-capable assistant.

You receive durable task_state as a system message near the start of context:
- original_request is the user's initial request. Treat it as the root task.
- todos is the current checklist of unresolved work.
- facts is durable information gathered or established while working.

Always respond using the required structured output schema:
- answer: the user-visible assistant answer for this turn.
- task_state_update: optional updates to apply to task_state after this turn.
  Omit this field when no task-state update is needed.

Use task_state_update to maintain context:
- Decompose original_request into concrete TODOs and push them to todos.
- Pop TODOs by zero-based index once they are complete or no longer relevant.
- Push durable gathered facts, decisions, constraints, and assumptions to facts.
- Pop facts by zero-based index only when they are wrong or obsolete.
- Keep updates concise and exact. Do not duplicate existing todos or facts.

Example of answer:
{
    "answer": "Tool call or final answer",
    "task_state_update": {
        "push_todos": ["inspect repo", "run targeted tests"],
        "pop_todos": [0],
        "push_facts": ["project uses uv"],
        "pop_facts": []
    }
}

Use tools proactively, without asking permission, when a tool is needed to
answer accurately or complete the task. Do not claim that you used a tool unless
a tool call was actually made. Keep using available tools until you have the
answer or the tools clearly fail.

Be direct, concise, and useful in answer. Distinguish facts from assumptions.
"""

# Number of items that stay untouched by summarization.
SUM_KEEP_RECENTS = 5

# Request token count above which history summarization is attempted.
SUMMARIZE_TOKEN_THRESHOLD = 8000

# Prompt used to summarize items in context.
SUMMARY_PROMPT = """Summarize the prior conversation and tool history into concise state for the next assistant turn.

Include only durable information needed to continue:
- user goals and constraints
- facts, decisions, and assumptions established so far
- unresolved tasks or next actions
- important tool calls and tool outputs

Do not answer the user. Do not invent details. Keep it brief and concrete.
"""

AgentMode = Literal["dummy", "structured"]


class Agent:
    """
    Own conversation context, call the model, execute tools, and return text.
    Initialized with an LLM client and a tool list.
    """

    def __init__(
        self,
        llm: LlmClient,
        tools: Sequence[Tool],
        *,
        instruction: str = "",
        max_step: int = 8,
        agent_mode: AgentMode = "dummy",
    ) -> None:
        if max_step < 0:
            raise ValueError("max_step must be non-negative")
        if agent_mode not in ("dummy", "structured"):
            raise ValueError(f"unsupported agent_mode: {agent_mode!r}")
        self.llm = llm
        self.tools = tuple(tools)
        self._tools_by_name = self._index_tools(self.tools)
        self.agent_mode = agent_mode
        self.llm_response_format = self._response_format_for_mode(agent_mode)
        self.system_instructions = [self._system_prompt_for_mode(agent_mode)]
        if instruction:
            self.system_instructions.append(instruction)
        self.max_step = max_step

    def run(
        self,
        prompt: str,
        *,
        context: ExecutionContext | None = None,
        container_env: "ContainerEnv | None" = None,
        trace_enabled: bool = False,
    ) -> AgentResult:
        """
        Run the agent until the model returns an assistant answer.

        @param prompt: user input
        @param context: optional caller initialized execution context. If None it's
            init here.
        @param container_env: the containerized environment where tools run.
                              agent don't start the container, must be start by caller.
        @param trace_enabled: store full LLM/tool diagnostics in event metadata.
        @return agent final answer.
        """

        # Create execution context.
        execution_context = context if context is not None else ExecutionContext()
        execution_context.current_step = 0
        execution_context.final_result = None
        execution_context.state.pop("agent_error", None)

        if execution_context.task_state is None:
            execution_context.task_state = TaskState(original_request=prompt)
        else:
            execution_context.add_user_message(prompt)

        while (
            execution_context.final_result is None
            and execution_context.current_step < self.max_step
        ):
            _ = self.step(
                execution_context,
                container_env=container_env,
                trace_enabled=trace_enabled,
            )
            if "agent_error" in execution_context.state:
                break

            # Check if the last event is a final response
            if execution_context.events:
                last_event = execution_context.events[-1]
                if self._is_final_response(last_event):
                    execution_context.final_result = self._extract_final_result(
                        last_event
                    )
        if execution_context.current_step >= self.max_step:
            logger.warning("reached max_step, return final_result=None")

        error = execution_context.state.get("agent_error")
        return AgentResult(
            output=execution_context.final_result,
            context=execution_context,
            status="error" if isinstance(error, str) else "complete",
        )

    def _is_final_response(self, event: Event) -> bool:
        """
        Check if this event contains a final response.
        Return true if no ToolCall nor AgentToolResult in event contents.
        """
        # TODO check final_answer tool call at this point.
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, AgentToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str:
        """
        Extract the final result from an event.
        Return the first assistant message in the event contents.
        """
        # TODO extract the output of final_answer tool
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return "Woops!!"

    @staticmethod
    def _system_prompt_for_mode(agent_mode: AgentMode) -> str:
        if agent_mode == "dummy":
            return SYSTEM_PROMPT_V1
        return SYSTEM_PROMPT_V2

    @staticmethod
    def _response_format_for_mode(
        agent_mode: AgentMode,
    ) -> type[AgentStructuredResponse] | None:
        if agent_mode == "dummy":
            return None
        return AgentStructuredResponse

    def step(
        self,
        context: ExecutionContext,
        *,
        container_env: "ContainerEnv | None" = None,
        trace_enabled: bool = False,
    ) -> None:
        """
        Perform one ReAct think-act cycle.

        @param context: execution context to update.
        @param container_env: the containerized environment where tools run.
        @return None.
        """
        request = self._prepare_llm_request(context, trace_enabled=trace_enabled)

        # Get LLM's decision
        response = self.think(request)

        logger.debug("response.content = [{}]", response.content)

        response_content = self._response_event_content(context, response)
        for i, content in enumerate(response_content):
            logger.debug("response_content[{}] : {} = {}", i, content.type, content)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author="agent",
            content=response_content,
            metadata=(
                {"llm": response.trace}
                if trace_enabled and response.trace is not None
                else {}
            ),
        )
        context.add_event(response_event)

        if response.error_message is not None:
            context.state["agent_error"] = response.error_message
            context.increment_step()
            return None

        tool_calls = [item for item in response_content if isinstance(item, ToolCall)]
        if tool_calls:
            _ = self.act(
                context,
                tool_calls,
                container_env=container_env,
                trace_enabled=trace_enabled,
            )
        context.increment_step()
        return None

    def think(self, request: LlmRequest) -> LlmResponse:
        """
        Ask the LLM for the next assistant message or tool call.

        @param request: request.
        @return parsed LLM response.
        """
        return self.llm.complete(request)

    def _task_state_message(self, task_state: TaskState) -> Message:
        """Format task state as a deterministic system message."""
        lines = [
            "Task state:",
            f"original_request: {task_state.original_request}",
        ]
        if task_state.todos:
            lines.append("todos:")
            lines.append(self._format_state_list(task_state.todos))
        if task_state.facts:
            lines.append("facts:")
            lines.append(self._format_state_list(task_state.facts))
        return Message(
            role="system",
            content="\n".join(lines),
        )

    def _prepare_llm_request(
        self,
        context: ExecutionContext,
        *,
        trace_enabled: bool = False,
    ) -> LlmRequest:
        """
        Build the complete LLM request for one agent turn.

        The caller is responsible for putting system instructions in content.
        Request content order is:
        1. configured system instructions
        2. task_state system message when present
        3. compacted execution history
        """
        prefix: list[ContentItem] = [
            Message(role="system", content=instruction)
            for instruction in self.system_instructions
        ]
        # TODO task_state should never be None
        if context.task_state is not None:
            task_state_msg = self._task_state_message(context.task_state)
            prefix.append(task_state_msg)

        compacted_history = self._compact_request_content(context.items())
        request_content = [*prefix, *compacted_history]
        tool_schemas = [tool.schema for tool in self.tools]

        pre_summary_request = LlmRequest(
            content=request_content,
            tool_schemas=tool_schemas,
        )
        token_count = self.llm.count_tokens(pre_summary_request)
        logger.debug(
            "request token count before summarization token_count={} threshold={}",
            token_count,
            SUMMARIZE_TOKEN_THRESHOLD,
        )

        # TODO: in the current state of thing, if we reach the threshold, summarization is done at each step, which
        # is inneficient and prone to errors...
        if token_count > SUMMARIZE_TOKEN_THRESHOLD:
            request_content = self._summarize_request_content(
                prefix=prefix,
                history=compacted_history,
            )
        else:
            logger.debug(
                "skip history summarization because request is below token threshold"
            )

        logger.info(
            "Forge LlmRequest len(content)={} len(tools)={} response_format={}",
            len(request_content),
            len(tool_schemas),
            self.llm_response_format,
        )
        return LlmRequest(
            content=request_content,
            tool_schemas=tool_schemas,
            response_format=self.llm_response_format,
            trace_enabled=trace_enabled,
        )

    @staticmethod
    def _format_state_list(items: Sequence[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def _response_event_content(
        self, context: ExecutionContext, response: LlmResponse
    ) -> list[ContentItem]:
        """Convert a structured LLM response into ordinary event content."""
        tool_calls = [item for item in response.content if isinstance(item, ToolCall)]
        if isinstance(response.parsed, AgentStructuredResponse):
            if (
                context.task_state is not None
                and response.parsed.task_state_update is not None
            ):
                context.task_state.apply_update(response.parsed.task_state_update)
            content: list[ContentItem] = []
            if response.parsed.answer:
                content.append(
                    Message(role="assistant", content=response.parsed.answer)
                )
            content.extend(tool_calls)
            return content

        return list(response.content)

    def _summarize_request_content(
        self,
        *,
        prefix: Sequence[ContentItem],
        history: Sequence[ContentItem],
    ) -> list[ContentItem]:
        """
        Summarize older request history without mutating execution context.
        Keep request prefix and last SUM_KEEP_RECENTS history items untouched.
        """
        if len(history) <= SUM_KEEP_RECENTS:
            logger.debug(
                "skip history summarization because history is short item_count={}",
                len(history),
            )
            return [*prefix, *history]

        summary_items = history[:-SUM_KEEP_RECENTS]
        recent_items = history[-SUM_KEEP_RECENTS:]
        summary = self._generate_history_summary(summary_items)
        if summary is None:  # In case of error use un-summarized content.
            return [*prefix, *history]

        logger.info(
            "history summarization completed summarized_count={} preserved_recent_count={} summary_chars={}",
            len(summary_items),
            len(recent_items),
            len(summary),
        )
        return [
            *prefix,
            Message(
                role="system",
                content=f"Conversation summary so far:\n{summary}",
            ),
            *recent_items,
        ]

    def _generate_history_summary(self, items: Sequence[ContentItem]) -> str | None:
        """
        Ask the model for a concise summary of older request content.
        Use SUMMARY_PROMPT.
        """
        try:
            response = self.llm.complete(
                LlmRequest(
                    content=[Message(role="system", content=SUMMARY_PROMPT), *items],
                    tool_schemas=[],
                )
            )
        except Exception as error:
            logger.warning("history summarization model call failed error={}", error)
            return None

        if response.error_message is not None:
            logger.warning(
                "history summarization response has error error={}",
                response.error_message,
            )
            return None

        for item in response.content:
            if isinstance(item, Message) and item.role == "assistant":
                summary = item.content.strip()
                if summary:
                    return summary
                logger.warning("history summarization returned empty assistant text")
                return None

        logger.error("history summarization returned no assistant message")
        return None

    def act(
        self,
        context: ExecutionContext,
        tool_calls: Sequence[ToolCall],
        *,
        container_env: "ContainerEnv | None" = None,
        trace_enabled: bool = False,
    ) -> None:
        """
        Execute tool calls and append their results to the context.

        @param context: execution context to update.
        @param tool_calls: tool calls emitted by the last think phase.
        @param container_env: the containerized environment where tools run.
        @return stored tool results.
        """
        tool_results: list[AgentToolResult] = []
        tool_traces: list[dict[str, object]] = []
        for tool_call in tool_calls:
            started = time.perf_counter()
            tool_result = self._execute_tool_call(
                tool_call,
                container_env=container_env,
            )
            elapsed = time.perf_counter() - started
            tool_results.append(tool_result)
            if trace_enabled:
                tool_traces.append(
                    {
                        "tool_call_id": tool_call.tool_call_id,
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                        "status": tool_result.status,
                        "elapsed_seconds": elapsed,
                        "error": (
                            str(tool_result.content[0])
                            if tool_result.status == "error" and tool_result.content
                            else None
                        ),
                    }
                )
        tool_event = Event(
            execution_id=context.execution_id,
            author="tool",
            content=tool_results,
            metadata={"tools": tool_traces} if trace_enabled else {},
        )
        context.add_event(tool_event)
        return None

    @classmethod
    def _index_tools(cls, tools: Sequence[Tool]) -> dict[str, Tool]:
        """
        Build a tool dictionary for easy access (by name).
        """
        indexed: dict[str, Tool] = {}
        for tool in tools:
            name = cls._tool_name(tool.schema)
            if name in indexed:
                raise ValueError(f"duplicate tool name {name!r}")
            indexed[name] = tool
        return indexed

    @staticmethod
    def _tool_name(schema: dict[str, object]) -> str:
        """
        Retrieve a tool name from a tool description dict.
        """
        function = schema.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool schema must include a function object")
        function_dict = cast(dict[str, object], function)
        name = function_dict.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool schema function must include a non-empty name")
        return name

    def _compact_request_content(
        self, items: Sequence[ContentItem]
    ) -> list[ContentItem]:
        """
        Return context items transformed by tool-specific request policies.

        The execution context remains the source of truth and is never mutated
        by this helper; compaction only affects the next LLM request. The last
        item is kept raw so the model sees the newest context without loss.
        """
        compacted_items: list[ContentItem] = []
        last_index = len(items) - 1
        for index, item in enumerate(items):
            if index == last_index:
                compacted_items.append(item)
            else:
                compacted_items.append(self._compact_request_item(item))
        return compacted_items

    def _compact_request_item(self, item: ContentItem) -> ContentItem:
        """Apply the matching tool context policy to one request item."""
        if isinstance(item, ToolCall):
            tool = self._tools_by_name.get(item.name)
            policy = None if tool is None else tool.context_policy
            compact_call = None if policy is None else policy.compact_call
            if compact_call is None:
                return item
            try:
                compacted = compact_call(item)
            except Exception as error:
                logger.warning(
                    "tool call compaction failed; using raw item tool={} error={}",
                    item.name,
                    error,
                )
                return item
            logger.debug(
                "compacted tool call tool={} before_chars={} after_chars={}",
                item.name,
                len(str(item)),
                len(str(compacted)),
            )
            return compacted

        if isinstance(item, AgentToolResult):
            tool = self._tools_by_name.get(item.name)
            policy = None if tool is None else tool.context_policy
            compact_answer = None if policy is None else policy.compact_answer
            if compact_answer is None:
                return item
            try:
                compacted = compact_answer(item)
            except Exception as error:
                logger.warning(
                    "tool answer compaction failed; using raw item tool={} error={}",
                    item.name,
                    error,
                )
                return item
            logger.debug(
                "compacted tool answer tool={} before_chars={} after_chars={}",
                item.name,
                len(str(item)),
                len(str(compacted)),
            )
            return compacted

        return item

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        container_env: "ContainerEnv | None" = None,
    ) -> AgentToolResult:
        """
        Lookup in tool dictionary and execute tool.
        @param tool_calls: tool calls emitted by the last think phase.
        @param container_env: the containerized environment where tools run.
        Return AgentToolResult.
        """
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            return AgentToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"unknown tool {tool_call.name!r}"],
            )

        try:
            result = execute_tool(tool, tool_call.arguments, container_env)
        except Exception as error:
            return AgentToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"{tool_call.name!r} failed: {error}"],
            )

        return AgentToolResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            status="success",
            content=[result],
        )
