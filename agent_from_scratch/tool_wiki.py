"""
Focused Wikimedia tools for model tool-use loops.

The module exposes three separate tools instead of one action-routed tool:
``find_wiki_page`` finds candidate pages, ``search_in_wiki_page`` searches
inside one page with vector search, and ``read_wiki_page`` reads one page with a
high guard limit.  The schemas are deliberately explicit because small local
models need step-by-step tool guidance.
"""

from __future__ import annotations

import copy
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import quote, unquote, urlparse

from loguru import logger
import requests

from .agent_context import AgentToolResult
from .tool_common import Tool, ToolContextPolicy
from .vector_search import SearchResult, TextEmbedder, vector_build_and_search

if TYPE_CHECKING:
    from .container_env import ContainerEnv

_DEFAULT_WIKI = "https://en.wikipedia.org"
_REQUEST_TIMEOUT_SECONDS = 10
_HEADERS = {
    "User-Agent": (
        "LLLM wiki tool/0.2 "
        "(https://github.com/leonmariotto/LLLM; contact: leon2mariotto@gmail.com)"
    )
}
_SUPPORTED_HOST_SUFFIXES = (
    ".wikipedia.org",
    ".wiktionary.org",
    ".wikibooks.org",
    ".wikiquote.org",
    ".wikisource.org",
    ".wikiversity.org",
    ".wikivoyage.org",
    ".wikinews.org",
    ".wikimedia.org",
)
_SUPPORTED_EXACT_HOSTS = {"www.wikidata.org"}
_CONTEXT_COMPACT_BODY_CHARS = 1200
_CONTEXT_COMPACT_MIN_SAVED_CHARS = 300
_READ_WIKI_PAGE_MAX_CHARS = 13000
_READ_GUARD_MARKER = (
    "\n[read_wiki_page guard: page was too long and was cut. "
    "Use search_in_wiki_page with a specific query to find details in this page.]"
)
_VECTOR_CHUNK_SIZE = 1000
_VECTOR_CHUNK_OVERLAP = 100

FIND_WIKI_PAGE_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "find_wiki_page",
        "description": (
            "Step 1 wiki tool. Use this when you do not know the exact Wikipedia "
            "or Wikimedia page title. Give a short topic or entity name. This "
            "tool only returns candidate pages and URLs. Do not answer from this "
            "tool alone. Next call read_wiki_page for broad reading, or "
            "search_in_wiki_page for a specific fact inside one returned page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Required. Short topic or page name to find, like "
                        "'Paris', 'CAC 40', or '2024 Summer Olympics'."
                    ),
                },
                "wiki": {
                    "type": "string",
                    "description": (
                        "Optional Wikimedia base URL, such as "
                        "https://fr.wikipedia.org. Defaults to English Wikipedia."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optional result count from 1 to 10. Default is 5.",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_IN_WIKI_PAGE_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "search_in_wiki_page",
        "description": (
            "Step 2 wiki tool for a specific question about one known page. "
            "Use this after you know the page title or URL. Put either the "
            "plain page title or the full Wikimedia page URL in the title "
            "parameter. It reads the page, "
            "splits it into chunks, and returns the most relevant passages using "
            "vector search. Use a short semantic query, not the whole chat. Good "
            "queries: 'market capitalization', 'host city', 'population'. If the "
            "answer is not in the passages, try another short query or read the "
            "page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Required. Short phrase for the fact to find.",
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Required. Wiki page title or full Wikimedia page URL. "
                        "Examples: '2024 Summer Olympics' or "
                        "'https://en.wikipedia.org/wiki/2024_Summer_Olympics'."
                    ),
                },
                "wiki": {
                    "type": "string",
                    "description": (
                        "Optional Wikimedia base URL used with title. Defaults to "
                        "English Wikipedia."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Optional number of relevant passages to return, from 1 "
                        "to 10. Default is 5."
                    ),
                },
            },
            "required": ["query", "title"],
        },
    },
}

READ_WIKI_PAGE_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "read_wiki_page",
        "description": (
            "Step 2 wiki tool for broad reading of one known page. Use this after "
            "you know the page title or URL. Put either the plain page title or "
            "the full Wikimedia page URL in the title parameter. It returns the page text up to a high "
            "guard limit. For long pages or specific facts, prefer "
            "search_in_wiki_page because it returns focused passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Required. Wiki page title or full Wikimedia page URL. "
                        "Examples: 'CAC 40' or "
                        "'https://en.wikipedia.org/wiki/CAC_40'."
                    ),
                },
                "wiki": {
                    "type": "string",
                    "description": (
                        "Optional Wikimedia base URL used with title. Defaults to "
                        "English Wikipedia."
                    ),
                },
            },
            "required": ["title"],
        },
    },
}


@dataclass(frozen=True)
class _WikiPage:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class _OpenTarget:
    base_url: str
    title: str


@dataclass(frozen=True)
class _FetchedPage:
    title: str
    url: str
    extract: str


def wiki_tools(embedder: TextEmbedder) -> tuple[Tool, Tool, Tool]:
    """
    Return ready-to-register Wikimedia tools.

    ``search_in_wiki_page`` needs the supplied embedder for vector search.
    ``find_wiki_page`` and ``read_wiki_page`` do not embed text, but they are
    returned together so callers register a coherent wiki workflow.
    """
    executor = _WikiExecutor(embedder)
    policy = ToolContextPolicy(compact_answer=_compact_wiki_answer)
    logger.info("creating split wiki tools with vector embedder={}", type(embedder))
    return (
        Tool(
            schema=copy.deepcopy(FIND_WIKI_PAGE_SCHEMA),
            execute=executor.find_wiki_page,
            context_policy=policy,
        ),
        Tool(
            schema=copy.deepcopy(SEARCH_IN_WIKI_PAGE_SCHEMA),
            execute=executor.search_in_wiki_page,
            context_policy=policy,
        ),
        Tool(
            schema=copy.deepcopy(READ_WIKI_PAGE_SCHEMA),
            execute=executor.read_wiki_page,
            context_policy=policy,
        ),
    )


class _WikiExecutor:
    """Executor shared by the split wiki tools."""

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    def find_wiki_page(
        self,
        arguments: dict[str, object],
        container_env: "ContainerEnv | None" = None,
    ) -> str:
        """Find candidate wiki pages for a short topic query."""
        del container_env
        query = _require_non_empty_string(arguments, "query")
        wiki = _optional_wiki(arguments)
        max_results = _validated_top_k(arguments, "max_results", default=5)
        logger.info(
            "find_wiki_page query={!r} wiki={} max_results={}",
            query,
            wiki,
            max_results,
        )
        return _execute_search(query, wiki, max_results)

    def search_in_wiki_page(
        self,
        arguments: dict[str, object],
        container_env: "ContainerEnv | None" = None,
    ) -> str:
        """Search one wiki page with vector search over page chunks."""
        del container_env
        target = _open_target(arguments)
        query = _require_non_empty_string(arguments, "query")
        top_k = _validated_top_k(arguments, "top_k", default=5)
        page = _fetch_page(target)
        logger.info(
            "search_in_wiki_page target_title={!r} url={} query={!r} page_chars={} "
            "top_k={} chunk_size={} chunk_overlap={}",
            page.title,
            page.url,
            query,
            len(page.extract),
            top_k,
            _VECTOR_CHUNK_SIZE,
            _VECTOR_CHUNK_OVERLAP,
        )
        results = vector_build_and_search(
            query,
            page.extract,
            self.embedder,
            top_k=top_k,
            chunk_size=_VECTOR_CHUNK_SIZE,
            chunk_overlap=_VECTOR_CHUNK_OVERLAP,
        )
        logger.info(
            "search_in_wiki_page returned {} vector results title={!r}",
            len(results),
            page.title,
        )
        return _format_vector_search_results(page, query, results)

    def read_wiki_page(
        self,
        arguments: dict[str, object],
        container_env: "ContainerEnv | None" = None,
    ) -> str:
        """Read one wiki page up to the fixed high guard limit."""
        del container_env
        target = _open_target(arguments)
        page = _fetch_page(target)
        output = f"URL: {page.url}\nTitle: {page.title}\n\n{page.extract}"
        guarded = _apply_read_guard(output)
        logger.info(
            "read_wiki_page title={!r} url={} page_chars={} returned_chars={} "
            "truncated={}",
            page.title,
            page.url,
            len(output),
            len(guarded),
            len(guarded) < len(output),
        )
        return guarded


def _compact_wiki_answer(result: AgentToolResult) -> AgentToolResult:
    """
    Return a request-only compacted wiki tool result.

    The stored execution context keeps the full result. This function preserves
    source/navigation metadata and trims long page bodies before the next LLM
    request is built.
    """
    if result.status != "success":
        logger.debug("skip wiki answer compaction for non-success status")
        return result

    compacted_content: list[object] = []
    changed = False
    for item in result.content:
        if not isinstance(item, str):
            compacted_content.append(item)
            continue
        compacted = _compact_wiki_text(item)
        changed = changed or compacted != item
        compacted_content.append(compacted)

    if not changed:
        logger.debug("skip wiki answer compaction because content is already short")
        return result

    logger.debug(
        "compacted wiki answer tool_call_id={} before_chars={} after_chars={}",
        result.tool_call_id,
        sum(len(str(item)) for item in result.content),
        sum(len(str(item)) for item in compacted_content),
    )
    return AgentToolResult(
        tool_call_id=result.tool_call_id,
        name=result.name,
        status=result.status,
        content=compacted_content,
    )


def _compact_wiki_text(text: str) -> str:
    """
    Compact a wiki response while preserving source and search metadata.

    Search results are usually shorter than this threshold and pass through
    unchanged. Long read/search answers keep metadata lines plus the first body
    segment that fits the request-context budget.
    """
    if len(text) <= _CONTEXT_COMPACT_BODY_CHARS + _CONTEXT_COMPACT_MIN_SAVED_CHARS:
        return text

    metadata_lines: list[str] = []
    body_lines: list[str] = []
    for line in text.splitlines():
        if _is_wiki_metadata_line(line):
            metadata_lines.append(line)
        elif line.startswith("[read_wiki_page guard:"):
            metadata_lines.append(line)
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    visible_body = body[:_CONTEXT_COMPACT_BODY_CHARS].rstrip()
    sections = [
        *metadata_lines,
        (f"[wiki answer compacted from {len(text)} to {len(visible_body)} body chars]"),
    ]
    if visible_body:
        sections.append(visible_body)
    return "\n".join(section for section in sections if section)


def _is_wiki_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("URL:", "Title:", "Query:", "Result ")):
        return True
    if re.match(r"^\d+\. .+", stripped):
        return True
    if stripped == (
        "These are only candidate pages. Next call read_wiki_page for broad "
        "reading, or search_in_wiki_page for a specific fact."
    ):
        return True
    return False


def _execute_search(
    query: str, wiki: str, max_results: int, include_snippet: bool = False
) -> str:
    payload = _api_get(
        wiki,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max_results),
            "format": "json",
            "utf8": "1",
        },
    )
    raw_results_object = _query_dict(payload).get("search", [])
    if not isinstance(raw_results_object, list) or not raw_results_object:
        logger.info("find_wiki_page returned 0 results query={!r}", query)
        return f"No wiki results found for: {query}"
    raw_results = cast(list[object], raw_results_object)

    results: list[_WikiPage] = []
    for raw_result in raw_results[:max_results]:
        if not isinstance(raw_result, dict):
            continue
        result_dict = cast(dict[str, object], raw_result)
        title = result_dict.get("title")
        if not isinstance(title, str) or not title:
            continue
        snippet = result_dict.get("snippet")
        results.append(
            _WikiPage(
                title=title,
                url=_page_url(wiki, title),
                snippet=_clean_snippet(snippet if isinstance(snippet, str) else ""),
            )
        )
    if not results:
        logger.info("find_wiki_page returned 0 valid results query={!r}", query)
        return f"No wiki results found for: {query}"

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\nURL: {result.url}")
        if result.snippet and include_snippet:
            lines.append(f"Snippet: {result.snippet}")
    lines.append(
        "These are only candidate pages. Next call read_wiki_page for broad "
        "reading, or search_in_wiki_page for a specific fact."
    )
    logger.info("find_wiki_page returned {} results query={!r}", len(results), query)
    return "\n".join(lines)


def _format_vector_search_results(
    page: _FetchedPage, query: str, results: Sequence[SearchResult]
) -> str:
    header = f"URL: {page.url}\nTitle: {page.title}\nQuery: {query}"
    if not results:
        return (
            f"{header}\n\n"
            "No relevant passages found in this page. Try read_wiki_page or a "
            "different short search query."
        )

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        text = re.sub(r"\s+", " ", result.sequence).strip()
        blocks.append(f"Result {index} (score={result.score:.4f}):\n{text}")
    return f"{header}\n\n" + "\n\n".join(blocks)


def _apply_read_guard(text: str) -> str:
    if len(text) <= _READ_WIKI_PAGE_MAX_CHARS:
        return text
    split_at = _READ_WIKI_PAGE_MAX_CHARS - len(_READ_GUARD_MARKER)
    return text[:split_at].rstrip() + _READ_GUARD_MARKER


def _validated_top_k(arguments: dict[str, object], name: str, *, default: int) -> int:
    value = _optional_int(arguments, name, default=default)
    if value < 1 or value > 10:
        raise ValueError(f"{name} must be between 1 and 10")
    return value


def _fetch_page(target: _OpenTarget) -> _FetchedPage:
    payload = _api_get(
        target.base_url,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": target.title,
            "format": "json",
            "utf8": "1",
        },
    )
    page = _first_page(payload)
    title = page.get("title")
    if not isinstance(title, str) or not title:
        title = target.title
    if page.get("missing") is not None:
        logger.error("Wiki page not found {}", target.title)
        raise ValueError(f"wiki page not found: {target.title}")
    extract = page.get("extract")
    if not isinstance(extract, str):
        logger.error("Wiki page has no extract {}", target.title)
        raise ValueError(f"wiki page has no extract: {title}")

    return _FetchedPage(
        title=title,
        url=_page_url(target.base_url, title),
        extract=extract,
    )


def _api_get(base_url: str, params: dict[str, str]) -> dict[str, object]:
    url = f"{base_url}/w/api.php"
    try:
        response = requests.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ValueError(f"wiki request failed: {error}") from error
    try:
        final_url = getattr(response, "url", url)
        if isinstance(final_url, str):
            _validate_wikimedia_base(_origin(final_url))
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError(f"wiki request failed: HTTP {response.status_code}")
        try:
            payload = cast(object, response.json())
        except ValueError as error:
            raise ValueError("wiki request failed: response was not JSON") from error
    finally:
        response.close()

    if not isinstance(payload, dict):
        raise ValueError("wiki request failed: response JSON was not an object")
    payload_dict = cast(dict[str, object], payload)
    if "error" in payload_dict:
        error_payload = payload_dict.get("error")
        if isinstance(error_payload, dict):
            error_dict = cast(dict[str, object], error_payload)
            info = error_dict.get("info")
            if isinstance(info, str) and info:
                raise ValueError(f"wiki API error: {info}")
        raise ValueError("wiki API error")
    return payload_dict


def _query_dict(payload: dict[str, object]) -> dict[str, object]:
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("wiki response missing query object")
    return cast(dict[str, object], query)


def _first_page(payload: dict[str, object]) -> dict[str, object]:
    pages_object = _query_dict(payload).get("pages")
    if not isinstance(pages_object, dict):
        raise ValueError("wiki response missing pages object")
    pages = cast(dict[str, object], pages_object)
    for page in pages.values():
        if isinstance(page, dict):
            return cast(dict[str, object], page)
    raise ValueError("wiki response contained no pages")


def _open_target(arguments: dict[str, object]) -> _OpenTarget:
    title = _require_non_empty_string(arguments, "title")
    if _looks_like_url(title):
        return _target_from_url(title)
    wiki = _optional_wiki(arguments)
    return _OpenTarget(base_url=wiki, title=title)


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme or parsed.netloc)


def _target_from_url(url: str) -> _OpenTarget:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an http or https URL")
    base_url = _origin(url)
    _validate_wikimedia_base(base_url)
    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        if title:
            return _OpenTarget(base_url=base_url, title=title)
    raise ValueError("url must be a Wikimedia page URL under /wiki/")


def _optional_wiki(arguments: dict[str, object]) -> str:
    value = arguments.get("wiki", _DEFAULT_WIKI)
    if not isinstance(value, str):
        raise ValueError("wiki must be a string")
    base_url = _origin(value.strip())
    _validate_wikimedia_base(base_url)
    return base_url


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("wiki must be an http or https URL")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def _validate_wikimedia_base(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if host in _SUPPORTED_EXACT_HOSTS:
        return
    if any(
        host.endswith(suffix) and host != suffix[1:]
        for suffix in _SUPPORTED_HOST_SUFFIXES
    ):
        return
    raise ValueError("wiki must be a supported Wikimedia wiki URL")


def _page_url(base_url: str, title: str) -> str:
    return f"{base_url}/wiki/{quote(title.replace(' ', '_'), safe=':_()')}"


def _clean_snippet(snippet: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", html.unescape(snippet))
    collapsed = re.sub(r"\s+", " ", without_tags).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", collapsed)


def _require_non_empty_string(arguments: dict[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _optional_int(arguments: dict[str, object], name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
