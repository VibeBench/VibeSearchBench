"""Tool implementations: search / visit / python.

Self-contained — uses Serper API for search & web crawling, an LLM
endpoint for summarization, and an HTTP code-sandbox for Python.

All tools expose a single ``async execute(args) -> str`` interface
and are grouped under :class:`ToolKit` for per-query lifecycle.
"""

import asyncio
import json
import logging
import os
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ============================================================================
# Retry helper
# ============================================================================


async def _retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    label: str = "",
) -> T:
    last: Optional[Exception] = None
    d = delay
    for i in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:
            last = e
            logger.warning("%s attempt %d/%d failed: %s", label, i, attempts, e)
            if i < attempts:
                await asyncio.sleep(d * random.uniform(0.8, 1.2))
                d *= backoff
    raise last  # type: ignore[misc]


# ============================================================================
# Serper search
# ============================================================================

_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"
_SERPER_SCRAPE_URL = "https://scrape.serper.dev/"

# Shown to the agent when scrape fails or page content is inaccessible
SCRAPE_FAILED_MESSAGE = "The webpage could not be accessed."


def _convert_date(desc: str) -> str:
    """Best-effort relative->absolute date conversion (no heavy deps)."""
    import re
    from datetime import datetime, timedelta
    m = re.match(r"(\d+)\s+(hour|day|week|month|year)s?\s+ago", (desc or "").strip(), re.I)
    if not m:
        return desc
    n, unit = int(m.group(1)), m.group(2).lower()
    now = datetime.now()
    if unit == "hour":
        return (now - timedelta(hours=n)).strftime("%b %d, %Y")
    if unit == "day":
        return (now - timedelta(days=n)).strftime("%b %d, %Y")
    if unit == "week":
        return (now - timedelta(weeks=n)).strftime("%b %d, %Y")
    if unit == "month":
        return (now - timedelta(days=30 * n)).strftime("%b, %Y")
    if unit == "year":
        return str(now.year - n)
    return desc


async def _serper_search(
    queries: List[str],
    api_key: str,
    topn: int = 10,
) -> str:
    """Search via Serper API, return markdown-formatted snippets."""
    timeout = aiohttp.ClientTimeout(total=30)
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    async def _one(session: aiohttp.ClientSession, query: str) -> str:
        async with session.post(
            _SERPER_SEARCH_URL,
            json={"q": query, "num": topn},
            headers=headers,
        ) as resp:
            if resp.status != 200:
                return f"Search error {resp.status}: {await resp.text()}"
            data = await resp.json()

        results = data.get("organic", [])
        if not results:
            return f"No results for '{query}'"

        lines = [f"A Google search for '{query}' found {len(results)} results:\n\n## Web Results"]
        for idx, r in enumerate(results):
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            date = _convert_date(r.get("date", ""))
            source = r.get("source", "")
            entry = f"{idx}. [{title}]({link})"
            if date:
                entry += f"\nDate published: {date}"
            if source:
                entry += f"\nSource: {source}"
            if snippet:
                entry += f"\n{snippet}"
            lines.append(entry)
        return "\n\n".join(lines)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        parts = await asyncio.gather(
            *(_one(session, q) for q in queries), return_exceptions=True
        )
    return "\n=======\n".join(str(p) for p in parts)


# ============================================================================
# Google Scholar search
# ============================================================================


def _contains_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


async def _serper_scholar_search(
    queries: List[str],
    api_key: str,
) -> str:
    """Search Google Scholar via Serper API, return markdown-formatted results."""
    timeout = aiohttp.ClientTimeout(total=30)
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    async def _one(session: aiohttp.ClientSession, query: str) -> str:
        if _contains_chinese(query):
            payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        else:
            payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}

        async with session.post(
            _SERPER_SCHOLAR_URL,
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                return f"Scholar search error {resp.status}: {await resp.text()}"
            data = await resp.json()

        results = data.get("organic", [])
        if not results:
            return f"No scholar results for '{query}'"

        lines = [f"A Google Scholar search for '{query}' found {len(results)} results:\n\n## Scholar Results"]
        for idx, r in enumerate(results):
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            date = r.get("date", "")
            source = r.get("source", "")
            entry = f"{idx}. [{title}]({link})"
            if date:
                entry += f"\nDate published: {date}"
            if source:
                entry += f"\nSource: {source}"
            if snippet:
                entry += f"\n{snippet}"
            lines.append(entry)
        return "\n\n".join(lines)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        parts = await asyncio.gather(
            *(_one(session, q) for q in queries), return_exceptions=True
        )
    return "\n=======\n".join(str(p) for p in parts)


# ============================================================================
# Visit (crawl + LLM summarize)
# ============================================================================

EXTRACTOR_PROMPT = (
    "Please process the following webpage content and user goal to extract "
    "relevant information:\n\n"
    "## **Webpage Content**\n{webpage_content}\n\n"
    "## **User Goal**\n{goal}\n\n"
    "## **Task Guidelines**\n"
    "1. **Content Scanning**: Locate sections directly related to the user's goal.\n"
    "2. **Key Extraction**: Extract the most relevant information — output the "
    "full original context as far as possible.\n"
    "3. **Summary**: Organize into a concise paragraph with logical flow.\n\n"
    '**Output JSON with "evidence" and "summary" fields**'
)


async def _scrape_url(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
) -> str:
    """Fetch page text via Serper scrape API. Returns SCRAPE_FAILED_MESSAGE on failure."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        async with session.post(
            _SERPER_SCRAPE_URL, json={"url": url}, headers=headers
        ) as resp:
            if resp.status != 200:
                return SCRAPE_FAILED_MESSAGE
            data = await resp.json()
        return data.get("text", "") or SCRAPE_FAILED_MESSAGE
    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return SCRAPE_FAILED_MESSAGE


def _truncate_tokens(text: str, max_chars: int = 150_000) -> str:
    """Rough char-based truncation (avoids heavy tokenizer dependency)."""
    return text[:max_chars] if len(text) > max_chars else text


async def _summarize_page(
    client,  # AsyncOpenAI
    model: str,
    url: str,
    content: str,
    goal: str,
) -> str:
    """Summarize page content with an LLM call."""
    if not content or len(content.strip()) < 20 or content.strip() == SCRAPE_FAILED_MESSAGE.strip():
        return (
            f"The useful information in {url} for user goal {goal} as follows:\n\n"
            f"Summary:\n{SCRAPE_FAILED_MESSAGE}\n"
        )

    content = _truncate_tokens(content)
    prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Summarize LLM call failed for %s: %s", url, e)
        raw = ""

    # Best-effort JSON parse
    try:
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_clean)
        evidence = str(parsed.get("evidence", ""))
        summary = str(parsed.get("summary", ""))
    except (json.JSONDecodeError, TypeError):
        evidence = ""
        summary = raw  # fall back to raw text

    out = f"The useful information in {url} for user goal {goal} as follows:\n\n"
    if evidence:
        out += f"Evidence in page:\n{evidence}\n\n"
    out += f"Summary:\n{summary}\n"
    return out


# ============================================================================
# Python sandbox
# ============================================================================


async def _run_python(code: str, sandbox_url: str) -> str:
    """Execute Python via an HTTP code-sandbox service."""
    payload = {
        "code": code,
        "language": "python",
        "compile_timeout": 60,
        "run_timeout": 120,
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=240)
        ) as session:
            async with session.post(sandbox_url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        run = data.get("run_result", {})
        parts = []
        if run.get("stdout"):
            parts.append(f"stdout:\n{run['stdout']}")
        if run.get("stderr"):
            parts.append(f"stderr:\n{run['stderr']}")
        if run.get("status") == "TimeLimitExceeded":
            parts.append("[Python Interpreter Error] TimeoutError: Execution timed out.")
        return "\n".join(parts).strip() or "(no output)"
    except Exception as e:
        logger.error("Python sandbox failed: %s", e)
        return f"[Python Interpreter Error]: {e}"


# ============================================================================
# BrowserToolKit — optional, requires gpt_oss
# ============================================================================


class BrowserToolKit:
    """Per-query adapter: wraps a shared BrowserPool behind the same
    ``call(fn_name, args) -> str`` interface as :class:`ToolKit`.

    BrowserPool is stateful (page stack, visited pages are keyed by qid), so
    one ``BrowserToolKit`` must be created per query — call :meth:`close` when
    the query is done to release the session.
    """

    def __init__(self, pool: Any, qid: Any) -> None:
        self._pool = pool
        self._qid = qid
        pool.init_session(qid)

    async def call(self, fn_name: str, arguments: dict) -> str:
        return await self._pool.call_simple(self._qid, fn_name, arguments)

    async def close(self) -> None:
        self._pool.cleanup(self._qid)


# ============================================================================
# ToolKit — groups tools into a per-query session
# ============================================================================


class ToolKit:
    """Stateless tool executor for search / visit / python.

    Parameters
    ----------
    serper_api_key : str
        Serper API key (for both search and scrape).
    summarize_client : AsyncOpenAI
        The *same* client the agent uses — avoid creating a second one.
    summarize_model : str
        Model name for the summarization LLM calls.
    sandbox_url : str or None
        HTTP endpoint for the Python sandbox.  ``None`` disables python.
    """

    def __init__(
        self,
        serper_api_key: str,
        summarize_client: Any,
        summarize_model: str,
        sandbox_url: Optional[str] = None,
    ):
        self._api_key = serper_api_key
        self._llm = summarize_client
        self._llm_model = summarize_model
        self._sandbox_url = sandbox_url

    async def call(self, fn_name: str, arguments: dict) -> str:
        """Dispatch a tool call by name.  Returns plain-text result."""
        if fn_name == "search":
            return await self._do_search(arguments)
        if fn_name == "scholar_search":
            return await self._do_scholar_search(arguments)
        if fn_name == "visit":
            return await self._do_visit(arguments)
        if fn_name == "python":
            return await self._do_python(arguments)
        return f"Unknown tool: {fn_name}"

    # ---- search ----

    async def _do_search(self, args: dict) -> str:
        queries = args.get("query", "")
        if isinstance(queries, str):
            queries = [queries]
        topn = args.get("topn", 10)
        return await _retry(
            lambda: _serper_search(queries, self._api_key, topn=topn),
            label="search",
        )

    # ---- scholar_search ----

    async def _do_scholar_search(self, args: dict) -> str:
        queries = args.get("query", "")
        if isinstance(queries, str):
            queries = [queries]
        return await _retry(
            lambda: _serper_scholar_search(queries, self._api_key),
            label="scholar_search",
        )

    # ---- visit ----

    async def _do_visit(self, args: dict) -> str:
        urls = args.get("url", [])
        if isinstance(urls, str):
            # model sometimes wraps list in a string
            if urls.strip().startswith("["):
                try:
                    urls = json.loads(urls)
                except json.JSONDecodeError:
                    urls = [urls]
            else:
                urls = [urls]
        goal = args.get("goal", "")

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            contents = await asyncio.gather(
                *(_scrape_url(session, u, self._api_key) for u in urls),
                return_exceptions=True,
            )

        summaries = await asyncio.gather(
            *(
                _summarize_page(self._llm, self._llm_model, url, str(c), goal)
                for url, c in zip(urls, contents)
            ),
            return_exceptions=True,
        )
        return "\n=======\n".join(str(s) for s in summaries)

    # ---- python ----

    async def _do_python(self, args: dict) -> str:
        if not self._sandbox_url:
            return "[Python Interpreter Error]: Sandbox not configured."
        code = args.get("code", "")
        return await _retry(
            lambda: _run_python(code, self._sandbox_url),
            attempts=2,
            label="python",
        )

    async def close(self) -> None:
        """No-op — ToolKit holds no stateful connections."""
