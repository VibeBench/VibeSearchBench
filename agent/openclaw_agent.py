"""OpenClaw agent: wraps the openclaw CLI into viberesearch's BaseAgent interface.

Communicates with openclaw via its gateway + CLI (subprocess), reading
responses from session JSONL files.  Supports all three benchmark modes
(direct / staged / simulated) with idle-polling and nudge logic.
"""

import asyncio
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from . import BaseAgent, register, summarize_time_stats

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

POLL_INTERVAL = 5
DEFAULT_IDLE_THRESHOLD = 300
DEFAULT_MAX_NUDGE = 3
NUDGE_MESSAGE = "你好，我没有看到你的回答，请问有结果了吗？"


# ============================================================================
# OpenclawDriver — gateway + CLI communication layer
# ============================================================================


class OpenclawDriver:
    """Manage an openclaw gateway process and send/receive messages via CLI."""

    def __init__(
        self,
        gateway_port: int,
        results_dir: str,
        source_dir: str,
        idle_threshold: int = DEFAULT_IDLE_THRESHOLD,
        openclaw_model: Optional[str] = None,
    ):
        self.gateway_port = gateway_port
        self.results_dir = results_dir
        self.source_dir = source_dir
        self.idle_threshold = idle_threshold
        self.openclaw_model = openclaw_model
        self.gateway_pid: Optional[int] = None
        self.state_dir: str = ""

    @property
    def session_dir(self) -> str:
        return f"{self.state_dir}/agents/main/sessions"

    # ── per-task workspace ──

    def setup_task(self, task_id: str):
        self.stop_gateway()
        self.state_dir = f"{self.results_dir}/{task_id}"

        for subdir in ("agents", "workspace", "openclaw.json"):
            p = f"{self.state_dir}/{subdir}"
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)

        os.makedirs(f"{self.state_dir}/agents/main/agent", exist_ok=True)
        os.makedirs(f"{self.state_dir}/agents/main/sessions", exist_ok=True)
        os.makedirs(f"{self.state_dir}/workspace", exist_ok=True)

        src_config = f"{self.source_dir}/openclaw.json"
        dst_config = f"{self.state_dir}/openclaw.json"
        with open(src_config) as f:
            cfg = json.load(f)
        defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        defaults["workspace"] = f"{self.state_dir}/workspace"
        if self.openclaw_model:
            defaults.setdefault("model", {})["primary"] = self.openclaw_model
        cfg.setdefault("gateway", {})["port"] = self.gateway_port
        with open(dst_config, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        for fname in ("auth-profiles.json", "auth-state.json", "models.json"):
            src = f"{self.source_dir}/agents/main/agent/{fname}"
            dst = f"{self.state_dir}/agents/main/agent/{fname}"
            if os.path.exists(src):
                subprocess.run(["cp", "-f", src, dst], check=False)

        self.start_gateway()

    # ── gateway lifecycle ──

    def _pids_on_gateway_port(self) -> set:
        pids: set = set()
        for cmd in (
            ["fuser", "-n", "tcp", str(self.gateway_port)],
            ["lsof", "-t", f"-i:{self.gateway_port}"],
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True)
                for m in re.findall(r"\b\d+\b", f"{out.stdout}\n{out.stderr}"):
                    pids.add(int(m))
                if pids:
                    return pids
            except FileNotFoundError:
                pass
        try:
            out = subprocess.run(
                ["ss", "-lptn", f"sport = :{self.gateway_port}"],
                capture_output=True, text=True,
            )
            for m in re.findall(r"pid=(\d+)", out.stdout):
                pids.add(int(m))
        except FileNotFoundError:
            pass
        return pids

    def _stop_processes_on_gateway_port(self):
        pids = self._pids_on_gateway_port()
        if not pids:
            return
        logger.info("port %d occupied by pids=%s, stopping", self.gateway_port, sorted(pids))
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        deadline = time.time() + 5
        remaining = set(pids)
        while time.time() < deadline and remaining:
            alive = set()
            for pid in remaining:
                try:
                    os.kill(pid, 0)
                    alive.add(pid)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    alive.add(pid)
            remaining = alive
            if remaining:
                time.sleep(0.5)

        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if remaining:
            time.sleep(1)

    def start_gateway(self):
        self._stop_processes_on_gateway_port()
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = self.state_dir
        proc = subprocess.Popen(
            ["openclaw", "gateway", "run", "--port", str(self.gateway_port), "--allow-unconfigured"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.gateway_pid = proc.pid
        logger.info("gateway starting pid=%d port=%d", proc.pid, self.gateway_port)

        import urllib.request
        for _ in range(60):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.gateway_port}/health", timeout=2
                )
                logger.info("gateway ready")
                time.sleep(2)
                return
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("gateway process exited during startup")
                time.sleep(1)
        raise RuntimeError("gateway failed to become ready in 60s")

    def stop_gateway(self):
        if self.gateway_pid:
            logger.info("stopping gateway pid=%d", self.gateway_pid)
            try:
                os.kill(self.gateway_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.gateway_pid = None
        self._stop_processes_on_gateway_port()

    # ── messaging ──

    def _run_agent(self, session_id: str, message: str):
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = self.state_dir
        env["OPENCLAW_GATEWAY_PORT"] = str(self.gateway_port)
        proc = subprocess.run(
            ["openclaw", "agent", "--json", "--session-id", session_id, "--message", message],
            env=env, capture_output=True,
        )
        logger.debug("agent stdout: %s", proc.stdout.decode()[:500])
        logger.debug("agent stderr: %s", proc.stderr.decode()[:500])

    def _resolve_jsonl(self, session_id: str) -> Optional[str]:
        sessions_json = f"{self.session_dir}/sessions.json"
        if not os.path.exists(sessions_json):
            return None
        with open(sessions_json) as f:
            data = json.load(f)
        for key, sess in data.items():
            if session_id.lower() in key.lower():
                real_id = sess.get("sessionId", "")
                path = f"{self.session_dir}/{real_id}.jsonl"
                if os.path.exists(path):
                    return path
        return None

    def _get_last_assistant_text(self, session_id: str) -> Optional[str]:
        jsonl_path = self._resolve_jsonl(session_id)
        if not jsonl_path:
            return None
        entries = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(line)
        for raw in reversed(entries):
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            has_tool = any(
                b.get("type") == "tool_use" for b in content if isinstance(b, dict)
            )
            if has_tool:
                continue
            texts = [
                b.get("text", "").strip()
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
            ]
            if texts:
                return "\n".join(texts)
        return None

    def send_and_get_response(self, session_id: str, message: str) -> Optional[str]:
        self._run_agent(session_id, message)
        return self.check_response(session_id)

    def check_response(self, session_id: str) -> Optional[str]:
        text = self._get_last_assistant_text(session_id)
        if text and text.strip() != "NO_REPLY":
            return text
        return None


# ============================================================================
# Nudge helpers
# ============================================================================


def _poll_for_response(
    driver: OpenclawDriver, session_id: str, timeout: float,
) -> Optional[str]:
    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        logger.debug("polling for response (%.0fs elapsed)", time.monotonic() - start)
        time.sleep(POLL_INTERVAL)
        resp = driver.check_response(session_id)
        if resp is not None:
            return resp
    return None


async def _send_with_nudge(
    driver: OpenclawDriver,
    session_id: str,
    message: str,
    max_nudge: int = DEFAULT_MAX_NUDGE,
    nudge_generator=None,
) -> tuple:
    """Send *message* and return ``(response_text, nudge_messages_sent)``.

    If *nudge_generator* (an async callable returning ``str``) is provided,
    nudge messages are generated by the user agent LLM; otherwise a fixed
    default message is used.
    """
    nudges_sent: List[str] = []

    response = await asyncio.to_thread(
        driver.send_and_get_response, session_id, message,
    )
    if response is not None:
        return response, nudges_sent

    response = await asyncio.to_thread(
        _poll_for_response, driver, session_id, driver.idle_threshold,
    )
    if response is not None:
        return response, nudges_sent

    for nudge_i in range(max_nudge):
        nudge_text = (await nudge_generator()) if nudge_generator else NUDGE_MESSAGE
        nudges_sent.append(nudge_text)
        logger.info("nudge #%d/%d: %s", nudge_i + 1, max_nudge, nudge_text[:80])
        response = await asyncio.to_thread(
            driver.send_and_get_response, session_id, nudge_text,
        )
        if response is not None:
            return response, nudges_sent
        response = await asyncio.to_thread(
            _poll_for_response, driver, session_id, driver.idle_threshold,
        )
        if response is not None:
            return response, nudges_sent

    logger.warning("no reply after %d nudges", max_nudge)
    return None, nudges_sent


# ============================================================================
# OpenclawAgent — BaseAgent subclass
# ============================================================================


@register("openclaw")
class OpenclawAgent(BaseAgent):
    """Evaluate openclaw through the viberesearch benchmark."""

    name = "Agent (OpenClaw)"

    def __init__(
        self,
        gateway_port: int = 18789,
        source_dir: str = "./openclaw_backup",
        openclaw_results_dir: str = "./openclaw_results",
        idle_threshold: int = DEFAULT_IDLE_THRESHOLD,
        max_nudge: int = DEFAULT_MAX_NUDGE,
        openclaw_model: Optional[str] = None,
        # user simulator LLM (for simulated mode nudges)
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        # accept and ignore GeneralAgent-specific kwargs
        **_kwargs,
    ):
        self._gateway_port = gateway_port
        self._source_dir = source_dir
        self._openclaw_results_dir = openclaw_results_dir
        self._idle_threshold = idle_threshold
        self._max_nudge = max_nudge
        self._openclaw_model = openclaw_model
        self._base_url = base_url
        self._api_key = api_key or "EMPTY"
        self._model = model_name
        self._driver: Optional[OpenclawDriver] = None

    async def setup(self) -> None:
        self._driver = OpenclawDriver(
            gateway_port=self._gateway_port,
            results_dir=self._openclaw_results_dir,
            source_dir=self._source_dir,
            idle_threshold=self._idle_threshold,
            openclaw_model=self._openclaw_model,
        )
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _cleanup_handler(signum, frame):
            logger.info("received signal %d, stopping gateway", signum)
            if self._driver:
                self._driver.stop_gateway()
            prev = self._prev_sigint if signum == signal.SIGINT else self._prev_sigterm
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _cleanup_handler)
        signal.signal(signal.SIGTERM, _cleanup_handler)

    async def teardown(self) -> None:
        if self._driver:
            self._driver.stop_gateway()
        signal.signal(signal.SIGINT, self._prev_sigint or signal.SIG_DFL)
        signal.signal(signal.SIGTERM, self._prev_sigterm or signal.SIG_DFL)

    # ---- helpers ----

    def _new_session_id(self, qid: Any) -> str:
        return f"vibe-{qid}-{int(time.time())}"

    def _setup_task_sync(self, qid: Any, sample_idx: int = 0):
        safe_qid = str(qid).replace("/", "_").replace("\\", "_")
        task_id = f"{safe_qid}_s{sample_idx}"
        self._driver.setup_task(task_id)

    @staticmethod
    def _append_to_trace(
        trace: List[dict], role: str, content: str,
    ):
        trace.append({"role": role, "content": content})

    @staticmethod
    def _prepend_system_prompt(system_prompt: str, message: str) -> str:
        return f"{system_prompt}\n\n---\n\n{message}"

    # ---- direct mode ----

    async def run_one(
        self,
        question: str,
        qid: Any,
        system_prompt_override: Optional[str] = None,
        sample_idx: int = 0,
        **_kwargs,
    ) -> dict:
        trace: List[dict] = []
        time_stats: Dict[str, list] = {"openclaw": []}
        t0 = time.time()

        await asyncio.to_thread(self._setup_task_sync, qid, sample_idx)
        session_id = self._new_session_id(qid)

        first_message = question
        if system_prompt_override:
            self._append_to_trace(trace, "system", system_prompt_override)
            first_message = self._prepend_system_prompt(system_prompt_override, question)
        self._append_to_trace(trace, "user", question)

        send_t0 = time.time()
        response, nudges = await _send_with_nudge(
            self._driver, session_id, first_message, self._max_nudge,
        )
        time_stats["openclaw"].append(time.time() - send_t0)

        for nudge_text in nudges:
            self._append_to_trace(trace, "user", nudge_text)

        if response:
            self._append_to_trace(trace, "assistant", response)
            termination = "answer"
        else:
            termination = "no_response"

        return {
            "messages": trace,
            "termination": termination,
            "time_stats": summarize_time_stats(time_stats),
            "turns": 1 + len(nudges),
            "prompt_tokens": 0,
        }

    # ---- staged mode ----

    async def run_one_staged(
        self,
        initial_query: str,
        sub_queries: List[str],
        qid: Any,
        system_prompt_override: Optional[str] = None,
        triple_request_prompt: Optional[str] = None,
        sample_idx: int = 0,
        **_kwargs,
    ) -> dict:
        trace: List[dict] = []
        time_stats: Dict[str, list] = {"openclaw": []}

        await asyncio.to_thread(self._setup_task_sync, qid, sample_idx)
        session_id = self._new_session_id(qid)

        if system_prompt_override:
            self._append_to_trace(trace, "system", system_prompt_override)

        termination = "answer"

        # Phase 1: initial query
        first_message = initial_query
        if system_prompt_override:
            first_message = self._prepend_system_prompt(system_prompt_override, initial_query)
        self._append_to_trace(trace, "user", initial_query)
        send_t0 = time.time()
        response, nudges = await _send_with_nudge(
            self._driver, session_id, first_message, self._max_nudge,
        )
        time_stats["openclaw"].append(time.time() - send_t0)
        for nudge_text in nudges:
            self._append_to_trace(trace, "user", nudge_text)
        if response:
            self._append_to_trace(trace, "assistant", response)
        else:
            termination = "no_response"

        # Phase 2: sequential sub-queries
        if termination != "no_response":
            for sq in sub_queries:
                self._append_to_trace(trace, "user", sq)
                send_t0 = time.time()
                response, nudges = await _send_with_nudge(
                    self._driver, session_id, sq, self._max_nudge,
                )
                time_stats["openclaw"].append(time.time() - send_t0)
                for nudge_text in nudges:
                    self._append_to_trace(trace, "user", nudge_text)
                if response:
                    self._append_to_trace(trace, "assistant", response)
                else:
                    termination = "no_response"
                    break

        # Phase 3: request triple output
        if termination != "no_response" and triple_request_prompt:
            self._append_to_trace(trace, "user", triple_request_prompt)
            send_t0 = time.time()
            response, nudges = await _send_with_nudge(
                self._driver, session_id, triple_request_prompt, self._max_nudge,
            )
            time_stats["openclaw"].append(time.time() - send_t0)
            for nudge_text in nudges:
                self._append_to_trace(trace, "user", nudge_text)
            if response:
                self._append_to_trace(trace, "assistant", response)
            else:
                termination = "no_response"

        total_user_msgs = sum(1 for m in trace if m["role"] == "user")
        return {
            "messages": trace,
            "termination": termination,
            "time_stats": summarize_time_stats(time_stats),
            "turns": total_user_msgs,
            "prompt_tokens": 0,
        }

    # ---- simulated mode ----

    async def run_one_simulated(
        self,
        initial_query: str,
        user_persona: str,
        qid: Any,
        system_prompt_override: Optional[str] = None,
        triple_request_prompt: Optional[str] = None,
        max_user_turns: int = 30,
        user_model_name: Optional[str] = None,
        user_model_url: Optional[str] = None,
        user_model_api_key: Optional[str] = None,
        sample_idx: int = 0,
        **_kwargs,
    ) -> dict:
        from .prompts import USER_SIMULATOR_SYSTEM_PROMPT

        trace: List[dict] = []
        time_stats: Dict[str, list] = {"openclaw": [], "user_sim": []}

        await asyncio.to_thread(self._setup_task_sync, qid, sample_idx)
        session_id = self._new_session_id(qid)

        if system_prompt_override:
            self._append_to_trace(trace, "system", system_prompt_override)

        # User simulator setup
        sim_system_prompt = USER_SIMULATOR_SYSTEM_PROMPT.format(
            user_persona=user_persona or "A curious researcher",
            initial_query=initial_query,
        )
        user_agent_msgs: List[dict] = [
            {"role": "system", "content": sim_system_prompt},
        ]
        sim_client = AsyncOpenAI(
            base_url=user_model_url or self._base_url,
            api_key=user_model_api_key or self._api_key or "EMPTY",
        )
        sim_model = user_model_name or self._model

        termination = "answer"

        try:
            # Phase 1: DR agent responds to initial query
            first_message = initial_query
            if system_prompt_override:
                first_message = self._prepend_system_prompt(system_prompt_override, initial_query)
            self._append_to_trace(trace, "user", initial_query)
            send_t0 = time.time()

            async def _nudge_gen():
                return await self._call_user_simulator_nudge(
                    user_agent_msgs, sim_client, sim_model,
                )

            response, nudges = await _send_with_nudge(
                self._driver, session_id, first_message,
                _nudge_gen, self._max_nudge,
            )
            time_stats["openclaw"].append(time.time() - send_t0)
            for nudge_text in nudges:
                self._append_to_trace(trace, "user", nudge_text)

            if not response:
                termination = "no_response"
            else:
                self._append_to_trace(trace, "assistant", response)
                dr_response = response

                # Phase 2: alternating turns
                for turn_num in range(1, max_user_turns + 1):
                    # User simulator generates follow-up
                    user_agent_msgs.append({"role": "user", "content": dr_response})
                    sim_t0 = time.time()
                    sim_response = await self._call_user_simulator(
                        user_agent_msgs, sim_client, sim_model,
                    )
                    time_stats["user_sim"].append(time.time() - sim_t0)
                    user_agent_msgs.append({"role": "assistant", "content": sim_response})

                    if "[DONE]" in sim_response:
                        logger.info("qid=%s: user simulator [DONE] at turn %d", qid, turn_num)
                        break

                    # Send user simulator message to openclaw
                    self._append_to_trace(trace, "user", sim_response)
                    send_t0 = time.time()
                    response, nudges = await _send_with_nudge(
                        self._driver, session_id, sim_response,
                        _nudge_gen, self._max_nudge,
                    )
                    time_stats["openclaw"].append(time.time() - send_t0)
                    for nudge_text in nudges:
                        self._append_to_trace(trace, "user", nudge_text)

                    if not response:
                        termination = "no_response"
                        break

                    self._append_to_trace(trace, "assistant", response)
                    dr_response = response

            # Phase 3: request triple output
            if termination != "no_response" and triple_request_prompt:
                self._append_to_trace(trace, "user", triple_request_prompt)
                send_t0 = time.time()
                response, nudges = await _send_with_nudge(
                    self._driver, session_id, triple_request_prompt,
                    _nudge_gen, self._max_nudge,
                )
                time_stats["openclaw"].append(time.time() - send_t0)
                for nudge_text in nudges:
                    self._append_to_trace(trace, "user", nudge_text)
                if response:
                    self._append_to_trace(trace, "assistant", response)
                else:
                    termination = "no_response"

        finally:
            await sim_client.close()

        total_user_msgs = sum(1 for m in trace if m["role"] == "user")
        return {
            "messages": trace,
            "termination": termination,
            "time_stats": summarize_time_stats(time_stats),
            "turns": total_user_msgs,
            "prompt_tokens": 0,
        }

    # ---- user simulator helpers ----

    @staticmethod
    async def _call_user_simulator(
        user_agent_msgs: List[dict],
        client: AsyncOpenAI,
        model: str,
    ) -> str:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=user_agent_msgs,
                temperature=0.7,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("User simulator call failed: %s", e)
            return "[DONE]"

    @staticmethod
    async def _call_user_simulator_nudge(
        user_agent_msgs: List[dict],
        client: AsyncOpenAI,
        model: str,
    ) -> str:
        nudge_instruction = {
            "role": "user",
            "content": (
                "助手已经过了一段时间还没有回复你。请保持你的人设角色，"
                "根据当前阶段生成一条简短自然的跟进消息催促助手（1句话即可）。"
            ),
        }
        try:
            msgs = list(user_agent_msgs) + [nudge_instruction]
            resp = await client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0.7,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = text.replace("[DONE]", "").strip()
            return text if text else NUDGE_MESSAGE
        except Exception as e:
            logger.warning("User simulator nudge call failed: %s", e)
            return NUDGE_MESSAGE

    # ---- response extraction ----

    def extract_response(self, messages: List[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content", "").strip():
                return msg["content"].strip()
        return ""
