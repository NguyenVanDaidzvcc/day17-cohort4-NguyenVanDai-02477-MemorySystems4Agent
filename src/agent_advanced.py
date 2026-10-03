from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .config import LabConfig, load_config
    from .memory_store import (
        CompactMemoryManager, UserProfileStore, estimate_message_tokens, estimate_tokens,
        extract_profile_updates, merge_facts, parse_facts,
    )
    from .model_provider import build_chat_model
    from .responses import SYSTEM_PROMPT, live_response, offline_response
except ImportError:
    from config import LabConfig, load_config
    from memory_store import (
        CompactMemoryManager, UserProfileStore, estimate_message_tokens, estimate_tokens,
        extract_profile_updates, merge_facts, parse_facts,
    )
    from model_provider import build_chat_model
    from responses import SYSTEM_PROMPT, live_response, offline_response


@dataclass
class AgentContext:
    user_id: str
    memory_path: str
    thread_id: str = ""
    latest_message: str = ""


class AdvancedAgent:
    """Short-term threads, atomic User.md persistence, and bounded compact summaries."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.thread_users.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError("A thread_id cannot be shared by different users.")
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        self._prepare_turn(user_id, thread_id, message)
        context = self.compact_memory.context(thread_id)
        prompt = self._estimate_prompt_context_tokens(user_id, thread_id)
        result = self.langchain_agent.invoke(
            {"messages": context["messages"]},
            context=AgentContext(user_id, str(self.profile_store.path_for(user_id)), thread_id, message),
        )
        answer, output, prompt = live_response(result, len(context["messages"]), prompt)
        self._finish_turn(thread_id, answer, output, prompt)
        return {"response": answer, "agent_tokens": output, "prompt_tokens": prompt, "mode": "live"}

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _prepare_turn(self, user_id: str, thread_id: str, message: str) -> None:
        updates = extract_profile_updates(message, self.config.profile_confidence_threshold)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)
        self.compact_memory.append(thread_id, "user", message)

    def _finish_turn(self, thread_id: str, answer: str, output: int, prompt: int) -> None:
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._prepare_turn(user_id, thread_id, message)
        prompt = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        output = estimate_tokens(answer)
        self._finish_turn(thread_id, answer, output, prompt)
        return {"response": answer, "agent_tokens": output, "prompt_tokens": prompt, "mode": "offline"}

    def _system_prompt(self, user_id: str, thread_id: str) -> str:
        summary = self.compact_memory.context(thread_id)["summary"]
        profile = self.profile_store.read_text(user_id) or "(Chưa có profile)"
        return (
            f"{SYSTEM_PROMPT}\nHồ sơ hiện tại (facts mới nhất):\n<profile>\n{profile}\n</profile>"
            f"\nTóm tắt thread, có thể chứa thông tin lịch sử:\n<summary>\n{summary}\n</summary>"
        )

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return estimate_tokens(self._system_prompt(user_id, thread_id)) + 4 + estimate_message_tokens(context["messages"])

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        context = self.compact_memory.context(thread_id)
        facts = parse_facts(context["summary"])
        for item in context["messages"]:
            if item["role"] == "user":
                facts = merge_facts(facts, extract_profile_updates(
                    item["content"], self.config.profile_confidence_threshold,
                ))
        # Current persisted assertions win over stale facts from another thread.
        facts.update(self.profile_store.facts(user_id))
        return offline_response(message, facts)

    def _maybe_build_langchain_agent(self):
        if self.force_offline or self.config.offline:
            return None
        from langchain.agents import create_agent
        from langchain.agents.middleware import dynamic_prompt
        from langchain.tools import ToolRuntime, tool

        def read_user_profile(runtime) -> str:
            """Read the current user's stable profile facts."""
            return self.profile_store.read_text(runtime.context.user_id) or "(Chưa có profile)"

        def remember_user_facts(runtime) -> str:
            """Write or edit stable facts asserted in the current user message."""
            updates = extract_profile_updates(
                runtime.context.latest_message, self.config.profile_confidence_threshold,
            )
            if updates:
                self.profile_store.upsert_facts(runtime.context.user_id, updates)
            return "Đã cập nhật facts rõ ràng." if updates else "Không có fact đủ tin cậy để ghi."

        # Resolve the lazy-imported runtime type before Pydantic inspects the tools.
        for function in (read_user_profile, remember_user_facts):
            function.__annotations__["runtime"] = ToolRuntime[AgentContext]

        @dynamic_prompt
        def memory_prompt(request) -> str:
            context = request.runtime.context
            return self._system_prompt(context.user_id, context.thread_id)

        # CompactMemoryManager is authoritative: each invocation receives only the
        # summary and recent messages. A second checkpointer would restore old history.
        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[tool(read_user_profile), tool(remember_user_facts)],
            context_schema=AgentContext,
            middleware=[memory_prompt],
        )
