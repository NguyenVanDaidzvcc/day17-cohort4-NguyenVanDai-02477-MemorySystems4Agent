from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from .config import LabConfig, load_config
    from .memory_store import estimate_message_tokens, estimate_tokens, extract_profile_updates, merge_facts
    from .model_provider import build_chat_model
    from .responses import SYSTEM_PROMPT, live_response, offline_response
except ImportError:
    from config import LabConfig, load_config
    from memory_store import estimate_message_tokens, estimate_tokens, extract_profile_updates, merge_facts
    from model_provider import build_chat_model
    from responses import SYSTEM_PROMPT, live_response, offline_response


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Keeps only this process's thread history. No persistent user profile."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.thread_users.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError("A thread_id cannot be shared by different users.")
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        session = self.sessions.setdefault(thread_id, SessionState())
        before_count = len(session.messages)
        session.messages.append({"role": "user", "content": message})
        estimate = estimate_tokens(SYSTEM_PROMPT) + 4 + estimate_message_tokens(session.messages)
        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                {"configurable": {"thread_id": thread_id}},
            )
            answer, output, prompt = live_response(result, before_count + 1, estimate)
        except Exception:
            session.messages.pop()
            raise
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += output
        session.prompt_tokens_processed += prompt
        return {"response": answer, "agent_tokens": output, "prompt_tokens": prompt, "mode": "live"}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt = estimate_tokens(SYSTEM_PROMPT) + 4 + estimate_message_tokens(session.messages)
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts = merge_facts(facts, extract_profile_updates(
                    item["content"], self.config.profile_confidence_threshold,
                ))
        answer = offline_response(message, facts)
        output = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += output
        session.prompt_tokens_processed += prompt
        return {"response": answer, "agent_tokens": output, "prompt_tokens": prompt, "mode": "offline"}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or self.config.offline:
            return None
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
