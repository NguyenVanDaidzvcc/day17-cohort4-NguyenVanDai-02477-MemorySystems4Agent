from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_FIELDS = (
    "name", "location", "profession", "response_style", "interests",
    "favorite_drink", "favorite_food", "pet",
)


def estimate_tokens(text: str) -> int:
    """Stable character heuristic, not a provider tokenizer or billing count."""
    return math.ceil(len(text.strip()) / 4) if text.strip() else 0


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(item["content"]) + 4 for item in messages)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonempty string.")
        # Every ID gets a digest, including safe-looking ones, so sanitizing an
        # unsafe ID cannot collide with another user's literal safe ID on Windows.
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:48].strip("_") or "user"
        slug += "-" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path must remain inside the profile directory.")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replacement is atomic; an interrupted write cannot leave a half profile.
        descriptor, temp_name = tempfile.mkstemp(prefix=".profile-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        text = self.read_text(user_id)
        if search_text not in text or search_text == replacement:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        return self.upsert_facts(user_id, {key: value})

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        if any(key not in PROFILE_FIELDS for key in updates):
            raise ValueError("Unknown profile field.")
        current = self.facts(user_id)
        cleaned = {key: " ".join(value.split())[:240] for key, value in updates.items() if value.strip()}
        merged = merge_facts(current, cleaned)
        path = self.path_for(user_id)
        if merged == current:
            return path
        lines = ["# User profile", "", "## Stable facts"]
        lines.extend(f"- {key}: {merged[key]}" for key in PROFILE_FIELDS if key in merged)
        return self.write_text(user_id, "\n".join(lines) + "\n")


def parse_facts(text: str) -> dict[str, str]:
    return {
        key: value.strip()
        for key, value in re.findall(r"^- ([a-z_]+): (.+)$", text, flags=re.MULTILINE)
        if key in PROFILE_FIELDS
    }


def merge_facts(current: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Single-valued fields use the latest assertion; interests/styles are merged."""
    result = dict(current)
    for key, value in updates.items():
        if key in {"interests", "response_style"} and key in result:
            items = result[key].split("; ")
            incoming = value.split("; ")
            if key == "response_style":
                if any(re.fullmatch(r"\d+ bullet", item) for item in incoming):
                    items = [item for item in items if not re.fullmatch(r"\d+ bullet", item)]
                if "chi tiết" in incoming:
                    items = [item for item in items if item != "ngắn gọn"]
                if "ngắn gọn" in incoming:
                    items = [item for item in items if item != "chi tiết"]
            result[key] = "; ".join(dict.fromkeys(items + incoming))
        else:
            result[key] = value
    return result


def _clean_value(value: str) -> str:
    value = re.split(
        r"\s+(?:và|chứ|dù|nhưng|để|cho|vì|trong|mỗi|nữa|không|thì)\b|[,;:]",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return " ".join(value.strip(" .!?\"'").split())[:160]


def extract_profile_updates(message: str, confidence_threshold: float = 0.85) -> dict[str, str]:
    """Conservative Vietnamese assertions; scores describe rule strength, not probabilities.

    Questions, hypothetical/joking clauses and third-person statements are not facts.
    Only fixed, structured fields are persisted, so news/logs do not grow User.md.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    message = unicodedata.normalize("NFC", message)
    updates: dict[str, str] = {}

    def put(key: str, value: str, confidence: float = 0.99) -> None:
        value = _clean_value(value) if key not in {"interests", "response_style"} else value
        if value and confidence >= confidence_threshold:
            updates.update(merge_facts(updates, {key: value}))

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", message):
        sentence = sentence.strip(" .!")
        lower = sentence.casefold()
        if not sentence or "?" in sentence:
            continue
        if re.search(r"\b(?:bạn|anh|chị|em|mẹ|bố|đồng nghiệp|công ty|team|nhóm) mình\s+(?:tên|ở|làm|nuôi)\b", lower):
            continue
        if re.search(r"\b(?:nếu|giả sử|giả dụ)\b", lower):
            # An assertion before an optional tail such as "nếu cần" is still usable.
            sentence = re.split(r"\b(?:nếu|giả sử|giả dụ)\b", sentence, maxsplit=1, flags=re.I)[0]
            lower = sentence.casefold()
        if re.search(r"\b(?:đùa|ví dụ cũ|không phải|không còn thích)\b", lower):
            continue
        if re.search(r"\b(?:là gì|tên gì|ở đâu|nghề gì|nhắc lại|hỏi lại|phải không|đúng không|có phải)\b", lower):
            continue

        for match in re.finditer(r"(?:\bmình tên(?: là)?|\btên mình là)\s+([^,.;!?]+)", sentence, re.I):
            put("name", match[1])
        location_patterns = (
            r"\bmình\s+(?:(?:hiện tại|hiện|vẫn|giờ|đang)\s+){0,3}(?:sống ở|ở|chuyển (?:đến|về))\s+([^,.;!?]+)",
            r"\bhiện(?: tại)? ở\s+([^,.;!?]+)",
            r"\bnơi ở(?: hiện tại)?(?: của mình)?(?: là|:)\s*([^,.;!?]+)",
            r"\bnơi ở đã cập nhật từ [^,.;!?]+? sang\s+([^,.;!?]+)",
        )
        # Collect matches by text position so the last correction wins.
        locations = sorted(
            (match.start(), match[1])
            for pattern in location_patterns for match in re.finditer(pattern, sentence, re.I)
        )
        for _, value in locations:
            put("location", value)

        profession_patterns = (
            r"\bmình\s+(?:(?:hiện tại|hiện|vẫn|đang|giờ)\s+){0,3}(?:làm(?: nghề)?|là)\s+([^,.;!?]+)",
            r"\bgiờ(?: mình)?\s+(?:chuyển sang|làm)\s+([^,.;!?]+)",
            r"\bnghề(?: nghiệp)?(?: hiện tại)?(?: của mình)?(?: thì)?(?: vẫn)?(?: là|:)\s*([^,.;!?]+)",
        )
        if re.search(r"\bmình\s+(?:(?:hiện tại|hiện|vẫn|đang|giờ)\s+){0,3}(?:ở|sống ở|tên)", lower):
            profession_patterns += (r"\bvà\s+(?:(?:đang|vẫn)\s+)?làm\s+([^,.;!?]+)",)
        professions = sorted(
            (match.start(), _clean_value(match[1]))
            for pattern in profession_patterns for match in re.finditer(pattern, sentence, re.I)
        )
        for _, value in professions:
            # "Mình đang làm sạch benchmark" is an activity, not an occupation.
            if re.search(r"(?i)\b(?:engineer|developer|manager|designer|teacher|bác sĩ|kỹ sư|giáo viên|lập trình viên|sinh viên|nhà thiết kế|nhân viên)\b", value):
                put("profession", value)

        for key, pattern in (
            ("favorite_drink", r"\bđồ uống yêu thích(?: của mình)?(?: là|:)\s*([^.;!?]+)"),
            ("favorite_food", r"\bmón ăn yêu thích(?: của mình)?(?: là|:)\s*([^.;!?]+)"),
            ("pet", r"\bmình nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?([^,.;!?]+)"),
        ):
            for match in re.finditer(pattern, sentence, re.I):
                put(key, match[1])

        if re.search(r"\bmình\b.*\b(?:thích|quan tâm)\b", lower):
            interests = [
                word for pattern, word in (
                    (r"\bpython\b", "Python"), (r"\bai\b", "AI"),
                    (r"\bmlops\b", "MLOps"), (r"\brag\b", "RAG"),
                    (r"\bbenchmark memory\b", "benchmark memory"),
                )
                if re.search(pattern, lower)
            ]
            if interests:
                put("interests", "; ".join(interests), 0.90)

        style_context = (
            re.search(r"(?:trả lời|giải thích|style)", lower)
            and re.search(r"(?:mình (?:vẫn )?(?:muốn|thích)|mình rất thích|mình không thích|hãy|bạn nên|style.*(?:giữ|ngắn|bullet))", lower)
        )
        if style_context:
            styles: list[str] = []
            if re.search(r"(?:ngắn|gọn|bullet ngắn)", lower):
                styles.append("ngắn gọn")
            elif "chi tiết" in lower:
                styles.append("chi tiết")
            count = re.search(r"(\d+)\s+bullet", lower)
            if count:
                styles.append(f"{count[1]} bullet")
            elif "bullet" in lower:
                styles.append("bullet")
            if re.search(r"ví dụ thực (?:tế|chiến)", lower):
                styles.append("có ví dụ thực tế / thực chiến")
            if "trade-off" in lower:
                styles.append("so sánh trade-off")
            if styles:
                put("response_style", "; ".join(styles), 0.95)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Merge earlier summaries rather than discarding facts at the next compaction."""
    if max_items < 1:
        return ""
    facts: dict[str, str] = {}
    notes: list[str] = []
    for item in messages:
        if item["role"] == "summary":
            facts = merge_facts(facts, parse_facts(item["content"]))
            notes.extend(re.findall(r"^- note: (.+)$", item["content"], flags=re.MULTILINE))
        elif item["role"] == "user":
            updates = extract_profile_updates(item["content"])
            facts = merge_facts(facts, updates)
            if not updates:
                note = " ".join(item["content"].split())
                if note:
                    notes.append(note[:160] + ("…" if len(note) > 160 else ""))
    notes = list(dict.fromkeys(notes))
    if len(notes) > max_items:
        anchor_count = min(2, max_items // 2)
        notes = notes[:anchor_count] + notes[-(max_items - anchor_count):]
    lines = ["Summary of older messages (notes may describe historical events):"]
    lines.extend(f"- {key}: {facts[key]}" for key in PROFILE_FIELDS if key in facts)
    lines.extend(f"- note: {note}" for note in notes)
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 32 or self.keep_messages < 1:
            raise ValueError("Compaction needs threshold_tokens >= 32 and keep_messages >= 1.")

    def _thread(self, thread_id: str) -> dict[str, Any]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages may enter thread memory.")
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        load = estimate_message_tokens(thread["messages"]) + estimate_tokens(thread["summary"])
        if load <= self.threshold_tokens or len(thread["messages"]) < 2:
            return
        # Keep the current message intact, even when it alone exceeds the threshold.
        keep = min(self.keep_messages, len(thread["messages"]) - 1)
        if len(thread["messages"]) <= self.keep_messages:
            keep = 1
        older = thread["messages"][:-keep]
        history = ([{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []) + older
        summary = summarize_messages(history)
        # A bounded summary prevents successive summaries from growing forever.
        budget = max(16, self.threshold_tokens // 3)
        lines: list[str] = []
        for line in summary.splitlines():
            candidate = "\n".join(lines + [line])
            if estimate_tokens(candidate) <= budget:
                lines.append(line)
        thread["summary"] = "\n".join(lines)
        thread["messages"] = thread["messages"][-keep:]
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        return deepcopy(self._thread(thread_id))

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
