from __future__ import annotations

import json
import unicodedata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from . import agent_advanced, agent_baseline, model_provider
    from .agent_advanced import AdvancedAgent
    from .agent_baseline import BaselineAgent
    from .benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
    from .config import LabConfig, load_config
    from .memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, parse_facts
    from .model_provider import ProviderConfig, build_chat_model, normalize_provider
except ImportError:
    import agent_advanced
    import agent_baseline
    import model_provider
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent
    from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
    from config import LabConfig, load_config
    from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, parse_facts
    from model_provider import ProviderConfig, build_chat_model, normalize_provider


def make_config(tmp_path: Path) -> LabConfig:
    root = Path(__file__).resolve().parent.parent
    model = ProviderConfig("openai", "test-model")
    return LabConfig(
        base_dir=root, data_dir=root / "data", state_dir=tmp_path / "state",
        compact_threshold_tokens=500, compact_keep_messages=2,
        model=model, judge_model=model, offline=True,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("lan") == ""
    assert store.file_size("lan") == 0
    path = store.write_text("lan", "# User\nTên: Lan\nNơi ở: Huế\n")
    assert path.name == "User.md"
    assert store.file_size("lan") == len(store.read_text("lan").encode("utf-8"))
    assert store.edit_text("lan", "Huế", "Hải Phòng")
    assert "Hải Phòng" in store.read_text("lan")
    assert not store.edit_text("lan", "không tồn tại", "test")
    assert not store.edit_text("lan", "", "test")
    assert not store.edit_text("missing", "test", "test")


def test_profiles_have_safe_distinct_paths(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    ids = ["../../escape", r"..\escape", "a/b", "a_b", "a?b", "alice", "Alice", "CON"]
    paths = [store.write_text(user_id, user_id) for user_id in ids]
    assert len({str(path).casefold() for path in paths}) == len(ids)
    assert all(path.is_relative_to((tmp_path / "profiles").resolve()) for path in paths)
    assert [store.read_text(user_id) for user_id in ids] == ids
    with pytest.raises(ValueError):
        store.path_for("")


def test_fact_upserts_are_bounded_and_replace_conflicts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.upsert_facts("lan", {"name": "Lan", "location": "Huế", "profession": "backend engineer"})
    store.upsert_facts("lan", {"location": "Hải Phòng", "profession": "data engineer"})
    before = store.read_text("lan")
    for _ in range(10):
        store.upsert_fact("lan", "location", "Hải Phòng")
    assert store.read_text("lan") == before
    assert "Huế" not in before and "backend engineer" not in before
    assert store.facts("lan")["profession"] == "data engineer"


@pytest.mark.parametrize("message", [
    "Mình tên là Lan phải không?",
    "Mình đang ở Hà Nội hay Huế?",
    "Đồ uống yêu thích là trà xanh?",
    "Nếu mình làm product manager thì sao?",
    "Nếu sau này mình ở Hà Nội, bạn nhớ giúp nhé.",
    "Bạn có biết DũngCT không?",
    "Bạn mình tên là Minh.",
    "Công ty mình ở Hà Nội.",
    "Mình tên là Lan phải không",
    "Mình đùa là chuyển sang product manager.",
    "Hà Nội chỉ là nơi đi họp chứ không phải nơi ở hiện tại.",
    "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa.",
])
def test_questions_hypotheticals_and_noise_are_not_profile_facts(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_extraction_handles_new_entities_and_corrections() -> None:
    updates = extract_profile_updates(
        "Mình tên là Trần An. Mình ở Hải Phòng và đang làm data engineer cho công ty X. "
        "Món ăn yêu thích là phở bò. Đồ uống yêu thích là trà xanh. Mình nuôi một bé mèo tên Mít."
    )
    assert updates["name"] == "Trần An"
    assert updates["location"] == "Hải Phòng"
    assert updates["profession"] == "data engineer"
    assert updates["favorite_food"] == "phở bò"
    assert updates["favorite_drink"] == "trà xanh"
    assert updates["pet"] == "mèo tên Mít"
    assert extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    )["profession"] == "MLOps engineer"
    assert extract_profile_updates(
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa."
    )["location"] == "Huế"


def test_confidence_threshold_and_unicode_normalization() -> None:
    text = "Mình tên là Trần An. Mình muốn bạn trả lời ngắn gọn."
    assert "response_style" in extract_profile_updates(text)
    conservative = extract_profile_updates(text, confidence_threshold=0.98)
    assert conservative == {"name": "Trần An"}
    assert extract_profile_updates(unicodedata.normalize("NFD", text)) == extract_profile_updates(text)


def test_style_count_can_be_corrected(tmp_path: Path) -> None:
    advanced = AdvancedAgent(make_config(tmp_path), force_offline=True)
    advanced.reply("lan", "t1", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet có ví dụ thực tế.")
    advanced.reply("lan", "t1", "Mình muốn bạn trả lời thành 5 bullet.")
    style = advanced.profile_store.facts("lan")["response_style"]
    assert "5 bullet" in style and "3 bullet" not in style


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply("lan", "long", f"Bản ghi {index}. " + "Dữ liệu kỹ thuật đang được kiểm tra. " * 30)
    context = agent.compact_memory.context("long")
    assert context["compactions"] >= 2
    assert context["summary"]
    assert len(context["messages"]) < 12 * 2
    assert any("Bản ghi 11" in item["content"] for item in context["messages"])
    assert len(context["summary"]) <= 4 * (agent.config.compact_threshold_tokens // 3)


def test_compact_preserves_old_facts_and_episodic_notes() -> None:
    manager = CompactMemoryManager(threshold_tokens=700, keep_messages=2)
    manager.append("t", "user", "Mình tên là Trần An. Mình ở Huế.")
    manager.append("t", "user", "Mã dự án hiện tại: ORION-42.")
    for index in range(20):
        manager.append("t", "user", f"Log {index}: " + "dữ liệu không ổn định " * 40)
        manager.append("t", "assistant", "Đã nhận.")
    summary = manager.context("t")["summary"]
    assert parse_facts(summary)["name"] == "Trần An"
    assert "ORION-42" in summary
    manager.append("t", "user", "Giờ mình đang ở Hải Phòng.")
    for _ in range(3):
        manager.append("t", "user", "thông tin mới " * 100)
    assert parse_facts(manager.context("t")["summary"])["location"] == "Hải Phòng"


def test_compact_context_is_a_snapshot_and_current_message_is_intact() -> None:
    manager = CompactMemoryManager(100, 2)
    message = "Nội dung dài cần giữ nguyên. " * 100
    manager.append("t", "user", message)
    context = manager.context("t")
    assert context["messages"][-1]["content"] == message
    context["messages"].clear()
    assert manager.context("t")["messages"][-1]["content"] == message
    assert manager.compaction_count("different") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply("lan", "t1", "Mình tên là Trần An. Đồ uống yêu thích là trà xanh.")
        within = agent.reply("lan", "t1", "Mình tên gì và đồ uống yêu thích là gì?")["response"]
        assert "Trần An" in within and "trà xanh" in within
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    baseline_answer = baseline.reply("lan", "t2", question)["response"]
    assert "Trần An" not in baseline_answer and "trà xanh" not in baseline_answer
    restarted = AdvancedAgent(config, force_offline=True)
    advanced_answer = restarted.reply("lan", "t2", question)["response"]
    assert "Trần An" in advanced_answer and "trà xanh" in advanced_answer


def test_recall_questions_do_not_change_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("lan", "a", "Mình tên là Trần An. Mình ở Huế và đang làm data engineer.")
    before = agent.profile_store.read_text("lan")
    answer = agent.reply("lan", "b", "Mình tên là Minh và mình ở Hà Nội phải không?")["response"]
    assert "Minh" not in answer and "Hà Nội" not in answer
    assert agent.profile_store.read_text("lan") == before


def test_latest_persistent_correction_wins_over_old_thread(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("lan", "old", "Mình ở Huế và đang làm backend engineer.")
    agent.reply("lan", "new", "Mình ở Hải Phòng và đang làm MLOps engineer.")
    answer = agent.reply("lan", "old", "Mình đang ở đâu và làm nghề gì?")["response"]
    assert "Hải Phòng" in answer and "MLOps engineer" in answer
    assert "Huế" not in answer and "backend engineer" not in answer


def test_users_are_isolated_and_thread_reuse_is_rejected(tmp_path: Path) -> None:
    for agent_class in (BaselineAgent, AdvancedAgent):
        agent = agent_class(make_config(tmp_path), force_offline=True)
        agent.reply("alice", "alice-thread", "Mình tên là Lan.")
        answer = agent.reply("bob", "bob-thread", "Mình tên gì?")["response"]
        assert "Lan" not in answer
        with pytest.raises(ValueError):
            agent.reply("bob", "alice-thread", "Mình tên gì?")


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Phần {index}: " + "Một bản ghi kỹ thuật dài để đo chi phí ngữ cảnh. " * 40
        for agent in (baseline, advanced):
            agent.reply("lan", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.6


def test_usage_accounts_for_prompt_before_response(tmp_path: Path) -> None:
    baseline = BaselineAgent(make_config(tmp_path), force_offline=True)
    first = baseline.reply("lan", "t", "Xin chào.")
    second = baseline.reply("lan", "t", "Mình tên gì?")
    assert baseline.token_usage("t") == first["agent_tokens"] + second["agent_tokens"]
    assert baseline.prompt_token_usage("t") == first["prompt_tokens"] + second["prompt_tokens"]
    assert second["prompt_tokens"] > first["prompt_tokens"]
    assert baseline.token_usage("unknown") == baseline.prompt_token_usage("unknown") == 0
    assert not (baseline.config.state_dir / "profiles").exists()
    assert estimate_tokens("  ") == 0


def test_recall_fraction_and_quality_proxy() -> None:
    assert recall_points("Python và AI", ["Python", "AI", "MLOps", "Huế"]) == 0.5
    assert recall_points("Huế", ["huế"]) == 1.0
    assert recall_points(unicodedata.normalize("NFD", "Huế"), ["Huế"]) == 1.0
    assert recall_points("Chưa biết.", []) == 0.0
    assert heuristic_quality("", ["Python"]) == 0.0
    assert heuristic_quality("Python", ["Python"]) > heuristic_quality("Chưa biết.", ["Python"])


@pytest.mark.parametrize("filename", ["conversations.json", "advanced_long_context.json"])
def test_real_benchmarks_have_correct_recall_and_stress_savings(tmp_path: Path, filename: str) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1200, compact_keep_messages=6)
    conversations = load_conversations(config.data_dir / filename)
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    advanced = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)
    assert baseline.recall_score == 0.0
    assert advanced.recall_score == 1.0
    assert baseline.memory_growth_bytes == baseline.compactions == 0
    assert advanced.memory_growth_bytes > 0
    if filename == "advanced_long_context.json":
        assert advanced.compactions >= 2
        assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed


def test_benchmark_uses_fresh_recall_threads_and_includes_their_usage(tmp_path: Path) -> None:
    class RecordingAgent:
        calls: list[tuple[str, str]] = []
        def reply(self, user_id, thread_id, message):
            self.calls.append((thread_id, message))
            return {"response": "Lan"}
        def token_usage(self, thread_id):
            return sum(t == thread_id for t, _ in self.calls) * 3
        def prompt_token_usage(self, thread_id):
            return sum(t == thread_id for t, _ in self.calls) * 5
        def compaction_count(self, thread_id):
            return 0

    conversations = [{
        "id": "c1", "user_id": "lan", "turns": ["Mình tên là Lan."],
        "recall_questions": [
            {"question": "Mình tên gì?", "expected_contains": ["Lan"]},
            {"question": "Tên mình là gì?", "expected_contains": ["Lan"]},
        ],
    }]
    agent = RecordingAgent()
    row = run_agent_benchmark("Recording", agent, conversations, make_config(tmp_path))
    assert len({thread for thread, _ in agent.calls}) == 3
    assert row.agent_tokens_only == 9 and row.prompt_tokens_processed == 15
    assert row.recall_score == 1.0


def test_dataset_validation(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([{"id": "bad"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_conversations(path)


@pytest.mark.parametrize("alias,expected", [
    (" Anthorpic ", "anthropic"), ("google", "gemini"),
    ("openai-compatible", "custom"), ("OPENAI", "openai"),
])
def test_provider_aliases(alias: str, expected: str) -> None:
    assert normalize_provider(alias) == expected


@pytest.mark.parametrize("provider,class_name", [
    ("openai", "ChatOpenAI"), ("custom", "ChatOpenAI"),
    ("gemini", "ChatGoogleGenerativeAI"), ("anthropic", "ChatAnthropic"),
    ("ollama", "ChatOllama"), ("openrouter", "ChatOpenRouter"),
])
def test_provider_factory_routes_without_network(monkeypatch, provider: str, class_name: str) -> None:
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return kwargs
    monkeypatch.setattr(model_provider, "import_module", lambda module: SimpleNamespace(**{class_name: constructor}))
    config = ProviderConfig(provider, "test-model", 0.2, "test-key", "http://localhost:8000/v1")
    build_chat_model(config)
    assert calls[0]["model"] == "test-model" and calls[0]["temperature"] == 0.2
    if provider != "ollama":
        assert calls[0]["api_key"] == "test-key"


def test_live_config_errors_are_explicit() -> None:
    with pytest.raises(ValueError):
        normalize_provider("unknown")
    with pytest.raises(ValueError, match="CUSTOM_BASE_URL"):
        build_chat_model(ProviderConfig("custom", "test"))
    with pytest.raises(ValueError, match="API key"):
        build_chat_model(ProviderConfig("openai", "test"))


def test_config_loads_paths_and_independent_judge(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "JUDGE_API_KEY", "JUDGE_BASE_URL",
        "COMPACT_THRESHOLD_TOKENS", "COMPACT_KEEP_MESSAGES", "PROFILE_CONFIDENCE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "local-test")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthorpic")
    monkeypatch.setenv("JUDGE_MODEL", "judge-test")
    monkeypatch.setenv("STATE_DIR", "isolated-state")
    monkeypatch.setenv("LAB_MODE", "offline")
    config = load_config(tmp_path)
    assert config.state_dir == tmp_path / "isolated-state"
    assert config.state_dir.exists()
    assert config.model.model_name == "local-test"
    assert config.judge_model.provider == "anthropic"
    assert config.judge_model.model_name == "judge-test"


def test_invalid_compact_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        replace(make_config(tmp_path), compact_threshold_tokens=0)
    with pytest.raises(ValueError):
        CompactMemoryManager(100, 0)


def test_live_graphs_memory_and_usage_without_api(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("langchain")
    pytest.importorskip("langgraph")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from pydantic import Field

    class RecordingModel(FakeMessagesListChatModel):
        recorded: list = Field(default_factory=list)
        def bind_tools(self, tools, **kwargs):
            return self
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.recorded.append(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    response = AIMessage(content="Đã nhận.", usage_metadata={
        "input_tokens": 111, "output_tokens": 7, "total_tokens": 118,
    })
    config = replace(make_config(tmp_path), offline=False)
    baseline_model = RecordingModel(responses=[response.model_copy(deep=True) for _ in range(3)])
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: baseline_model)
    baseline = BaselineAgent(config)
    baseline.reply("lan", "one", "Mình tên là Lan.")
    baseline.reply("lan", "one", "Mình tên gì?")
    assert any("Mình tên là Lan" in str(message.content) for message in baseline_model.recorded[-1])
    baseline.reply("lan", "two", "Mình tên gì?")
    assert not any("Mình tên là Lan" in str(message.content) for message in baseline_model.recorded[-1])
    assert baseline.token_usage("one") == 14
    assert baseline.prompt_token_usage("one") == 222

    advanced_model = RecordingModel(responses=[response.model_copy(deep=True) for _ in range(2)])
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: advanced_model)
    advanced = AdvancedAgent(config)
    advanced.reply("lan", "a", "Mình tên là Lan.")
    restarted = AdvancedAgent(config)
    result = restarted.reply("lan", "b", "Mình tên gì?")
    assert any("name: Lan" in str(message.content) for message in advanced_model.recorded[-1])
    assert result["agent_tokens"] == 7 and result["prompt_tokens"] == 111


def test_live_profile_tool_is_scoped_and_usage_includes_tool_calls(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("langchain")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "read_user_profile", "args": {}, "id": "call-1"}],
                  usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}),
        AIMessage(content="Tên: Lan.",
                  usage_metadata={"input_tokens": 120, "output_tokens": 5, "total_tokens": 125}),
    ])
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    agent = AdvancedAgent(replace(make_config(tmp_path), offline=False))
    result = agent.reply("lan", "t", "Mình tên là Lan.")
    assert result["response"] == "Tên: Lan."
    assert result["agent_tokens"] == 15 and result["prompt_tokens"] == 220
    assert agent.profile_store.facts("lan") == {"name": "Lan"}
