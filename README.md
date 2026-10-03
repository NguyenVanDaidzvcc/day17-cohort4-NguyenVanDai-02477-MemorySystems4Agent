# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Bản triển khai trong `src/` đã hoàn thiện và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

Kết quả đo và phân tích nằm trong [ANALYSIS.md](ANALYSIS.md), số liệu có thể đọc bằng máy trong [results/benchmark.json](results/benchmark.json). Xem [src/README.md](src/README.md) để dùng API agent.

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md
├── Guide.md
├── Rubric.md
├── ANALYSIS.md             # kết quả đo, trade-off, bonus và giới hạn
├── .env.example
├── requirements.txt        # test + đọc .env
├── requirements-live.txt   # LangChain/LangGraph và sáu provider
├── data/
│   ├── conversations.json
│   └── advanced_long_context.json
├── results/
│   └── benchmark.json
└── src/
    ├── __init__.py
    ├── README.md
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── responses.py        # responder chung, không truy cập expected_contains
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (`state/profiles/<user-slug>-<hash>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự phụ thuộc:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `responses.py` | Sinh câu trả lời offline dùng chung và đọc usage metadata live | `offline_response()`, `live_response()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent mặc định có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Runtime live hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Benchmark xuất các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: proxy heuristic về độ phủ facts và độ gọn; có thể dùng LLM judge trong chế độ live
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Dùng Python `>= 3.11`. Benchmark offline chỉ dùng thư viện chuẩn; file dependency tối thiểu thêm `pytest` để test và `python-dotenv` để đọc `.env`.

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

Linux/macOS:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Nếu không activate môi trường trên Windows, dùng `.\.venv\Scripts\python.exe` thay cho `python`.

Chế độ live là tùy chọn:

~~~powershell
python -m pip install -r requirements-live.txt
Copy-Item .env.example .env
~~~

Điền API key và model phù hợp vào `.env`. File này đã được ignore. `LLM_PROVIDER` chọn một trong sáu provider; `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_API_KEY` cấu hình judge độc lập. `custom` cần `CUSTOM_BASE_URL`; `ollama` cần model đã có trong Ollama và có thể đặt `OLLAMA_BASE_URL`.

| Biến môi trường | Mặc định / ý nghĩa |
| --- | --- |
| `LAB_MODE` | `offline`; đặt `live` để các instance agent gọi LLM |
| `STATE_DIR` | `state`, đường dẫn tương đối tính từ root repo |
| `COMPACT_THRESHOLD_TOKENS` | `1200` |
| `COMPACT_KEEP_MESSAGES` | `6` message gần nhất sau mỗi lần compact |
| `PROFILE_CONFIDENCE_THRESHOLD` | `0.85`; độ mạnh của rule trích fact |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / `gpt-4o-mini` |
| `LLM_TEMPERATURE` | `0` |
| `LLM_BASE_URL` / `LLM_API_KEY` | Tùy chọn, ưu tiên hơn cấu hình theo provider |
| `JUDGE_*` | Kế thừa model chính nếu không override |

## Chạy benchmark và test

Từ root repo:

~~~bash
python src/benchmark.py
python -m pytest src/test_agents.py -v
~~~

Lưu số liệu hoặc chạy dưới dạng module:

~~~bash
python -m src.benchmark --output results/benchmark.json
~~~

Chạy model thật và bật judge tùy chọn:

~~~bash
python src/benchmark.py --live
python src/benchmark.py --live --judge --output results/live-benchmark.json
~~~

`--live` gọi dịch vụ đã cấu hình và có thể phát sinh phí. Benchmark luôn mặc định offline, kể cả khi `LAB_MODE=live`; cần truyền `--live` để dùng API.

Benchmark in đủ hai bảng và sáu chỉ số. Mỗi bộ dữ liệu dùng một thư mục memory tạm, bắt đầu trống và được dọn sau khi chạy. Profile của agent dùng trực tiếp trong `state/profiles/` được giữ nguyên. Các câu recall được hỏi ngay sau từng conversation, mỗi câu trong một thread mới.

Token offline dùng `ceil(len(text.strip()) / 4)` cộng 4 token overhead cho mỗi message; `Agent tokens only` tính output và `Prompt tokens processed` tính input, đều gồm lượt conversation và recall. Live ưu tiên usage metadata từ tất cả lượt gọi model, kể cả tool call; nếu SDK không trả metadata thì dùng ước lượng. Token judge không cộng vào usage của agent. Compact dùng heuristic, không gọi model tóm tắt nên không phát sinh token LLM riêng.

Điểm recall là tỷ lệ các chuỗi kỳ vọng xuất hiện trong câu trả lời, trung bình theo câu hỏi. Điểm quality mặc định là `0.85 × recall + 0.15 × concise`, không phải đánh giá chất lượng ngôn ngữ độc lập. Xem [ANALYSIS.md](ANALYSIS.md) để đọc các giới hạn.

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
