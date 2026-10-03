# Memory Systems for AI Agent

`src/` đã hoàn thiện hai agent cùng lớp memory, cấu hình provider, benchmark và test. Chạy script từ root repo hoặc import package `src`.

## Ví dụ dùng agent

~~~python
from src.agent_advanced import AdvancedAgent
from src.agent_baseline import BaselineAgent
from src.config import load_config

config = load_config()
baseline = BaselineAgent(config, force_offline=True)
advanced = AdvancedAgent(config, force_offline=True)

for agent in (baseline, advanced):
    agent.reply("demo-user", "session-1", "Mình tên là Lan. Mình ở Hải Phòng.")
    print(agent.reply("demo-user", "session-2", "Mình tên gì và hiện đang ở đâu?")["response"])

# Tạo lại agent vẫn đọc được User.md:
restarted = AdvancedAgent(config, force_offline=True)
print(restarted.reply("demo-user", "session-3", "Mình tên gì?")["response"])
print(restarted.profile_store.path_for("demo-user"))
~~~

Baseline trả các trường chưa biết ở session mới; Advanced trả Lan và Hải Phòng từ profile. `reply()` trả `response`, `agent_tokens`, `prompt_tokens` và `mode`. Hai bộ đếm token là số của lượt hiện tại; `token_usage(thread_id)` và `prompt_token_usage(thread_id)` trả tổng của thread.

Một `thread_id` thuộc đúng một `user_id`; dùng thread của người khác gây `ValueError`. ID người dùng được làm sạch và thêm hash khi xây đường dẫn file để tránh traversal và trùng tên trên Windows.

## Ba lớp memory

- Baseline giữ toàn bộ messages của mỗi thread trong RAM, không ghi profile và không compact.
- Advanced ghi facts ổn định vào `User.md`: `name`, `location`, `profession`, `response_style`, `interests`, `favorite_drink`, `favorite_food`, `pet`. Các trường đơn dùng bản đính chính mới nhất; interests và style được gộp có giới hạn.
- Advanced giữ recent messages và summary của mỗi thread trong `CompactMemoryManager`. Summary kế thừa facts của summary trước, giữ một số note đầu/gần nhất và bị giới hạn độ dài. Threshold là ngưỡng kích hoạt, không phải hard limit: message hiện tại luôn được giữ nguyên.

Profile bền vững qua restart; recent messages và summary nằm trong RAM nên chỉ sống theo instance agent. `UserProfileStore` cung cấp read/write/edit, đọc facts, upsert và đo byte thực tế. Write dùng file tạm cùng thư mục rồi atomic replace.

## Chế độ live

`LAB_MODE=live` bật live cho agent; `force_offline=True` luôn ưu tiên. SDK được import khi chạy live, lỗi cấu hình hoặc API được báo rõ, không tự chuyển sang offline.

Baseline dùng `create_agent` với `InMemorySaver`. Advanced dùng graph với hai tool đọc/upsert profile, runtime gắn user hiện tại và middleware prompt động. Tool ghi chỉ trích facts từ message thật của lượt hiện tại, không nhận một profile tùy ý do model tạo. Advanced truyền summary và recent messages vào mỗi invocation, không gắn thêm checkpointer mang lại lịch sử đã compact.

Cách lưu thread bằng checkpointer theo [tài liệu LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory). Hai agent dùng chung responder offline, giúp kết quả khác nhau do lượng memory có sẵn.

## Kiểm tra và kết quả

~~~bash
python -m pytest src/test_agents.py -v
python src/benchmark.py --output results/benchmark.json
~~~

Các test xác minh persistence sau restart, recall trong/ngoài thread, correction, chống lưu câu hỏi và nhiễu, cách ly người dùng, profile không tăng khi upsert lặp lại, nhiều lần compact, giữ facts/notes cũ và giảm prompt load. Test live dùng graph thật với model giả lập, không cần key; tự skip nếu chưa cài LangChain/LangGraph.

[README](../README.md) có setup, biến môi trường và lệnh live. [ANALYSIS](../ANALYSIS.md) ghi kết quả và trade-off.
