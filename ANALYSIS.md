# Kết quả và phân tích Day 17

Đo ngày 03/10/2026, Python 3.11.0, chế độ offline. Cấu hình: threshold `1200` token, giữ `6` messages sau compact, confidence threshold `0.85`. Số liệu gốc: [results/benchmark.json](results/benchmark.json).

~~~bash
python src/benchmark.py --output results/benchmark.json
python -m pytest src/test_agents.py -v
~~~

## Kết quả đo

Standard Benchmark: 10 conversation, 101 lượt người dùng và 14 câu recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1084 | 23803 | 0.0% | 15.0% | 0 | 0 |
| Advanced | 1085 | 34986 | 100.0% | 100.0% | 328 | 0 |

Long-Context Stress Benchmark: 1 conversation, 16 lượt dài và 3 câu recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 238 | 23805 | 0.0% | 15.0% | 0 | 0 |
| Advanced | 259 | 15615 | 100.0% | 100.0% | 243 | 3 |

Mỗi suite bắt đầu với memory trống. Hai agent nhận cùng lượt nhập, dùng cùng responder offline; bộ sinh câu trả lời không đọc `expected_contains` hay file dữ liệu. Recall được hỏi ngay sau conversation tương ứng, mỗi câu trong thread mới, trước các correction của conversation tiếp theo.

## Vì sao Advanced nhớ tốt hơn

Baseline chỉ truy xuất user messages của thread đang hỏi. Các câu recall mở thread mới nên không có tên, nghề, nơi ở hay sở thích để trả lời. Đây là hành vi mong muốn của baseline; trong thread cũ, cả hai agent đều có thể nhắc đúng facts.

Advanced đọc `User.md` trong mọi thread của cùng user. Tên và sở thích đã xác nhận tồn tại ngay cả khi tạo lại instance. Correction ghi đè một trường đơn: Standard chuyển nơi ở Đà Nẵng → Huế và nghề backend engineer → MLOps engineer; Stress chuyển nơi ở Huế → Đà Nẵng. Hà Nội là nơi đi họp và product manager là câu đùa nên không thành profile facts.

Prompt profile chứa facts hiện tại; summary được ghi rõ có thể chứa lịch sử. Offline ưu tiên persistent facts mới nhất, kể cả khi quay lại thread cũ sau khi đã sửa thông tin trong thread khác.

## Vì sao hội thoại ngắn tốn thêm

Standard không vượt ngưỡng compact. Advanced kéo theo `User.md` và phần bao prompt qua mỗi lượt, nên prompt tăng từ 23.803 lên 34.986 token ước lượng, tương đương **46,98%**. Output gần như ngang nhau: 1.084 so với 1.085.

Đây là chi phí cho recall dài hạn. Thêm profile và logic memory không tự động giảm số token trong mọi loại hội thoại. Trong bộ ngắn, compaction chưa tạo lợi ích để bù phần context tăng thêm.

## Vì sao compact có lợi ở hội thoại dài

Baseline gửi lại toàn bộ lịch sử đang tăng qua từng lượt, nên cùng đoạn dài bị xử lý nhiều lần. Advanced compact 3 lần, thay phần cũ bằng summary có giới hạn, giữ messages gần đây để xử lý follow-up.

Prompt giảm từ 23.805 xuống 15.615, tức **34,40%**, trong khi recall profile vẫn đạt 100% trên bộ mẫu. Output lại tăng 8,82% (238 → 259) vì câu trả lời chứa thông tin đã nhớ. Lợi ích đo được chủ yếu ở prompt input, không phải số token sinh ra.

Compact dùng heuristic nên không gọi LLM tóm tắt. Nếu thay bằng LLM, cần cộng cả token input/output của các lần tóm tắt vào chi phí tổng. Ngưỡng ở đây là điều kiện kích hoạt; một message hiện tại quá dài vẫn được giữ nguyên nên không đảm bảo context tuyệt đối luôn dưới ngưỡng.

Summary chỉ tồn tại trong thread hiện tại. Recall chéo phiên nhờ `User.md`, không phải nhờ chuyển summary của thread cũ sang thread mới.

## Memory tăng trưởng và rủi ro

Profile cuối Standard là 328 byte, Stress là 243 byte. Đây là chênh lệch byte trên đĩa giữa cuối và đầu suite, không đo tổng RAM của history/summary hay lượng ghi tích lũy.

Các field có cấu trúc và upsert tránh lặp nguyên văn cả hội thoại: sửa nghề hoặc địa điểm thay giá trị cũ, nhắc lại cùng fact không thêm dòng. News/logs chỉ ở thread/summary nên không làm profile tăng theo mỗi lượt. Interests chỉ dùng tập từ khóa kỹ thuật giới hạn; style gộp các thuộc tính nhưng thay số bullet khi có yêu cầu mới.

Trong ứng dụng lớn, nhiều loại entity hơn và nhiều người dùng vẫn tăng dung lượng tổng. Lưu sai một fact khiến lỗi được lặp lại qua nhiều phiên; giữ lâu thông tin tạm thời cũng có thể gây lỗi. Atomic replace bảo vệ file khỏi write dở, nhưng read–modify–write chưa có lock cho nhiều process ghi đồng thời. Short-term state nằm trong RAM và chưa có chính sách thu hồi thread cũ.

## Bonus đã triển khai

**Entity extraction có cấu trúc:** tám field ổn định, thay vì append hội thoại vào markdown. Các test dùng thêm tên Trần An, Hải Phòng, trà xanh, phở bò và mèo Mít để kiểm tra việc trích fact không phụ thuộc một bộ tên mẫu.

**Conflict handling:** một giá trị hiện tại cho tên, nghề, địa điểm; cập nhật mới thay giá trị cũ. Số bullet cũng được cập nhật, ví dụ 3 → 5. Test xác minh old thread không lấn át correction mới trong profile.

**Confidence threshold:** rule nhận identity trực tiếp có điểm 0.99, style 0.95, interests 0.90; chỉ persist khi vượt `PROFILE_CONFIDENCE_THRESHOLD`. Đây là điểm độ mạnh của rule, chưa được hiệu chỉnh thành xác suất thống kê. Tăng threshold có thể tránh lưu preference yếu nhưng làm giảm recall.

**Chống lưu sai:** bỏ câu hỏi, giả định, câu đùa, các phát biểu về người khác và lời nhắc đến fact cũ. Những câu recall chứa tên hoặc thành phố gây nhiễu không làm thay đổi profile. Rule bảo thủ có thể bỏ sót một fact thật trong câu phức tạp; cần thêm đánh giá false-positive/false-negative khi mở rộng dữ liệu.

Memory decay chưa triển khai; thông tin đã xác nhận còn tồn tại đến khi có correction. Đây là một hướng mở rộng cho preference hoặc trạng thái có thời hạn.

## Phạm vi kiểm chứng

47/47 test đã pass, gồm các test cốt lõi và tình huống lỗi. Luồng live được kiểm tra với graph LangChain thật, model giả lập và usage metadata, gồm checkpoint thread, prompt chứa profile qua restart và tool đọc profile. Đã khởi tạo thành công sáu lớp provider bằng key giả, không gửi request tới dịch vụ.

Chưa đo hành vi hay chất lượng của LLM thật. Chạy `--live` để đo provider đã cấu hình và `--live --judge` để chấm bằng model judge độc lập.

Estimator offline là `ceil(len(text.strip()) / 4)` với 4 token overhead mỗi message; không phải tokenizer hoặc hóa đơn API. Output và prompt đều bao gồm conversation lẫn recall. Live ưu tiên usage metadata từng lượt gọi model, kể cả tool call; khi thiếu metadata thì dùng ước lượng. Usage của judge không cộng vào agent.

Recall là khớp chuỗi không phân biệt hoa/thường và chuẩn hóa Unicode, trung bình theo câu hỏi. Nó chưa phát hiện đầy đủ câu trả lời có expected substring trong ngữ cảnh phủ định hay thông tin thừa sai. Quality là `0.85 × recall + 0.15 × concise`, nên điểm 100% chủ yếu phản ánh trả đủ facts trong câu ngắn; điểm 15% của Baseline chỉ phản ánh câu trả lời ngắn, không có recall đúng.

Summary heuristic giữ facts cùng một số note đầu/gần nhất, có thể mất chi tiết ở giữa. Các câu recall của Stress chỉ đo profile bền vững, chưa chứng minh recall đầy đủ bốn chủ đề news sau compact. Tin tức trong dataset được coi là nội dung hội thoại mẫu, không được xác minh như một nguồn tin thực tế.
