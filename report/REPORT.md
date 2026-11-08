# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh (hoặc vài nhãn nếu multi-label), ví dụ traffic hoặc bus. Đơn vị nhãn là toàn bộ ảnh | Top-1 cab (~0.51) chiếm ưu thế nhưng không khớp nội dung chính của ảnh là xe buýt. Các lớp còn lại (minibus, police_van, recreational_vehicle, streetcar) đều dưới ~0.2 và là lớp gần nghĩa. Ảnh nhiều đối tượng, có thanh đen letterbox và khói/sương mù, còn top-5 không có lớp traffic hay bus | Chọn nhãn theo quy tắc đã thống nhất (đối tượng chiếm ưu thế hoặc nhãn cảnh). Ảnh nhiều đối tượng thì dùng multi-label hoặc gắn cờ “mơ hồ” | So nhãn với nội dung chính của ảnh, không dựa vào độ tin cậy của mô hình để quyết định đúng/sai. Chú ý các trường hợp mô hình tự tin nhưng sai như ảnh này |
| Phát hiện vật thể | Mỗi vật thể một bounding box (x, y, w, h) kèm lớp (person, bowl, oven...). Đơn vị nhãn là từng vật thể | Số vật thể thay đổi theo ngưỡng: 17 (0.20), 11 (0.35), 6 (0.60), nên “đúng” phụ thuộc ngưỡng. Nhiều box độ tin cậy thấp (bowl 0.46, cup 0.45) bám vào vật nhỏ hoặc mờ. person 0.61 chỉ là bàn tay/cánh tay ở mép ảnh, khó quyết định có phải một người hay không. oven bên trái bao cả vùng bếp lớn, ranh giới không rõ. Các bowl chồng lấn nhau ở bàn bột, và nhiều vật khác (chảo, nồi treo tường, con lăn) không được phát hiện. | Vẽ box chặt quanh từng vật, gán đúng lớp, và xử lý vật bị cắt/che theo quy ước (gắn cờ “truncated”). Không sao chép dự đoán của mô hình làm nhãn mà phải tự kiểm tra từng box | So box mô hình với ảnh: box có sát vật không, lớp có đúng không (có phải thật sự là bowl, oven), có sót vật nào không, có box thừa không. Không dùng độ tin cậy để quyết định đúng/sai mà phải nhìn ảnh |
| Instance segmentation | Mỗi vật thể một mặt nạ (polygon/mask) riêng kèm lớp (person, bowl, dining table...). Hai bowl vẫn là hai instance khác nhau. Đơn vị nhãn là pixel của từng instance | Mask dining table 0.58 tràn xuống gầm bàn (xô, thùng) và đè lên các vật đặt trên bàn, nên khó quyết định phần nào thuộc bàn. Mask oven 0.49 là vùng tối mơ hồ, không rõ có phải lò không, và bị bàn cắt ngang. potted plant 0.63 thực chất là bó thảo mộc treo tường, lớp sai. person 0.43 chỉ là bàn tay ở mép ảnh, mask nhỏ, còn person 0.90 có viền lệch ở tay/vai. Các bowl (0.37 đến 0.74) chồng lấn nhau và mask của bát nhỏ dễ thô. Nhiều vật khác (chảo, nồi treo) không có mask | Vẽ polygon bám viền từng vật, tách riêng từng instance, cắt phần bàn khỏi vật nằm trên bàn và khỏi gầm bàn, gán lại lớp đúng (bó thảo mộc không gán potted plant nếu quy tắc không cho), xử lý vật bị che/cắt theo quy ước, không để hai mask chồng nhau. Dùng SAM gợi ý được, nhưng phải chỉnh tay | Mask có bám viền không, có tràn sang vật cạnh hoặc nền không, mỗi vật đúng một instance không, mask phần bị che có nhất quán không, lớp có đúng không (bắt các lỗi như potted plant, oven). Không dựa vào độ tin cậy mà phải nhìn ảnh |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng ảnh công khai hoặc ảnh mẫu của bài lab, không có thông tin cá nhân, và không đưa dữ liệu nhạy cảm hay khóa bí mật lên nơi công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Mình sẽ dừng lại, không xử lý hay lưu ảnh đó, và báo cho giảng viên (người phụ trách bài lab).
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
