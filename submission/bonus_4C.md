# Bonus 4C — Val lật gương: metric nào đã che lỗi `flip_idx`?

Link notebook đã chạy (ô 4C): https://github.com/tam253211-a11y/Track04-Day18-DangHuuTam-2A202602940-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

## Thiết lập

- Hai model YOLO26n-pose, train trên tiger-pose cùng cấu hình: 40 epoch, imgsz 640, batch 16, seed 0, GPU T4.
  - **Giải phẫu:** `flip_idx = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]`.
  - **Đồng nhất:** `flip_idx = [0, 1, …, 11]` (YAML gốc).
- Hai tập đánh giá:
  - **Val gốc:** 53 ảnh, 100% hổ quay phải. Tập train cũng vậy: 210 quay phải, 0 quay trái.
  - **Val lật gương:** cùng 53 ảnh lật ngang, nhãn đổi chỗ theo quy ước giải phẫu. Tập này giả lập hổ quay trái lúc triển khai.

## Kết quả (từ lần chạy của tôi)

| Model | Tập val | Box mAP50-95 | Pose P | Pose R | Pose mAP50 | Pose mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| flip_idx giải phẫu | gốc | 0.930 | 0.999 | 1.000 | 0.995 | **0.457** |
| flip_idx giải phẫu | lật gương | 0.918 | 0.999 | 1.000 | 0.995 | **0.439** |
| flip_idx đồng nhất | gốc | 0.904 | 0.999 | 1.000 | 0.995 | **0.417** |
| flip_idx đồng nhất | lật gương | 0.894 | 0.924 | 0.925 | 0.878 | **0.298** |

Thay đổi khi chuyển từ val gốc sang val lật gương:
- Model giải phẫu: Pose mAP50 giữ nguyên 0.995; Pose mAP50-95 giảm 0.018 (−4%).
- Model đồng nhất: Pose mAP50 giảm từ 0.995 xuống 0.878; Pose mAP50-95 giảm 0.119 (−29%).

## Metric nào đã che lỗi?

**Pose mAP50 trên val gốc che lỗi hoàn toàn.** Hai model đều đạt 0.995, cùng P = 0.999 và R = 1.000, nên nếu chỉ nhìn bảng 4B thì không thể phân biệt model nào gán sai trái/phải. Có ba lý do:

1. **Val có cùng thiên lệch với train.** Mọi con hổ trong val đều quay phải. Trên ảnh quay phải, chân phía camera đúng là `right_*`, nên model đồng nhất vẫn trả đúng tên. Lỗi chỉ xuất hiện khi con vật quay trái, mà val không có ca nào như vậy.
2. **Ngưỡng OKS 0.5 quá rộng.** Với σ = 1/12, một dự đoán chỉ cần nằm "gần đúng chỗ" là được tính TP ở ngưỡng 0.5. Hai chân cùng phía của hổ nằm sát nhau, nên dự đoán lẫn trái/phải đôi khi vẫn qua ngưỡng 0.5.
3. **Box mAP không nhạy với nhãn trái/phải.** Box của con hổ không đổi khi đổi tên chân. Box mAP50-95 của model đồng nhất trên val lật gương vẫn là 0.894, nên metric này không bao giờ lộ lỗi `flip_idx`.

Trên val gốc, chỉ **Pose mAP50-95** có dấu hiệu: 0.417 so với 0.457. Tuy vậy chênh 0.04 đủ nhỏ để bị coi là nhiễu giữa hai lần train. Lỗi chỉ lộ rõ khi **đổi phân bố hướng quay** (val lật gương). Khi đó model đồng nhất tụt cả mAP50 (0.878) lẫn mAP50-95 (0.298), còn model giải phẫu gần như giữ nguyên.

Nguyên nhân lỗi: với `flip_idx` đồng nhất, augmentation `fliplr=0.5` tạo ảnh hổ quay trái nhưng nhãn `right_*` vẫn gắn vào chân phía camera. Model vì thế học quy tắc "chân phía camera = right", trong khi về giải phẫu, chân phía camera của con hổ quay trái là chân **trái**. Lúc triển khai gặp hổ quay trái, model đảo trái/phải. Mọi ứng dụng dựa trên từng chân (dáng đi, chân bị thương, so sánh hai bên) đều sai, dù mAP50 trên tập val vẫn đẹp.

## Thiết kế tập val tốt hơn

- **Phủ đủ các điều kiện triển khai.** Tập val cần có cả hai hướng quay, cùng các góc camera và điều kiện che khuất sẽ gặp thực tế. Nếu thiếu dữ liệu thật, thêm val lật gương với nhãn đổi theo `flip_idx` giải phẫu như ở thí nghiệm này.
- **Báo cáo metric theo nhóm, không chỉ một con số tổng:** tách mAP theo hướng quay trái/phải, và theo mức che khuất.
- **Ưu tiên Pose mAP50-95 hoặc OKS ở ngưỡng chặt (0.75) thay vì mAP50.** Đo thêm tỉ lệ "đảo trái/phải": số ảnh mà đổi nhãn trái ↔ phải thì OKS lại cao hơn. Notebook đã tính chỉ số này ở 4B (4/53 ảnh với model giải phẫu).
- **Ước lượng `kpt_oks_sigmas` riêng cho từng keypoint** thay cho σ = 1/12 đều nhau. Bàn chân là điểm nhỏ, dễ lẫn, nên cần σ phù hợp để metric phạt đúng các lỗi định vị chân.
