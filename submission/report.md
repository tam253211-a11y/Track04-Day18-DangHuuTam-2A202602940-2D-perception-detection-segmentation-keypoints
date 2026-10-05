# Lab 18 — 2D Perception: Detection · Segmentation · Keypoints

**Học viên:** Đặng Hữu Tâm · 2A202602940

Link notebook đã chạy: https://github.com/tam253211-a11y/Track04-Day18-DangHuuTam-2A202602940-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

Môi trường: Google Colab, GPU Tesla T4 · torch 2.11.0+cu130 · ultralytics 8.4.171. Notebook đã chạy `Restart session and run all`, không ô nào lỗi.

## Nội dung thư mục `submission/`

| File | Nội dung |
|---|---|
| `ket_qua.json` | Kết quả do `final_report()` xuất ra (giữ nguyên, không sửa tay) |
| `autolabel/bus.txt` | Nhãn YOLO-seg tạo bằng YOLO26 → SAM 2.1 → polygon (5 object) |
| `report.md` | Báo cáo này |
| `bonus_4C.md` | Bonus 4C: val gốc và val lật gương, metric nào che lỗi `flip_idx` |
| `bonus_onnx.md` | Bài tập về nhà 3: export ONNX, đo latency CPU của hai head |

## Kết quả chính

| Mục | Kết quả |
|---|---|
| 1B | `box_iou`, `nms`, `batched_nms` đạt bộ kiểm tra (không dùng phao) |
| 1C | NMS tự viết giữ 5 box, khớp Ultralytics; camera đếm 4 người, 1 xe buýt; bảng latency đủ 4 cấu hình |
| 1D ⭐ | `average_precision` đạt, đường PR của ví dụ trên slide |
| 2B | `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg` đạt; Hungarian ghép 5 cặp, mask IoU 0.842–0.939 |
| 2C | `autolabel/bus.txt` hợp lệ; IoU polygon so với mask SAM: 0.967–0.983 |
| 3B, 3C | `oks`, `joint_angle` đạt; luật "ngã": ảnh gốc nghiêng 1° (đứng), ảnh xoay 83–94° (NGÃ?) |
| 4A | `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` (quy ước giải phẫu) |
| 4B | 40 epoch, imgsz 640, GPU: Box mAP50-95 0.930 · Pose mAP50 0.995 · Pose mAP50-95 0.457 · 4.6 phút |
| 4C ⭐ | Xem `bonus_4C.md` |
| Bài tập về nhà ⭐ | Xem `bonus_onnx.md` |
| Q1–Q12 | 12/12 câu đã trả lời trong notebook |
