# Bài tập về nhà 3 — Export ONNX và đo latency trên CPU

Link notebook đã chạy (ô ngay sau 4C): https://github.com/tam253211-a11y/Track04-Day18-DangHuuTam-2A202602940-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

## Cách làm

- Export `yolo26n.pt` sang ONNX (opset 18, ảnh 640×640 cố định) bằng `model.export(format="onnx")`, ra hai biến thể:
  - **one-to-many:** export mặc định. Output `[1, 84, 8400]`; NMS chạy ở bước hậu xử lý.
  - **one-to-one:** `export(format="onnx", nms=False)`. Output `[1, 300, 6]`, tức 300 box × (xyxy, conf, cls), không cần NMS. Ở Ultralytics 8.4.171, tham số `end2end` đã bị thay bằng `nms`.
- Chạy bằng ONNX Runtime 1.30.0 (CPUExecutionProvider) trên CPU Colab: Intel Xeon @ 2.00GHz, 2 luồng.
- Đo trên `bus.jpg` bằng hàm `bench` của mục 1C: 1 lần warm-up, lấy trung bình 30 lần, tách thời gian preprocess / inference / postprocess.

## Kết quả

| Cấu hình (ONNX, CPU) | preprocess (ms) | inference (ms) | postprocess (ms) | tổng (ms) | số box |
|---|---:|---:|---:|---:|---:|
| one-to-many + NMS, conf 0.25 | 4.89 | 76.53 | 1.46 | 82.88 | 5 |
| one-to-many + NMS, conf 0.001 | 6.87 | 105.73 | 2.79 | 115.39 | 186 |
| one-to-one NMS-free, conf 0.25 | 4.87 | 79.86 | 0.46 | 85.19 | 5 |
| one-to-one NMS-free, conf 0.001 | 4.89 | 86.25 | 0.49 | 91.63 | 177 |

Để so sánh, cùng phép đo trên GPU T4 (PyTorch, mục 1C):
- one-to-many: postprocess 1.27 ms ở conf 0.25 và 1.47 ms ở conf 0.001;
- one-to-one: postprocess 0.44 ms ở cả hai mức conf.

## Nhận xét

1. **Postprocess là cột phân biệt rõ hai head.**
   - Với one-to-one, postprocess gần như hằng số (0.46 → 0.49 ms) dù số box tăng từ 5 lên 177. Bước này chỉ lọc theo conf trên tối đa 300 box có sẵn.
   - Với one-to-many, postprocess tăng gần gấp đôi khi hạ conf (1.46 → 2.79 ms), vì NMS phải xử lý nhiều ứng viên hơn: 186 box được giữ, còn số ứng viên trước NMS nhiều hơn thế.
   - Ở conf 0.001, NMS chậm hơn one-to-one khoảng **5.7 lần** (2.79 so với 0.49 ms). Trên GPU, tỉ lệ này chỉ khoảng 3.3 lần.
2. **Inference chiếm khoảng 90% thời gian trên CPU và dao động mạnh.** One-to-many dùng cùng một đồ thị ONNX cho hai mức conf, nhưng đo được 76.5 ms ở lần này và 105.7 ms ở lần kia. Conf không ảnh hưởng tới inference, nên chênh lệch khoảng 30 ms là nhiễu của CPU dùng chung trên Colab, chỉ có 2 luồng. Vì vậy tôi không kết luận được head nào có inference nhanh hơn. Hai head dùng chung backbone, chỉ khác phần đầu ra; head one-to-one có thêm bước chọn top-k trong đồ thị.
3. **Ý nghĩa khi triển khai.** Trong phép đo này, NMS của Ultralytics chạy bằng torchvision (C++), và trên CPU hai nhân chỉ chiếm 1–3 ms trong tổng khoảng 80–115 ms. Lợi ích chính của one-to-one nằm ở chỗ khác:
   - Toàn bộ pipeline nằm trong một file ONNX. Không cần cài đặt NMS riêng trên thiết bị đích như NPU, TensorRT hay mobile, nơi NMS thường không được hỗ trợ trong đồ thị hoặc phải chạy về CPU.
   - Postprocess không phụ thuộc số box, nên latency ổn định ở cảnh đông và khi dùng conf thấp.
   - Không có ngưỡng IoU của NMS, nên tránh được lỗi xoá nhầm người đứng sát nhau (Q3).
4. **Hạn chế của phép đo:** mỗi cấu hình chỉ đo một lần 30 vòng trên máy dùng chung. Muốn so sánh inference đáng tin cần chạy nhiều lần, cố định số luồng ONNX Runtime và báo cáo median kèm độ lệch chuẩn.
