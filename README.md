<div align="center">

# 🚗 Multi-View BEV Segmentation with Spatial Transformer

*Phân đoạn ảnh góc nhìn chim từ 4 camera với Spatial Transformer & Deep Learning hiện đại*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[📖 Giới thiệu](#-giới-thiệu) • [✨ Tính năng](#-tính-năng-nổi-bật) • [🗂️ Cấu trúc](#-cấu-trúc-dự-án) • [🚀 Cài đặt](#-cài-đặt-nhanh) • [🔬 Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng) • [📊 Kết quả](#-đánh-giá--kết-quả) • [🤝 Đóng góp](#-đóng-góp) • [👤 Liên hệ](#-liên-hệ)

</div>

---

## 📖 Giới thiệu

**Multi-View BEV Segmentation** là dự án nghiên cứu & phát triển giải pháp phân đoạn ảnh góc nhìn chim (BEV) từ 4 camera (front, left, rear, right) trên xe tự hành. Dự án ứng dụng các kỹ thuật:
- Deep Learning (UNet, Segformer backbone)
- Spatial Transformer (chuyển đổi góc nhìn thông minh)
- Multi-View Fusion

**Mục tiêu:**  
Dự đoán bản đồ BEV+Occlusion chất lượng cao từ 4 ảnh đầu vào, giúp xe tự hành hiểu rõ môi trường xung quanh, hỗ trợ các bài toán perception, planning, ADAS.

---

## ✨ Tính năng nổi bật

| 🚘 Đa góc nhìn | 🧠 Spatial Transformer | 🏆 Nhiều mô hình mạnh | 📊 Đánh giá chuẩn |
|:---:|:---:|:---:|:---:|
| Xử lý đồng thời 4 camera | Học phép biến đổi không gian | UNet, UNet+ST, UNet+Segformer | mIoU, Pixel Acc, log chi tiết |

- **3 mô hình so sánh:**  
  - UNet thuần  
  - UNet + Spatial Transformer  
  - UNet + Segformer backbone + Fusion  
- **Training/Validation script rõ ràng, dễ mở rộng**
- **Visualization kết quả trực quan**
- **Hỗ trợ custom dataset, class, palette**
- **Tối ưu cho showcase, nghiên cứu, thực chiến**

---

## 🗂️ Cấu trúc dự án

```
FinalDemo_KLTN/
├── Dataset/
│   ├── front/ left/ rear/ right/ bev+occlusion/
├── IPM/                        # Xử lý chuyển đổi góc nhìn
├── Model_Unet/                 # UNet thuần
├── UnetXST/                    # UNet + Spatial Transformer
├── UnetXST_With_SegformerBackbone/ # UNet + Segformer backbone
└── README.md
```

- **Input:** 4 ảnh RGB (front, left, rear, right)
- **Label:** Ảnh mask BEV+Occlusion
- **Số lớp:** 7 (định nghĩa trong palette.py)

---

## 🚀 Cài đặt nhanh

### 📋 Yêu cầu hệ thống

- Python >= 3.8
- PyTorch >= 1.10 (CUDA khuyến nghị)
- OpenCV, numpy, PIL, tqdm, matplotlib

### ⚡ Cài đặt một lệnh

```bash
git clone <link-github>
cd FinalDemo_KLTN
python -m venv venv
# Kích hoạt venv (Windows): venv\Scripts\activate
pip install torch torchvision opencv-python numpy pillow tqdm matplotlib
```

---

## 🔬 Hướng dẫn sử dụng

### 1️⃣ Chuẩn bị dữ liệu

- Đặt dữ liệu vào thư mục `Dataset/` theo cấu trúc đã nêu.
- Chỉnh đường dẫn dữ liệu trong các file train/test cho phù hợp với máy của bạn.

### 2️⃣ Train/Test từng mô hình

#### 🔹 UNet thuần

```bash
cd Model_Unet
python train_thuan.py         # Train
python predict_visualize.py   # Hiển thị kết quả
```
- Checkpoint tốt nhất: `4camoriginal_best_checkpoint.pth`

#### 🔹 UNet + Spatial Transformer

```bash
cd ../UnetXST
python train_unet_custom.py   # Train
python test_valid.py          # Đánh giá mIoU, Pixel Acc
```
- Checkpoint tốt nhất: `best_model_unet_custom_new_weights_4.pt`

#### 🔹 UNet + Segformer Backbone

```bash
cd ../UnetXST_With_SegformerBackbone
python train_multiview.py     # Train
python test_valid.py          # Đánh giá mIoU, Pixel Acc
```
- Checkpoint tốt nhất: `best_model_segformer_newweights_4.pt`

### 3️⃣ Visualize kết quả

- Sử dụng script `predict_visualize.py` (Model_Unet) để hiển thị ảnh, mask groundtruth, mask dự đoán.
- Kết quả mIoU, Pixel Accuracy sẽ được lưu log chi tiết.

---

## 📊 Đánh giá & Kết quả

- **Chỉ số đánh giá:**  
  - 🎯 Pixel Accuracy  
  - 🏅 mIoU từng class  
  - 🏆 mIoU tổng thể  
- **Log chi tiết:**  
  - Kết quả in ra màn hình & lưu file log sau khi test.
- **Visualization:**  
  - So sánh mask dự đoán & groundtruth trực quan.

---

## 🛠️ Công nghệ sử dụng

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

</div>

---

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh!  
- Fork repo, tạo branch mới, commit và mở Pull Request.
- Nếu phát hiện lỗi, vui lòng tạo issue kèm mô tả chi tiết, log, screenshot (nếu có).

---

## 📄 License

Dự án phát hành theo giấy phép MIT. Xem file `LICENSE` để biết chi tiết.

---

## 👤 Liên hệ

<div align="center">

**[Đoàn Văn Minh Phước]**  
[![Email](https://img.shields.io/badge/Email-phuocdoan333@gmail.com-red?style=for-the-badge&logo=gmail)](mailto:phuocdoan333@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/minphhuoc1)

</div>

---

<div align="center">

⭐ Nếu bạn thấy dự án hữu ích, hãy cho mình một star nhé!  
**[⬆ Về đầu trang](#-multi-view-bev-segmentation-with-spatial-transformer)**

</div>

---
