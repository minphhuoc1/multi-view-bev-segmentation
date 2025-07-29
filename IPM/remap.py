import numpy as np
import cv2
import os

def bilinear_sampler(imgs, pix_coords):
    img_h, img_w, img_c = imgs.shape
    pix_h, pix_w, _ = pix_coords.shape
    out_shape = (pix_h, pix_w, img_c)

    pix_x, pix_y = np.split(pix_coords, [1], axis=-1)
    pix_x = pix_x.astype(np.float32)
    pix_y = pix_y.astype(np.float32)

    pix_x0 = np.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = np.floor(pix_y)
    pix_y1 = pix_y0 + 1

    x_max = img_w - 1
    y_max = img_h - 1

    pix_x0 = np.clip(pix_x0, 0, x_max)
    pix_x1 = np.clip(pix_x1, 0, x_max)
    pix_y0 = np.clip(pix_y0, 0, y_max)
    pix_y1 = np.clip(pix_y1, 0, y_max)

    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    base_y0 = pix_y0 * img_w
    base_y1 = pix_y1 * img_w

    idx00 = (pix_x0 + base_y0).astype(np.int32).flatten()
    idx01 = (pix_x0 + base_y1).astype(np.int32).flatten()
    idx10 = (pix_x1 + base_y0).astype(np.int32).flatten()
    idx11 = (pix_x1 + base_y1).astype(np.int32).flatten()

    imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
    im00 = imgs_flat[idx00].reshape(out_shape)
    im01 = imgs_flat[idx01].reshape(out_shape)
    im10 = imgs_flat[idx10].reshape(out_shape)
    im11 = imgs_flat[idx11].reshape(out_shape)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output

def remap_bilinear(image, map_x, map_y):
    pix_coords = np.concatenate([
        np.expand_dims(map_x, -1),
        np.expand_dims(map_y, -1)
    ], axis=-1)
    bilinear_output = bilinear_sampler(image, pix_coords)
    return np.clip(np.round(bilinear_output), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # === Đường dẫn tới ảnh ===
    image_path = r"D:\KLTN\cam2bev-data-master\cam2bev-data-master\bev_output\bev_result.png"
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không tìm thấy ảnh. Hãy kiểm tra đường dẫn.")
        exit()

    # === Tạo map_x, map_y mẫu (giả lập) ===
    h, w = 860, 800  # Kích thước BEV ảnh đầu ra
    map_x, map_y = np.meshgrid(
        np.linspace(0, image.shape[1] - 1, w),
        np.linspace(0, image.shape[0] - 1, h)
    )
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # === Dùng remap OpenCV (bilinear interpolation)
    output_cv2 = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # === Dùng remap thủ công với bilinear
    output_manual = remap_bilinear(image, map_x, map_y)

    # === So sánh L1/L2 Loss
    mask = (output_cv2 > [0, 0, 0])
    output_cv2 = output_cv2.astype(np.float32)
    output_manual = output_manual.astype(np.float32)

    l1 = np.mean(np.abs(output_cv2[mask] - output_manual[mask]))
    l2 = np.mean((output_cv2[mask] - output_manual[mask]) ** 2)

    print("✅ L1 Loss (opencv vs custom):", l1)
    print("✅ L2 Loss (opencv vs custom):", l2)

    # === Lưu ảnh kết quả
    cv2.imwrite("bev_remap_cv2_linear.jpg", output_cv2)
    cv2.imwrite("bev_remap_manual_linear.jpg", output_manual)
    print("✅ Ảnh đã được lưu: bev_remap.jpg và bev_remap_manual.jpg")
