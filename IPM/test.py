import cv2
from camera_param import load_camera_params
from lut import generate_direct_backward_mapping, generate_bev_image

if __name__ == "__main__":
    # Load camera parameters
    extrinsic, intrinsic = load_camera_params()

    world_x_min = 0
    world_x_max = 50
    world_x_interval = 0.07324

    world_y_min = -20
    world_y_max = 20
    world_y_interval = 0.07324

    # Sinh map_x, map_y
    map_x, map_y = generate_direct_backward_mapping(
        world_x_min, world_x_max, world_x_interval,
        world_y_min, world_y_max, world_y_interval,
        extrinsic, intrinsic
    )

    # Đọc ảnh camera gốc
    input_image = cv2.imread(r"D:\KLTN\cam2bev-data-master\cam2bev-data-master\1_FRLR\train\front\t_6_1_0202000.png")
    if input_image is None:
        print("⚠️ Không tìm thấy ảnh ")
        exit()

    # Tạo ảnh BEV
    bev_image = generate_bev_image(input_image, map_x, map_y)

    # Hiển thị ảnh BEV
    # Lưu ảnh BEV
    output_path = r"D:\KLTN\cam2bev-data-master\cam2bev-data-master\bev_output\bev_result.png"
    cv2.imwrite(output_path, bev_image)
    print(f"✅ Ảnh BEV đã được lưu tại: {output_path}")

