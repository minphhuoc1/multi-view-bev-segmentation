import numpy as np
import cv2

def generate_direct_backward_mapping(
    world_x_min, world_x_max, world_x_interval,
    world_y_min, world_y_max, world_y_interval,
    extrinsic, intrinsic
):
    world_x_coords = np.arange(world_x_max, world_x_min, -world_x_interval)
    world_y_coords = np.arange(world_y_max, world_y_min, -world_y_interval)

    output_height = len(world_x_coords)
    output_width = len(world_y_coords)

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)

    for i, world_x in enumerate(world_x_coords):
        for j, world_y in enumerate(world_y_coords):
            world_coord = np.array([world_x, world_y, 0, 1], dtype=np.float32)
            camera_coord = extrinsic[:3, :] @ world_coord
            uv_coord = intrinsic[:3, :3] @ camera_coord
            uv_coord /= uv_coord[2]

            map_x[i, j] = uv_coord[0]
            map_y[i, j] = uv_coord[1]

    return map_x, map_y

def generate_bev_image(input_image, map_x, map_y):
    return cv2.remap(
        input_image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
