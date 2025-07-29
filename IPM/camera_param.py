import numpy as np

def rotation_from_euler(roll=0.0, pitch=0.0, yaw=0.0):
    """
    Tạo ma trận xoay từ các góc Euler (radian).
    """
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R

def translation_matrix(vector):
    """
    Tạo ma trận tịnh tiến từ vector (x, y, z).
    """
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M

def load_camera_params():
    """
    Tạo ma trận nội tại (K) và ngoại tại (RT) từ thông số hiệu chỉnh camera.
    Trả về:
        RT: Ma trận ngoại tại (4x4)
        K:  Ma trận nội tại (4x4)
    """
 
    fx = 278.283
    fy = 408.1295
    u0 = 482
    v0 = 302

    roll = 0
    pitch = 0
    yaw = 0

    x = 1.7
    y = 0
    z = 1.4

    # Ma trận nội tại
    K = np.array([
        [fx,  0,  u0, 0],
        [0,  fy,  v0, 0],
        [0,   0,   1, 0],
        [0,   0,   0, 1]
    ])

    # Ma trận ngoại tại
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x, -y, -z))  # dịch ngược lại từ thế giới về camera

    # Chuẩn hóa hướng camera
    R_align = np.array([
        [0., -1.,  0., 0.],
        [0.,  0., -1., 0.],
        [1.,  0.,  0., 0.],
        [0.,  0.,  0., 1.]
    ])

    RT = R_align @ R_veh2cam @ T_veh2cam

    return RT, K

if __name__ == "__main__":
    RT, K = load_camera_params()
    print("=== Ma trận nội tại (K): ===")
    print(K)
    print("\n=== Ma trận ngoại tại (RT): ===")
    print(RT)
