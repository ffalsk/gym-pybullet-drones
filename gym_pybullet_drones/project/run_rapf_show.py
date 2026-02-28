import time
import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.utils import sync


# ==============================================================================
# 1. 纯 RAPF 算法核心函数
# ==============================================================================
def calculate_rapf(pos, goal, obs_list, other_drones_pos, drone_idx):
    """
    计算基于新型旋转人工势场法 (RAPF) 的 3D 合力
    """
    # [参数设置]
    k_att = 2.0  # 引力系数
    k_rep = 0.8  # 障碍物斥力系数
    k_in = 0.5  # 无人机互斥系数 (防互撞)
    k_rot = 1.2  # 旋转力系数

    d_safe_obs = 0.6  # 障碍物安全距离
    d_safe_in = 0.2  # 无人机安全距离 (设置大一点防止螺旋桨打架)

    F_res = np.zeros(3)

    # 1. 计算引力 (Attractive Force)
    dist_to_goal = np.linalg.norm(goal - pos)
    F_att = k_att * (goal - pos)
    F_res += F_att

    # 2. 计算障碍物斥力与旋转力 (由于 obs_list 为空，这里实际上不会执行)
    for obs in obs_list:
        obs_pos = obs["pos"]
        d_vec = pos - obs_pos
        d_obs = np.linalg.norm(d_vec)

        if d_obs < d_safe_obs:
            dir_obs = d_vec / d_obs
            force_mag = (1.0 / d_obs - 1.0 / d_safe_obs) * (1.0 / (d_obs**2))
            F_rep = k_rep * force_mag * dir_obs
            F_res += F_rep

            v_to_goal = goal - pos
            cross_z = d_vec[0] * v_to_goal[1] - d_vec[1] * v_to_goal[0]
            direction = 1 if cross_z > 0 else -1

            tangent = np.array([-d_vec[1] * direction, d_vec[0] * direction, 0])
            if np.linalg.norm(tangent) > 0:
                tangent = tangent / np.linalg.norm(tangent)
                F_rot = k_rot * force_mag * tangent
                F_res += F_rot

    # 3. 计算无人机互斥力 (Inter-agent Force) 防碰撞
    for j, other_pos in enumerate(other_drones_pos):
        if j == drone_idx:
            continue  # 跳过自己

        d_vec_in = pos - other_pos
        d_in = np.linalg.norm(d_vec_in)
        if d_in < d_safe_in and d_in > 0.01:
            dir_in = d_vec_in / d_in
            force_in_mag = (1.0 / d_in - 1.0 / d_safe_in) * (1.0 / (d_in**2))
            F_in = k_in * force_in_mag * dir_in
            F_res += F_in

    return F_res


# ==============================================================================
# 2. 目标形状生成函数 (垂直心形)
# ==============================================================================
def get_heart_shape_targets(num_drones, center_y=0.0, base_z=1.5, scale=0.06):
    targets = np.zeros((num_drones, 3))
    for i in range(num_drones):
        t = i * (2 * np.pi / num_drones)
        x_param = 16 * math.sin(t) ** 3
        y_param = (
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )

        targets[i, 0] = x_param * scale
        targets[i, 1] = center_y
        targets[i, 2] = base_z + (y_param + 5) * scale

    return targets


# ==============================================================================
# 3. 主运行程序
# ==============================================================================
if __name__ == "__main__":
    NUM_DRONES = 16

    # 1. 初始位置：在 Y = -2.0 的地面处排成 4x4 方阵
    INIT_XYZS = np.zeros((NUM_DRONES, 3))
    for i in range(NUM_DRONES):
        INIT_XYZS[i, :] = [(i % 4) * 0.3 - 0.45, -2.0 + (i // 4) * 0.3, 0.1]

    # 2. 目标位置：前方 Y = 0.0 的垂直空中，高度 1.5 米
    TARGET_POS = get_heart_shape_targets(
        NUM_DRONES, center_y=0.0, base_z=1.5, scale=0.06
    )

    # 3. 初始化环境
    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        gui=True,
        obstacles=False,
    )

    PYB_CLIENT = env.getPyBulletClient()
    # obs, info = env.reset()

    # 【关键修改】：这里将障碍物列表设置为空，不生成任何障碍物
    obs_list = []

    # 4. 开始仿真主循环
    start_time = time.time()
    action = np.zeros((NUM_DRONES, 4))

    print("[INFO] 开始无障碍物测试！验证心形阵列集结与机间避碰功能。")

    # 调整 PyBullet 视角
    p.resetDebugVisualizerCamera(
        cameraDistance=3.5,
        cameraYaw=0,
        cameraPitch=-20,
        cameraTargetPosition=[0, -0.5, 1.0],
        physicsClientId=PYB_CLIENT,
    )

    for step in range(100000):
        all_drones_pos = np.array(
            [env._getDroneStateVector(i)[0:3] for i in range(NUM_DRONES)]
        )

        for i in range(NUM_DRONES):
            current_pos = all_drones_pos[i]
            goal_pos = TARGET_POS[i]

            # --- 核心：利用 RAPF 计算期望合力 ---
            F_res = calculate_rapf(current_pos, goal_pos, obs_list, all_drones_pos, i)

            # --- 桥梁：将合力映射为动作 ---
            f_norm = np.linalg.norm(F_res)
            v_dir = F_res / f_norm if f_norm > 0 else np.zeros(3)

            # 速度截断
            speed_fraction = np.clip(f_norm * 0.2, 0.0, 1.0)

            # 到达容差：贴近目标后悬停
            if np.linalg.norm(goal_pos - current_pos) < 0.05:
                speed_fraction = 0.0

            action[i, 0:3] = v_dir
            action[i, 3] = speed_fraction

        obs, reward, terminated, truncated, info = env.step(action)
        sync(step, start_time, env.CTRL_TIMESTEP)

    env.close()
