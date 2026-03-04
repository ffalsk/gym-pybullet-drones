import time
import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.utils import sync


# ==============================================================================
# 1. 纯 RAPF 算法核心函数 (升级真3D旋转力 + 长距离引力截断)
# ==============================================================================
def calculate_rapf(pos, goal, obs_list, other_drones_pos, drone_idx):
    """
    计算基于新型旋转人工势场法 (RAPF) 的 3D 合力
    """
    k_att = 2.0  # 引力系数
    k_rep = 1.5  # 增加一点斥力系数，应对高速飞行
    k_in = 0.5  # 无人机互斥系数
    k_rot = 2.0  # 强化 3D 旋转力

    d_safe_in = 0.25  # 无人机互斥安全距离

    F_res = np.zeros(3)

    # 1. 计算引力 (Attractive Force)
    dist_to_goal = np.linalg.norm(goal - pos)
    if dist_to_goal > 0:
        # 【关键改进】：引力截断！
        # 如果距离很远，引力大小被限制在 max_att_dist (比如 1.5) 以内
        # 这样防止了距离过远导致引力无限大，从而无视了障碍物的斥力
        effective_dist = min(dist_to_goal, 1.5)
        F_att = k_att * (goal - pos) / dist_to_goal * effective_dist
    else:
        F_att = np.zeros(3)
    F_res += F_att

    # 2. 计算障碍物斥力与 真·3D 旋转力
    for obs in obs_list:
        obs_pos = obs["pos"]
        obs_rad = obs["radius"]

        # 动态安全距离：球体半径 + 0.4米安全缓冲
        d_safe_obs = obs_rad + 0.4

        d_vec = pos - obs_pos
        d_obs = np.linalg.norm(d_vec)

        if d_obs < d_safe_obs:
            # --- 经典斥力 (沿半径向外推) ---
            dir_obs = d_vec / d_obs
            force_mag = (1.0 / d_obs - 1.0 / d_safe_obs) * (1.0 / (d_obs**2))
            F_rep = k_rep * force_mag * dir_obs
            F_res += F_rep

            # --- 真·3D 旋转力 (绕障碍物边缘滑行) ---
            v_to_goal = goal - pos

            # 第一步：计算旋转轴 (垂直于 障碍物向量 和 目标向量 构成的平面)
            rotation_axis = np.cross(d_vec, v_to_goal)

            # 防死锁保护：如果刚好在一条直线上 (叉乘为0)，强制给一个 Z轴向上的旋转轴
            if np.linalg.norm(rotation_axis) < 1e-5:
                rotation_axis = np.array([0.0, 0.0, 1.0])
            else:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # 第二步：计算 3D 切向向量 (围绕旋转轴旋转，且垂直于斥力方向)
            tangent = np.cross(rotation_axis, d_vec)

            if np.linalg.norm(tangent) > 0:
                tangent = tangent / np.linalg.norm(tangent)
                # 施加旋转力，引导飞机绕开
                F_rot = k_rot * force_mag * tangent
                F_res += F_rot

    # 3. 计算无人机互斥力防碰撞
    for j, other_pos in enumerate(other_drones_pos):
        if j == drone_idx:
            continue

        d_vec_in = pos - other_pos
        d_in = np.linalg.norm(d_vec_in)
        if 0.01 < d_in < d_safe_in:
            dir_in = d_vec_in / d_in
            force_in_mag = (1.0 / d_in - 1.0 / d_safe_in) * (1.0 / (d_in**2))
            F_in = k_in * force_in_mag * dir_in
            F_res += F_in

    return F_res


# ==============================================================================
# 2. 目标形状生成函数 (垂直心形)
# ==============================================================================
def get_heart_shape_targets(num_drones, center_y=2.0, base_z=1.5, scale=0.1):
    """
    目标放到前方 Y = 2.0 的位置，并适当放大阵型
    """
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

    # 1. 初始位置：在 Y = -4.0 的远端地面起飞，排成宽松的 4x4 方阵
    INIT_XYZS = np.zeros((NUM_DRONES, 3))
    for i in range(NUM_DRONES):
        INIT_XYZS[i, :] = [(i % 4) * 0.5 - 0.75, -4.0 + (i // 4) * 0.5, 0.1]

    # 2. 目标位置：前方 Y = 2.0 的高空
    TARGET_POS = get_heart_shape_targets(
        NUM_DRONES, center_y=2.0, base_z=1.5, scale=0.1
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
    obs, info = env.reset()

    # 4. 【高能预警】在半空中悬浮多个 3D 球形障碍物
    sphere_configs = [
        # (坐标 [x, y, z], 半径, 颜色 [R, G, B, A])
        ([0.0, -1.5, 1.2], 0.4, [1, 0, 0, 0.8]),  # 正中间的大红球 (阻挡低空直飞)
        ([0.6, -0.5, 1.8], 0.25, [0, 1, 0, 0.8]),  # 右侧高空的绿球
        ([-0.6, 0.5, 1.0], 0.3, [0, 0, 1, 0.8]),  # 左侧低空的蓝球
    ]

    obs_list = []
    for pos, radius, color in sphere_configs:
        col_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=PYB_CLIENT
        )
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=PYB_CLIENT
        )
        # baseMass=0 表示它是固定在空中的，不会掉下来
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            physicsClientId=PYB_CLIENT,
        )
        obs_list.append({"pos": np.array(pos), "radius": radius})

    # 5. 调整 PyBullet 视角 (拉远、拉高一点以看清全局)
    p.resetDebugVisualizerCamera(
        cameraDistance=6.5,
        cameraYaw=25,
        cameraPitch=-25,
        cameraTargetPosition=[0, -1.0, 1.0],
        physicsClientId=PYB_CLIENT,
    )

    # 6. 开始仿真主循环
    start_time = time.time()
    action = np.zeros((NUM_DRONES, 4))

    print("[INFO] 3D 长途奔袭 + 多球体避障 + 动态变阵测试开始！")

    for step in range(100000):
        all_drones_pos = np.array(
            [env._getDroneStateVector(i)[0:3] for i in range(NUM_DRONES)]
        )

        for i in range(NUM_DRONES):
            current_pos = all_drones_pos[i]
            goal_pos = TARGET_POS[i]

            # --- 核心：利用 3D RAPF 计算期望合力 ---
            F_res = calculate_rapf(current_pos, goal_pos, obs_list, all_drones_pos, i)

            # --- 桥梁：将合力映射为动作 ---
            f_norm = np.linalg.norm(F_res)
            v_dir = F_res / f_norm if f_norm > 0 else np.zeros(3)

            # 速度截断 (最高限制给到 0.3 的比例，让它飞得稍微快点)
            speed_fraction = np.clip(f_norm * 1, 0.0, 1.0)

            # 到达容差：贴近目标后悬停
            if np.linalg.norm(goal_pos - current_pos) < 0.05:
                speed_fraction = 0.0

            action[i, 0:3] = v_dir
            action[i, 3] = speed_fraction

        obs, reward, terminated, truncated, info = env.step(action)
        sync(step, start_time, env.CTRL_TIMESTEP)

    env.close()
