import time
import math
import random
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.utils import sync


# ==============================================================================
# 1. 纯 RAPF 算法核心 (真3D + 结界限高)
# ==============================================================================
def calculate_rapf(pos, goal, obs_list, other_drones_pos, drone_idx):
    k_att = 2.0
    k_rep = 2.0
    k_in = 0.6
    k_rot = 3.0

    d_safe_in = 0.25
    Z_CEILING = 4.0
    Z_CEILING_SAFE = 3.9
    X_BOUND = 3.0
    X_BOUND_SAFE = 2.95

    F_res = np.zeros(3)

    # 1. 截断引力
    dist_to_goal = np.linalg.norm(goal - pos)
    if dist_to_goal > 0:
        effective_dist = min(dist_to_goal, 2.0)
        F_att = k_att * (goal - pos) / dist_to_goal * effective_dist
    else:
        F_att = np.zeros(3)
    F_res += F_att

    # --- 结界斥力 ---
    if pos[2] > Z_CEILING_SAFE:
        F_res[2] -= k_rep * (1.0 / (Z_CEILING - pos[2] + 1e-3) - 1.0 / 0.2) ** 2
    if pos[0] > X_BOUND_SAFE:
        F_res[0] -= k_rep * (1.0 / (X_BOUND - pos[0] + 1e-3) - 1.0 / 0.2) ** 2
    if pos[0] < -X_BOUND_SAFE:
        F_res[0] += k_rep * (1.0 / (pos[0] - (-X_BOUND) + 1e-3) - 1.0 / 0.2) ** 2

    # 2. 异构障碍物斥力与 真·3D 统一旋转力
    F_rep_total = np.zeros(3)

    for obs in obs_list:
        obs_type = obs["type"]
        d_safe_obs = obs["radius"] + 0.35

        if obs_type == "sphere":
            closest_point = obs["pos"]
            d_vec = pos - closest_point
        elif obs_type == "cylinder":
            closest_point = np.array([obs["pos"][0], obs["pos"][1], pos[2]])
            d_vec = pos - closest_point

        d_obs = np.linalg.norm(d_vec)

        if 1e-4 < d_obs < d_safe_obs:
            dir_obs = d_vec / d_obs
            force_mag = (1.0 / d_obs - 1.0 / d_safe_obs) * (1.0 / (d_obs**2))
            F_rep_total += k_rep * force_mag * dir_obs

    # 统一 3D 旋转力
    total_rep_mag = np.linalg.norm(F_rep_total)
    if total_rep_mag > 0:
        F_res += F_rep_total

        rep_dir = F_rep_total / total_rep_mag
        v_to_goal = goal - pos

        rotation_axis = np.cross(rep_dir, v_to_goal)
        if np.linalg.norm(rotation_axis) < 1e-5:
            rotation_axis = np.array([0.1, 0.1, 1.0])
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        tangent = np.cross(rotation_axis, rep_dir)
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
            F_rot = k_rot * total_rep_mag * tangent

            if abs(tangent[2]) > 0.8:
                tangent[2] = 0
                if np.linalg.norm(tangent) > 0:
                    tangent = tangent / np.linalg.norm(tangent)
                    F_rot = k_rot * total_rep_mag * tangent

            F_res += F_rot

    # 3. 机间互斥力
    for j, other_pos in enumerate(other_drones_pos):
        if j == drone_idx:
            continue
        d_vec_in = pos - other_pos
        d_in = np.linalg.norm(d_vec_in)
        if 0.01 < d_in < d_safe_in:
            dir_in = d_vec_in / d_in
            force_in_mag = (1.0 / d_in - 1.0 / d_safe_in) * (1.0 / (d_in**2))
            dir_in[2] = max(dir_in[2], 0.0) if pos[2] < 0.2 else dir_in[2]
            F_in = k_in * force_in_mag * dir_in
            F_res += F_in

    return F_res


# ==============================================================================
# 2. 环境生成器 (长途赛道 + 纯点云/圆柱障碍)
# ==============================================================================
def create_boundary(client, obs_list):
    """半透明结界，长达 30 米！"""
    color = [0.5, 0.5, 0.5, 0.15]

    def add_bound(center, half_ext):
        c_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_ext, physicsClientId=client
        )
        v_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_ext, rgbaColor=color, physicsClientId=client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=c_id,
            baseVisualShapeIndex=v_id,
            basePosition=center,
            physicsClientId=client,
        )

    # 覆盖 Y 从 -5 到 25 的巨大通道
    add_bound([0, 10, 4.05], [3.0, 15, 0.05])  # 天花板
    add_bound([-3.05, 10, 2.0], [0.05, 15, 2.0])  # 左墙
    add_bound([3.05, 10, 2.0], [0.05, 15, 2.0])  # 右墙


def build_cylinder_forest(client, obs_list, y_start, y_end, num=20):
    """第一重：错综复杂的参天圆柱阵列"""
    for _ in range(num):
        x, y = random.uniform(-2.5, 2.5), random.uniform(y_start, y_end)
        radius, height = random.uniform(0.15, 0.3), 4.0
        c_id = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client
        )
        v_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.8, 0.4, 0.2, 0.9],
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=c_id,
            baseVisualShapeIndex=v_id,
            basePosition=[x, y, height / 2],
            physicsClientId=client,
        )
        obs_list.append(
            {"type": "cylinder", "pos": np.array([x, y, 0]), "radius": radius}
        )


def build_sphere_mines(client, obs_list, y_start, y_end, num=30):
    """第二重：悬浮的 3D 点云地雷"""
    for _ in range(num):
        x, y, z = (
            random.uniform(-2.5, 2.5),
            random.uniform(y_start, y_end),
            random.uniform(0.5, 3.5),
        )
        radius = random.uniform(0.2, 0.4)
        c_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=client
        )
        v_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.2, 0.8, 0.2, 0.8],
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=c_id,
            baseVisualShapeIndex=v_id,
            basePosition=[x, y, z],
            physicsClientId=client,
        )
        obs_list.append(
            {"type": "sphere", "pos": np.array([x, y, z]), "radius": radius}
        )


def get_heart_targets(num_drones, center_y=35.0, base_z=0.3, scale=0.1):
    """生成终点心形"""
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
        targets[i, :] = [x_param * scale, center_y, base_z + (y_param + 17) * scale]
    return targets


# ==============================================================================
# 3. 主运行程序
# ==============================================================================
if __name__ == "__main__":
    NUM_DRONES = 16

    # 1. 初始起飞点：Y = -3.0
    INIT_XYZS = np.zeros((NUM_DRONES, 3))
    for i in range(NUM_DRONES):
        INIT_XYZS[i, :] = [(i % 4) * 0.6 - 0.9, -3.0 + (i // 4) * 0.6, 0.1]

    # 2. 目标终点：Y = 25.0 (适中的距离，展现赛道感)
    TARGET_POS = get_heart_targets(NUM_DRONES, center_y=24.0, scale=0.12, base_z=0.4)

    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        gui=True,
        obstacles=False,
    )
    PYB_CLIENT = env.getPyBulletClient()
    # obs, info = env.reset()

    obs_list = []
    np.random.seed(int(time.time()))

    # ------------------------------------------------------------------
    # 【纯净版：双重长途试炼场】
    # ------------------------------------------------------------------
    create_boundary(PYB_CLIENT, obs_list)

    build_cylinder_forest(PYB_CLIENT, obs_list, y_start=2.0, y_end=12.0, num=40)

    build_sphere_mines(PYB_CLIENT, obs_list, y_start=12.0, y_end=22.0, num=120)

    start_time = time.time()
    action = np.zeros((NUM_DRONES, 4))

    for step in range(100000):
        all_drones_pos = np.array(
            [env._getDroneStateVector(i)[0:3] for i in range(NUM_DRONES)]
        )

        for i in range(NUM_DRONES):
            current_pos = all_drones_pos[i]
            goal_pos = TARGET_POS[i]

            F_res = calculate_rapf(current_pos, goal_pos, obs_list, all_drones_pos, i)

            f_norm = np.linalg.norm(F_res)
            v_dir = F_res / f_norm if f_norm > 0 else np.zeros(3)

            speed_fraction = np.clip(f_norm * 0.4, 0.0, 1.0)
            if np.linalg.norm(goal_pos - current_pos) < 0.1:
                speed_fraction = 0.0

            action[i, 0:3] = v_dir
            action[i, 3] = speed_fraction * 0.6  # 调整最高速度

        obs, reward, terminated, truncated, info = env.step(action)
        sync(step, start_time, env.CTRL_TIMESTEP)

    env.close()
