import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from env.env422 import PowerNetEnv_IEEE123
from stable_baselines3 import PPO     # 记得安装 SB3
# === 载入已训练模型 ===
model = PPO.load(r"../数据保存-IEEE123/Bowl_4.24_2/ppo_model2.zip")

# ----------------------------------------------------------------------------------
# 通用保存函数：增加 pv_q / svc_q
# ----------------------------------------------------------------------------------
def save_data_to_csv(voltage_rec: List[np.ndarray],
                     pv_q_rec: List[np.ndarray],
                     svc_q_rec: List[np.ndarray],
                     save_path: str,
                     prefix: str = "2012_05_01_4.24"):
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame(voltage_rec).to_csv(os.path.join(save_path, f"voltage_{prefix}.csv"), index=False)
    pd.DataFrame(pv_q_rec   ).to_csv(os.path.join(save_path, f"pv_q_{prefix}.csv"),       index=False)
    pd.DataFrame(svc_q_rec  ).to_csv(os.path.join(save_path, f"svc_q_{prefix}.csv"),      index=False)
    print(f"✅ 数据已保存至 {save_path}")

# ----------------------------------------------------------------------------------
# 带控制的仿真函数
# ----------------------------------------------------------------------------------
def run_simulation_with_control(env: PowerNetEnv_IEEE123
                               ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    代理按训练策略控制 PV / SVC / 电池。
    返回记录：电压、PV 无功、SVC 无功
    """
    voltage_rec, pv_q_rec, svc_q_rec = [], [], []

    # 固定日期
    env.year, env.month, env.day = 2012, 5, 1
    obs, _ = env.reset()

    while True:
        # 预测动作并施加
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        # 记录：潮流结果中的电压 / sgen.q_mvar / shunt.q_mvar
        voltage_rec.append(env.net.res_bus.vm_pu.values.copy())
        pv_q_rec .append(env.net.sgen['q_mvar'].values.copy())
        svc_q_rec.append(env.net.shunt['q_mvar'].values.copy())

        if terminated or truncated:
            break

    return voltage_rec, pv_q_rec, svc_q_rec

# ----------------------------------------------------------------------------------
# 绘图：电压 & 无功
# ----------------------------------------------------------------------------------
def plot_control_results(voltage_rec: List[np.ndarray],
                         pv_q_rec: List[np.ndarray],
                         svc_q_rec: List[np.ndarray],
                         dt_min: int):
    t_axis = np.arange(len(voltage_rec))*dt_min

    # —— 电压 —— #
    plt.figure(figsize=(11,5))
    for idx in range(voltage_rec[0].size):
        plt.plot(t_axis, [v[idx] for v in voltage_rec], label=f'Bus {idx}')
    plt.title("Voltage (p.u.) with Control")
    plt.xlabel("Time (min)"); plt.ylabel("V (p.u.)"); plt.grid(True); plt.tight_layout()
    plt.show()

    # —— PV / SVC 无功 —— #
    plt.figure(figsize=(11,5))
    for i in range(pv_q_rec[0].size):
        plt.plot(t_axis, [q[i] for q in pv_q_rec], label=f'PV{i} q')
    for i in range(svc_q_rec[0].size):
        plt.plot(t_axis, [q[i] for q in svc_q_rec], '--', label=f'SVC{i} q')
    plt.title("PV & SVC Reactive Power (Mvar)")
    plt.xlabel("Time (min)"); plt.ylabel("q (Mvar)"); plt.grid(True); plt.legend(ncol=4); plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------------
if __name__ == "__main__":

    env_config_Bowl = {
        "battery_list": [11, 33, 55, 80],
        "year": 2012,
        "month": 1,
        "day": 1,
        "train": False,
        "reward_fun_type": "Bowl",
        "network_info": {
            'pv_buses': [5, 11, 12, 25, 32, 35, 45, 51, 56, 66, 75, 82, 85, 101, 110],
            'pv_q_mvar': [0.05, 0.07, 0.06, 0.05, 0.07, 0.06, 0.05, 0.07, 0.06, 0.05, 0.05, 0.07, 0.06, 0.05, 0.07],
            'svc_buses': [31, 50, 74, 110],
            'q_mvar': [0.05, 0.05, 0.05, 0.05],
        },
        "activate_power_data_path": "../data/active_power-data/IEEE123/load_active.csv",
        "reactivate_power_data_path": "../data/reactive_power-data/IEEE123/load_reactive.csv",
        "pv_data_path": "../data/pv-data/IEEE123/pv_active.csv"
    }

    env_Bowl = PowerNetEnv_IEEE123(env_config_Bowl)
    # ——— 仿真 ———
    voltage_rec, pv_q_rec, svc_q_rec = run_simulation_with_control(env_Bowl)

    # ——— 保存 ———
    save_dir = r"../数据保存-IEEE123/测试数据2012_5_1-4.24"
    save_data_to_csv(voltage_rec, pv_q_rec, svc_q_rec, save_dir)

    # ——— 可视化 ———
    plot_control_results(voltage_rec, pv_q_rec, svc_q_rec,
                         dt_min=env_Bowl.activate_power_data.time_interval)