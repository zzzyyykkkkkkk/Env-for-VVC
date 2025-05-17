import os, copy, numpy as np, pandas as pd, matplotlib.pyplot as plt
import pandapower as pp
from typing import Tuple, List, Dict
from env.env422 import PowerNetEnv_IEEE123

# ----------------------------------------------------------------------------------
# 数据保存：电压 + 负荷/光伏
# ----------------------------------------------------------------------------------
def save_data_to_csv(voltage_rec: List[np.ndarray],
                     power_rec: List[Dict[str, np.ndarray]],
                     save_path: str,
                     prefix: str = "2012_05_01-4.24"):
    os.makedirs(save_path, exist_ok=True)

    pd.DataFrame(voltage_rec).to_csv(os.path.join(save_path, f"voltage_{prefix}.csv"), index=False)

    # 分离 active / reactive / pv_active
    ap  = np.vstack([d["active_power"]   for d in power_rec])
    rp  = np.vstack([d["reactive_power"] for d in power_rec])
    pv  = np.vstack([d["pv_active_power"]for d in power_rec])
    pd.DataFrame(ap).to_csv(os.path.join(save_path, f"load_p_{prefix}.csv"), index=False)
    pd.DataFrame(rp).to_csv(os.path.join(save_path, f"load_q_{prefix}.csv"), index=False)
    pd.DataFrame(pv).to_csv(os.path.join(save_path, f"pv_p_{prefix}.csv"),   index=False)

    print(f"✅ 数据已保存到 {save_path}")


# ----------------------------------------------------------------------------------
# 无控制仿真：固定 2012‑05‑01
# ----------------------------------------------------------------------------------
def run_simulation_without_control(env: PowerNetEnv_IEEE123,
                                   year: int = 2012,
                                   month: int = 5,
                                   day: int = 1
                                   ) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:

    # —— 1. 固定日期并硬复位 —— #
    env.year, env.month, env.day = year, month, day
    env.current_time = 0
    env.net = copy.deepcopy(env.base_net)

    voltage_rec, power_rec = [], []

    for _ in range(env.episode_length):
        t = env.current_time

        # ---- 当日负荷 & PV 数据 ----
        ap = 0.75 * env.activate_power_data.select_timeslot_data(year, month, day, t).astype(float)
        rp = 0.75 * env.reactivate_power_data.select_timeslot_data(year, month, day, t).astype(float)
        pv = 2 * env.pv_power_data.select_timeslot_data(year, month, day, t).astype(float)

        # ---- 更新负荷 ----
        # ——————— 2. 更新负荷 p/q ——————— #
        for i, row in env.net.load.iterrows():
            env.net.load.at[i, 'p_mw'] = ap[i]
            env.net.load.at[i, 'q_mvar'] = rp[i]

        # ---- 更新 PV ----
        for i, bus in enumerate(env.network_info['pv_buses']):
            if i < len(env.net.sgen):
                env.net.sgen.at[i, 'p_mw'] = pv[i]

        # ---- 潮流计算 ----
        pp.runpp(env.net, algorithm='nr', numba=False)

        # ---- 记录 ----
        voltage_rec.append(env.net.res_bus.vm_pu.values.copy())
        power_rec.append(dict(active_power   = ap.copy(),
                              reactive_power = rp.copy(),
                              pv_active_power= pv.copy()))

        env.current_time += 1

    return voltage_rec, power_rec


# ----------------------------------------------------------------------------------
# 可视化：电压 & 负荷 / PV
# ----------------------------------------------------------------------------------
def plot_results(voltage_rec: List[np.ndarray],
                 power_rec: List[Dict[str, np.ndarray]],
                 time_interval: int):

    t_axis = np.arange(len(voltage_rec)) * time_interval

    # —— 电压 —— #
    plt.figure(figsize=(11,5))
    for idx in range(voltage_rec[0].size):
        plt.plot(t_axis, [v[idx] for v in voltage_rec], label=f'Bus{idx}')
    plt.title("Voltage Profile on 2012‑05‑01 (No Control)")
    plt.xlabel("Time (min)"); plt.ylabel("V (p.u.)"); plt.grid(True)
    plt.tight_layout(); plt.show()

    # —— 负荷有功 & PV 有功 —— #
    ap  = np.vstack([d["active_power"]    for d in power_rec])
    pv  = np.vstack([d["pv_active_power"] for d in power_rec])

    plt.figure(figsize=(11,5))
    plt.plot(t_axis, ap.sum(axis=1), label="Total Load P", linestyle='--')
    plt.plot(t_axis, pv.sum(axis=1), label="Total PV P",   linestyle='-.')
    plt.title("System Active Power on 2012‑05‑01 (Scaled)")
    plt.xlabel("Time (min)"); plt.ylabel("P (MW)"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    env_config_Bowl = {
        "battery_list": [11, 33, 55, 80],
        "year": 2012, "month": 1, "day": 1,
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

    env = PowerNetEnv_IEEE123(env_config_Bowl)
    # —— 仿真 —— #
    voltage_rec, power_rec = run_simulation_without_control(env)

    # —— 保存 —— #
    save_dir = r"../数据保存-IEEE123/优化前4.22"
    save_data_to_csv(voltage_rec, power_rec, save_dir)

    # —— 绘图 —— #
    plot_results(voltage_rec, power_rec, env.activate_power_data.time_interval)
