import random
import numpy as np
import pandapower as pp
import pandas as pd
import copy
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
# 忽略一些不影响运行的警告⚠️
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from data_manager.data_manager import GeneralPowerDataManager  # 数据管理模块
from make_net.pandapower_net import create_pandapower_net  # pandapower网络形成模块
from make_net.BatteryUnit import BatteryUnit, battery_parameters  # 电池模块

# 环境配置
env_config = {
    "battery_list": [11, 15, 26, 29, 33],
    "year": 2012,
    "month": 1,
    "day": 1,
    "reward_fun_type": "Bowl",
    "train": True,
    "network_info": {
        'pv_buses': [10, 25, 60],
        'pv_q_mvar': [0.05, 0.07, 0.06],
        'svc_buses': [30, 70],
        'q_mvar': [0.02, 0.03],
    },
    "activate_power_data_path": "../data/active_power-data/IEEE123/load_active.csv",
    "reactivate_power_data_path": "../data/reactive_power-data/IEEE123/load_reactive.csv",
    "pv_data_path": "../data/pv-data/IEEE123/pv_active.csv"
}


class PowerNetEnv_IEEE123(gym.Env):
    """
    自定义环境：模拟配电网无功电压控制。
    代理通过控制光伏(PV)、静态无功补偿器(SVC)和电池的功率来缓解电压波动。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config: dict = env_config) -> None:

        super(PowerNetEnv_IEEE123, self).__init__()

        """初始化环境配置。"""
        config = env_config
        self.battery_list: list = config['battery_list']
        self.year: int = config['year']
        self.month: int = config['month']
        self.day: int = config['day']
        self.train: bool = config['train']
        self.network_info: dict = config['network_info']
        self.reward_fun_type = config['reward_fun_type']

        if not self.network_info:
            raise ValueError("请输入配电网信息用于构建配电网。")
        self.base_net = create_pandapower_net(self.network_info)
        self.net = copy.deepcopy(self.base_net)

        if not self.battery_list:
            raise ValueError("没有添加储能!")

        # 初始化电池对象
        for node_index in self.battery_list:
            setattr(self, f"battery_{node_index}", BatteryUnit(battery_parameters))

        # 动作空间：PV、SVC和电池的操作
        n_actions = len(self.network_info['pv_buses']) + len(self.network_info['svc_buses']) + len(self.battery_list)
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

        # 初始化数据管理器
        self.activate_power_data = GeneralPowerDataManager(config['activate_power_data_path'])
        self.reactivate_power_data = GeneralPowerDataManager(config['reactivate_power_data_path'])
        self.pv_power_data = GeneralPowerDataManager(config['pv_data_path'])
        self.episode_length: int = int(24 * 60 / self.activate_power_data.time_interval)

        # 状态空间：包括电池SOC、所有节点的电压、有功和无功功率
        self.observation_length = (len(self.battery_list) +  # 电池SOC
                                   len(self.base_net.bus) +  # 电压
                                   len(self.base_net.bus) * 2 +  # 节点P和Q
                                   len(self.network_info['pv_buses']) * 2 +  # 光伏P和Q
                                   len(self.network_info['svc_buses']))  # SVC Q
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.observation_length,), dtype=np.float32)

        self.current_time: int = 0
        self.after_control: Optional[np.ndarray] = None

    def reset(self, *, seed: int | None = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng, _ = gym.utils.seeding.np_random(seed)

        # -------- 随机挑选日期 --------
        if self.train:
            dates = self.activate_power_data.train_dates
            self.year, self.month, self.day = random.choice(dates)
        else:
            self.year, self.month, self.day = 2012, 5, 1

        # -------- 清空时间指针与网、储能 --------
        self.current_time = 0
        self.net = copy.deepcopy(self.base_net)
        for node in self.battery_list:
            getattr(self, f"battery_{node}").reset()

        obs = self._build_state().astype(np.float32)
        info: dict = {}
        return obs, info

    def _build_state(self) -> np.ndarray:
        """构建当前状态观测值。"""
        try:
            obs = self._get_obs()
            vm_pu = np.array(list(obs['node_data']['voltage'].values()))
            active_power = np.array(list(obs['node_data']['active_power'].values()))
            reactive_power = np.array(list(obs['node_data']['reactive_power'].values()))
            pv_active_power = np.array(list(obs['node_data']['pv_active_power'].values()))
            pv_reactive_power = np.array(list(obs['node_data']['pv_reactive_power'].values()))
            svc_reactive_power = np.array(list(obs['node_data']['svc_reactive_power'].values()))
            soc_list = np.array([obs['battery_data']['soc'][f'battery_{node}'] for node in self.battery_list])

            # 状态向量拼接
            state = np.concatenate(
                (vm_pu, active_power, reactive_power, pv_active_power, pv_reactive_power, svc_reactive_power, soc_list))
            return state
        except Exception as e:
            raise RuntimeError(f"Failed to build state: {str(e)}")

    def _get_obs(self) -> Dict:
        """
        通过运行潮流计算来获取观测值。
        返回：
            Dict：包含网络和电池数据的观测字典。
        """
        try:
            # ——————— 1. 安全读取功率数据 ——————— #
            def safe_array(dm, y, m, d, t, scale):
                raw = dm.select_timeslot_data(y, m, d, t)
                values = pd.to_numeric(raw, errors="coerce").astype(float)
                return np.nan_to_num(values, nan=0.0) * scale  # NaN→0 再统一缩放

            active_power = safe_array(self.activate_power_data, self.year, self.month, self.day, self.current_time,
                                      0.85)
            reactive_power = safe_array(self.reactivate_power_data, self.year, self.month, self.day, self.current_time,
                                        0.85)
            pv_power = safe_array(self.pv_power_data, self.year, self.month, self.day, self.current_time, 1.25)

            # ——————— 2. 更新负荷 p/q ——————— #
            for i, row in self.net.load.iterrows():
                self.net.load.at[i, 'p_mw'] = active_power[i]
                self.net.load.at[i, 'q_mvar'] = reactive_power[i]

            # ——————— 3. 更新 PV & SVC ——————— #
            for i, bus in enumerate(self.network_info['pv_buses']):
                if i < len(self.net.sgen):
                    self.net.sgen.at[i, 'p_mw'] = pv_power[i]
                    if self.current_time == 0:
                        self.net.sgen.at[i, 'q_mvar'] = 0.0
            if self.current_time == 0:
                for i, _ in enumerate(self.network_info['svc_buses']):
                    if i < len(self.net.shunt):
                        self.net.shunt.at[i, 'q_mvar'] = 0.0

            # ——————— 4. 潮流计算 ——————— #
            try:
                pp.runpp(self.net, numba=False)
            except Exception as e:
                msg = (f"潮流计算失败: {e}\n"
                       f"bus:\n{self.net.bus.head()}\n"
                       f"load:\n{self.net.load.head()}\n"
                       f"sgen:\n{self.net.sgen.head()}\n"
                       f"shunt:\n{self.net.shunt.head()}")
                raise RuntimeError(msg)

            # ——————— 5. 组织观测字典 ——————— #
            obs = {
                'node_data': {
                    'voltage': {},
                    'active_power': {},
                    'reactive_power': {},
                    'pv_active_power': {},
                    'pv_reactive_power': {},
                    'svc_reactive_power': {}
                },
                'battery_data': {'soc': {}}
            }
            # ——————— 6. 将负荷数据扩充为节点数据，就是在没有负荷的节点默认设置为0.0 ——————— #
            active_vals = np.zeros(len(self.net.bus))
            reactive_vals = np.zeros(len(self.net.bus))
            load_buses = self.net.load['bus'].unique()
            # 映射到对应的 bus 上
            for i, bus_id in enumerate(load_buses):
                active_vals[bus_id] = active_power[i]
                reactive_vals[bus_id] = reactive_power[i]

            # ——————— 7. 获取数据 ——————— #
            # 获取节点数据（节点电压，节点有功，节点无功）
            for idx, bus in self.net.bus.iterrows():
                node_id = bus['name']
                obs['node_data']['voltage'][f'node_{node_id}'] = self.net.res_bus.vm_pu.at[node_id]
                obs['node_data']['active_power'][f'node_{node_id}'] = active_vals[idx]
                obs['node_data']['reactive_power'][f'node_{node_id}'] = reactive_vals[idx]

            # 获取光伏数据（有功，无功）
            for i, bus in enumerate(self.network_info['pv_buses']):
                obs['node_data']['pv_active_power'][f'node_{bus}'] = pv_power[i]
                obs['node_data']['pv_reactive_power'][f'node_{bus}'] = self.net.sgen.at[i, 'q_mvar']

            # 获取SVC数据（无功）
            for i, bus in enumerate(self.network_info['svc_buses']):
                obs['node_data']['svc_reactive_power'][f'node_{bus}'] = self.net.shunt.at[i, 'q_mvar']

            # 获取电池SOC数据（SOC）
            for node in self.battery_list:
                obs['battery_data']['soc'][f'battery_{node}'] = getattr(self, f"battery_{node}").get_state()

            return obs
        except Exception as e:
            raise RuntimeError(f"潮流计算失败：{str(e)}")

    '''
    策略网络输出一般是在(−1,1)区间内的“归一化动作”，这有利于网络训练的数值稳定性。
    但在实际电力系统、电机控制、储能调度等应用中，我们需要将其“反归一化”成实际动作。
    🌟🌟🌟 安全强化学习的实现关键点也是在这里通过先验在反归一化的过程中保证实际动作的安全。
    '''

    def _denormalize_q_pv(self, normalized_action: float) -> float:
        """反归一化光伏无功功率动作。"""
        q_min, q_max = -max(self.network_info['pv_q_mvar']), max(self.network_info['pv_q_mvar'])
        return q_min + (normalized_action + 1) / 2 * (q_max - q_min)

    def _denormalize_q_svc(self, normalized_action: float) -> float:
        """反归一化SVC无功功率动作。"""
        q_min, q_max = -max(self.network_info['q_mvar']), max(self.network_info['q_mvar'])
        return q_min + (normalized_action + 1) / 2 * (q_max - q_min)

    def _apply_actions(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """将控制动作应用于网络，返回节省的能量和电压。"""
        n_pv, n_svc = len(self.network_info['pv_buses']), len(self.network_info['svc_buses'])
        pv_actions = action[:n_pv]
        svc_actions = action[n_pv:n_pv + n_svc]
        bat_actions = action[n_pv + n_svc:]

        power_before = self.net.res_ext_grid['p_mw'].sum()

        # 应用PV、SVC、和电池的动作(➕为充电（相当于负荷所以对负荷来说是减），➖为放电（相当于电源所以对负荷来说是减）)
        for i, bus in enumerate(self.network_info['pv_buses']):
            self.net.sgen.at[i, 'q_mvar'] = self._denormalize_q_pv(pv_actions[i])
        for i, bus in enumerate(self.network_info['svc_buses']):
            self.net.shunt.at[i, 'q_mvar'] = self._denormalize_q_svc(svc_actions[i])
        for i, node in enumerate(self.battery_list):
            battery = getattr(self, f"battery_{node}")
            p_mw = (bat_actions[i])
            actual_power, _ = battery.step(p_mw, self.activate_power_data.time_interval / 60)
            self.net.load.at[node, 'p_mw'] += actual_power

        # 运行潮流计算
        pp.runpp(self.net, algorithm='nr', numba=False)
        self.after_control = self.net.res_bus.vm_pu.to_numpy(dtype=float)
        power_after = self.net.res_ext_grid['p_mw'].sum()

        return power_before - power_after, self.after_control

    def step(self, action: np.ndarray):
        """推进一个时间步，符合 SB3 的 (obs, reward, terminated, truncated, info) 接口。"""
        saved_energy, vm_pu = self._apply_actions(action)
        reward = self._calculate_reward(vm_pu, saved_energy, action)  # ← 传入 action

        self.current_time += 1
        terminated = self.current_time >= self.episode_length
        truncated = False  # 这里不额外使用截断逻辑

        obs = self._build_state()  # **不要** 在 step 里调用 reset()
        info = {}  # 如需调试指标可在此处放入

        return obs, reward, terminated, truncated, info

    '''
    奖励函数：
        - 使用碗状 (Bowl-shape) 电压屏障函数，限制电压偏离安全范围
        - 惩罚无功功率变化量的平方和，避免过大调整

        - 使用多段双曲正切电压屏障函数(Tanh)
        - 在安全范围内外，分别采用不同的梯度惩罚机制
        - 惩罚无功功率变化量的平方和，避免过大调整
    '''

    def _calculate_reward(self, vm_pu: np.ndarray, saved_power: float, action: np.ndarray) -> float:
        if self.reward_fun_type == 'Bowl':
            voltages = vm_pu  # 获取当前电压
            v_ref = 1.0  # 参考电压
            safe_range = 0.05  # 碗状函数的安全范围 (|v_k - v_ref| <= 0.05)
            a, b, c, d = 2.0, 0.095, 0.01, 0.04  # 超参数，可根据需求调整

            # 正态分布密度函数
            def normal_density(v, mean, std):
                return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - mean) / std) ** 2)

            # 碗状电压屏障函数
            def voltage_barrier(v_k):
                deviation = np.abs(v_k - v_ref)
                if deviation > safe_range:
                    # 电压超出安全范围时，采用线性惩罚
                    return a * deviation - b
                else:
                    # 电压在安全范围内，采用负的正态分布密度函数惩罚
                    return -c * normal_density(v_k, v_ref, 0.1) + d

            # 计算每个母线的电压屏障值
            voltage_penalty = np.sum([voltage_barrier(v) for v in voltages])

            # 无功功率调整惩罚项
            action_penalty = np.square(action).sum()

            # 奖励：负的电压屏障值和动作惩罚值
            reward = -1 * voltage_penalty - 0.01 * action_penalty - 0.1 * saved_power  # 权重可根据需要调整
            return float(reward) * 0.5

        elif self.reward_fun_type == 'Tanh':
            voltages = vm_pu
            v_ref = 1.0
            safe_lower = 0.95
            safe_upper = 1.05
            inner_lower = 0.975
            inner_upper = 1.025
            a, b, c, d = 2.0, 0.5, 0.01, 0.04

            def voltage_barrier(v_k):
                if inner_lower <= v_k <= inner_upper:
                    return d * np.tanh(a * np.abs(v_k - v_ref))
                elif safe_lower <= v_k < inner_lower or inner_upper < v_k <= safe_upper:
                    return b * np.abs(v_k - v_ref) + c
                else:
                    return a * np.abs(v_k - v_ref)

            voltage_penalty = np.sum([voltage_barrier(v) for v in voltages])
            action_penalty = np.square(action).sum()

            # 增加电压接近参考值的正向奖励
            voltage_reward = np.sum(np.exp(-a * (voltages - v_ref) ** 2))

            reward = voltage_reward - 0.5 * voltage_penalty - 0.05 * action_penalty - 0.05 * saved_power
            return float(reward) * 0.5

    def render(self, mode: str = "human"):
        """简单打印关键运行信息。"""
        max_dev = np.max(np.abs(self.net.res_bus.vm_pu.values - 1.0))
        print(f"[t={self.current_time:03d}]  max|ΔV|={max_dev:.4f}")


if __name__ == '__main__':
    # 创建环境
    env = PowerNetEnv_IEEE123(env_config)

    # 环境初始化
    state = env.reset()
    print(f"Initial State: {state}")

    # 进行多个步骤的测试
    for step in range(10):  # 测试10个时间步
        # 随机选择一个动作
        action = np.random.uniform(low=-1, high=1, size=env.action_space.shape)

        # 执行动作
        next_state, reward, done, _, info = env.step(action)

        # 输出信息
        print(f"Step {step + 1}:")
        print(f"Action: {action}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        if done:
            print("Environment reset after reaching the end of the episode.")
            state = env.reset()  # 重新开始一个新的回合
        else:
            state = next_state  # 更新当前状态为下一个状态