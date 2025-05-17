import random
import numpy as np
import pandapower as pp
import pandas as pd
import copy
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import cvxpy as cp  # cvxpy 是一个用于定义和解决凸优化问题的 Python 库

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # 忽略一些不影响运行的警告⚠️

from data_manager.data_manager import GeneralPowerDataManager  # 数据管理模块
from make_net.pandapower_net import create_pandapower_net  # pandapower网络形成模块
from make_net.BatteryUnit import BatteryUnit, battery_parameters  # 电池模块

# 环境配置
env_config = {
    "battery_list": [11, 33, 55, 80],
    "year": 2012,
    "month": 1,
    "day": 1,
    "train": True,
    "reward_fun_type": "Bowl",
    "network_info": {
        'pv_buses': [5, 11, 12, 25, 32, 35, 45, 51, 56, 66, 75, 82, 85, 101, 110],
        'pv_q_mvar': [0.10, 0.14, 0.12, 0.10, 0.14, 0.12, 0.10, 0.14, 0.12, 0.10, 0.10, 0.14, 0.12, 0.10, 0.14],
        'svc_buses': [31, 50, 74, 110],
        'q_mvar': [0.15, 0.15, 0.15, 0.15],
    },
    "activate_power_data_path": "../data/active_power-data/IEEE123/load_active.csv",
    "reactivate_power_data_path": "../data/reactive_power-data/IEEE123/load_reactive.csv",
    "pv_data_path": "../data/pv-data/IEEE123/pv_active.csv"
}

def safe_array(data_manager: GeneralPowerDataManager, year: int, month: int, day: int, time_idx: int, scale: float) -> np.ndarray:
    """
    安全读取指定时间槽的数据，遇到非数字时置零，并进行缩放。
    参数：
      data_manager - 数据管理器实例
      year, month, day - 年月日
      time_idx - 时段索引
      scale - 缩放因子
    返回：
      经过处理和缩放后的 numpy 数组
    """
    raw = data_manager.select_timeslot_data(year, month, day, time_idx)
    values = pd.to_numeric(raw, errors="coerce").astype(float)
    return np.nan_to_num(values, nan=0.0) * scale

class Safe_PowerNetEnv_IEEE123(gym.Env):
    """
    自定义环境：在 IEEE-123 配电网中，代理通过控制光伏(PV)、静态无功补偿器(SVC)
    和储能电池来调节无功功率，以保持母线电压在安全范围内。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config: dict) -> None:
        super().__init__()

        # --- 读取配置参数 ---
        self.battery_list = env_config['battery_list']           # 电池节点列表
        self.network_info = env_config['network_info']           # 网络信息，包括PV/SVC参数
        self.year = env_config['year']                           # 起始年份
        self.month = env_config['month']                         # 起始月份
        self.day = env_config['day']                             # 起始日期
        self.train = env_config['train']                         # 训练/测试标志
        self.reward_fun_type = env_config['reward_fun_type']     # 奖励函数类型

        # --- 参数校验 ---
        if not self.network_info:
            raise ValueError("必须提供网络信息。")
        if not self.battery_list:
            raise ValueError("至少需要一个电池单元。")

        # --- 构建 Pandapower 网并备份 ---
        self.base_net = create_pandapower_net(self.network_info)
        self.net = copy.deepcopy(self.base_net)

        # --- 初始化电池对象 ---
        for node in self.battery_list:
            setattr(self, f"battery_{node}", BatteryUnit(battery_parameters))

        # --- 定义动作空间：PV、SVC、储能动作均归一化到[-1,1] ---
        n_pv = len(self.network_info['pv_buses'])
        n_svc = len(self.network_info['svc_buses'])
        n_bat = len(self.battery_list)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(n_pv + n_svc + n_bat,), dtype=np.float32
        )

        # --- 初始化数据管理器，用于获取时序功率数据 ---
        self.activate_power_data = GeneralPowerDataManager(env_config['activate_power_data_path'])
        self.reactivate_power_data = GeneralPowerDataManager(env_config['reactivate_power_data_path'])
        self.pv_power_data = GeneralPowerDataManager(env_config['pv_data_path'])
        self.episode_length = int(24 * 60 / self.activate_power_data.time_interval)

        # --- 定义观测空间：母线电压、负荷有功/无功、PV有功/无功、SVC无功、电池SOC ---
        n_bus = len(self.base_net.bus)
        obs_len = (
            n_bus +               # 母线电压
            2 * n_bus +           # 负荷有功+无功
            2 * n_pv +            # PV有功+无功
            n_svc +               # SVC无功
            n_bat                 # 电池SOC
        )
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(obs_len,), dtype=np.float32
        )

        # --- 模拟状态变量 ---
        self.current_time = 0
        self.v_min = 0.95     # 最小安全电压(p.u.)
        self.v_max = 1.05     # 最大安全电压(p.u.)
        self.epsilon = 0.02   # 安全裕度

    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境：随机挑选训练日期或使用固定测试日期，重置电池和网络状态。
        """
        super().reset(seed=seed)
        if self.train:
            dates = self.activate_power_data.train_dates
            self.year, self.month, self.day = random.choice(dates)
        else:
            self.year, self.month, self.day = 2012, 5, 1  # 测试使用固定日期

        self.current_time = 0
        self.net = copy.deepcopy(self.base_net)
        for node in self.battery_list:
            getattr(self, f"battery_{node}").reset()

        obs = self._build_state().astype(np.float32)
        return obs, {}
    def _build_state(self) -> np.ndarray:
        """
        根据当前网络和电池状态，构建并返回观测向量。
        """
        obs_dict = self._get_obs()
        voltages = np.array(list(obs_dict['node_data']['voltage'].values()))
        p_load = np.array(list(obs_dict['node_data']['active_power'].values()))
        q_load = np.array(list(obs_dict['node_data']['reactive_power'].values()))
        p_pv = np.array(list(obs_dict['node_data']['pv_active_power'].values()))
        q_pv = np.array(list(obs_dict['node_data']['pv_reactive_power'].values()))
        q_svc = np.array(list(obs_dict['node_data']['svc_reactive_power'].values()))
        socs = np.array([obs_dict['battery_data']['soc'][f'battery_{n}'] for n in self.battery_list])
        return np.concatenate([voltages, p_load, q_load, p_pv, q_pv, q_svc, socs])
    def _get_obs(self) -> Dict:
        """
        运行潮流计算，读取并返回母线电压、负荷、PV/SVC注入和电池SOC数据。
        """
        # 1) 读取并缩放功率时序数据
        P_load = safe_array(self.activate_power_data, self.year, self.month, self.day,
                             self.current_time, scale=0.65)
        Q_load = safe_array(self.reactivate_power_data, self.year, self.month, self.day,
                             self.current_time, scale=0.65)
        P_pv   = safe_array(self.pv_power_data, self.year, self.month, self.day,
                             self.current_time, scale=1.75)

        # 2) 更新网络中的负荷数据
        for idx, _ in self.net.load.iterrows():
            self.net.load.at[idx, 'p_mw'] = P_load[idx]
            self.net.load.at[idx, 'q_mvar'] = Q_load[idx]

        # 3) 更新 PV 和 SVC
        for i, bus in enumerate(self.network_info['pv_buses']):
            self.net.sgen.at[i, 'p_mw'] = P_pv[i]
            if self.current_time == 0:
                self.net.sgen.at[i, 'q_mvar'] = 0.0
        if self.current_time == 0:
            for i in range(len(self.network_info['svc_buses'])):
                self.net.shunt.at[i, 'q_mvar'] = 0.0

        # 4) 潮流计算
        try:
            pp.runpp(self.net, numba=False)
        except Exception as exc:
            raise RuntimeError(f"潮流计算失败: {exc}")

        # 5) 构建观测字典
        n_bus = len(self.net.bus)
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

        # 将负荷分配到对应母线上，无负荷母线为 0
        P_bus = np.zeros(n_bus); Q_bus = np.zeros(n_bus)
        load_buses = list(self.net.load.bus)
        for i, bus in enumerate(load_buses):
            P_bus[bus] = P_load[i]
            Q_bus[bus] = Q_load[i]

        # 填充母线数据
        for idx, bus in self.net.bus.iterrows():
            vid = int(bus.name)
            obs['node_data']['voltage'][f'node_{vid}'] = self.net.res_bus.vm_pu.at[vid]
            obs['node_data']['active_power'][f'node_{vid}'] = P_bus[idx]
            obs['node_data']['reactive_power'][f'node_{vid}'] = Q_bus[idx]

        # 填充 PV 数据
        for i, bus in enumerate(self.network_info['pv_buses']):
            obs['node_data']['pv_active_power'][f'node_{bus}'] = P_pv[i]
            obs['node_data']['pv_reactive_power'][f'node_{bus}'] = self.net.sgen.at[i, 'q_mvar']

        # 填充 SVC 数据
        for i, bus in enumerate(self.network_info['svc_buses']):
            obs['node_data']['svc_reactive_power'][f'node_{bus}'] = self.net.shunt.at[i, 'q_mvar']

        # 填充电池 SOC
        for n in self.battery_list:
            obs['battery_data']['soc'][f'battery_{n}'] = getattr(self, f'battery_{n}').get_state()

        return obs

    def _denormalize_action(self, normalized_action: float, q_max: float) -> float:
        """将归一化动作从[-1, 1]反归一化到[-q_max, q_max]。"""
        return normalized_action * q_max

    def _apply_action_to_net(self, net, action):
        """将动作应用于指定网络并运行潮流计算。"""
        n_pv = len(self.network_info['pv_buses'])
        n_svc = len(self.network_info['svc_buses'])
        pv_actions = action[:n_pv]
        svc_actions = action[n_pv:n_pv + n_svc]
        bat_actions = action[n_pv + n_svc:]

        for i, bus in enumerate(self.network_info['pv_buses']):
            q_max = self.network_info['pv_q_mvar'][i]
            net.sgen.at[i, 'q_mvar'] = self._denormalize_action(pv_actions[i], q_max)
        for i, bus in enumerate(self.network_info['svc_buses']):
            q_max = self.network_info['q_mvar'][i]
            net.shunt.at[i, 'q_mvar'] = self._denormalize_action(svc_actions[i], q_max)
        for i, node in enumerate(self.battery_list):
            battery = getattr(self, f"battery_{node}")
            p_mw = bat_actions[i]
            actual_power, _ = battery.step(p_mw, self.activate_power_data.time_interval / 60)
            net.load.at[node, 'p_mw'] += actual_power

        pp.runpp(net, calculate_voltage_angles=True, trafo_model="t", numba=False)

    def _is_safe(self, net):
        """检查网络中所有电压是否在安全范围内。"""
        return all(self.v_min <= vm <= self.v_max for vm in net.res_bus.vm_pu)

    def _find_safe_action(self, action):
        """使用电压灵敏度矩阵和优化调整特定设备动作，增强鲁棒性。"""
        try:
            net_copy = copy.deepcopy(self.net)
            self._apply_action_to_net(net_copy, action)
            V_A = net_copy.res_bus.vm_pu.values
            print(f"安全约束前最大电压差值 {np.max(np.abs(V_A - 1.0)):.4f}")
            if all(self.v_min <= v <= self.v_max for v in V_A):
                return action

            action_current = action.copy()
            n_pv = len(self.network_info['pv_buses'])
            n_svc = len(self.network_info['svc_buses'])
            bat_action = action[n_pv + n_svc:]
            M = 1000.0
            max_iters = 50

            for i in range(1, max_iters + 1):
                net_temp = copy.deepcopy(self.net)
                self._apply_action_to_net(net_temp, action_current)
                V_temp = net_temp.res_bus.vm_pu.values

                J = net_temp._ppc.get("internal", {}).get("J", None)
                if J is None:
                    print(f"[第{i}轮] 警告：Jacobian 未生成，返回保守动作")
                    return np.zeros_like(action)
                try:
                    J_dense = J.toarray()
                    n = net_temp.bus.shape[0] - 1
                    B = J_dense[n:, n:]
                    S_vq = np.linalg.pinv(B)
                except Exception as e:
                    print(f"[第{i}轮] Jacobian处理异常：{e}")
                    return action_current

                pv_buses = self.network_info['pv_buses']
                svc_buses = self.network_info['svc_buses']
                C = pv_buses + svc_buses
                C_idx = [net_temp.bus.index.get_loc(bus) for bus in C]
                m = len(C)
                pv_sgen_idx = [net_temp.sgen.index[net_temp.sgen.bus == bus].tolist()[0] for bus in pv_buses]
                svc_shunt_idx = [net_temp.shunt.index[net_temp.shunt.bus == bus].tolist()[0] for bus in svc_buses]
                Q_current_pv = [net_temp.sgen.at[idx, 'q_mvar'] for idx in pv_sgen_idx]
                Q_current_svc = [net_temp.shunt.at[idx, 'q_mvar'] for idx in svc_shunt_idx]
                Q_current = Q_current_pv + Q_current_svc
                Q_max_pv = self.network_info['pv_q_mvar']
                Q_max_svc = self.network_info['q_mvar']
                Q_max = list(Q_max_pv) + list(Q_max_svc)
                D_min = [idx for idx, v in enumerate(V_temp) if v < self.v_min]
                D_max = [idx for idx, v in enumerate(V_temp) if v > self.v_max]

                x = cp.Variable(m)
                s = cp.Variable(len(D_min) + len(D_max), nonneg=True)
                objective = cp.Minimize(cp.sum_squares(x) + M * cp.sum(s))
                constraints = []
                for j in range(m):
                    lb = -Q_max[j] - Q_current[j]
                    ub = Q_max[j] - Q_current[j]
                    constraints += [x[j] >= lb, x[j] <= ub]

                for idx_k, k in enumerate(D_min):
                    coeffs = [S_vq[k - 1, C_idx[j]] for j in range(m)]
                    constraints.append(
                        sum(coeffs[j] * x[j] for j in range(m)) + s[idx_k] >= self.v_min - V_temp[k] + self.epsilon)
                for idx_k, k in enumerate(D_max):
                    coeffs = [S_vq[k - 1, C_idx[j]] for j in range(m)]
                    constraints.append(
                        sum(coeffs[j] * x[j] for j in range(m)) <= self.v_max - V_temp[k] - self.epsilon + s[
                            len(D_min) + idx_k])

                try:
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, warm_start=True)
                except Exception as e:
                    print(f"[第{i}轮] 优化求解异常：{e}")
                    return np.zeros_like(action)

                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"[第{i}轮] 优化失败（状态：{prob.status}），返回动作")
                    return action_current if i > 1 else np.zeros_like(action)

                Delta_Q = x.value
                Q_safe = [Q_current[j] + Delta_Q[j] for j in range(m)]
                A_pv = [Q_safe[j] / Q_max_pv[j] for j in range(len(pv_buses))]
                A_svc = [Q_safe[len(pv_buses) + j] / Q_max_svc[j] for j in range(len(svc_buses))]
                action_new = np.concatenate([A_pv, A_svc, bat_action])
                net_safe = copy.deepcopy(self.net)
                self._apply_action_to_net(net_safe, action_new)
                V_new = net_safe.res_bus.vm_pu.values

                slack_sum = float(np.sum(s.value)) if s.value is not None else 0.0
                V_min_val = float(np.min(V_new))
                V_max_val = float(np.max(V_new))
                still_violation = not all(self.v_min <= v <= self.v_max for v in V_new)

                print(  f"[第{i}轮] 优化成功，Slack={slack_sum:.4f}, V范围=[{V_min_val:.4f}, {V_max_val:.4f}], 仍越限: {'是' if still_violation else '否'}")

                if not still_violation:
                    return action_new
                action_current = action_new.copy()

            print(f"[警告] 迭代{max_iters}次仍有越限，返回当前最优动作")
            return action_current

        except Exception as e:
            print(f"[致命错误] _find_safe_action 执行异常：{e}")
            return np.zeros_like(action)

    def _apply_actions(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """将控制动作应用于网络，返回实际动作和电压。"""
        n_pv, n_svc = len(self.network_info['pv_buses']), len(self.network_info['svc_buses'])
        pv_actions = action[:n_pv]
        svc_actions = action[n_pv:n_pv + n_svc]
        bat_actions = action[n_pv + n_svc:]

        action_real = np.concatenate([pv_actions, svc_actions, bat_actions])

        for i, bus in enumerate(self.network_info['pv_buses']):
            q_max = self.network_info['pv_q_mvar'][i]
            self.net.sgen.at[i, 'q_mvar'] = self._denormalize_action(pv_actions[i], q_max)
        for i, bus in enumerate(self.network_info['svc_buses']):
            q_max = self.network_info['q_mvar'][i]
            self.net.shunt.at[i, 'q_mvar'] = self._denormalize_action(svc_actions[i], q_max)
        for i, node in enumerate(self.battery_list):
            battery = getattr(self, f"battery_{node}")
            p_mw = bat_actions[i]
            actual_power, _ = battery.step(p_mw, self.activate_power_data.time_interval / 60)
            self.net.load.at[node, 'p_mw'] += actual_power

        pp.runpp(self.net, algorithm='nr', numba=False)
        self.after_control = self.net.res_bus.vm_pu.to_numpy(dtype=float)

        return action_real, self.after_control

    def _get_max_voltage_deviation(self) -> float:
        """|vm_pu − 1.0| 的最大值"""
        return float(np.max(np.abs(self.net.res_bus.vm_pu.values - 1.0)))

    def _get_soc_stats(self) -> Tuple[float, float, float]:
        """
        返回 (mean, min, max) SOC，便于记录标量
        """
        socs = np.array([getattr(self, f"battery_{n}").get_state()
                         for n in self.battery_list], dtype=float)
        return float(socs.mean()), float(socs.min()), float(socs.max())

    def step(self, action):
        """执行一步模拟，使用安全层调整动作。"""
        # 1. 原始动作接收与归一化
        action = np.array(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)  # 将动作限制在有效范围 [-1, 1] 内
        # 2. 安全层检查与动作修正
        safe_action = self._find_safe_action(action)
        used_safe_action = not np.allclose(action, safe_action)  # 判断安全层是否启用了（原始动作是否被调整）
        action_diff_norm = float(np.linalg.norm(action - safe_action))  # 原始动作与安全动作的欧几里得距离
        slack_sum = 0.0  # slack 变量总和（若安全层未使用 slack 则为 0.0）
        # 3. 动作实际应用与状态更新
        action_real, voltages = self._apply_actions(safe_action)  # 应用最终安全动作到网络并执行潮流计算
        # 4. 奖励计算（含新增惩罚项）
        base_reward = self._calculate_reward(voltages, action_real)
        reward = base_reward
        alpha1, alpha2 = 0.5, 0.1  # 惩罚系数 (alpha1 用于动作差距，alpha2 用于 slack)
        if used_safe_action:
            reward -= alpha1 * action_diff_norm + alpha2 * slack_sum  # 若启用了安全层，按照差距和 slack 总量施加额外惩罚
        # 5. 是否终止判断与信息返回
        self.current_time += 1
        terminated = self.current_time >= self.episode_length
        next_state = self._build_state()  # 获取下一时刻的状态观测
        V = self.net.res_bus.vm_pu.values
        print(f"安全约束后最大电压差值 {np.max(np.abs(V - 1.0)):.4f}")
        dv_max = self._get_max_voltage_deviation()
        soc_mean, soc_min, soc_max = self._get_soc_stats()
        # 在 info 中记录诊断信息
        info = {
            'loss_mw': self.net.res_line.pl_mw.sum(),
            'dv_max': dv_max,
            'soc_mean': soc_mean,
            'soc_min': soc_min,
            'soc_max': soc_max,
            'used_safe_action': used_safe_action,
            'action_diff_norm': action_diff_norm,
            'slack_sum': slack_sum
        }
        return next_state, float(reward), terminated, False, info

    def _calculate_reward(self, vm_pu: np.ndarray, action_real: np.ndarray) -> float:
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
            action_penalty = np.square(action_real).sum()
            line_loss = self.net.res_line.pl_mw.sum()
            # 奖励：负的电压屏障值和动作惩罚值
            reward = -1 * voltage_penalty - 0.001 * action_penalty - 0.01 * line_loss  # 权重可根据需要调整
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
            action_penalty = np.square(action_real).sum()

            # 增加电压接近参考值的正向奖励
            voltage_reward = np.sum(np.exp(-a * (voltages - v_ref) ** 2))

            reward = voltage_reward - 0.5 * voltage_penalty - 0.01 * action_penalty
            return float(reward) * 0.5


    def render(self, mode: str = "human"):
        """简单打印关键运行信息。"""
        max_dev = np.max(np.abs(self.net.res_bus.vm_pu.values - 1.0))
        print(f"[t={self.current_time:03d}]  max|ΔV|={max_dev:.4f}")

if __name__ == '__main__':
    # 创建环境
    env = Safe_PowerNetEnv_IEEE123(env_config)

    # 环境初始化
    state = env.reset()
    print(f"Initial State: {state}")

    # 进行多个步骤的测试
    for step in range(480):  # 测试480个时间步也就是一天
        # 随机选择一个动作
        action = np.random.uniform(low=-1, high=1, size=env.action_space.shape)

        # 执行动作
        next_state, reward, done, _, info = env.step(action)

        # 输出信息
        print(f"Step {step + 1}:")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        if done:
            print("Environment reset after reaching the end of the episode.")
            state = env.reset()  # 重新开始一个新的回合
        else:
            state = next_state  # 更新当前状态为下一个状态
