from typing import Tuple

battery_parameters={
'capacity_kwh':300,# kW.h
'max_charge_kw':20, # kW
'max_discharge_kw':20, #kW
'charge_efficiency':0.95,
'discharge_efficiency':0.95,
'initial_soc':0.4}

class BatteryUnit():
    """
    BatteryUnit 表示环境中的一个电池单元，支持充放电动作并维护状态。

    Attributes:
        capacity_kwh (float): 电池容量（kWh）。
        max_charge_kw (float): 最大充电功率（kW）。
        max_discharge_kw (float): 最大放电功率（kW）。
        charge_efficiency (float): 充电效率（0~1）。
        discharge_efficiency (float): 放电效率（0~1）。
        soc (float): 当前状态荷电量 (State of Charge)，0~1 之间。
    """

    def __init__(self, parameters):
        self.capacity_kwh = parameters['capacity_kwh']
        self.max_charge_kw = parameters['max_charge_kw']
        self.max_discharge_kw = parameters['max_discharge_kw']
        self.charge_efficiency = parameters['charge_efficiency']
        self.discharge_efficiency = parameters['discharge_efficiency']
        self.initial_soc = parameters['initial_soc']
        self.soc = self.initial_soc
        self._validate_parameters()

    def _validate_parameters(self):
        assert 0.0 <= self.soc <= 1.0, "initial_soc 必须在 [0,1] 之间"
        assert 0.0 < self.capacity_kwh, "容量必须大于 0"
        assert 0.0 < self.max_charge_kw, "最大充电功率必须大于 0"
        assert 0.0 < self.max_discharge_kw, "最大放电功率必须大于 0"
        assert 0.0 < self.charge_efficiency <= 1.0, "充电效率必须在 (0,1] 之间"
        assert 0.0 < self.discharge_efficiency <= 1.0, "放电效率必须在 (0,1] 之间"

    def reset(self, initial_soc: float = None):
        """
        重置电池状态。
        Args:
            initial_soc (float, optional): 重置后的 SOC，如果为 None 则保持当前 soc。
        """
        initial_soc = self.initial_soc
        if initial_soc is not None:
            assert 0.0 <= initial_soc <= 1.0, "initial_soc 必须在 [0,1] 之间"
            self.soc = initial_soc

    def get_state(self) -> float:
        """
        返回当前的 SOC（状态荷电量）。
        """
        return self.soc

    def step(self, action: float, dt_h: float) -> Tuple[float, float]:
        """
        执行动作并更新 SOC。

        Args:
            action (float): 充放电命令，[-1, 1] 区间。正值表示充电（最大充电功率），负值表示放电（最大放电功率）。
            dt_h (float): 时间步长，单位小时（h）。

        Returns:
            Tuple[float, float]:
                actual_power_kw: 实际充放电功率（正为充电，负为放电，单位 kW）。
                soc (float): 更新后的 SOC。
        """
        # 限幅
        action = max(-1.0, min(1.0, action))
        if action >= 0:
            # 充电
            target_power = self.max_charge_kw
            energy_in = target_power * dt_h * self.charge_efficiency  # 考虑效率后的入库能量（kWh）
            max_energy = (1.0 - self.soc) * self.capacity_kwh
            energy_stored = min(energy_in, max_energy)
            actual_power = energy_stored / (dt_h * self.charge_efficiency)
            # 更新 SOC
            self.soc += energy_stored / self.capacity_kwh
        else:
            # 放电
            target_power = self.max_discharge_kw
            energy_out = -target_power * dt_h / self.discharge_efficiency  # 考虑效率后的出库能量（kWh）
            max_energy = self.soc * self.capacity_kwh
            energy_drawn = min(energy_out, max_energy)
            actual_power = - (energy_drawn * self.discharge_efficiency) / dt_h
            # 更新 SOC
            self.soc -= energy_drawn / self.capacity_kwh

        # 保证边界
        self.soc = max(0.0, min(1.0, self.soc))
        return actual_power*1e-3, self.soc  # kw➡️mw换算