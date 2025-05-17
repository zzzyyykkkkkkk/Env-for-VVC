import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt

def create_pandapower_net(network_info: dict):
    """
    基于已有网络文件加载基础网架，并添加光伏与SVC设备，构建增强型配电网模型。

    参数:
        network_info (dict): 字典中可包含以下可选字段：
            - 'pv_buses': 添加光伏的节点编号列表（bus index）
            - 'pv_p_mw': 每个光伏的有功出力（可为列表或单值）
            - 'svc_buses': 添加SVC的节点编号列表
            - 'q_mvar': 每个SVC的无功设定值（可为列表或单值）

    返回:
        net: 完善后的 pandapower 网络模型
    """
    # 1. 加载基础网络
    base_net = pp.converter.from_mpc('../data/net-data/IEEE-123/case123_2025_4_15.mat', f_hz=50, casename_mpc_file='case123', validate_conversion=False)
    # 1.1 设置所有 bus 的电压幅值限制在 0.90 ~ 1.10 pu 之间
    base_net.bus.loc[1:, 'max_vm_pu'] = 1.1
    base_net.bus.loc[1:, 'min_vm_pu'] = 0.90

    # 2. 添加光伏设备（使用 sgen 表示）
    if 'pv_buses' in network_info:
        pv_buses = network_info['pv_buses']
        pv_p_mw = network_info.get('pv_p_mw', 0.05)  # 默认每台光伏 0.05MW

        for i, bus in enumerate(pv_buses):
            p_mw = pv_p_mw[i] if isinstance(pv_p_mw, list) else pv_p_mw
            pp.create_sgen(base_net, bus=bus, p_mw=p_mw, q_mvar=0.0, name=f"PV_{bus}", type='PV')

    # 3. 添加 SVC（使用 shunt 或 static generator 实现）
    if 'svc_buses' in network_info:
        svc_buses = network_info['svc_buses']
        q_mvar = network_info.get('q_mvar', 0)  # 默认SVC无功容量为 0 MVar

        for i, bus in enumerate(svc_buses):
            q = q_mvar[i] if isinstance(q_mvar, list) else q_mvar
            pp.create_shunt(base_net, bus=bus, q_mvar=q, p_mw=0.0, name=f"SVC_{bus}")

    # 修改支路电阻和电抗为原来的 0.85 倍
    base_net.line["r_ohm_per_km"] *= 0.85
    base_net.line["x_ohm_per_km"] *= 0.85

    return base_net



def plot_pandapower_results(net):
    """
    绘制 Pandapower 潮流结果，包括：
    - 节点电压曲线
    - 节点有功负荷曲线
    - 节点无功负荷曲线

    Args:
        net (pandapower.network): 经过潮流计算的 Pandapower 网络
    """
    # 获取节点索引
    buses = net.bus.index.tolist()

    # 获取结果数据
    voltage = net.res_bus.vm_pu.values
    try:
        p_mw = net.res_load.p_mw.values
        q_mvar = net.res_load.q_mvar.values
        load_buses = net.load.bus.values  # 获取 load 连接的 bus
    except:
        p_mw = np.zeros(len(buses))
        q_mvar = np.zeros(len(buses))
        load_buses = []

    # 映射有功无功到对应 bus（初始化为 0）
    p_curve = np.zeros(len(buses))
    q_curve = np.zeros(len(buses))
    for i, bus in enumerate(load_buses):
        p_curve[bus] += p_mw[i]
        q_curve[bus] += q_mvar[i]

    # 绘图
    plt.figure(figsize=(12, 6))

    plt.plot(buses, voltage, marker='o', label='Voltage (p.u.)')
    plt.plot(buses, p_curve, marker='s', label='Active Power (MW)')
    plt.plot(buses, q_curve, marker='^', label='Reactive Power (MVAr)')

    plt.title("Bus Voltage and Power Curves")
    plt.xlabel("Bus Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
测试上述函数的代码
'''

if __name__ == "__main__":
    # 网络配置路径（你可以根据实际路径修改）
    network_info = {
        'pv_buses': [10, 25, 60],
        'pv_p_mw': [0.05, 0.07, 0.06],
        'svc_buses': [30, 70],
        'q_mvar': [0.02, 0.03]
    }

    # 创建网络
    print("🔧 正在创建 Pandapower 网络...")
    net = create_pandapower_net(network_info)
    print("✅ 网络创建完成")


    # 可视化网络（可选）
    try:
        print("🖼️ 尝试绘制网络图...")
        pp.runpp(net,numba = False)
        plot_pandapower_results(net)
    except Exception as e:
        print(f"⚠️ 绘图失败：{e}")