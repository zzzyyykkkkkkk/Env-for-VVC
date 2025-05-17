import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_manager.data_manager import GeneralPowerDataManager  # 数据管理模块
from make_net.pandapower_net import create_pandapower_net  # pandapower网络形成模块
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

env_config_Bowl = {
    "battery_list": [11, 33, 55, 80],
    "year": 2012,
    "month": 1,
    "day": 1,
    "train": True,
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

# ==== 1. 构建 IEEE 123 节点网络 ====
net = create_pandapower_net(env_config_Bowl["network_info"])

# ==== 2. 数据生成函数 ====
def generate_samples(net_template, n_samples=3000):
    inputs = []
    outputs = []
    failures = 0

    activate_power_data = GeneralPowerDataManager(env_config_Bowl['activate_power_data_path'])
    reactivate_power_data = GeneralPowerDataManager(env_config_Bowl['reactivate_power_data_path'])
    pv_power_data = GeneralPowerDataManager(env_config_Bowl['pv_data_path'])
    episode_length: int = int(24 * 60 / activate_power_data.time_interval)

    for t in range(n_samples):
        year, month, day = activate_power_data.random_date()


        for _ in range(episode_length):
            current_time = 0
            net = net_template.deepcopy()

            # ---- 当日负荷 & PV 数据 ----
            ap = 0.75 * activate_power_data.select_timeslot_data(year, month, day, current_time).astype(float)
            rp = 0.75 * reactivate_power_data.select_timeslot_data(year, month, day, current_time).astype(float)
            pv = 2 * pv_power_data.select_timeslot_data(year, month, day, current_time).astype(float)

            # 修改负荷
            # ——————— 2. 更新负荷 p/q ——————— #
            for i, row in net.load.iterrows():
                net.load.at[i, 'p_mw'] = ap[i]
                net.load.at[i, 'q_mvar'] = rp[i]

            # ---- 更新 PV ----
            for i, bus in enumerate(env_config_Bowl["network_info"]['pv_buses']):
                if i < len(net.sgen):
                    net.sgen.at[i, 'p_mw'] = pv[i]
                    net.sgen.at[i, 'q_mvar'] = np.random.uniform(-0.05, 0.05)

            try:
                pp.runpp(net, init="auto", numba=False)
                print(f"✅ 第{t*episode_length + _ + 1} 次潮流计算成功")

                # 计算节点净注入功率（发电为正，负荷为负）
                p = np.zeros(len(net.bus))
                q = np.zeros(len(net.bus))

                # 加入负荷（为负注入）
                for _, load in net.load.iterrows():
                    p[load.bus] -= load.p_mw
                    q[load.bus] -= load.q_mvar

                # 加入光伏sgen（为正注入）
                for _, sgen in net.sgen.iterrows():
                    p[sgen.bus] += sgen.p_mw
                    q[sgen.bus] += sgen.q_mvar

                # （可选）加入shunt
                for _, shunt in net.shunt.iterrows():
                    q[shunt.bus] += shunt.q_mvar

                v = net.res_bus.vm_pu.values
                inputs.append(np.concatenate([p, q]))
                outputs.append(v)
                current_time += 1
            except Exception as e:
                print(f"❌ 第{i + 1} 次 runpp 失败: {e}")
                failures += 1
                continue
    print(f"\n✅ 最终：成功 {len(inputs)} / {n_samples}，失败 {failures} 次")
    return np.array(inputs), np.array(outputs)


X_raw, y_raw = generate_samples(net, n_samples=20)
print("样本维度：", X_raw.shape, y_raw.shape)

# ==== 3. 数据预处理 ====
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# ==== 4. 划分训练/测试集 ====
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==== 5. PyTorch 数据准备 ====
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# ==== 6. 构建 surrogate model ====
class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


model = SurrogateModel(input_dim=X.shape[1], hidden_dims=[128, 64], output_dim=y.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 7. 训练模型 ====
epochs = 1000
loss_list = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# ==== 7.1 Save训练模型 ====
torch.save(model.state_dict(), "surrogate_model_IEEE123.pth")
print("✅ 模型已保存为 surrogate_model_IEEE123.pth")

# # ==== 7.2 保存 scaler ====
# import joblib
# joblib.dump(X_scaler, "X_scaler.save")
# joblib.dump(y_scaler, "y_scaler.save")
# print("✅ 归一化器已保存")

'''
# # 加载模型时使用代码
# model = SurrogateModel(input_dim=66, hidden_dims=[128, 64], output_dim=33)
# model.load_state_dict(torch.load("surrogate_model.pth"))
# model.eval()
# 
# # 加载 scaler
# X_scaler = joblib.load("X_scaler.save")
# y_scaler = joblib.load("y_scaler.save")
'''

# ==== 8. 测试与评估 ====
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()
    y_pred_test_inv = y_scaler.inverse_transform(y_pred_test)
    y_test_inv = y_scaler.inverse_transform(y_test)

mae = np.mean(np.abs(y_pred_test_inv - y_test_inv))
print("Test MAE:", mae)

# ==== 9. 可视化 ====
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()

# ==== 10. 查看预测效果（选取一个样本）====
i = 10
plt.plot(y_test_inv[i], label="True")
plt.plot(y_pred_test_inv[i], label="Predicted")
plt.legend()
plt.title("Voltage Profile Prediction")
plt.xlabel("Bus Index")
plt.ylabel("Voltage (p.u.)")
plt.grid()
plt.show()

# ==== 10. 保存 Loss 和预测结果 ====
# 保存 loss 曲线
pd.DataFrame(loss_list, columns=["MSE Loss"]).to_csv("loss_curve.csv", index_label="Epoch")
print("✅ 已保存 loss 曲线到 loss_curve.csv")

df_all = pd.DataFrame(
    np.hstack([y_test_inv, y_pred_test_inv]),
    columns=[f"True_{i}" for i in range(y_test_inv.shape[1])] + [f"Pred_{i}" for i in range(y_pred_test_inv.shape[1])]
)
df_all.to_csv("all_test_predictions.csv", index=False)
print("✅ 已保存全部预测结果到 all_test_predictions.csv")

# ==== 12. 计算误差并保存误差数据 ====
# 计算每个节点的误差（绝对误差）
errors = y_pred_test_inv - y_test_inv

# 保存所有样本所有节点的误差展开成一列（便于画误差直方图）
error_flat = errors.flatten()
pd.DataFrame(error_flat, columns=["Absolute Error"]).to_csv("error_histogram_data.csv", index=False)
print("✅ 已保存误差直方图数据到 error_histogram_data.csv")
