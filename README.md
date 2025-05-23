
# Safe Deep Reinforcement Learning Environment for Voltage Control in Distribution Networks

A Python toolkit built on OpenAI Gym for **active voltage regulation** in the IEEE‑123 distribution feeder.  
Agents learn to coordinate PV inverters, static var compensators (SVCs) and battery storage so that all bus voltages stay within limits while network losses are minimised.

The Safeenv512.py module defines a custom Gym environment named Safe_PowerNetEnv_IEEE123, designed for safe reinforcement learning in reactive power and voltage control within the IEEE-123 bus distribution network. This environment integrates photovoltaic (PV) systems, static VAR compensators (SVCs), and battery energy storage systems (BESS), with additional emphasis on safety constraints and optimization-based action correction mechanisms.

---

## ✨ Key Features

| Category | Highlights |
|----------|------------|
| **Gym‑compatible environments** | `PowerNetEnv_IEEE123` (standard) & `Safe_PowerNetEnv_IEEE123` (with convex‑optimisation safety‑layer). |
| **Data management** | `GeneralPowerDataManager` loads / cleans multi‑year load & PV CSVs, interpolates to 3‑min resolution and delivers training batches. |
| **Pandapower network builder** | `create_pandapower_net` constructs the IEEE‑123 feeder, injects branch parameters and attaches PV, SVC and battery devices. |
| **Training callbacks** | Custom Stable‑Baselines3 callbacks stream episode reward, volt‑dev, losses & SOC to CSV + TensorBoard. |

---

## 🛠️ Installation

1. Install [Anaconda](https://www.anaconda.com/products/individual#Downloads).
2. After cloning or downloading this repository, assure that the current directory is `[your own parent path]/Env-for-VVC`.
3. on Windows OS, please execute the following command. **Note that please launch the Anaconda shell by the permission of Administration.**
   ```bash
   conda env create -f environment.yml
   ```
4. Activate the installed virtual environment using the following command.
    ```bash
    conda activate VVC_Env
    ```


### Core Python packages

* `gymnasium`
* `stable-baselines3`
* `pandapower`
* `torch`
* `numpy`, `pandas`, `cvxpy`
* `matplotlib`, `tqdm`, `scikit-learn`

### Prepare data

Place CSVs under:

```
data/
├─ active_power-data/IEEE123/
├─ reactive_power-data/IEEE123/
└─ pv-data/IEEE123/
```

(The data download link is：[LINK](https://drive.google.com/file/d/1-q9oLUa-WNJggB0y5h_SorzMKEeUT8rG/view?usp=drive_link).  Modify paths in `env_config` if required.)

---

## 📁 Project Structure

```
.
├── data.py                    # Data Storage(You need to download and place it in this location)
├── data_manager.py            # time‑series loader / pre‑processor
├── pandapower_net.py          # IEEE‑123 builder
├── env/
│   ├── env422.py              # standard DRL Gym env   
│   └── Safeenv512.py          # env with safety layer
├── train.py                   # PPO training example
├── test.py                    # evaluation script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Test environment

```python
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
```


---

## ⚙️ Configuration

All environment parameters (battery buses, PV/SVC limits, data paths, reward design) live in the `env_config` dict at the top of each env file.
Adapt these to different feeders or hardware inverters.

---


---

## 📄 License

Distributed under the **MIT License**.
See [`LICENSE`](LICENSE) for full text.

---

## 🙏 Acknowledgements

* [Pandapower](https://www.pandapower.org/) for distribution‑system modelling
* [Stable‑Baselines3](https://github.com/DLR-RM/stable-baselines3) for DRL algorithms
* Inspired by [RL-ADN](https://github.com/ShengrenHou/RL-ADN) for reinforcement learning in active distribution networks
* Built upon concepts from [MAPDN](https://github.com/Future-Power-Networks/MAPDN) for multi-agent ADN control frameworks
* This work also uses a lot of AI help. For example, the README of this article is generated by AI.
* I have learned a lot of useful knowledge from other people's open source projects, so I thought about making some of my own projects open source to help more people and also help my past self.

Happy voltage‑controlling! ⚡

```
```
