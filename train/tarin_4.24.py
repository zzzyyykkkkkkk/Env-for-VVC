import os
import pandas as pd
from env.env422 import PowerNetEnv_IEEE123
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from matplotlib import rcParams


# 设置字体为 SimHei（黑体）以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号


class CustomCallback(BaseCallback):
    """
    合并了 RewardStepCallback 和 MetricsCallback 的功能：
    记录每个 episode 的奖励、步数和累积步数，同时记录网损、最大电压偏差和电池SOC等指标。
    将这些数据记录到 TensorBoard 和 CSV 文件。
    """
    def __init__(self, reward_save_path, metrics_save_path, verbose=0):
        super().__init__(verbose)
        # 初始化数据列表
        self.episode_rewards = []             # 存储每个 episode 的奖励
        self.episode_steps = []               # 存储每个 episode 的步数
        self.cumulative_steps = []           # 存储累积的步数
        self.current_cumulative_steps = 0    # 当前的累积步数

        # 数据保存路径
        self.reward_save_path = reward_save_path     # 奖励和步数保存的路径
        self.metrics_save_path = metrics_save_path   # 指标（网损、最大电压偏差等）保存的路径
        self.buf = []  # 存储每个步骤的指标数据

    def _on_step(self) -> bool:
        """
        在每一步记录 episode 信息，更新奖励、步数、累积步数和相关指标
        """
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                reward = ep_info.get('r', 0)  # 获取奖励
                steps = ep_info.get('l', 0)   # 获取步数
                self.episode_rewards.append(reward)
                self.episode_steps.append(steps)
                self.current_cumulative_steps += steps
                self.cumulative_steps.append(self.current_cumulative_steps)

                # 获取其他指标
                loss = info.get("loss_mw")
                dv = info.get("dv_max")
                soc_m = info.get("soc_mean")
                soc_lo = info.get("soc_min")
                soc_hi = info.get("soc_max")

                # ---- 记录到 TensorBoard ----
                self.logger.record('custom/episode_reward', reward)
                self.logger.record('custom/episode_steps', steps)
                self.logger.record('custom/cumulative_steps', self.current_cumulative_steps)

                if loss is not None:
                    self.logger.record("env/loss_mw", loss)
                    self.logger.record("env/dv_max", dv)
                    self.logger.record("env/soc_mean", soc_m)

                # ---- 记录指标数据到 buf ----
                if None not in (loss, dv, soc_m):
                    self.buf.append([self.num_timesteps, loss, dv, soc_m, soc_lo, soc_hi])

        return True

    def _on_training_end(self):
        """
        训练结束时，保存数据到 CSV 文件中
        """
        # 保存奖励数据到 CSV
        if len(self.episode_rewards) > 0:
            reward_df = pd.DataFrame({
                'episode': range(1, len(self.episode_rewards) + 1),
                'reward': self.episode_rewards,
                'steps': self.episode_steps,
                'cumulative_steps': self.cumulative_steps
            })
            reward_df.to_csv(self.reward_save_path, index=False)
            if self.verbose > 0:
                print(f"Episode 数据已保存至: {self.reward_save_path}")
        else:
            print("没有记录到任何 episode 信息。")

        # 保存指标数据到 CSV
        if len(self.buf) > 0:
            metrics_df = pd.DataFrame(
                self.buf,
                columns=["step", "loss_mw", "dv_max", "soc_mean", "soc_min", "soc_max"]
            )
            metrics_df.to_csv(self.metrics_save_path, index=False)
            if self.verbose > 0:
                print(f"指标数据已保存至: {self.metrics_save_path}")


def train_model(env, hyperparams, model_name, save_dir, tensorboard_log):
    model_path = os.path.join(save_dir, model_name)
    reward_cb = CustomCallback(
        reward_save_path=os.path.join(save_dir, f"{model_name}_episode.csv"),
        metrics_save_path=os.path.join(save_dir, f"{model_name}_metrics.csv"),
        verbose=0
    )

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        **hyperparams
    )

    model.learn(total_timesteps=120000, callback=reward_cb)
    model.save(model_path)

    return pd.read_csv(reward_cb.reward_save_path), pd.read_csv(reward_cb.metrics_save_path)


def main(env, save_path):
    os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）
    env = Monitor(env, info_keywords=("loss_mw", "dv_max", "soc_mean", "soc_min", "soc_max"))

    hyperparams1 = {
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.7704782551401501,
        'learning_rate': 0.0007716667865490209,
        'gae_lambda': 0.9206651873783704,
        'clip_range': 0.02972906412802212,
        'ent_coef': 0.09060142380113331,
    }

    hyperparams2 = {
        'batch_size': 256,
        'n_epochs': 7,
        'gamma': 0.6909697435938972,
        'learning_rate': 0.002130902642914204,
        'gae_lambda': 0.9161022097962338,
        'clip_range': 0.28258823625451196,
        'ent_coef': 0.014910642765512067,
    }

    tensorboard_log_dir1 = os.path.join(save_path, 'tensorboard_model1')
    tensorboard_log_dir2 = os.path.join(save_path, 'tensorboard_model2')

    print("训练模型1...")
    df1 = train_model(env, hyperparams1, 'ppo_model1', save_path, tensorboard_log_dir1)

    print("训练模型2...")
    df2 = train_model(env, hyperparams2, 'ppo_model2', save_path, tensorboard_log_dir2)

    print("训练完成。")


if __name__ == "__main__":
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
    # 创建环境
    env_Bowl = PowerNetEnv_IEEE123(env_config_Bowl)
    main(env=env_Bowl, save_path="../数据保存-IEEE123/Bowl_4.24_2")
