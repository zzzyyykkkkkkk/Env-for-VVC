import os
import pandas as pd
from env.Safeenv512 import Safe_PowerNetEnv_IEEE123
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from matplotlib import rcParams
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class CustomCallback(BaseCallback):
    def __init__(self, reward_save_path, metrics_save_path, verbose=0, total_timesteps=None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_steps = []
        self.cumulative_steps = []
        self.current_cumulative_steps = 0
        self.reward_save_path = reward_save_path
        self.metrics_save_path = metrics_save_path
        self.buf = []  # 存储 [step, loss, dv, soc_mean, soc_min, soc_max, action_diff_norm]

        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        if self.total_timesteps is None:
            print("⚠️  未指定总步数，tqdm 进度条无法正确显示")
        else:
            self.pbar = tqdm(total=self.total_timesteps, desc="训练进度", dynamic_ncols=True)

    def _on_step(self) -> bool:
        # 更新进度条
        if self.pbar:
            self.pbar.update(1)

        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                reward = ep_info.get('r', 0)
                steps = ep_info.get('l', 0)
                self.episode_rewards.append(reward)
                self.episode_steps.append(steps)
                self.current_cumulative_steps += steps
                self.cumulative_steps.append(self.current_cumulative_steps)

                loss = info.get("loss_mw")
                dv = info.get("dv_max")
                soc_m = info.get("soc_mean")
                soc_lo = info.get("soc_min")
                soc_hi = info.get("soc_max")
                action_diff_norm = info.get("action_diff_norm")

                # 记录到 TensorBoard
                self.logger.record('custom/episode_reward', reward)
                self.logger.record('custom/episode_steps', steps)
                self.logger.record('custom/cumulative_steps', self.current_cumulative_steps)
                if None not in (loss, dv, soc_m, action_diff_norm):
                    self.logger.record("env/loss_mw", loss)
                    self.logger.record("env/dv_max", dv)
                    self.logger.record("env/soc_mean", soc_m)
                    self.logger.record("env/action_diff_norm", action_diff_norm)
                    # 将行数据写入 buf
                    self.buf.append([
                        self.num_timesteps,
                        loss,
                        dv,
                        soc_m,
                        soc_lo,
                        soc_hi,
                        action_diff_norm
                    ])

        return True

    def _on_training_end(self):
        # 关闭进度条
        if self.pbar:
            self.pbar.close()

        # 保存 episode 数据
        if self.episode_rewards:
            reward_df = pd.DataFrame({
                'episode': range(1, len(self.episode_rewards) + 1),
                'reward': self.episode_rewards,
                'steps': self.episode_steps,
                'cumulative_steps': self.cumulative_steps
            })
            reward_df.to_csv(self.reward_save_path, index=False)
            if self.verbose > 0:
                print(f"🎉 Episode 数据已保存至: {self.reward_save_path}")
        else:
            print("❗ 没有记录到任何 episode 信息。")

        # 保存指标数据（注意这里有 7 列）
        if self.buf:
            metrics_df = pd.DataFrame(
                self.buf,
                columns=[
                    "step",
                    "loss_mw",
                    "dv_max",
                    "soc_mean",
                    "soc_min",
                    "soc_max",
                    "action_diff_norm"  # 新增第七列
                ]
            )
            metrics_df.to_csv(self.metrics_save_path, index=False)
            if self.verbose > 0:
                print(f"📈 指标数据已保存至: {self.metrics_save_path}")


def train_model(env, hyperparams, model_name, save_dir, tensorboard_log):
    os.makedirs(save_dir, exist_ok=True)
    reward_cb = CustomCallback(
        reward_save_path=os.path.join(save_dir, f"{model_name}_episode.csv"),
        metrics_save_path=os.path.join(save_dir, f"{model_name}_metrics.csv"),
        verbose=1,
        total_timesteps=100000
    )

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        **hyperparams
    )

    model.learn(total_timesteps=100000, callback=reward_cb)
    model.save(os.path.join(save_dir, model_name))

    return pd.read_csv(reward_cb.reward_save_path), pd.read_csv(reward_cb.metrics_save_path)


def main(env, save_path):
    os.makedirs(save_path, exist_ok=True)
    env = Monitor(env, info_keywords=("loss_mw", "dv_max", "soc_mean", "soc_min", "soc_max", "action_diff_norm"))

    hyperparams1 = {
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.41434940445298885,
        'learning_rate': 0.0021350783533327453,
        'gae_lambda': 0.904031187822231,
        'clip_range': 0.12934548234029952,
        'ent_coef': 0.010917253270041623,
    }
    hyperparams2 = {}

    print("训练模型1...")
    train_model(env, hyperparams1, 'ppo_model1', save_path, os.path.join(save_path, 'tensorboard_model1'))
    print("训练模型2...")
    train_model(env, hyperparams2, 'ppo_model2', save_path, os.path.join(save_path, 'tensorboard_model2'))
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
            'pv_q_mvar': [0.10,0.14,0.12,0.10,0.14,0.12,0.10,0.14,0.12,0.10,0.10,0.14,0.12,0.10,0.14],
            'svc_buses': [31, 50, 74, 110],
            'q_mvar': [0.15, 0.15, 0.15, 0.15],
        },
        "activate_power_data_path": "../data/active_power-data/IEEE123/load_active.csv",
        "reactivate_power_data_path": "../data/reactive_power-data/IEEE123/load_reactive.csv",
        "pv_data_path": "../data/pv-data/IEEE123/pv_active.csv"
    }
    # 创建环境
    env_Bowl = Safe_PowerNetEnv_IEEE123(env_config_Bowl)
    main(env=env_Bowl, save_path="../安全强化学习数据_IEEE123/Bowl_5.13")
