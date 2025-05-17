from env.env422 import PowerNetEnv_IEEE123
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rcParams



# 设置字体为 SimHei（黑体）以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

class RewardStepCallback(BaseCallback):
    """
    自定义回调，用于记录每个 episode 的奖励、步数和累积步数，并在训练完成后保存到 CSV 文件。
    同时，将奖励和步数记录到 TensorBoard。
    """

    def __init__(self, save_path, verbose=0):
        super(RewardStepCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_steps = []
        self.cumulative_steps = []
        self.current_cumulative_steps = 0
        self.save_path = save_path

    def _on_step(self) -> bool:
        # 从 infos 中提取 episode 信息
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
                if self.verbose > 0:
                    print(
                        f"捕获到 Episode {len(self.episode_rewards)}: 奖励={reward}, 步数={steps}, 累积步数={self.current_cumulative_steps}")

                # 记录到 TensorBoard
                self.logger.record('custom/episode_reward', reward)
                self.logger.record('custom/episode_steps', steps)
                self.logger.record('custom/cumulative_steps', self.current_cumulative_steps)
        return True

    def _on_training_end(self):
        """
        训练结束时调用，将记录的数据保存到 CSV 文件中。
        """
        if len(self.episode_rewards) == 0:
            print("没有记录到任何 episode 信息。请检查环境是否正确包装为 Monitor，并确保训练过程中有完成的 episodes。")
            return

        # 创建 DataFrame
        df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'steps': self.episode_steps,
            'cumulative_steps': self.cumulative_steps
        })

        # 保存到 CSV 文件
        df.to_csv(self.save_path, index=False)

        if self.verbose > 0:
            print(f"Episode 数据已保存至: {self.save_path}")


class MetricsCallback(BaseCallback):
    """
    记录 loss_mw / dv_max / soc_mean 等指标到
    1) TensorBoard
    2) 本地 CSV/SVC
    """
    def __init__(self, csv_path: str, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.buf = []            # list of [step, loss, dv, soc_m, soc_min, soc_max]

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]          # 单环境 -> 只看第 0 个
        loss   = info.get("loss_mw")
        dv     = info.get("dv_max")
        soc_m  = info.get("soc_mean")
        soc_lo = info.get("soc_min")
        soc_hi = info.get("soc_max")

        # ---- 1. 写 TensorBoard ----
        if loss is not None:
            self.logger.record("env/loss_mw", loss)
            self.logger.record("env/dv_max",  dv)
            self.logger.record("env/soc_mean", soc_m)

        # ---- 2. 暂存，之后写 CSV/SVC ----
        if None not in (loss, dv, soc_m):
            self.buf.append([
                self.num_timesteps, loss, dv, soc_m, soc_lo, soc_hi
            ])
        return True

    def _on_training_end(self):
        if self.buf:
            import pandas as pd
            df = pd.DataFrame(
                self.buf,
                columns=["step", "loss_mw", "dv_max",
                         "soc_mean", "soc_min", "soc_max"]
            )
            # 你提到“SVC 文件”，这里默认用 .svc 扩展名，
            # 但内部仍是普通 CSV 格式
            df.to_csv(self.csv_path, index=False)
            if self.verbose:
                print(f"[MetricsCallback] 指标已保存 -> {self.csv_path}")

def train_model(env, hyperparams, model_name, save_dir, tensorboard_log):
    model_path = os.path.join(save_dir, model_name)
    # ----- 回调 1：Episode reward/steps -----
    reward_cb = RewardStepCallback(
        save_path=os.path.join(save_dir, f"{model_name}_episode.csv"),
        verbose=0
    )
    # ----- 回调 2：我们的指标 -----
    metrics_cb = MetricsCallback(
        csv_path=os.path.join(save_dir, f"{model_name}_metrics.csv"),
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        **hyperparams
    )

    model.learn(
        total_timesteps=480_000,
        callback=[reward_cb, metrics_cb]   # <<<< 传入 list
    )
    model.save(model_path)
    return (
        pd.read_csv(reward_cb.save_path),    # df1
        pd.read_csv(metrics_cb.csv_path)     # df2
    )



def moving_average(series, window=100):
    """
    计算数据的移动平均，以平滑曲线。

    参数：
        series: Pandas Series
        window: 窗口大小
    返回：
        移动平均后的 Pandas Series
    """
    return series.rolling(window=window).mean()


def plot_metrics(df1, df2, save_dir):
    """
    绘制并保存奖励和步数对比图。

    参数：
        df1: 第一个模型的 DataFrame
        df2: 第二个模型的 DataFrame
        save_dir: 保存图表的目录
    """
    window = 100  # 移动平均窗口大小

    # 计算移动平均
    df1['reward_ma'] = moving_average(df1['reward'], window)
    df2['reward_ma'] = moving_average(df2['reward'], window)
    df1['cumulative_steps_ma'] = moving_average(df1['cumulative_steps'], window)
    df2['cumulative_steps_ma'] = moving_average(df2['cumulative_steps'], window)

    df1['steps_ma'] = moving_average(df1['steps'], window)
    df2['steps_ma'] = moving_average(df2['steps'], window)

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df1['cumulative_steps_ma'], df1['reward_ma'], label='model_1 - PPO-Bowl', color='blue')
    plt.plot(df2['cumulative_steps_ma'], df2['reward_ma'], label='model_2 - AutoRL-PPO-Bowl', color='red')
    plt.xlabel('Training steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.grid(True)
    # 保存奖励对比图
    reward_plot_path = os.path.join(save_dir, 'reward_comparison.png')
    plt.savefig(reward_plot_path)
    print(f"奖励对比图已保存至: {reward_plot_path}")
    plt.close()

    # 绘制步数曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df1['cumulative_steps_ma'], df1['steps_ma'], label='model_1 - Average number of steps per episode',
             color='green')
    plt.plot(df2['cumulative_steps_ma'], df2['steps_ma'], label='model_2 - Average number of steps per episode',
             color='red')
    plt.xlabel('Training steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.grid(True)
    # 保存步数对比图
    steps_plot_path = os.path.join(save_dir, 'steps_comparison.png')
    plt.savefig(steps_plot_path)
    print(f"步数对比图已保存至: {steps_plot_path}")
    plt.close()


def main(env, save_path):
    os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）
    env = Monitor(env,info_keywords=("loss_mw",
                                     "dv_max",
                                     "soc_mean",
                                     "soc_min",
                                     "soc_max")
                  )  # 使用 Monitor 包装环境

    # 超参数组1
    hyperparams1 = {
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.7704782551401501,
        'learning_rate': 0.0007716667865490209,
        'gae_lambda': 0.9206651873783704,
        'clip_range': 0.02972906412802212,
        'ent_coef': 0.09060142380113331,
    }

    # 超参数组2
    hyperparams2 = {
        'batch_size': 256,
        'n_epochs': 7,
        'gamma': 0.6909697435938972,
        'learning_rate': 0.002130902642914204,
        'gae_lambda': 0.9161022097962338,
        'clip_range': 0.28258823625451196,
        'ent_coef': 0.014910642765512067,
    }

    # 指定 TensorBoard 日志目录
    tensorboard_log_dir1 = os.path.join(save_path, 'tensorboard_model1')
    tensorboard_log_dir2 = os.path.join(save_path, 'tensorboard_model2')

    # 训练第一个模型
    print("训练模型1...")
    df1 = train_model(
        env,
        hyperparams1,
        'ppo_model1',
        save_path,
        tensorboard_log=tensorboard_log_dir1
    )

    # 训练第二个模型
    print("训练模型2...")
    df2 = train_model(
        env,
        hyperparams2,
        'ppo_model2',
        save_path,
        tensorboard_log=tensorboard_log_dir2
    )

    # 绘制并保存对比图
    plot_metrics(df1, df2, save_path)

    print("训练和绘图完成。")


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
    main(env=env_Bowl, save_path="C:\\Users\\zhaoyukang\\Desktop\\DNN for PF\\DRL_env_VVC\\数据保存-IEEE123\\Bowl_4.23")
