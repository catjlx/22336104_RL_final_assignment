#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

from vmas import make_env
from vmas.simulator.environment import Wrapper


scenario_name = "balance"

# Scenario specific variables.
n_agents = 3

# Common variables
continuous_actions = True
max_steps = 200
num_vectorized_envs = 96
num_workers = 5
vmas_device = "cpu"  # or cuda

MARL_ALGO = "IPPO"
# MARL_ALGO = "CPPO"
# MARL_ALGO = "MAPPO"

# VMAS + Wrapper.RLLIB只能使用单 policy PPO
# 三种算法区别仅体现在critic设计

print(f"Using MARL algorithm: {MARL_ALGO}")


# 根据配置初始化场景，返回 RLlib 兼容的向量化环境
def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env


# 初始化 Ray
if not ray.is_initialized():
    ray.init()
    print("Ray init!")

register_env(scenario_name, lambda config: env_creator(config))

# 回调：记录评估指标
class EvaluationCallbacks(DefaultCallbacks):
    # 记录每个智能体的额外信息，用于训练监控
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                episode.user_data.setdefault(
                    f"{a_key}/{b_key}", []
                ).append(info[a_key][b_key])

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()

# 渲染视频
class RenderingCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

    def on_episode_end(self, *, worker, base_env, episode, **kwargs):
        vid = np.transpose(self.frames, (0, 3, 1, 2))
        episode.media["rendering"] = wandb.Video(
            vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
        )
        self.frames = []


def train():
    RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0
    num_gpus_per_worker = 0

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 400},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,

        config={
            "framework": "torch",
            "env": scenario_name,

            # PPO 超参数
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.9,
            "clip_param": 0.2,
            "entropy_coeff": 0.0,
            "vf_loss_coeff": 1.0,
            "vf_clip_param": float("inf"),
            "kl_coeff": 0.01,
            "kl_target": 0.01,

            # batch
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,

            # resources
            "num_workers": num_workers,
            "num_envs_per_worker": num_vectorized_envs,
            "num_gpus": num_gpus,
            "num_gpus_per_worker": 0,

            # env
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },

            "callbacks": EvaluationCallbacks,
            
            # evaluation（不渲染）
            "evaluation_interval": 5,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {"num_envs": 1},
                "callbacks": EvaluationCallbacks,
            },
        },
    )


if __name__ == "__main__":
    train()
