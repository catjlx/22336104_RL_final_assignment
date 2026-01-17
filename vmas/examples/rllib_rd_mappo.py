#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Dict

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune import register_env

from vmas import make_env
from vmas.simulator.environment import Wrapper


scenario_name = "balance"
n_agents = 3
continuous_actions = True
max_steps = 200
num_vectorized_envs = 96
num_workers = 5
vmas_device = "cpu"  # or "cuda"


# RD-MAPPO 超参数
REWARD_ALPHA = 0.8
print("Using algorithm: RD-MAPPO (Reward-Decomposed MAPPO)")
print(f"Reward alpha = {REWARD_ALPHA}")


def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        **config["scenario_config"],
    )
    return env


if not ray.is_initialized():
    ray.init()
    print("Ray init!")

register_env(scenario_name, lambda config: env_creator(config))



# 回调：Reward Decomposition + 记录自定义指标
class RewardDecompositionCallbacks(DefaultCallbacks):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        infos = episode.last_info_for() or {}
        # 获取 agent 列表
        agent_ids = infos.keys() if infos else []

        new_rewards = {}
        for agent_id in agent_ids:
            r_global = episode.last_reward_for(agent_id) 
            info = infos.get(agent_id, {})
            local_reward = -abs(info["distance_to_center"]) if "distance_to_center" in info else 0.0
            r_new = self.alpha * r_global + (1 - self.alpha) * local_reward
            new_rewards[agent_id] = r_new

            # 保存 info
            for k, v in info.items():
                episode.user_data.setdefault(f"{agent_id}/{k}", []).append(v)

        # 替换原始奖励
        episode._last_raw_rewards = new_rewards


    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        for agent_key, data in episode.user_data.items():
            metric = np.array(data)
            episode.custom_metrics[agent_key] = np.sum(metric).item()



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
            "num_gpus_per_worker": num_gpus_per_worker,

            # 环境配置
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                "scenario_config": {"n_agents": n_agents},
                "reward_alpha": REWARD_ALPHA,
            },

            # 回调
            "callbacks": lambda: RewardDecompositionCallbacks(alpha=REWARD_ALPHA),

            # evaluation
            "evaluation_interval": 5,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {"num_envs": 1},
                "callbacks": lambda: RewardDecompositionCallbacks(alpha=REWARD_ALPHA),
            },
        },
    )


if __name__ == "__main__":
    train()
