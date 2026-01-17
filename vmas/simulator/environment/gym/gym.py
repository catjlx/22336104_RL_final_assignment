#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

import gym
import numpy as np

from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.gym.base import BaseGymWrapper


class GymWrapper(gym.Env, BaseGymWrapper):
    # VMAS环境的Gym接口包装器
    # 将VMAS的向量化环境适配为标准OpenAI Gym非向量化接口，便于集成现有单环境RL算法
    metadata = Environment.metadata # 继承VMAS环境的元数据

    def __init__(
        self,
        env: Environment, # 原始VMAS环境
        return_numpy: bool = True,
    ):
        super().__init__(env, return_numpy=return_numpy, vectorized=False)
        # 断言VMAS环境为单环境模式，因为Gym接口不支持向量化，论文中提到VMAS支持批量但是Gym适配需限制为1个环境
        assert (
            env.num_envs == 1
        ), f"GymEnv wrapper is not vectorised, got env.num_envs: {env.num_envs}"
        # 断言禁用终止/截断标志
        assert (
            not self._env.terminated_truncated
        ), "GymWrapper is not compatible with termination and truncation flags. Please set `terminated_truncated=False` in the VMAS environment."
        # 继承VMAS环境的观测空间和动作空间
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def unwrapped(self) -> Environment:
        # 返回原始VMAS环境，用于保持环境灵活性，允许用户访问VMAS的向量化特性和自定义功能
        return self._env

    def step(self, action): # Gym标准step接口（单环境动作执行）
        # 将Gym动作（numpy/列表）转换为VMAS所需的Tensor格式
        action = self._action_list_to_tensor(action)
        # 调用VMAS环境的step方法
        obs, rews, done, info = self._env.step(action)
        # 转换VMAS输出数据为Gym格式 Tensor→numpy、批量维度压缩
        env_data = self._convert_env_data(
            obs=obs, # 观测（numpy数组，符合Gym标准）
            rews=rews, # 奖励（标量，符合Gym标准）
            info=info, # 终止标志（布尔值）
            done=done, # 额外信息（字典）
        )
        return env_data.obs, env_data.rews, env_data.done, env_data.info

    def reset(
        # Gym标准render接口
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset_at(index=0)
        env_data = self._convert_env_data(obs=obs)
        return env_data.obs

    def render(
        self,
        mode="human",
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            mode=mode,
            env_index=0, # 渲染第0个环境（单环境模式）
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
