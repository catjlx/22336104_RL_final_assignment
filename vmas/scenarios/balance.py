#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # 创建世界（3.2节Scenario核心函数）
        # 功能：初始化智能体、地标、物理参数并定义场景规则
        # batch_dim: 并行环境数量 device：计算设备 kwargs：场景参数 return：初始化后的向量化世界
        self.n_agents = kwargs.pop("n_agents", 3) # 协作智能体数量，需要大于1，依赖团队协作
        self.package_mass = kwargs.pop("package_mass", 5) # 包裹质量，单个智能体无法移动，需要激励协作
        self.random_package_pos_on_line = kwargs.pop("random_package_pos_on_line", True) # 包裹是否随机放置在线段上
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert self.n_agents > 1
        # 实体尺寸参数
        self.line_length = 0.8 # 可旋转线段长度
        self.agent_radius = 0.03 # 球体半径
        # 奖励函数参数
        self.shaping_factor = 100 # 全局塑形奖励系数，用于提升样本效率
        self.fall_reward = -10 # 落地惩罚（线段 / 包裹接触地面时触发）

        self.visualize_semidims = False

        # Make world 创建向量化世界，gravity=(0.0, -0.05)定义了3.3节中垂直向下的重力，y_semidim=1即Y轴边界
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # Add agents 
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7, # 动作力放大系数
            )
            world.add_agent(agent)
        # 添加目标地标，即包裹需要抵达的位置
        goal = Landmark(
            name="goal",
            collide=False, # 不可碰撞，即仅作为目标点
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        # 添加核心实体：需要可移动、可碰撞、质量高到可协作推动
        self.package = Landmark(
            name="package",
            collide=True, # 可碰撞
            movable=True, # 可移动
            shape=Sphere(), 
            mass=self.package_mass, # 质量高，足以协作推动
            color=Color.RED,
        )
        self.package.goal = goal # 绑定目标，用于计算奖励
        world.add_landmark(self.package)
        # Add landmarks 
        # 添加可旋转线段：承载包裹，需智能体协作控制旋转
        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True, # 支持碰撞，可与智能体、包裹、地面交互
            movable=True, # 可移动
            rotatable=True, # 可旋转
            mass=5, # 线段的质量，需要多智能体协作推动，更像一根棍子
            color=Color.BLACK,
        )
        world.add_landmark(self.line)
        # 添加地面：防止实体坠落，落地触发惩罚
        self.floor = Landmark(
            name="floor",
            collide=True, # 支持碰撞，用于检测线段和包裹落地
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)
        # 向量化奖励存储
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32) # 位置奖励
        self.ground_rew = self.pos_rew.clone() # 落地惩罚

        return world

    def reset_world_at(self, env_index: int = None):
        # 重置指定批量环境或全量环境（论文3.2节核心函数）
        # 功能：随机生成目标、线段、包裹位置，初始化实体状态
        # env_index: 需重置的环境索引
        # 目标位置随机生成
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        # 线段初始位置，位于环境底部、智能体上方
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ), # X 轴避免线段超出边界
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ), # Y 轴在环境底部，智能体上方
            ],
            dim=1,
        )
        # 包裹相对线段的初始位置
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    # 随机 X 坐标（避免包裹超出线段）
                    (
                        -self.line_length / 2 + self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                    (
                        self.line_length / 2 - self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        # 智能体初始位置
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [ # 智能体沿线段X轴均匀分布
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        # 重置线段位置
        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        # 重置目标位置
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        # 重置旋转角（初始无旋转）
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        # 重置包裹位置
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )
        # 重置地面位置
        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        # 检测初始是否落地
        self.compute_on_the_ground()
        # 基于包裹与目标的初始距离初始化全局塑形奖励
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def compute_on_the_ground(self):
        # 用于检测线段或包裹是否落地
        # 依赖VMAS内置的向量化碰撞检测
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def reward(self, agent: Agent):
        # 论文3.2节核心函数：计算每个智能体的奖励
        # 奖励逻辑：包裹靠近目标奖励为+，落地为-
        # agent为当前计算奖励的智能体，但是仅首个智能体执行计算，其余共享结果即可
        # 返回批量环境中每个环境的将离职
        is_first = agent == self.world.agents[0] # 仅首个智能体执行计算（避免重复）

        if is_first:
            self.pos_rew[:] = 0 # 重置位置奖励
            self.ground_rew[:] = 0 # 重置落地惩罚

            self.compute_on_the_ground() # 检测是否落地
            # 计算包裹与目标的距离
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            # 落地惩罚
            self.ground_rew[self.on_the_ground] = self.fall_reward
            # 位置奖励：全局塑形当前距离与初始距离的差值，激励包裹靠近目标
            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

        return self.ground_rew + self.pos_rew # 总奖励 = 落地惩罚 + 位置奖励

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        # 论文 3.2 节核心函数：计算智能体的观测
        # agent：当前智能体
        # 返回批量环境中每个环境的观测值
        return torch.cat(
            [
                agent.state.pos, # 智能体自身位置
                agent.state.vel, # 智能体自身速度
                agent.state.pos - self.package.state.pos, # 智能体与包裹的相对位置
                agent.state.pos - self.line.state.pos, # 智能体与线段的相对位置
                self.package.state.pos - self.package.goal.state.pos, # 包裹与目标的相对位置
                self.package.state.vel, # 包裹速度
                self.line.state.vel, # 线段速度
                self.line.state.ang_vel, # 线段角速度
                self.line.state.rot % torch.pi, # 线段旋转角，取模 [-pi, pi]
            ],
            dim=-1,
        )

    def done(self):
        # 终止条件函数：落地或包裹抵达目标即终止
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}
        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
    )
