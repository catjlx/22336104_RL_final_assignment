#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.joints import Joint
from vmas.simulator.physics import (
    _get_closest_box_box,
    _get_closest_line_box,
    _get_closest_point_box,
    _get_closest_point_line,
    _get_closest_points_line_line,
    _get_inner_point_box,
)
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    COLLISION_FORCE,
    Color,
    DRAG,
    JOINT_FORCE,
    LINE_MIN_DIST,
    LINEAR_FRICTION,
    Observable,
    override,
    TorchUtils,
    TORQUE_CONSTRAINT_FORCE,
    X,
    Y,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class TorchVectorizedObject(object):
    def __init__(self, batch_dim: int = None, device: torch.device = None):
        # batch dim 批量环境数量
        self._batch_dim = batch_dim
        # device
        self._device = device

    @property
    def batch_dim(self):
        return self._batch_dim

    @batch_dim.setter
    def batch_dim(self, batch_dim: int):
        assert self._batch_dim is None, "You can set batch dim only once"
        self._batch_dim = batch_dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    def _check_batch_index(self, batch_index: int): # 验证批量环境索引合法性
        if batch_index is not None:
            assert (
                0 <= batch_index < self.batch_dim
            ), f"Index must be between 0 and {self.batch_dim}, got {batch_index}"

    def to(self, device: torch.device): # 张量从CPU迁移到GPU，支持GPU直接进行运算
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                self.__dict__[attr] = value.to(device)


class Shape(ABC): # 实体形状抽象类：支持球体、立方体、线段三种形状
    @abstractmethod
    def moment_of_inertia(self, mass: float):
        # 计算转动惯量，计算方法见论文公式（3）
        raise NotImplementedError

    @abstractmethod
    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        # 获取锚点相对形状中心的偏移
        raise NotImplementedError

    @abstractmethod
    def get_geometry(self):
        # 获取渲染几何形状
        raise NotImplementedError

    @abstractmethod
    def circumscribed_radius(self):
        # 计算外接圆半径
        raise NotImplementedError


class Box(Shape): # 立方体
    def __init__(self, length: float = 0.3, width: float = 0.1, hollow: bool = False):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width
        self.hollow = hollow # 是否空心：关系到碰撞检测时内部点处理

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        # 基于形状中心的相对位置计算锚点偏移
        return anchor[X] * self.length / 2, anchor[Y] * self.width / 2

    def moment_of_inertia(self, mass: float):
        # 立方体转动惯量 I = (1/12) m (l²+w²)
        return (1 / 12) * mass * (self.length**2 + self.width**2)

    def circumscribed_radius(self):
        # 外接圆半径（即对角线的一半，用于快速碰撞筛选）
        return math.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)

    def get_geometry(self) -> "Geom":
        # 生成渲染的图形（可视化）
        from vmas.simulator import rendering

        l, r, t, b = (
            -self.length / 2,
            self.length / 2,
            self.width / 2,
            -self.width / 2,
        )
        return rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])


class Sphere(Shape):
    # 球体形状 / 默认形状
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius # 半径

    @property
    def radius(self):
        return self._radius

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        # 锚点便宜：限制在球体内
        delta = torch.tensor([anchor[X] * self.radius, anchor[Y] * self.radius]).to(
            torch.float32
        )
        delta_norm = torch.linalg.vector_norm(delta)
        if delta_norm > self.radius:
            delta /= delta_norm * self.radius
        return tuple(delta.tolist())

    def moment_of_inertia(self, mass: float):
        # 球体转动惯量公式：I = (1/2) mr²
        return (1 / 2) * mass * self.radius**2

    def circumscribed_radius(self):
        # 外接圆半径 即 球体半径
        return self.radius

    def get_geometry(self) -> "Geom":
        # 生成渲染的图形
        from vmas.simulator import rendering

        return rendering.make_circle(self.radius)


class Line(Shape):
    # 线段形状，如转动轴
    def __init__(self, length: float = 0.5):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        self._length = length 
        self._width = 2 # 渲染宽度

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float):
        # 线段转动惯量I = (1/12) ml²
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self):
        # 外接圆半径 = 长度一半
        return self.length / 2

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        # 锚点偏移：仅沿线段 X 轴
        return anchor[X] * self.length / 2, 0.0

    def get_geometry(self) -> "Geom":
        # 生成渲染用线段
        from vmas.simulator import rendering

        return rendering.Line(
            (-self.length / 2, 0),
            (self.length / 2, 0),
            width=self.width,
        )


class EntityState(TorchVectorizedObject):
    # 实体状态类：用张量存储批量环境中实体的物理状态：位置、速度、旋转角、角速度
    def __init__(self):
        super().__init__()
        # physical position (x, y)，shape: (batch_dim, 2)
        self._pos = None
        # physical velocity shape: (batch_dim, 2)
        self._vel = None
        # physical rotation -- from -pi to pi shape: (batch_dim, 1)
        self._rot = None
        # angular velocity shape: (batch_dim, 1)
        self._ang_vel = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Tensor):
        # 设置位置
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            pos.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {pos.shape[0]}, expected {self._batch_dim}"
        if self._vel is not None:
            assert (
                pos.shape == self._vel.shape
            ), f"Position shape must match velocity shape, got {pos.shape} expected {self._vel.shape}"

        self._pos = pos.to(self._device)

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, vel: Tensor):
        # 设置速度
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            vel.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {vel.shape[0]}, expected {self._batch_dim}"
        if self._pos is not None:
            assert (
                vel.shape == self._pos.shape
            ), f"Velocity shape must match position shape, got {vel.shape} expected {self._pos.shape}"

        self._vel = vel.to(self._device)

    @property
    def ang_vel(self):
        return self._ang_vel

    @ang_vel.setter
    def ang_vel(self, ang_vel: Tensor):
         # 设置角速度
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            ang_vel.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {ang_vel.shape[0]}, expected {self._batch_dim}"

        self._ang_vel = ang_vel.to(self._device)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot: Tensor):
        # 设置旋转角
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            rot.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {rot.shape[0]}, expected {self._batch_dim}"

        self._rot = rot.to(self._device)

    def _reset(self, env_index: typing.Optional[int]):
        # 重置指定环境的状态，scenarios中的reset_world_at依赖于该函数
        for attr_name in ["pos", "rot", "vel", "ang_vel"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )

    def zero_grad(self):
        # 梯度清零，用以支持可微分仿真，适配强化学习反向传播
        for attr_name in ["pos", "rot", "vel", "ang_vel"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())

    def _spawn(self, dim_c: int, dim_p: int):
        # 初始化状态张量
        self.pos = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.vel = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.rot = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.ang_vel = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )


class AgentState(EntityState):
    # 智能体状态类：继承实体状态，扩展通信、动作力、扭矩状态，适配智能体力控制特性
    def __init__(
        self,
    ):
        super().__init__()
        # communication utterance 支持连续 / 离散通信shape: (batch_dim, dim_c)
        self._c = None
        # Agent force from actions shape: (batch_dim, 2)
        self._force = None
        # Agent torque from actions shape: (batch_dim, 1)
        self._torque = None

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c: Tensor):
        # 设置通信信息
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"

        self._c = c.to(self._device)

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, value):
        # 设置动作力：动作->力转换
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._force = value.to(self._device)

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, value):
        # 设置扭矩：碰撞力->扭矩转换
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._torque = value.to(self._device)

    @override(EntityState)
    def _reset(self, env_index: typing.Optional[int]):
        # 重置智能体专属状态（通信、力、扭矩）
        for attr_name in ["c", "force", "torque"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )
        super()._reset(env_index)

    @override(EntityState)
    def zero_grad(self):
        # 智能体状态梯度清零
        for attr_name in ["c", "force", "torque"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())
        super().zero_grad()

    @override(EntityState)
    def _spawn(self, dim_c: int, dim_p: int):
        # 初始化智能体专属状态张量
        if dim_c > 0:
            self.c = torch.zeros(
                self.batch_dim, dim_c, device=self.device, dtype=torch.float32
            )
        self.force = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.torque = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        super()._spawn(dim_c, dim_p)


# action of an agent
class Action(TorchVectorizedObject):
    # 智能体动作：管理物理动作（力控制）和通信动作，支持动作噪声、范围限制、放大系数
    def __init__(
        self,
        u_range: Union[float, Sequence[float]],
        u_multiplier: Union[float, Sequence[float]],
        u_noise: Union[float, Sequence[float]],
        action_size: int,
    ):
        super().__init__()
        # physical motor noise amount
        self._u_noise = u_noise
        # control range：动作剪辑到 [-u_range, u_range]
        self._u_range = u_range
        # agent action is a force multiplied by this amount：动作→力转换的缩放因子
        self._u_multiplier = u_multiplier
        # Number of actions：物理动作 + 通信动作维度
        self.action_size = action_size

        # physical action：对应力控制信号
        self._u = None
        # communication_action：智能体间通信信号
        self._c = None
        # 缓存张量格式的动作参数，加速批量运算
        self._u_range_tensor = None
        self._u_multiplier_tensor = None
        self._u_noise_tensor = None

        self._check_action_init()

    def _check_action_init(self):
        # 验证动作参数维度一致性，参数需与动作维度匹配
        for attr in (self.u_multiplier, self.u_range, self.u_noise):
            if isinstance(attr, List):
                assert len(attr) == self.action_size, (
                    "Action attributes u_... must be either a float or a list of floats"
                    " (one per action) all with same length"
                )

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u: Tensor):
        # 设置物理动作
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an agent to the world before setting its action"
        assert (
            u.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {u.shape[0]}, expected {self._batch_dim}"

        self._u = u.to(self._device)

    @property
    def c(self):
        # 设置通信动作
        return self._c

    @c.setter
    def c(self, c: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an agent to the world before setting its action"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"

        self._c = c.to(self._device)

    @property
    def u_range(self):
        return self._u_range

    @property
    def u_multiplier(self):
        return self._u_multiplier

    @property
    def u_noise(self):
        return self._u_noise

    @property
    def u_range_tensor(self):
        # 获取张量格式的动作范围
        if self._u_range_tensor is None:
            self._u_range_tensor = self._to_tensor(self.u_range)
        return self._u_range_tensor

    @property
    def u_multiplier_tensor(self):
        # 获取张量格式的动作放大系数
        if self._u_multiplier_tensor is None:
            self._u_multiplier_tensor = self._to_tensor(self.u_multiplier)
        return self._u_multiplier_tensor

    @property
    def u_noise_tensor(self):
        # 获取张量格式的动作噪声
        if self._u_noise_tensor is None:
            self._u_noise_tensor = self._to_tensor(self.u_noise)
        return self._u_noise_tensor

    def _to_tensor(self, value):
        # 将标量和列表转换为张量以适配批量环境
        return torch.tensor(
            value if isinstance(value, Sequence) else [value] * self.action_size,
            device=self.device,
            dtype=torch.float,
        )

    def _reset(self, env_index: typing.Optional[int]):
        # 重置动作状态
        for attr_name in ["u", "c"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )

    def zero_grad(self):
        for attr_name in ["u", "c"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())


# properties and state of physical world entity
class Entity(TorchVectorizedObject, Observable, ABC):
    # 实体抽象基类，统一管理智能体和地标的公共属性，支持碰撞检测、状态更新、渲染等核心功能
    def __init__(
        self,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: typing.Union[float, Tensor] = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        if shape is None:
            shape = Sphere()

        TorchVectorizedObject.__init__(self)
        Observable.__init__(self)
        # name
        self._name = name
        # entity can move / be pushed
        self._movable = movable
        # entity can rotate
        self._rotatable = rotatable
        # entity collides with others
        self._collide = collide
        # material density (affects mass)
        self._density = density
        # mass
        self._mass = mass
        # max speed
        self._max_speed = max_speed
        self._v_range = v_range
        # color
        self._color = color
        # shape
        self._shape = shape
        # is joint
        self._is_joint = is_joint
        # collision filter
        self._collision_filter = collision_filter
        # state
        self._state = EntityState()
        # drag
        self._drag = drag
        # friction 线性摩擦 + 角摩擦
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        # gravity
        if isinstance(gravity, Tensor):
            self._gravity = gravity
        else:
            self._gravity = (
                torch.tensor(gravity, device=self.device, dtype=torch.float32)
                if gravity is not None
                else gravity
            )
        # entity goal
        self._goal = None
        # Render the entity
        self._render = None

    @TorchVectorizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        # 设置批量维度时，同步初始化状态的批量维度
        TorchVectorizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @property
    def is_rendering(self):
        # 获取渲染开关状态
        if self._render is None:
            self.reset_render()
        return self._render

    def reset_render(self):
        # 重置渲染开关
        self._render = torch.full((self.batch_dim,), True, device=self.device)

    def collides(self, entity: Entity):
        # 是否碰撞
        if not self.collide:
            return False
        return self._collision_filter(entity)

    @property
    def is_joint(self):
        return self._is_joint

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass: float):
        self._mass = mass

    @property
    def moment_of_inertia(self):
        # 计算转动惯量：扭矩->角加速度转换依赖
        return self.shape.moment_of_inertia(self.mass)

    @property
    def state(self):
        return self._state

    @property
    def movable(self):
        return self._movable

    @property
    def collide(self):
        return self._collide

    @property
    def shape(self):
        return self._shape

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def v_range(self):
        return self._v_range

    @property
    def name(self):
        return self._name

    @property
    def rotatable(self):
        return self._rotatable

    @property
    def color(self):
        # 获取渲染颜色
        if isinstance(self._color, Color):
            return self._color.value
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def goal(self):
        return self._goal

    @property
    def drag(self):
        return self._drag

    @property
    def linear_friction(self):
        return self._linear_friction

    @linear_friction.setter
    def linear_friction(self, value):
        self._linear_friction = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value

    @property
    def angular_friction(self):
        return self._angular_friction

    @goal.setter
    def goal(self, goal: Entity):
        self._goal = goal

    @property
    def collision_filter(self):
        return self._collision_filter

    @collision_filter.setter
    def collision_filter(self, collision_filter: Callable[[Entity], bool]):
        self._collision_filter = collision_filter

    def _spawn(self, dim_c: int, dim_p: int):
        # 初始化实体状态
        self.state._spawn(dim_c, dim_p)

    def _reset(self, env_index: int):
        # 重置实体状态，在环境重置时调用
        self.state._reset(env_index)

    def zero_grad(self):
        self.state.zero_grad()

    def set_pos(self, pos: Tensor, batch_index: int):
        # 设置位置
        self._set_state_property(EntityState.pos, self.state, pos, batch_index)

    def set_vel(self, vel: Tensor, batch_index: int):
        # 设置速度
        self._set_state_property(EntityState.vel, self.state, vel, batch_index)

    def set_rot(self, rot: Tensor, batch_index: int):
        # 设置旋转角
        self._set_state_property(EntityState.rot, self.state, rot, batch_index)

    def set_ang_vel(self, ang_vel: Tensor, batch_index: int):
        # 设置角速度
        self._set_state_property(EntityState.ang_vel, self.state, ang_vel, batch_index)

    def _set_state_property(
        self, prop, entity: EntityState, new: Tensor, batch_index: int
    ):
    # 通用状态设置
        assert (
            self.batch_dim is not None
        ), f"Tried to set property of {self.name} without adding it to the world"
        self._check_batch_index(batch_index)
        new = new.to(self.device)
        if batch_index is None:
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                prop.fset(entity, new)
            else:
                prop.fset(entity, new.repeat(self.batch_dim, 1))
        else:
            value = prop.fget(entity)
            value[batch_index] = new
        self.notify_observers()

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        # 设备迁移：同步状态和实体属性的设备
        super().to(device)
        self.state.to(device)

    def render(self, env_index: int = 0) -> "List[Geom]":
        # 渲染实体：交互式渲染
        from vmas.simulator import rendering

        if not self.is_rendering[env_index]:
            return []
        geom = self.shape.get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)

        xform.set_translation(*self.state.pos[env_index])
        xform.set_rotation(self.state.rot[env_index])

        color = self.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color)

        return [geom]


# properties of landmark entities
class Landmark(Entity):
    # 地标类：非智能体实体
    # 继承实体基类，无额外扩展
    def __init__(
        self,
        name: str,
        shape: Shape = None,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            gravity,
            collision_filter,
        )


# properties of agent entities
class Agent(Entity):
    # 智能体类，定义可控制的实体（MARL的核心交互对象）
    # 扩展传感器、通信、动力学模型、动作脚本等智能体专属功能
    # 支持连续 / 离散动作，适配Gym / RLlib接口
    def __init__(
        self,
        name: str,
        shape: Shape = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: Union[float, Sequence[float]] = 0.0,
        u_range: Union[float, Sequence[float]] = 1.0,
        u_multiplier: Union[float, Sequence[float]] = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: List[Sensor] = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: List[
            int
        ] = None,  # Defaults to 3-way discretization if discrete actions are chosen (stay, decrement, increment)
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint=False,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"
        # 验证动作维度与离散动作向量一致性
        if action_size is not None and discrete_action_nvec is not None:
            if action_size != len(discrete_action_nvec):
                raise ValueError(
                    f"action_size {action_size} is inconsistent with discrete_action_nvec {discrete_action_nvec}"
                )
        if discrete_action_nvec is not None:
            if not all(n > 1 for n in discrete_action_nvec):
                raise ValueError(
                    f"All values in discrete_action_nvec must be greater than 1, got {discrete_action_nvec}"
                )

        # cannot observe the world 观测范围：限制传感器有效距离
        self._obs_range = obs_range
        # observation noise 观测噪声：添加观测噪声提升鲁棒性
        self._obs_noise = obs_noise
        # force constraints 力约束：力的范围限制 + 最大力
        self._f_range = f_range
        self._max_f = max_f
        # torque constraints 扭矩约束：扭矩的范围限制 + 最大扭矩
        self._t_range = t_range
        self._max_t = max_t
        # script behavior to execute 脚本行为：启发式策略，用于基准对比
        self._action_script = action_script
        # agents sensors 传感器列表：支持激光雷达等自定义传感器
        self._sensors = []
        if sensors is not None:
            [self.add_sensor(sensor) for sensor in sensors]
        # non differentiable communication noise 通信噪声
        self._c_noise = c_noise
        # cannot send communication signals
        self._silent = silent
        # render the agent action force 可视化动作力的方向
        self._render_action = render_action
        # is adversary 是否味对抗性智能体
        self._adversary = adversary
        # Render alpha 渲染透明度：区分不同智能体
        self._alpha = alpha
 
        # Dynamics 动力学模型：默认完整约束运动，聚焦高层协调
        self.dynamics = dynamics if dynamics is not None else Holonomic()
        # Action
        if action_size is not None:
            self.action_size = action_size
        elif discrete_action_nvec is not None:
            self.action_size = len(discrete_action_nvec)
        else:
            self.action_size = self.dynamics.needed_action_size
        # 离散动作配置：停留、递减、递增
        if discrete_action_nvec is None:
            self.discrete_action_nvec = [3] * self.action_size
        else:
            self.discrete_action_nvec = discrete_action_nvec
        self.dynamics.agent = self
        # 动作对象：管理物理动作和通信动作
        self._action = Action(
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
            action_size=self.action_size,
        )

        # state 智能体状态：扩展通信、力、扭矩
        self._state = AgentState()

    def add_sensor(self, sensor: Sensor):
        # 添加传感器
        sensor.agent = self
        self._sensors.append(sensor)

    @Entity.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        # 设置批量维度时，同步初始化动作的批量维度
        Entity.batch_dim.fset(self, batch_dim)
        self._action.batch_dim = batch_dim

    @property
    def action_script(self) -> Callable[[Agent, World], None]:
        return self._action_script

    def action_callback(self, world: World):
        # 执行动作脚本：启发式策略执行
        self._action_script(self, world)
        # 验证静默智能体不通信
        if self._silent or world.dim_c == 0:
            assert (
                self._action.c is None
            ), f"Agent {self.name} should not communicate but action script communicates"
        # 验证动作已设置且维度正确
        assert (
            self._action.u is not None
        ), f"Action script of {self.name} should set u action"
        assert (
            self._action.u.shape[1] == self.action_size
        ), f"Scripted action of agent {self.name} has wrong shape"
        # 验证动作在允许范围内
        assert (
            (self._action.u / self.action.u_multiplier_tensor).abs()
            <= self.action.u_range_tensor
        ).all(), f"Scripted physical action of {self.name} is out of range"

    @property
    def u_range(self):
        return self.action.u_range

    @property
    def obs_noise(self):
        return self._obs_noise if self._obs_noise is not None else 0

    @property
    def action(self) -> Action:
        return self._action

    @property
    def u_multiplier(self):
        return self.action.u_multiplier

    @property
    def max_f(self):
        return self._max_f

    @property
    def f_range(self):
        return self._f_range

    @property
    def max_t(self):
        return self._max_t

    @property
    def t_range(self):
        return self._t_range

    @property
    def silent(self):
        return self._silent

    @property
    def sensors(self) -> List[Sensor]:
        return self._sensors

    @property
    def u_noise(self):
        return self.action.u_noise

    @property
    def c_noise(self):
        return self._c_noise

    @property
    def adversary(self):
        return self._adversary

    @override(Entity)
    def _spawn(self, dim_c: int, dim_p: int):
        # 初始化智能体状态，含通信、力、扭矩
        if dim_c == 0:
            assert (
                self.silent
            ), f"Agent {self.name} must be silent when world has no communication"
        if self.silent:
            dim_c = 0
        super()._spawn(dim_c, dim_p)

    @override(Entity)
    def _reset(self, env_index: int):
        # 重置智能体状态，含动作和动力学模型
        self.action._reset(env_index)
        self.dynamics.reset(env_index)
        super()._reset(env_index)

    def zero_grad(self):
        self.action.zero_grad()
        self.dynamics.zero_grad()
        super().zero_grad()

    @override(Entity)
    def to(self, device: torch.device):
        # 设备迁移，含动作和传感器
        super().to(device)
        self.action.to(device)
        for sensor in self.sensors:
            sensor.to(device)

    @override(Entity)
    def render(self, env_index: int = 0) -> "List[Geom]":
        # 渲染智能体，含传感器和动作力可视化
        from vmas.simulator import rendering

        geoms = super().render(env_index)
        if len(geoms) == 0:
            return geoms
        # 设置透明度
        for geom in geoms:
            geom.set_color(*self.color, alpha=self._alpha)
        # 渲染传感器
        if self._sensors is not None:
            for sensor in self._sensors:
                geoms += sensor.render(env_index=env_index)
        # 渲染动作力
        if self._render_action and self.state.force is not None:
            velocity = rendering.Line(
                self.state.pos[env_index],
                self.state.pos[env_index]
                + self.state.force[env_index] * 10 * self.shape.circumscribed_radius(),
                width=2,
            )
            velocity.set_color(*self.color)
            geoms.append(velocity)

        return geoms


# Multi-agent world
class World(TorchVectorizedObject):
    # 多智能体世界类，管理所有实体（智能体+地标）
    # 执行力计算、状态积分、通信状态更新
    def __init__(
        self,
        batch_dim: int,
        device: torch.device,
        dt: float = 0.1,
        substeps: int = 1,  # if you use joints, higher this value to gain simulation stability
        drag: float = DRAG,
        linear_friction: float = LINEAR_FRICTION,
        angular_friction: float = ANGULAR_FRICTION,
        x_semidim: float = None,
        y_semidim: float = None,
        dim_c: int = 0,
        collision_force: float = COLLISION_FORCE,
        joint_force: float = JOINT_FORCE,
        torque_constraint_force: float = TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        gravity: Tuple[float, float] = (0.0, 0.0),
    ):
        assert batch_dim > 0, f"Batch dim must be greater than 0, got {batch_dim}"

        super().__init__(batch_dim, device)
        # list of agents and entities (can change at execution-time!)
        self._agents = []
        self._landmarks = []
        # world dims: no boundaries if none
        self._x_semidim = x_semidim
        self._y_semidim = y_semidim
        # position dimensionality
        self._dim_p = 2
        # communication channel dimensionality
        self._dim_c = dim_c
        # simulation timestep
        self._dt = dt
        # 子步数量：关节约束时增大以提升稳定性
        self._substeps = substeps
        self._sub_dt = self._dt / self._substeps
        # drag coefficient
        self._drag = drag
        # gravity
        self._gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32)
        # friction coefficients
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        # constraint response parameters：碰撞力强度 c；关节力；扭矩约束力；碰撞接触裕度
        self._collision_force = collision_force
        self._joint_force = joint_force
        self._contact_margin = contact_margin
        self._torque_constraint_force = torque_constraint_force
        # joints
        self._joints = {}
        # Pairs of collidable shapes 预定义支持的碰撞组合
        self._collidable_pairs = [
            {Sphere, Sphere},
            {Sphere, Box},
            {Sphere, Line},
            {Line, Line},
            {Line, Box},
            {Box, Box},
        ]
        # Map to save entity indexes
        self.entity_index_map = {}

    def add_agent(self, agent: Agent):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent.to(self._device)
        agent._spawn(dim_c=self._dim_c, dim_p=self.dim_p)
        self._agents.append(agent)

    def add_landmark(self, landmark: Landmark):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark.to(self._device)
        landmark._spawn(dim_c=self.dim_c, dim_p=self.dim_p)
        self._landmarks.append(landmark)

    def add_joint(self, joint: Joint):
        assert self._substeps > 1, "For joints, world substeps needs to be more than 1"
        if joint.landmark is not None:
            self.add_landmark(joint.landmark)
        for constraint in joint.joint_constraints:
            self._joints.update(
                {
                    frozenset(
                        {constraint.entity_a.name, constraint.entity_b.name}
                    ): constraint
                }
            )

    def reset(self, env_index: int):
        for e in self.entities:
            e._reset(env_index)

    def zero_grad(self):
        for e in self.entities:
            e.zero_grad()

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def landmarks(self) -> List[Landmark]:
        return self._landmarks

    @property
    def x_semidim(self):
        return self._x_semidim

    @property
    def dt(self):
        return self._dt

    @property
    def y_semidim(self):
        return self._y_semidim

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def joints(self):
        return self._joints.values()

    # return all entities in the world
    @property
    def entities(self) -> List[Entity]:
        # 获取所有实体，包括地标和智能体，碰撞检测遍历要用到
        return self._landmarks + self._agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> List[Agent]:
        # 获取外部策略控制的智能体：MARL训练对象
        return [agent for agent in self._agents if agent.action_script is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> List[Agent]:
        # 获取脚本控制的智能体：启发式基准策略
        return [agent for agent in self._agents if agent.action_script is not None]

    def _cast_ray_to_box(
        self,
        box: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        # 射线与立方体碰撞检测，激光雷达传感器核心
        # 基于AABB盒相交算法，返回射线距离
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(box.shape, Box)

        pos_origin = ray_origin - box.state.pos
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box.state.rot)
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box.state.rot)

        tx1 = (-box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx2 = (box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)

        ty1 = (-box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty2 = (box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)

        intersect_aabb = tmin.unsqueeze(1) * ray_dir_aabb + pos_aabb
        intersect_world = (
            TorchUtils.rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
        )

        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=1)
        dist[~collision] = max_range
        return dist

    def _cast_rays_to_box(
        self,
        box_pos,
        box_rot,
        box_length,
        box_width,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        # 批量射线与立方体碰撞检测：向量化优化，支持批量环境 + 多射线
        # 适配激光雷达多光束并行检测，提升仿真效率
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert box_pos.shape[:-2] == batch_size
        assert box_pos.shape[-1] == 2
        assert box_rot.shape[:-1] == batch_size
        assert box_width.shape[:-1] == batch_size
        assert box_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_boxes = box_pos.shape[-2]

        # Expand input to [*batch_size, n_boxes, num_angles, 2]
        # 扩展输入张量维度：适配批量环境 + 多立方体 + 多射线
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_boxes, num_angles, 2)
        )
        box_pos_expanded = box_pos.unsqueeze(-2).expand(
            *batch_size, n_boxes, num_angles, 2
        )
        # Expand input to [*batch_size, n_boxes, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_boxes, num_angles
        )
        box_rot_expanded = box_rot.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )
        box_width_expanded = box_width.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )
        box_length_expanded = box_length.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )

        # Compute pos_origin and pos_aabb 计算相对位置和 AABB 空间位置
        pos_origin = ray_origin - box_pos_expanded
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box_rot_expanded)

        # Calculate ray_dir_world 计算射线方向(世界空间→AABB 空间）
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )

        # Calculate ray_dir_aabb
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box_rot_expanded)

        # Calculate tx, ty, tmin, and tmax 射线与 AABB 的相交参数
        tx1 = (-box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx2 = (box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)

        ty1 = (-box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty2 = (box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)

        # Compute intersect_aabb and intersect_world
        # 计算相交点并转换回世界空间
        intersect_aabb = tmin.unsqueeze(-1) * ray_dir_aabb + pos_aabb
        intersect_world = (
            TorchUtils.rotate_vector(intersect_aabb, box_rot_expanded)
            + box_pos_expanded
        )

        # Calculate collision and distances 计算相交点并转换回世界空间
        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=-1)
        dist[~collision] = max_range
        return dist

    def _cast_ray_to_sphere(
        # 射线与球体碰撞检测
        self,
        sphere: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )
        test_point_pos = sphere.state.pos
        line_rot = ray_direction
        line_length = max_range
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        closest_point = _get_closest_point_line(
            line_pos,
            line_rot.unsqueeze(-1),
            line_length,
            test_point_pos,
            limit_to_line_length=False,
        )

        d = test_point_pos - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=1)
        ray_intersects = d_norm < sphere.shape.radius
        a = sphere.shape.radius**2 - d_norm**2
        m = torch.sqrt(torch.where(a > 0, a, 1e-8))

        u = test_point_pos - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1 点积判断球体是否在射线前方
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range

        return dist

    def _cast_rays_to_sphere(
        self,
        sphere_pos,
        sphere_radius,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        # 批量射线与球体碰撞检测
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert sphere_pos.shape[:-2] == batch_size
        assert sphere_pos.shape[-1] == 2
        assert sphere_radius.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_spheres = sphere_pos.shape[-2]

        # Expand input to [*batch_size, n_spheres, num_angles, 2]
        # 扩展输入维度（适配批量环境 + 多球体 + 多射线）
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_spheres, num_angles, 2)
        )
        sphere_pos_expanded = sphere_pos.unsqueeze(-2).expand(
            *batch_size, n_spheres, num_angles, 2
        )
        # Expand input to [*batch_size, n_spheres, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_spheres, num_angles
        )
        sphere_radius_expanded = sphere_radius.unsqueeze(-1).expand(
            *batch_size, n_spheres, num_angles
        )

        # Calculate ray_dir_world 计算射线方向和线段位置
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )

        line_rot = ray_direction.unsqueeze(-1)

        # line_length remains scalar and will be broadcasted as needed
        line_length = max_range

        # Calculate line_pos
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        # Call the updated _get_closest_point_line function 计算射线与球体的最近点
        closest_point = _get_closest_point_line(
            line_pos,
            line_rot,
            line_length,
            sphere_pos_expanded,
            limit_to_line_length=False,
        )

        # Calculate distances and other metrics 计算碰撞距离
        d = sphere_pos_expanded - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=-1)
        ray_intersects = d_norm < sphere_radius_expanded
        a = sphere_radius_expanded**2 - d_norm**2
        m = torch.sqrt(torch.where(a > 0, a, 1e-8))

        u = sphere_pos_expanded - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=-1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range

        return dist

    def _cast_ray_to_line(
        self,
        line: Entity, # 目标线段实体
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float, # 射线最大检测距离
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        # 单挑射线与线段的碰撞检测，也即激光雷达核心字逻辑
        # 基于线段相较算法返回射线到线段的距离，无碰撞则为max_range
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(line.shape, Line)
        # 线段状态提取：位置、旋转方向向量（pos/rot）
        p = line.state.pos # 线段中心位置（batch_dim, 2）
        r = (
            torch.stack(
                [
                    torch.cos(line.state.rot.squeeze(1)),
                    torch.sin(line.state.rot.squeeze(1)),
                ],
                dim=-1,
            )
            * line.shape.length # 线段方向向量 × 长度
        )

        # 射线状态转换：起点→向量，方向角→单位向量
        q = ray_origin # 射线起点 (batch_dim, 2)
        s = torch.stack(
            [
                torch.cos(ray_direction),
                torch.sin(ray_direction),
            ],
            dim=-1,
        ) # 射线单位方向向量 (batch_dim, 2)

        # 线段-射线相交计算（基于向量叉乘判断方向关系）
        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - p, s / rxs)
        u = TorchUtils.cross(q - p, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1) # 计算碰撞点距离射线起点的距离
       
        # 边缘情况处理
        perpendicular = rxs == 0.0 # 射线与线段平行，无相交
        above_line = t > 0.5 # 碰撞点超出线段范围（上侧）
        below_line = t < -0.5 # 碰撞点超出线段范围（下侧）
        behind_line = u < 0.0 # 碰撞点在射线起点后方
        # 无碰撞时距离设为max_range
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def _cast_rays_to_line(
        self,
        line_pos, # 批量线段位置 (batch_dim, n_lines, 2)
        line_rot, # 批量线段旋转角 (batch_dim, n_lines, 1)
        line_length, # 批量线段长度 (batch_dim, n_lines)
        ray_origin: Tensor, # 批量射线起点 (batch_dim, n_angles, 2)
        ray_direction: Tensor, # 批量射线方向（角度）(batch_dim, n_angles)
        max_range: float, # 射线最大检测距离
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        # 批量射线与批量线段的向量化碰撞检测
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim # 确保批量维度一致
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert line_pos.shape[:-2] == batch_size
        assert line_pos.shape[-1] == 2
        assert line_rot.shape[:-1] == batch_size
        assert line_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1] # 每条激光雷达的射线数量
        n_lines = line_pos.shape[-2] # 批量处理的线段数量

        # Expand input to [*batch_size, n_lines, num_angles, 2]
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_lines, num_angles, 2)
        )
        line_pos_expanded = line_pos.unsqueeze(-2).expand(
            *batch_size, n_lines, num_angles, 2
        )
        # Expand input to [*batch_size, n_lines, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_lines, num_angles
        )
        line_rot_expanded = line_rot.unsqueeze(-1).expand(
            *batch_size, n_lines, num_angles
        )
        line_length_expanded = line_length.unsqueeze(-1).expand(
            *batch_size, n_lines, num_angles
        )

        # Expand line state variables批量计算线段方向向量（旋转角→单位向量 × 长度）
        r = torch.stack(
            [
                torch.cos(line_rot_expanded),
                torch.sin(line_rot_expanded),
            ],
            dim=-1,
        ) * line_length_expanded.unsqueeze(-1)

        # Calculate q and s 计算射线方向向量（角度→单位向量）
        q = ray_origin
        s = torch.stack(
            [
                torch.cos(ray_direction),
                torch.sin(ray_direction),
            ],
            dim=-1,
        )

        # Calculate rxs, t, u, and d 计算线段-射线相交参数
        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - line_pos_expanded, s / rxs)
        u = TorchUtils.cross(q - line_pos_expanded, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1)

        # Handle edge cases 批量处理边缘情况
        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def cast_ray(
        # 单条射线批量检测，适配单个实体的激光雷达观测
        self,
        entity: Entity, # 发射射线的实体，可能是智能体或地标
        angles: Tensor, # 射线角度列表
        max_range: float, # 射线最大检测距离
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ):
        pos = entity.state.pos # 射线发射起点（实体中心位置）
        # 向量化批量处理要求验证输入维度
        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = [
            torch.full((self.batch_dim,), fill_value=max_range, device=self.device)
        ]
        # 遍历所有实体，检测射线与可碰撞实体的相交
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            # 根据实体形状调用对应的射线检测函数
            if isinstance(e.shape, Box):
                d = self._cast_ray_to_box(e, pos, angles, max_range)
            elif isinstance(e.shape, Sphere):
                d = self._cast_ray_to_sphere(e, pos, angles, max_range)
            elif isinstance(e.shape, Line):
                d = self._cast_ray_to_line(e, pos, angles, max_range)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")
            dists.append(d)
        # 取每条射线的最近碰撞距离
        dist, _ = torch.min(torch.stack(dists, dim=-1), dim=-1)
        return dist

    def cast_rays(
        # 批量射线检测：激光雷达核心接口
        self,
        entity: Entity,
        angles: Tensor,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,# 实体过滤函数
    ):
        pos = entity.state.pos

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = torch.full_like(
            angles, fill_value=max_range, device=self.device
        ).unsqueeze(-1)
        boxes = []
        spheres = []
        lines = []
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            if isinstance(e.shape, Box):
                boxes.append(e)
            elif isinstance(e.shape, Sphere):
                spheres.append(e)
            elif isinstance(e.shape, Line):
                lines.append(e)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")

        # Boxes 批量检测立方体
        if len(boxes):
            pos_box = []
            rot_box = []
            length_box = []
            width_box = []
            for box in boxes:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
            # 堆叠为批量张量
            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_boxes = self._cast_rays_to_box(
                pos_box,
                rot_box.squeeze(-1),
                length_box,
                width_box,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_boxes.transpose(-1, -2)], dim=-1)
        # Spheres 批量检测球体
        if len(spheres):
            pos_s = []
            radius_s = []
            for s in spheres:
                pos_s.append(s.state.pos)
                radius_s.append(torch.tensor(s.shape.radius, device=self.device))
            pos_s = torch.stack(pos_s, dim=-2)
            radius_s = (
                torch.stack(
                    radius_s,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_spheres = self._cast_rays_to_sphere(
                pos_s,
                radius_s,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_spheres.transpose(-1, -2)], dim=-1)
        # Lines 批量检测线段
        if len(lines):
            pos_l = []
            rot_l = []
            length_l = []
            for line in lines:
                pos_l.append(line.state.pos)
                rot_l.append(line.state.rot)
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            pos_l = torch.stack(pos_l, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            length_l = (
                torch.stack(
                    length_l,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_lines = self._cast_rays_to_line(
                pos_l,
                rot_l.squeeze(-1),
                length_l,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_lines.transpose(-1, -2)], dim=-1)

        # 每条射线的最近碰撞距离
        dist, _ = torch.min(dists, dim=-1)
        return dist

    def get_distance_from_point(
        # 计算实体到指定点的距离，用于碰撞预检测核心
        self, entity: Entity, test_point_pos, env_index: int = None
    ):
        self._check_batch_index(env_index)

        if isinstance(entity.shape, Sphere):
            delta_pos = entity.state.pos - test_point_pos
            dist = torch.linalg.vector_norm(delta_pos, dim=-1)
            return_value = dist - entity.shape.radius
        elif isinstance(entity.shape, Box):
            closest_point = _get_closest_point_box(
                entity.state.pos,
                entity.state.rot,
                entity.shape.width,
                entity.shape.length,
                test_point_pos,
            )
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST # 立方体表面到点的距离
        elif isinstance(entity.shape, Line):
            closest_point = _get_closest_point_line(
                entity.state.pos,
                entity.state.rot,
                entity.shape.length,
                test_point_pos,
            )
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST # 线段表面到点的距离
        else:
            raise RuntimeError("Distance not computable for given entity")
        # 仅返回指定环境的结果
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    def get_distance(self, entity_a: Entity, entity_b: Entity, env_index: int = None):
        # 计算两个实体间的最小距离，碰撞检测的核心依赖
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        # 不同形状组合的距离计算
        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere):
            dist = self.get_distance_from_point(entity_a, entity_b.state.pos, env_index)
            return_value = dist - b_shape.radius
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(box, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
            is_overlapping = self.is_overlapping(entity_a, entity_b)
            return_value[is_overlapping] = -1 # 重叠时强制设为负
        elif (
            isinstance(entity_a.shape, Line)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Line)
            and isinstance(entity_a.shape, Sphere)
        ):
            line, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(line, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
            point_a, point_b = _get_closest_points_line_line(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.length,
            )
            dist = torch.linalg.vector_norm(point_a - point_b, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Line)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Line)
        ):
            box, line = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Line)
                else (entity_b, entity_a)
            )
            point_box, point_line = _get_closest_line_box(
                box.state.pos,
                box.state.rot,
                box.shape.width,
                box.shape.length,
                line.state.pos,
                line.state.rot,
                line.shape.length,
            )
            dist = torch.linalg.vector_norm(point_box - point_line, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = _get_closest_box_box(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.width,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.width,
                entity_b.shape.length,
            )
            dist = torch.linalg.vector_norm(point_a - point_b, dim=-1)
            return_value = dist - LINE_MIN_DIST
        else:
            raise RuntimeError("Distance not computable for given entities")
        return return_value

    def is_overlapping(self, entity_a: Entity, entity_b: Entity, env_index: int = None):
        # 判断两个实体是否重叠
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        self._check_batch_index(env_index)
        
        # Sphere sphere, sphere line, line line, line box, box box
        if (
            (isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere))
            or (
                (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                )
            )
            or (isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line))
            or (
                isinstance(entity_a.shape, Box)
                and isinstance(entity_b.shape, Line)
                or isinstance(entity_b.shape, Box)
                and isinstance(entity_a.shape, Line)
            )
            or (isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box))
        ):
            return self.get_distance(entity_a, entity_b, env_index) < 0
        # 立方体-球体组合的特殊判断
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            closest_point = _get_closest_point_box(
                box.state.pos,
                box.state.rot,
                box.shape.width,
                box.shape.length,
                sphere.state.pos,
            )

            distance_sphere_closest_point = torch.linalg.vector_norm(
                sphere.state.pos - closest_point, dim=-1
            )
            distance_sphere_box = torch.linalg.vector_norm(
                sphere.state.pos - box.state.pos, dim=-1
            )
            distance_closest_point_box = torch.linalg.vector_norm(
                box.state.pos - closest_point, dim=-1
            )
            dist_min = sphere.shape.radius + LINE_MIN_DIST
            return_value = (distance_sphere_box < distance_closest_point_box) + (
                distance_sphere_closest_point < dist_min
            )
        else:
            raise RuntimeError("Overlap not computable for give entities")
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    # update state of the world
    def step(self):
        # 世界仿真核心步骤：单次仿真步
        # 流程：实体索引映射 → 子步迭代（力计算→碰撞力→状态积分）→ 通信状态更新
        self.entity_index_map = {e: i for i, e in enumerate(self.entities)} # 实体索引映射

        # 子步迭代
        for substep in range(self._substeps):
            # 初始化批量力和扭矩张量（向量化存储，适配batch_dim个环境）
            self.forces_dict = {
                e: torch.zeros(
                    self._batch_dim,
                    self._dim_p,
                    device=self.device,
                    dtype=torch.float32,
                )
                for e in self.entities
            }
            self.torques_dict = {
                e: torch.zeros(
                    self._batch_dim,
                    1,
                    device=self.device,
                    dtype=torch.float32,
                )
                for e in self.entities
            }

            # 1. 应用实体力：动作力、重力、摩擦力
            for entity in self.entities:
                if isinstance(entity, Agent):
                    # apply agent force controls
                    self._apply_action_force(entity)
                    # apply agent torque controls
                    self._apply_action_torque(entity)
                # apply friction
                self._apply_friction_force(entity)
                # apply gravity
                self._apply_gravity(entity)
            # 2. 应用向量化环境力：碰撞力、关节力
            self._apply_vectorized_enviornment_force()
            # 3. 物理状态积分：半隐式欧拉法，更新位置、速度、旋转角、角速度
            for entity in self.entities:
                # integrate physical state
                self._integrate_state(entity, substep)

        # update non-differentiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    # gather agent action forces
    def _apply_action_force(self, agent: Agent):
        # 应用智能体的动作力
        # 流程：1. 力约束（最大力、力范围）→ 2. 累加到实体力字典
        if agent.movable:
            # 力大小约束（最大力、力范围）
            if agent.max_f is not None:
                agent.state.force = TorchUtils.clamp_with_norm(
                    agent.state.force, agent.max_f
                )
            if agent.f_range is not None:
                agent.state.force = torch.clamp(
                    agent.state.force, -agent.f_range, agent.f_range
                )
            # 累加动作力到实体力字典
            self.forces_dict[agent] = self.forces_dict[agent] + agent.state.force

    def _apply_action_torque(self, agent: Agent):
        # 应用智能体的动作扭矩
        # 流程：1. 扭矩约束 → 2. 累加到实体扭矩字典
        if agent.rotatable:
            # 扭矩大小约束
            if agent.max_t is not None:
                agent.state.torque = TorchUtils.clamp_with_norm(
                    agent.state.torque, agent.max_t
                )
            if agent.t_range is not None:
                agent.state.torque = torch.clamp(
                    agent.state.torque, -agent.t_range, agent.t_range
                )
            # 累加扭矩到实体扭矩字典
            self.torques_dict[agent] = self.torques_dict[agent] + agent.state.torque

    def _apply_gravity(self, entity: Entity):
        # 应用重力：f_i^g = m_i g
        if entity.movable:
            # 应用全局重力
            if not (self._gravity == 0.0).all():
                self.forces_dict[entity] = (
                    self.forces_dict[entity] + entity.mass * self._gravity
                )
            # 应用实体自定义重力
            if entity.gravity is not None:
                self.forces_dict[entity] = (
                    self.forces_dict[entity] + entity.mass * entity.gravity
                )

    def _apply_friction_force(self, entity: Entity):
        # 应用摩擦力：线性摩擦+角摩擦
        def get_friction_force(vel, coeff, force, mass):
            # 计算摩擦力：静摩擦→动摩擦转换
            speed = torch.linalg.vector_norm(vel, dim=-1)
            static = speed == 0 # 静摩擦状态（速度为0）
            static_exp = static.unsqueeze(-1).expand(vel.shape)
             # 摩擦系数转换为张量
            if not isinstance(coeff, Tensor):
                coeff = torch.full_like(force, coeff, device=self.device)
            coeff = coeff.expand(force.shape)

            friction_force_constant = coeff * mass # 最大静摩擦力
            # 计算动摩擦力
            friction_force = -(
                vel / torch.where(static, 1e-8, speed).unsqueeze(-1)
            ) * torch.minimum(
                friction_force_constant, (vel.abs() / self._sub_dt) * mass
            )
            friction_force = torch.where(static_exp, 0.0, friction_force) # 静摩擦时无摩擦力

            return friction_force
        # 应用线性摩擦力（影响线速度）
        if entity.linear_friction is not None:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(
                entity.state.vel,
                entity.linear_friction,
                self.forces_dict[entity],
                entity.mass,
            )
        elif self._linear_friction > 0:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(
                entity.state.vel,
                self._linear_friction,
                self.forces_dict[entity],
                entity.mass,
            )
        # 应用角摩擦力（影响角速度）
        if entity.angular_friction is not None:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                entity.angular_friction,
                self.torques_dict[entity],
                entity.moment_of_inertia,
            )
        elif self._angular_friction > 0:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                self._angular_friction,
                self.torques_dict[entity],
                entity.moment_of_inertia,
            )

    def _apply_vectorized_enviornment_force(self):
        # 应用向量化环境力（碰撞力+关节力）
        # 流程：1. 实体对分组（按形状+关节）→ 2. 批量计算约束force/torque → 3. 应用到实体
        # 按形状组合和关节分组实体对
        s_s = [] # 球+球
        l_s = [] # 线+球
        b_s = [] # 盒+球
        l_l = [] # 线+线
        b_l = [] # 盒+线
        b_b = [] # 盒+盒
        joints = [] # 关节约束
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue # 避免重复处理实体对（a-b和b-a）
                # 关节约束优先处理
                joint = self._joints.get(
                    frozenset({entity_a.name, entity_b.name}), None
                )
                if joint is not None:
                    joints.append(joint)
                    if joint.dist == 0:
                        continue
                # 碰撞与检测（粗检测，过滤不可能碰撞的实体对）
                if not self.collides(entity_a, entity_b):
                    continue
                # 按形状组合分组
                if isinstance(entity_a.shape, Sphere) and isinstance(
                    entity_b.shape, Sphere
                ):
                    s_s.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    line, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    l_s.append((line, sphere))
                elif isinstance(entity_a.shape, Line) and isinstance(
                    entity_b.shape, Line
                ):
                    l_l.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    box, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    b_s.append((box, sphere))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Line)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Line)
                ):
                    box, line = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Line)
                        else (entity_b, entity_a)
                    )
                    b_l.append((box, line))
                elif isinstance(entity_a.shape, Box) and isinstance(
                    entity_b.shape, Box
                ):
                    b_b.append((entity_a, entity_b))
                else:
                    raise AssertionError()
        # Joints
        self._vectorized_joint_constraints(joints)

        # 处理碰撞力
        # Sphere and sphere
        self._sphere_sphere_vectorized_collision(s_s)
        # Line and sphere
        self._sphere_line_vectorized_collision(l_s)
        # Line and line
        self._line_line_vectorized_collision(l_l)
        # Box and sphere
        self._box_sphere_vectorized_collision(b_s)
        # Box and line
        self._box_line_vectorized_collision(b_l)
        # Box and box
        self._box_box_vectorized_collision(b_b)

    def update_env_forces(self, entity_a, f_a, t_a, entity_b, f_b, t_b):
        # 批量更新实体的力和扭矩
        if entity_a.movable:
            self.forces_dict[entity_a] = self.forces_dict[entity_a] + f_a
        if entity_a.rotatable:
            self.torques_dict[entity_a] = self.torques_dict[entity_a] + t_a
        if entity_b.movable:
            self.forces_dict[entity_b] = self.forces_dict[entity_b] + f_b
        if entity_b.rotatable:
            self.torques_dict[entity_b] = self.torques_dict[entity_b] + t_b

    def _vectorized_joint_constraints(self, joints):
        # 向量化关节约束力和扭矩计算
        # 基于罚函数法的软约束，避免硬约束的数值震荡
        if len(joints):
            # 收集批量关节参数
            pos_a = []
            pos_b = []
            pos_joint_a = []
            pos_joint_b = []
            dist = []
            rotate = []
            rot_a = []
            rot_b = []
            joint_rot = []
            for joint in joints:
                entity_a = joint.entity_a
                entity_b = joint.entity_b
                pos_joint_a.append(joint.pos_point(entity_a)) # 关节在A上的作用点
                pos_joint_b.append(joint.pos_point(entity_b)) # 关节在B的作用点
                pos_a.append(entity_a.state.pos)
                pos_b.append(entity_b.state.pos)
                dist.append(torch.tensor(joint.dist, device=self.device)) # 关节目标距离
                rotate.append(torch.tensor(joint.rotate, device=self.device)) # 是否允许旋转
                rot_a.append(entity_a.state.rot)
                rot_b.append(entity_b.state.rot)
                # 关节固定旋转角
                joint_rot.append(
                    torch.tensor(joint.fixed_rotation, device=self.device)
                    .unsqueeze(-1)
                    .expand(self.batch_dim, 1)
                    if isinstance(joint.fixed_rotation, float)
                    else joint.fixed_rotation
                )
            # 堆叠为批量张量
            pos_a = torch.stack(pos_a, dim=-2)
            pos_b = torch.stack(pos_b, dim=-2)
            pos_joint_a = torch.stack(pos_joint_a, dim=-2)
            pos_joint_b = torch.stack(pos_joint_b, dim=-2)
            rot_a = torch.stack(rot_a, dim=-2)
            rot_b = torch.stack(rot_b, dim=-2)
            dist = (
                torch.stack(
                    dist,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            rotate_prior = torch.stack(
                rotate,
                dim=-1,
            )
            rotate = rotate_prior.unsqueeze(0).expand(self.batch_dim, -1).unsqueeze(-1)
            joint_rot = torch.stack(joint_rot, dim=-2)
            # 计算关节约束力：吸引力+排斥力，软约束核心
            (force_a_attractive, force_b_attractive,) = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=True, # 吸引力：距离>目标时拉回
                force_multiplier=self._joint_force,
            )
            force_a_repulsive, force_b_repulsive = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=False, # 排斥力：距离<目标时推开
                force_multiplier=self._joint_force,
            )
            force_a = force_a_attractive + force_a_repulsive
            force_b = force_b_attractive + force_b_repulsive

            # 计算扭矩：力臂×力，力臂=关节作用点-实体中心
            r_a = pos_joint_a - pos_a
            r_b = pos_joint_b - pos_b

            torque_a_rotate = TorchUtils.compute_torque(force_a, r_a)
            torque_b_rotate = TorchUtils.compute_torque(force_b, r_b)

            # 计算旋转约束扭矩
            torque_a_fixed, torque_b_fixed = self._get_constraint_torques(
                rot_a, rot_b + joint_rot, force_multiplier=self._torque_constraint_force
            )

            # 根据关节旋转自由度选择扭矩
            torque_a = torch.where(
                rotate, torque_a_rotate, torque_a_rotate + torque_a_fixed
            )
            torque_b = torch.where(
                rotate, torque_b_rotate, torque_b_rotate + torque_b_fixed
            )

            # 批量应用关节力和扭矩
            for i, joint in enumerate(joints):
                self.update_env_forces(
                    joint.entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    joint.entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _sphere_sphere_vectorized_collision(self, s_s):
        # 球-球向量化碰撞检测与碰撞力计算
        # 基于球心距离判断穿透，计算排斥力
        if len(s_s):
            # 收集批量球体参数
            pos_s_a = []
            pos_s_b = []
            radius_s_a = []
            radius_s_b = []
            for s_a, s_b in s_s:
                pos_s_a.append(s_a.state.pos)
                pos_s_b.append(s_b.state.pos)
                radius_s_a.append(torch.tensor(s_a.shape.radius, device=self.device))
                radius_s_b.append(torch.tensor(s_b.shape.radius, device=self.device))
            # 堆叠为批量张量
            pos_s_a = torch.stack(pos_s_a, dim=-2)
            pos_s_b = torch.stack(pos_s_b, dim=-2)
            radius_s_a = (
                torch.stack(
                    radius_s_a,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            radius_s_b = (
                torch.stack(
                    radius_s_b,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            # 计算碰撞力
            force_a, force_b = self._get_constraint_forces(
                pos_s_a,
                pos_s_b,
                dist_min=radius_s_a + radius_s_b,
                force_multiplier=self._collision_force,
            )

            # 应用碰撞力（球体碰撞无扭矩）
            for i, (entity_a, entity_b) in enumerate(s_s):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    0,
                    entity_b,
                    force_b[:, i],
                    0,
                )

    def _sphere_line_vectorized_collision(self, l_s):
        # 线-球向量化碰撞检测与碰撞力计算
        # 核心：先求球心到线段的最近点，再计算碰撞力
        if len(l_s):
            # 收集批量线和球的参数
            pos_l = []
            pos_s = []
            rot_l = []
            radius_s = []
            length_l = []
            for line, sphere in l_s:
                pos_l.append(line.state.pos)
                pos_s.append(sphere.state.pos)
                rot_l.append(line.state.rot)
                radius_s.append(torch.tensor(sphere.shape.radius, device=self.device))
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            # 堆叠为批量张量
            pos_l = torch.stack(pos_l, dim=-2)
            pos_s = torch.stack(pos_s, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            radius_s = (
                torch.stack(
                    radius_s,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            length_l = (
                torch.stack(
                    length_l,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            # 计算球心到线段的最近点
            closest_point = _get_closest_point_line(pos_l, rot_l, length_l, pos_s)
            # 计算碰撞力
            force_sphere, force_line = self._get_constraint_forces(
                pos_s,
                closest_point,
                dist_min=radius_s + LINE_MIN_DIST, # 最小安全距离=球半径+线最小距离
                force_multiplier=self._collision_force,
            )
            # 计算线段的扭矩（力臂=最近点-线段中心）
            r = closest_point - pos_l
            torque_line = TorchUtils.compute_torque(force_line, r)
            # 应用碰撞力和扭矩
            for i, (entity_a, entity_b) in enumerate(l_s):
                self.update_env_forces(
                    entity_a,
                    force_line[:, i],
                    torque_line[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _line_line_vectorized_collision(self, l_l):
        # 线-线向量化碰撞检测与碰撞力计算
        # 求两条线段的最近点对，计算碰撞力和扭矩
        if len(l_l):
            # 收集批量线的参数
            pos_l_a = []
            pos_l_b = []
            rot_l_a = []
            rot_l_b = []
            length_l_a = []
            length_l_b = []
            for l_a, l_b in l_l:
                pos_l_a.append(l_a.state.pos)
                pos_l_b.append(l_b.state.pos)
                rot_l_a.append(l_a.state.rot)
                rot_l_b.append(l_b.state.rot)
                length_l_a.append(torch.tensor(l_a.shape.length, device=self.device))
                length_l_b.append(torch.tensor(l_b.shape.length, device=self.device))
            # 堆叠为批量张量
            pos_l_a = torch.stack(pos_l_a, dim=-2)
            pos_l_b = torch.stack(pos_l_b, dim=-2)
            rot_l_a = torch.stack(rot_l_a, dim=-2)
            rot_l_b = torch.stack(rot_l_b, dim=-2)
            length_l_a = (
                torch.stack(
                    length_l_a,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            length_l_b = (
                torch.stack(
                    length_l_b,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            # 计算两条线段的最近点对
            point_a, point_b = _get_closest_points_line_line(
                pos_l_a,
                rot_l_a,
                length_l_a,
                pos_l_b,
                rot_l_b,
                length_l_b,
            )
            # 计算碰撞力
            force_a, force_b = self._get_constraint_forces(
                point_a,
                point_b,
                dist_min=LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            # 计算扭矩（力臂=最近点-线段中心）
            r_a = point_a - pos_l_a
            r_b = point_b - pos_l_b

            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
            # 应用碰撞力和扭矩
            for i, (entity_a, entity_b) in enumerate(l_l):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _box_sphere_vectorized_collision(self, b_s):
        # 盒-球向量化碰撞检测与碰撞力计算
        # 求球心到旋转盒的最近点，处理实心盒内部重叠情况
        if len(b_s):
            # 收集批量盒和球的参数
            pos_box = []
            pos_sphere = []
            rot_box = []
            length_box = []
            width_box = []
            not_hollow_box = []
            radius_sphere = []
            for box, sphere in b_s:
                pos_box.append(box.state.pos)
                pos_sphere.append(sphere.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                radius_sphere.append(
                    torch.tensor(sphere.shape.radius, device=self.device)
                )
            # 堆叠为批量张量
            pos_box = torch.stack(pos_box, dim=-2)
            pos_sphere = torch.stack(pos_sphere, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            radius_sphere = (
                torch.stack(
                    radius_sphere,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            # 计算球心到旋转盒的最近点
            closest_point_box = _get_closest_point_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_sphere,
            )

            # 处理实心盒：球在盒内时取盒内最近点
            inner_point_box = closest_point_box
            d = torch.zeros_like(radius_sphere, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    pos_sphere, closest_point_box, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = torch.where(not_hollow_box, d_hollow, d)

            # 计算碰撞力
            force_sphere, force_box = self._get_constraint_forces(
                pos_sphere,
                inner_point_box,
                dist_min=radius_sphere + LINE_MIN_DIST + d, # 含实心盒补偿距离
                force_multiplier=self._collision_force,
            )
            # 计算盒的扭矩（力臂=最近点-盒中心）
            r = closest_point_box - pos_box
            torque_box = TorchUtils.compute_torque(force_box, r)
            # 应用碰撞力和扭矩
            for i, (entity_a, entity_b) in enumerate(b_s):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _box_line_vectorized_collision(self, b_l):
        # 盒-线向量化碰撞检测与碰撞力计算
        # 求盒与线段的最近点对，处理实心盒内部重叠
        if len(b_l):
            # 收集批量盒和线的参数
            pos_box = []
            pos_line = []
            rot_box = []
            rot_line = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_line = []
            for box, line in b_l:
                pos_box.append(box.state.pos)
                pos_line.append(line.state.pos)
                rot_box.append(box.state.rot)
                rot_line.append(line.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                length_line.append(torch.tensor(line.shape.length, device=self.device))
            # 堆叠为批量张量
            pos_box = torch.stack(pos_box, dim=-2)
            pos_line = torch.stack(pos_line, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            rot_line = torch.stack(rot_line, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            length_line = (
                torch.stack(
                    length_line,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            # 计算盒与线段的最近点对
            point_box, point_line = _get_closest_line_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_line,
                rot_line,
                length_line,
            )

            # 处理实心盒：线段在盒内时取盒内最近点
            inner_point_box = point_box
            d = torch.zeros_like(length_line, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_line, point_box, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = torch.where(not_hollow_box, d_hollow, d)
            
            # 计算碰撞力
            force_box, force_line = self._get_constraint_forces(
                inner_point_box,
                point_line,
                dist_min=LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            # 计算扭矩（力臂=最近点-实体中心）
            r_box = point_box - pos_box
            r_line = point_line - pos_line

            torque_box = TorchUtils.compute_torque(force_box, r_box)
            torque_line = TorchUtils.compute_torque(force_line, r_line)
            # 应用碰撞力和扭矩
            for i, (entity_a, entity_b) in enumerate(b_l):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_line[:, i],
                    torque_line[:, i],
                )

    def _box_box_vectorized_collision(self, b_b):
        # 盒-盒向量化碰撞检测与碰撞力计算
        # 两个旋转盒的最近点对，处理实心盒内部重叠，向量化批量处理
        if len(b_b):
            # 收集批量盒的参数
            pos_box = []
            pos_box2 = []
            rot_box = []
            rot_box2 = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_box2 = []
            width_box2 = []
            not_hollow_box2 = []
            for box, box2 in b_b:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                pos_box2.append(box2.state.pos)
                rot_box2.append(box2.state.rot)
                length_box2.append(torch.tensor(box2.shape.length, device=self.device))
                width_box2.append(torch.tensor(box2.shape.width, device=self.device))
                not_hollow_box2.append(
                    torch.tensor(not box2.shape.hollow, device=self.device)
                )
            # 堆叠为批量张量
            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            pos_box2 = torch.stack(pos_box2, dim=-2)
            rot_box2 = torch.stack(rot_box2, dim=-2)
            length_box2 = (
                torch.stack(
                    length_box2,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box2 = (
                torch.stack(
                    width_box2,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box2_prior = torch.stack(
                not_hollow_box2,
                dim=-1,
            )
            not_hollow_box2 = not_hollow_box2_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            # 基于分离轴定理（SAT）计算两个旋转盒的最近点对
            point_a, point_b = _get_closest_box_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_box2,
                rot_box2,
                width_box2,
                length_box2,
            )
            # 处理实心盒A：盒2在盒1内时取盒1内最近点
            inner_point_a = point_a
            d_a = torch.zeros_like(length_box, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_b, point_a, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_a.shape)
                inner_point_a = torch.where(cond, inner_point_box_hollow, inner_point_a)
                d_a = torch.where(not_hollow_box, d_hollow, d_a)
            # 处理实心盒B：盒1在盒2内时取盒2内最近点
            inner_point_b = point_b
            d_b = torch.zeros_like(length_box2, device=self.device, dtype=torch.float)
            if not_hollow_box2_prior.any():
                inner_point_box2_hollow, d_hollow2 = _get_inner_point_box(
                    point_a, point_b, pos_box2
                )
                cond = not_hollow_box2.unsqueeze(-1).expand(inner_point_b.shape)
                inner_point_b = torch.where(
                    cond, inner_point_box2_hollow, inner_point_b
                )
                d_b = torch.where(not_hollow_box2, d_hollow2, d_b)
            # 计算碰撞力
            force_a, force_b = self._get_constraint_forces(
                inner_point_a,
                inner_point_b,
                dist_min=d_a + d_b + LINE_MIN_DIST, # 含两个实心盒的补偿距离
                force_multiplier=self._collision_force,
            )
            # 计算扭矩（力臂=最近点-盒中心）
            r_a = point_a - pos_box
            r_b = point_b - pos_box2
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
            # 应用碰撞力和扭矩
            for i, (entity_a, entity_b) in enumerate(b_b):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def collides(self, a: Entity, b: Entity) -> bool:
        # 碰撞预检测（粗检测）
        # 过滤逻辑：自身/非碰撞实体 → 静态实体对 → 形状组合不支持 → 距离超出外接圆半径和
        if (not a.collides(b)) or (not b.collides(a)) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        # 过滤静态实体对
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        # 过滤不支持的形状组合
        if not {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return False
        # 外接圆粗检测（距离>外接圆半径和则无碰撞）
        if not (
            torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=-1)
            <= a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        ).any():
            return False

        return True

    def _get_constraint_forces(
        # 计算约束力（如碰撞力和关节力）：force = sign × force_multiplier × (delta_pos/dist) × penetration
        self,
        pos_a: Tensor,
        pos_b: Tensor,
        dist_min,
        force_multiplier: float,
        attractive: bool = False,
    ) -> Tensor:
        min_dist = 1e-6 # 避免除零
        delta_pos = pos_a - pos_b # 相对位置向量
        dist = torch.linalg.vector_norm(delta_pos, dim=-1) # 欧氏距离
        sign = -1 if attractive else 1 # 吸引力为负，排斥力为正

        # softmax penetration 
        k = self._contact_margin # 接触裕度，控制力的平滑程度
        penetration = (
            torch.logaddexp( # logaddexp实现平滑过渡
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                (dist_min - dist) * sign / k,
            )
            * k
        )
        # 计算约束 force（方向为相对位置单位向量，大小与穿透深度成正比）
        force = (
            sign
            * force_multiplier
            * delta_pos
            / torch.where(dist > 0, dist, 1e-8).unsqueeze(-1) # 归一化方向
            * penetration.unsqueeze(-1) # 乘以穿透深度
        )
        # 距离过小时置零（避免数值爆炸）
        force = torch.where((dist < min_dist).unsqueeze(-1), 0.0, force)
        # 仅在需要时生效：排斥力（dist < dist_min），吸引力（dist > dist_min）
        if not attractive:
            force = torch.where((dist > dist_min).unsqueeze(-1), 0.0, force)
        else:
            force = torch.where((dist < dist_min).unsqueeze(-1), 0.0, force)
        return force, -force # 作用力与反作用力

    def _get_constraint_torques(
        # 关节固定旋转时计算旋转约束扭矩
        # 与约束 force 计算一致，通过软穿透实现平滑的扭矩调节
        self,
        rot_a: Tensor, # 实体A的旋转角张量 (batch_dim, ..., 1)
        rot_b: Tensor, # 目标旋转角张量(batch_dim, ..., 1)
        force_multiplier: float = TORQUE_CONSTRAINT_FORCE, # 扭矩放大系数
    ) -> Tensor:
        min_delta_rot = 1e-9 # 避免旋转角差过小时的数值异常
        delta_rot = rot_a - rot_b # 旋转角差值（当前角度 - 目标角度）
        abs_delta_rot = torch.linalg.vector_norm(delta_rot, dim=-1).unsqueeze(-1) # 旋转角差的绝对值

        # softmax penetration
        # 软穿透计算：用指数函数实现平滑的穿透深度，避免扭矩突变
        k = 1 # 平滑系数，调节扭矩随角度差的变化速率
        penetration = k * (torch.exp(abs_delta_rot / k) - 1)
        # 计算约束扭矩：方向与旋转角差一致，大小与软穿透深度、放大系数成正比
        torque = force_multiplier * delta_rot.sign() * penetration
        # 旋转角差过小时置零扭矩，避免数值抖动
        torque = torch.where((abs_delta_rot < min_delta_rot), 0.0, torque)

        return -torque, torque # 作用力与反作用力

    # integrate physical state
    # uses semi-implicit euler with sub-stepping
    # 根据累积的力和扭矩，更新实体的位置、速度、旋转角、角速度，包含速度阻尼和边界约束
    def _integrate_state(self, entity: Entity, substep: int):
        if entity.movable:
            # Compute translation
            # 线状态积分
            if substep == 0:
                # 速度阻尼
                if entity.drag is not None:
                    entity.state.vel = entity.state.vel * (1 - entity.drag)
                else:
                    entity.state.vel = entity.state.vel * (1 - self._drag)
            # 加速度计算：F=ma → a=F/m
            accel = self.forces_dict[entity] / entity.mass
            # 速度更新：v = v0 + a·Δt
            entity.state.vel = entity.state.vel + accel * self._sub_dt
            # 速度约束
            if entity.max_speed is not None:
                entity.state.vel = TorchUtils.clamp_with_norm(
                    entity.state.vel, entity.max_speed
                )
            if entity.v_range is not None:
                entity.state.vel = entity.state.vel.clamp(
                    -entity.v_range, entity.v_range
                )
            # 位置更新：x = x0 + v·Δt
            new_pos = entity.state.pos + entity.state.vel * self._sub_dt
            # 边界约束：限制实体在环境范围内
            entity.state.pos = torch.stack(
                [
                    (
                        new_pos[..., X].clamp(-self._x_semidim, self._x_semidim)
                        if self._x_semidim is not None
                        else new_pos[..., X]
                    ),
                    (
                        new_pos[..., Y].clamp(-self._y_semidim, self._y_semidim)
                        if self._y_semidim is not None
                        else new_pos[..., Y]
                    ),
                ],
                dim=-1,
            )

        if entity.rotatable:
            # Compute rotation
            if substep == 0:
                # 角速度阻尼
                if entity.drag is not None:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - entity.drag)
                else:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - self._drag)
            # 角加速度计算：τ=Iα → α=τ/I
            entity.state.ang_vel = (
                entity.state.ang_vel
                + (self.torques_dict[entity] / entity.moment_of_inertia) * self._sub_dt
            )
            # 旋转角更新：θ = θ0 + ω·Δt
            entity.state.rot = entity.state.rot + entity.state.ang_vel * self._sub_dt

    def _update_comm_state(self, agent):
        # set communication state (directly for now)
        # 将智能体的通信动作（action.c）赋值给状态（state.c），支持连续/离散通信
        # 注意：非静默智能体才更新通信状态
        if not agent.silent:
            agent.state.c = agent.action.c

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device) # # 迁移世界自身的批量维度和设备属性
        # # 迁移所有实体（智能体+地标）的张量
        for e in self.entities:
            e.to(device)
