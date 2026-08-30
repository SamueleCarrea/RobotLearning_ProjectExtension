"""Implementation of the Hopper environment supporting
domain randomization optimization."""

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class CustomHopper(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        domain_randomization: bool = False,
        randomization_range: float = 0.15,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            domain_randomization,
            randomization_range,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._domain_randomization = domain_randomization
        self._randomization_range = randomization_range

        if xml_file == "hopper.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        self.original_masses = np.copy(self.model.body_mass[1:])  # Default link masses

        if (
            domain == "source"
        ):  # Source environment has an imprecise torso mass (1kg shift)
            self.model.body_mass[1] -= 1.0

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        # Custom domain randomization
        if self._domain_randomization:
            self.set_random_parameters()

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses by sampling from the domain randomization distribution"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        N.B. You can't change the mass of the torso (first link)

        Randomizes the masses of thigh, leg, and foot using uniform distributions
        around their original values. The torso mass remains fixed at the source value.

        Returns:
            np.array: Array of 4 masses [torso, thigh, leg, foot]
        """
        # Original masses: [torso, thigh, leg, foot]
        original = self.original_masses.copy()

        # Define randomization ranges (as percentage of original mass)
        # the range is a hyperparameter, configurable per env id
        randomization_range = self._randomization_range

        # Sample new masses for thigh, leg, and foot (NOT torso)
        # Torso mass stays at original_masses[0] which is already adjusted for source domain
        sampled_masses = np.zeros(4)
        sampled_masses[0] = self.model.body_mass[
            1
        ]  # Keep current torso mass (source or target)

        # Randomize thigh, leg, foot masses
        for i in range(1, 4):  # indices 1, 2, 3 correspond to thigh, leg, foot
            low = original[i] * (1.0 - randomization_range)
            high = original[i] * (1.0 + randomization_range)
            sampled_masses[i] = self.np_random.uniform(low, high)
        return sampled_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.model.body_mass[1:] = task


"""
    Registered environments
"""
gym.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"},
)

gym.register(
    id="CustomHopper-source-udr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source", "domain_randomization": True},
)

# stessa UDR con ampiezze diverse, per provare piu' distribuzioni
for _pct in (5, 25, 50):
    gym.register(
        id=f"CustomHopper-source-udr{_pct}-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={
            "domain": "source",
            "domain_randomization": True,
            "randomization_range": _pct / 100.0,
        },
    )
