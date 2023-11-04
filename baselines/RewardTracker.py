from dataclasses import dataclass
from typing import Iterable


@dataclass
class RewardCfg:
    base_value: float = 0.0
    reward_per_value: float = 1.0



class RewardTracker:
    """
    A class for tracking and scaling rewards.

    This tracker maintains configurations for different types of rewards, computes deltas
    for certain reward types, applies scaling factors, and aggregates total and step-based
    rewards for use in the learning process.

    Attributes:
        _reward_scale (float): A global scaling factor applied to the sum of all step rewards.
        _reward_config (dict[str, RewardCfg]): A dictionary mapping reward names to their respective RewardCfg instances.
        _values_total (dict): A dictionary to hold the total accumulated values for rewards.
        _values_step (dict): A dictionary to hold the step-specific values for rewards.

    The key concepts are:
    - value: The raw registered values for each reward type before scaling.
    - reward: The scaled representation of values as specified by the reward configurations.
    - absolute: A reward type where the value is registered at regular intervals. The reward
      for a step is the delta between the current and the previous registered value, subject
      to scaling by the reward_per_value factor.
    - one-time: A reward type where values are expected to be transitory and not necessarily
      based on a difference from a previous value. One-time rewards are summed over time,
      with each registration being treated as a reward for the current step.

    The class provides methods for registering absolute and one-time values, calculating and
    retrieving step and total rewards, and resetting state. The following categories are present:

    - total_x: Methods that calculate and return the total accumulated rewards or values. For
      absolute rewards, the base or previous value is considered, whereas for one-time rewards,
      it is a sum over all registered values.
    - step_x: Methods that provide the reward or value for the current step. For absolute rewards,
      this is the delta computed from the change since the last step. For one-time rewards, it is
      the value registered in the current step.

    Scaling is applied in two stages:
    - global reward scale: Applied to the sum of step rewards and is transparent to the user. This
      can be used for global adjustment of reward magnitude.
    - reward_per_value: Configured per reward and applied as a factor to the raw values before they
      are summed and scaled globally. This allows for fine-tuning of individual reward contributions.
    """

    _reward_scale: float
    _reward_config: dict[str, RewardCfg]
    _values_total: dict
    _values_step: dict


    def __init__(self, scale: float = 1.0, rew_config: dict[str, RewardCfg] = {}):
        self._reward_scale = scale
        self._reward_config = rew_config
        self.reset()


    def reset(self):
        self._values_total = {name: cfg.base_value for name, cfg in self._reward_config.items()}
        self._values_step = {}


    def register_absolute(self, name: str, value: float, decrease_allowed: bool = True):
        """Save the total value for a reward.

        The step reward will be the difference from the previous to the registered value, multiplied by the configured factor.
        If no previous value is known, the registered value will be saved and the step reward will be zero.

        If `decrease_allowed` is `False`,
        the registered value will be ignored if it is lower than the previous value."""
        prev = self._values_total.setdefault(name, value)
        if value > prev or decrease_allowed:
            self._values_step[name] = value - prev


    def register_onetime(self, name: str, value: float = 1.0):
        """Add a one-time value for reward.

        The step reward will be the registered value, multiplied by the configured factor.
        If used multiple times per step with the same `name`, the value will be overwritten."""
        self._values_total.setdefault(name, 0.0)
        self._values_step[name] = value


    def apply_deltas_and_get_scaled_step_reward(self) -> float:
        """Mark the end of the current step and return the scaled sum of registered step rewards.

        Call this and the end of the environment step."""
        # Apply step values
        self._values_total = self.total_values()

        # Scale, sum and reset step rewards
        sum_individually_scaled_step_values = 0
        for name, step_val in self._values_step.items():
            step_rew = step_val
            rew_cfg = self._reward_config.get(name)
            if rew_cfg is not None:
                step_rew *= rew_cfg.reward_per_value
            sum_individually_scaled_step_values += step_rew

        step_reward = self._reward_scale * sum_individually_scaled_step_values
        self._values_step = {}

        return step_reward


    def step_rewards(self) -> dict[str, float]:
        """Return the scaled step reward values by names."""
        scaled_step_rewards = {}
        for name, step_val in self.step_values().items():
            rew_cfg = self._reward_config.get(name, RewardCfg())
            scaled_step_rewards[name] = step_val * rew_cfg.reward_per_value
        return scaled_step_rewards


    def step_reward(self, name: str = None, default: float = 0.0) -> float:
        """Return the scaled step reward value for `name`.

        If `name` is `None`, returns the sum of all scaled step rewards."""
        if name is None:
            return sum(self.step_rewards().values())
        else:
            rew_cfg = self._reward_config.get(name, RewardCfg())
            return (self.step_values().get(name, default)) * rew_cfg.reward_per_value


    def total_rewards(self) -> dict[str, float]:
        """Return the scaled and base-adjusted total reward values by names."""
        scaled_total_rewards = {}
        for name, total_val in self.total_values().items():
            rew_cfg = self._reward_config.get(name, RewardCfg())
            scaled_total_rewards[name] = (total_val - rew_cfg.base_value) * rew_cfg.reward_per_value
        return scaled_total_rewards


    def total_reward(self, name: str = None, default: float = 0.0) -> float:
        """Return the scaled and base-adjusted total reward value for `name`.

        If `name` is `None`, returns the sum of all scaled and base-adjusted total rewards."""
        if name is None:
            return sum(self.total_rewards().values())
        else:
            rew_cfg = self._reward_config.get(name, RewardCfg())
            total_val = self.total_values().get(name, default)
            return (total_val - rew_cfg.base_value) * rew_cfg.reward_per_value


    def step_values(self) -> dict[str, float]:
        """The unscaled step reward values by names.

        The step value is the the difference between the current and the last registerd absolute value.
        For one-time rewards, the step value is the value registered this step."""
        return self._values_step


    def step_value(self, name: str = None, default: float = 0.0) -> float:
        """The unscaled step reward value for `name`.

        Returns the sum of unscaled step rewards if `name` is `None`"""
        if name is None:
            return sum(self._values_step.values())
        else:
            return self._values_step.get(name, default)


    def total_values(self) -> dict[str, float]:
        """Total unscaled total reward values by names.

        The total value is the last registerd absolute value.
        For one-time rewards, the total value is the sum."""
        return {name: self._values_total[name] + self._values_step.get(name, 0)
                for name in self._values_total.keys()}


    def total_value(self, name: str = None, default: float = 0.0) -> float:
        """The unscaled total reward value for `name`.

        Returns the sum of unscaled total rewards if `name` is `None`"""
        if name is None:
            return sum(self.total_values().values())
        else:
            return self._values_total.get(name, default) + self._values_step.get(name, 0)
