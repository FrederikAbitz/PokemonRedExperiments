from typing import SupportsFloat
import numpy as np


class RecentRewardMemory:

    _channel_size: int
    _render_height: int
    _render_width: int
    _n_channels: int
    _reward_scaling_factor: float
    _channel_data: np.ndarray


    def __init__(self,
                 render_height: int = 8,
                 render_width: int = 40,
                 n_channels: int = 3,
                 reward_scaling_factor: float = 64,
                 ):
        self._height = render_height
        self._width = render_width
        self._channel_size = self._height * self._width
        self._n_channels = n_channels
        self._reward_scaling_factor = reward_scaling_factor
        self.reset()


    def reset(self):
        self._channel_data = np.zeros((self._channel_size, self._n_channels), dtype=np.uint8)


    def step_channels(self, delta_rewards_by_channel: tuple[SupportsFloat, ...]):
        # Roll channels
        self._channel_data = np.roll(self._channel_data, self._n_channels)

        # Add this tick's rewards
        assert len(delta_rewards_by_channel) == self._n_channels
        for ch, reward in enumerate(delta_rewards_by_channel):
            reward_mapped = int(reward * self._reward_scaling_factor)
            clamped_reward = max(0, min(255, reward_mapped))
            self._channel_data[0, ch] = clamped_reward


    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._height, self._width, self._n_channels)


    def render_observation(self) -> np.ndarray:
        """
        Renders the RecentRewardMemory data into an 8-bit array.
        """
        # Reshape to (height, width, channels)
        grid = self._channel_data.reshape((self._height, self._width, self._n_channels), order="F")
        return grid
