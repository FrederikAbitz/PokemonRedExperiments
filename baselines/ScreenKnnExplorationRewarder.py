from typing import SupportsFloat
import numpy as np
import hnswlib


class ScreenEnnExplorationRewarder:

    _knn_index: hnswlib.Index
    _knn_vec_dim: int
    _knn_max_elements: int
    _knn_similar_frame_dist: float
    _reward_per_element: float

    _total_reward_curr: float
    _total_reward_prev: float
    _delta_reward_step: float


    def __init__(self,
                 vec_dim: int = 4320,
                 max_elements: int = 20000,
                 similar_frame_dist: float = 20000,
                 reward_per_element: float = 0.01
                 ):

        self._knn_vec_dim = vec_dim
        self._knn_max_elements = max_elements
        self._knn_similar_frame_dist = similar_frame_dist
        self._reward_per_element = reward_per_element
        self.reset()


    def reset(self):
        self._total_reward_curr = 0
        self._total_reward_prev = 0
        self._delta_reward_step = 0
        self.reset_knn_index()


    def reset_knn_index(self):
        """Reset the k-NN index, keep current total reward.

        Previously explored areas whose frames were part of the k-NN index and counted towards the reward
        will be rewarded again and counted towards the total. This reproduces the behavior from @PWhiddy's
        original implementation of the `RedGymEnv` class,
        e.g. at commit `f8a7fdfa08b01fa89d0cb76f06d681506ac4115d`."""
        self._knn_index = hnswlib.Index(space='l2', dim=self._knn_vec_dim)
        self._knn_index.init_index(
            max_elements=self._knn_max_elements,
            ef_construction=100,
            M=16
            )


    def register_frame(self, flattened_frame: np.ndarray) -> bool:
        self._total_reward_prev = self._total_reward_curr
        reward = 0.0
        add_frame = False

        if self._knn_index.get_current_count() == 0:
            add_frame = True
        else:
            labels, distances = self._knn_index.knn_query(flattened_frame, k = 1)
            if distances[0][0] > self._knn_similar_frame_dist:
                add_frame = True

        if add_frame:
            self._knn_index.add_items(flattened_frame, np.array([self._knn_index.get_current_count()]))
            reward = self._reward_per_element

        self._delta_reward_step = reward
        self._total_reward_curr = self._total_reward_prev + reward
        return add_frame


    def get_total_reward(self) -> float:
        return self._total_reward_curr


    def get_delta_reward(self) -> float:
        return self._delta_reward_step


    def get_current_count(self) -> int:
        return self._knn_index.get_current_count()


    @property
    def reward_per_element(self) -> float:
        return self._reward_per_element

    @reward_per_element.setter
    def reward_per_element(self, reward_per_element: float):
        self._reward_per_element = reward_per_element

