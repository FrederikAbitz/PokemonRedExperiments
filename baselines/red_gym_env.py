
from dataclasses import dataclass
import math
import sys
import uuid
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from typing import List
from collections import deque

from MapTracker import MapTracker
from RecentRewardMemory import RecentRewardMemory
from ScreenKnnExplorationRewarder import ScreenEnnExplorationRewarder
from GamePositionExplorationRewarder import GamePositionExplorationRewarder
from RewardTracker import RewardCfg, RewardTracker
from GameStateValueTracker import GameStateValueTracker, ValueCfg


MAP_LOCATIONS = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    3: "Cerulean City",
    12: "Route 1",
    13: "Route 2",
    14: "Route 3",
    15: "Route 4",
    33: "Route 22",
    37: "Red house first",
    38: "Red house second",
    39: "Blues house",
    40: "oaks lab",
    41: "Pokémon Center (Viridian City)",
    42: "Poké Mart (Viridian City)",
    43: "School (Viridian City)",
    44: "House 1 (Viridian City)",
    47: "Gate (Viridian City/Pewter City) (Route 2)",
    49: "Gate (Route 2)",
    50: "Gate (Route 2/Viridian Forest) (Route 2)",
    51: "viridian forest",
    52: "Pewter Museum (floor 1)",
    53: "Pewter Museum (floor 2)",
    54: "Pokémon Gym (Pewter City)",
    55: "House with disobedient Nidoran♂ (Pewter City)",
    56: "Poké Mart (Pewter City)",
    57: "House with two Trainers (Pewter City)",
    58: "Pokémon Center (Pewter City)",
    59: "Mt. Moon (Route 3 entrance)",
    60: "Mt. Moon",
    61: "Mt. Moon",
    68: "Pokémon Center (Route 4)",
    193: "Badges check gate (Route 22)"
}


@dataclass(frozen=True)
class SeenCoordinate:
    map_id: int
    x: int
    y: int

class RedGymEnv(Env):


    def __init__(
        self,
        config
        ):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.headless = config['headless']
        self.knn_vec_dim = 4320 #1000
        self.knn_max_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.save_video_framebuffer_size = 600
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        """Factor by which the PyBoy frames are downsampled along width and height"""
        self.frame_stacks = 3

        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.explore_reward_scale = 1 if 'explore_weight' not in config else config['explore_weight']

        self.use_knn_exploration_reward = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.knn_similar_frame_dist = config['sim_frame_dist']
        self.use_position_exploration_reward = not self.use_knn_exploration_reward

        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []


        ### Actions and Action Space

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))


        ### Emulator

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)

        ### Observations and Observation Space

        SCREEN_WIDTH, SCREEN_HEIGHT = self.screen.raw_screen_buffer_dims()
        FRAME_SHAPE = np.array((SCREEN_HEIGHT, SCREEN_WIDTH, 3))

        #self.lowres_frame_shape = np.divide(FRAME_SHAPE, ) (36, 40, 3)
        self.lowres_frame_shape = (36, 40, 3)
        self.mem_padding = 2
        self.bars_lvl_hp_explore_heigt = 8
        self.memory_height = 8
        self.map_tracker_height = self.lowres_frame_shape[0]  # Height of the MapTracker (movement + presence view)

        # Height of the observation space
        obs_height = (self.map_tracker_height + self.mem_padding + 
                            self.memory_height + self.mem_padding + 
                            self.bars_lvl_hp_explore_heigt + self.mem_padding + 
                            self.lowres_frame_shape[0] * self.frame_stacks)

        # Width of the observation space
        self.output_full = (obs_height, self.lowres_frame_shape[1], self.lowres_frame_shape[2])
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)


        ### Game State and Reward Tracking

        # Movement and Presence Tracker
        self.map_tracker = MapTracker()

        # State Tracker
        self.state_tracker = GameStateValueTracker(val_config={
            'pos': ValueCfg(data_type=tuple, default_value=(0, 0)),
            'map_id': ValueCfg(data_type=int, default_value=0),
            'money': ValueCfg(data_type=int),
            'party_size': ValueCfg(data_type=int, metrics={
                'changed?': lambda curr, prev, metric_prev, count: curr != prev
                }),
            'party_levels': ValueCfg(data_type=tuple),
            'opponent_party_levels': ValueCfg(data_type=tuple, default_value=5),
            'party_hp_max': ValueCfg(data_type=tuple),
            'party_hp_curr': ValueCfg(data_type=tuple),
            'party_hp_fraction': ValueCfg(data_type=float),
            'badges': ValueCfg(data_type=tuple, default_value=([False] * 8)),
        })

        # Recent Reward Memory
        self.recent_reward_memory = RecentRewardMemory(
            render_height=self.memory_height,
            render_width=self.lowres_frame_shape[1],
            n_channels=3,
            reward_scaling_factor=64
        )

        # Exploration Reward Management
        self.knn_exploration_rewarder = ScreenEnnExplorationRewarder(
            vec_dim=self.knn_vec_dim,
            max_elements=self.knn_max_elements,
            similar_frame_dist=self.knn_similar_frame_dist,
            reward_per_element=0.005
        )

        self.position_exploration_rewarder = GamePositionExplorationRewarder(
            reward_per_cell=0.08,
            cell_size=2,
        )

        self.rewards = RewardTracker(
            scale = self.reward_scale * 0.1,
            rew_config = {
                'event': RewardCfg(reward_per_value = 4),
                'level': RewardCfg(base_value = 6),
                'heal': RewardCfg(reward_per_value = 4),
                'op_lvl': RewardCfg(reward_per_value = 0.2),
                'dead': RewardCfg(reward_per_value = -0.1),
                'badge': RewardCfg(reward_per_value = 20),
                'explore_pos': RewardCfg(reward_per_value = self.explore_reward_scale),
                'explore_knn': RewardCfg(reward_per_value = self.explore_reward_scale),
                }
            )


        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.reset()


    def reset(self, seed=None):
        self.seed = seed
        self.pixels = np.zeros((*self.screen.raw_screen_buffer_dims(), 3))
        self.map_tracker.reset()
        self.state_tracker.reset()
        self.recent_reward_memory.reset()

        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.use_knn_exploration_reward:
            self.knn_exploration_rewarder.reset()
            self.knn_exploration_rewarder.reward_per_element = 0.005

        if self.use_position_exploration_reward:
            self.position_exploration_rewarder.reset()
            self.position_exploration_rewarder.reward_per_cell = 0.08

        self.recent_frames = np.zeros(
            (self.frame_stacks, self.lowres_frame_shape[0], 
             self.lowres_frame_shape[1], self.lowres_frame_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
            self.full_frame_buffer = deque()
            self.model_frame_buffer = deque()
       
        self.exploration_reset_level_threshold = 28
        self.exploration_reset_level_threshold_reached = False
        self.died_count = 0
        self.step_count = 0
        self.rewards.reset()
        self.reset_count += 1
        return self.render_observation(), {}


    def render(self):
        return self.render_observation()


    def render_observation(self):
        # Generate Map Tracker View
        x, y = self.state_tracker.curr('pos')
        map_tracker_view = self.map_tracker.get_view_8bit_rgb(
            self.state_tracker.curr('map_id'),
            x,
            y,
        )

        # Padding
        pad = np.zeros(shape=(self.mem_padding, self.lowres_frame_shape[1], 3), dtype=np.uint8)

        # Combine components
        combined_view = np.concatenate(
            (
                self.render_bars_lvl_hp_exlore(),
                pad,
                self.recent_reward_memory.render_observation(),
                pad,
                map_tracker_view,
                pad,
                *self.recent_frames
            ),
            axis=0)

        return combined_view


    def get_pixels(self, reduce_res):
        """Game screen pixels as int8 ndarray, shape=(x, y, 3)"""
        # Get game pixels
        game_pixels_render = self.pixels
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.lowres_frame_shape)).astype(np.uint8)

        return game_pixels_render



    @staticmethod
    def make_reward_channel(r_val, w: int, h: int, col_steps: int):
        max_r_val = (w-1) * h * col_steps
        # truncate progress bar. if hitting this
        # you should scale down the reward in group_rewards!
        r_val = min(r_val, max_r_val)
        row = floor(r_val / (h * col_steps))
        memory = np.zeros(shape=(h, w), dtype=np.uint8)
        memory[:, :row] = 255
        row_covered = row * h * col_steps
        col = floor((r_val - row_covered) / col_steps)
        memory[:col, row] = 255
        col_covered = col * col_steps
        last_pixel = floor(r_val - row_covered - col_covered) 
        memory[col, row] = last_pixel * (255 // col_steps)
        return memory


    def render_bars_lvl_hp_exlore(self):
        w = self.lowres_frame_shape[1]
        h = self.memory_height

        level_reward = self.rewards.total_value('level') * 100 
        hp_reward = self.state_tracker.curr('party_hp_fraction', 1.0, True) * 2000
        explore_reward = 150 * (self.rewards.total_value('explore_pos') + self.rewards.total_value('explore_knn'))

        full_memory = np.stack((
            self.make_reward_channel(level_reward, w, h, 16),
            self.make_reward_channel(hp_reward, w, h, 16),
            self.make_reward_channel(explore_reward, w, h, 16)
        ), axis=-1)

        if sum(self.state_tracker.curr('badges')) > 0:
            full_memory[:, -1, :] = 255

        return full_memory


    def register_emulator_values(self):
        # Register the player's position
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        self.state_tracker.register_value('pos', (x_pos, y_pos))

        # Register the map ID
        self.state_tracker.register_value('map_id', self.read_m(0xD35E))

        # Register player's money
        offsets = (0xD347, 0xD348, 0xD349)
        money = (100 * 100 * ((self.read_m(offsets[0]) >> 4) * 10 + (self.read_m(offsets[0]) & 0xF)) +
                 100 * ((self.read_m(offsets[1]) >> 4) * 10 + (self.read_m(offsets[1]) & 0xF)) +
                 ((self.read_m(offsets[2]) >> 4) * 10 + (self.read_m(offsets[2]) & 0xF)))
        self.state_tracker.register_value('money', money)

        # Register party size
        self.state_tracker.register_value('party_size', self.read_m(0xD163))

        # Register party pokemon
        party_pokemon = tuple(self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169])
        self.state_tracker.register_value('party_pokemon', party_pokemon)

        # Register party levels
        party_levels = tuple(max(self.read_m(addr), 0) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268])
        self.state_tracker.register_value('party_levels', party_levels)

        # Register opponent party levels
        opponent_party_levels = tuple(max(self.read_m(addr), 0) for addr in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1])
        self.state_tracker.register_value('opponent_party_levels', opponent_party_levels)

        # Helper function to read HP using the formula from read_hp method
        read_hp = lambda start_addr: 256 * self.read_m(start_addr) + self.read_m(start_addr + 1)

        # Register HP details for party
        party_hp_max = tuple(read_hp(addr) for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269])
        party_hp_curr = tuple(read_hp(addr) for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248])
        self.state_tracker.register_value('party_hp_max', party_hp_max)
        self.state_tracker.register_value('party_hp_curr', party_hp_curr)

        # Calculate and register total HP fraction for the party
        if sum(party_hp_max) > 0:
            party_hp_fraction = sum(party_hp_curr) / sum(party_hp_max)
        else:
            party_hp_fraction = 0
        self.state_tracker.register_value('party_hp_fraction', party_hp_fraction)

        # Convert the badge byte into a tuple of booleans and register
        badge_byte = self.read_m(0xD356)
        badges = tuple((badge_byte & (1 << i)) != 0 for i in range(8))
        self.state_tracker.register_value('badges', badges)


    def calculate_rewards(self):
        # Event reward
        self.rewards.register_absolute('event', self.read_num_events_for_reward_calc(), decrease_allowed=False)

        # HP fraction reward (???; Only for HP bars probably. Can be removed here?)
        # self.rewards.register_absolute('hp_fraction', self.state_tracker.curr('party_hp_fraction'))

        # Levels reward
        lvl_threshold = 28
        excess_lvl_factor = .25
        lvl_sum = sum(self.state_tracker.curr("party_levels"))
        levels_for_reward = min(lvl_sum, lvl_threshold) + excess_lvl_factor * max(0, lvl_sum - lvl_threshold)
        self.rewards.register_absolute('level', levels_for_reward, decrease_allowed=False)

        # Max opponent level reward
        opponent_level = max(self.state_tracker.curr('opponent_party_levels')) - 5
        self.rewards.register_absolute('op_lvl', opponent_level, decrease_allowed=False)

        # Exploration (screen k-NN) reward
        if self.use_knn_exploration_reward:
            self.rewards.register_absolute('explore_knn', self.knn_exploration_rewarder.get_total_reward())

        # Exploration (coords) reward
        if self.use_position_exploration_reward:
            self.rewards.register_absolute('explore_pos', self.position_exploration_rewarder.get_total_reward())

        # Badge reward
        self.rewards.register_absolute('badge', sum(self.state_tracker.curr("badges")))

        # Healing and death reward
        old_health = self.state_tracker.prev('party_hp_fraction', 1.0)
        new_health = self.state_tracker.curr('party_hp_fraction', 1.0)
        party_size_changed = self.state_tracker.metric('party_size', 'changed?', False)

        if (new_health > old_health > 0 and not party_size_changed):
            # HP fraction increased while party size didn't change.
            # Therefore, the player must have healed his pokemon.
            healed_hp_fraction = new_health - old_health
            self.rewards.register_onetime('heal', healed_hp_fraction)

            if healed_hp_fraction > 0.5:
                print(f'healed: {healed_hp_fraction}')
                self.save_screenshot('healing')

        if (new_health > old_health == 0 and not party_size_changed):
            # HP fraction increased, which was 0 before.
            # Therefore, the player died and was revived.
            self.died_count += 1
            self.rewards.register_onetime('dead')


    def step(self, action):

        ### Execute action

        # Press and release button, tick emulator
        self.run_action_on_emulator(action)

        ### Read from emulator and update cache

        # Screen
        self.pixels = self.screen.screen_ndarray()  # (144, 160, 3)

        # Update rolling frame buffer
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.recent_frames[0] = self.get_pixels(reduce_res=True)

        # Read and cache game state
        self.register_emulator_values()

        # Update movement tracker
        self.map_tracker.register_step_to(
            self.state_tracker.curr("map_id"),
            self.state_tracker.curr("pos")
            )

        # Update data for exploration reward: k-NN index or seen coordinates
        if (sum(self.state_tracker.curr("party_levels")) >= self.exploration_reset_level_threshold
                and not self.exploration_reset_level_threshold_reached):
            self.exploration_reset_level_threshold_reached = True
            if self.use_knn_exploration_reward:
                self.knn_exploration_rewarder.reset_knn_index()
                self.knn_exploration_rewarder.reward_per_element = 0.01
            if self.use_position_exploration_reward:
                self.position_exploration_rewarder.reset_visited_cells()
                self.position_exploration_rewarder.reward_per_cell = 0.01

        if self.use_knn_exploration_reward:
            pixels_flat = self.get_pixels(reduce_res=True).flatten().astype(np.float32)
            self.knn_exploration_rewarder.register_frame(pixels_flat)

        if self.use_position_exploration_reward:
            self.position_exploration_rewarder.register_step(
                self.state_tracker.curr("map_id"),
                self.state_tracker.curr("pos")
            )

        ### Reward Calculation

        self.calculate_rewards()

        ### Build Observation

        # Update Recent Reward Short Term Memory
        self.recent_reward_memory.step_channels((
            self.rewards.step_reward('level'),
            self.rewards.step_reward('hp_fraction'),
            150 * (self.rewards.step_reward('explore_pos') + self.rewards.step_reward('explore_knn'))
        ))

        # Render observation
        obs = self.render_observation()

        ### Statistics, Logging, Housekeeping, Any Extra Functionality

        # Update agent stats
        agent_stats = {
            'step': self.step_count,
            'x': self.state_tracker.curr('pos')[0],
            'y': self.state_tracker.curr('pos')[1],
            'map': self.state_tracker.curr('map_id'),
            'map_location': MAP_LOCATIONS.get(self.state_tracker.curr('map_id'), "Unknown Location"),
            'last_action': action,
            'pcount': self.state_tracker.curr('party_size'),
            'levels': self.state_tracker.curr('party_levels'),
            'levels_sum': sum(self.state_tracker.curr('party_levels')),
            'ptypes': self.state_tracker.curr('party_pokemon'),
            'hp': self.state_tracker.curr('party_hp_fraction'),
            'deaths': self.died_count,
            'badge': self.state_tracker.metric('badges', 'earned'),
            'event': self.rewards.total_reward('event'),
            'healr': self.rewards.total_reward('heal')
        }
        if self.use_knn_exploration_reward:
            agent_stats['frames'] = self.knn_exploration_rewarder.get_current_count()
        if self.use_position_exploration_reward:
            agent_stats['coord_count'] = self.position_exploration_rewarder.get_visited_cell_count()

        self.agent_stats.append(agent_stats)


        done_ep_is_over = self.check_if_done()

        # Save Screenshot if reward went down without dying
        if sum(self.rewards.step_values().values()) < 0 and self.rewards.step_value('dead') == 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')


        # Save Video
        if self.save_video:
            frame_full = self.screen.screen_ndarray()
            frame_small = obs

            # Add frames to buffer
            self.full_frame_buffer.append(frame_full)
            self.model_frame_buffer.append(frame_small)

            # Write to file when episode over or buffer full
            if done_ep_is_over or len(self.full_frame_buffer) >= self.save_video_framebuffer_size:
                while self.full_frame_buffer:
                    self.full_frame_writer.add_image(self.full_frame_buffer.popleft())
                while self.model_frame_buffer:
                    self.model_frame_writer.add_image(self.model_frame_buffer.popleft())

            # Close file when episode is over
            if done_ep_is_over:
                self.full_frame_writer.close()
                self.model_frame_writer.close()


        self.save_and_print_info(done_ep_is_over, obs)

        self.step_count += 1

        return obs, self.rewards.apply_deltas_and_get_scaled_step_reward(), False, done_ep_is_over, {}


    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()


    @staticmethod
    def calculate_seen_coords_scaling(num_seen: int) -> float:
        """ Starts with steeper, almost linear ascent and gets more flat as the number grows,
        to avoid exploiting explore reward too much
        f\left(x\right)=\frac{\left(x+3x\cdot e^{-0.002x}\right)}{4}
        """
        return (num_seen + 3 * num_seen * math.exp(-0.0025 * num_seen)) / 4


    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_reward_memory._channel_data.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        return done


    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            reward_string = f'step: {self.step_count:6d}'
            for key, val in self.rewards.total_rewards().items():
                reward_string += f' {key}: {val:5.2f}'
            reward_string += f' sum: {self.rewards.total_reward():5.2f}'
            print(f'\r{reward_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.get_pixels(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.rewards.total_reward():.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.rewards.total_reward():.4f}_{self.reset_count}_full.jpeg'), 
                    self.get_pixels(reduce_res=False))

        if done:
            self.all_runs.append(self.rewards.total_rewards())
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')


    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)


    def read_bit(self, addr, bit: int) -> bool:
        return (self.read_m(addr) & (1 << bit)) != 0


    def read_num_events_for_reward_calc(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )


    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.rewards.total_reward():.4f}_{self.reset_count}_{name}.jpeg'), 
            self.get_pixels(reduce_res=False))
        plt.imsave(
            ss_dir / Path(f'observation{self.instance_id}_r{self.rewards.total_reward():.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render_observation())


    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
