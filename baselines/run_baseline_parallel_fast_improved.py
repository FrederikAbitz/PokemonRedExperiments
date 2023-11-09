import argparse
import uuid
import json
from os.path import exists, join
from pathlib import Path
from glob import glob

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

default_config = {
    "ep_length": 20480,
    "headless": True,
    "reward_scale": 4,
    "explore_weight": 3,
    "init_state": '../has_pokedex_nballs.state',
    "n_envs": 4,
    "ep_updates": 8,
    'extra_buttons': False,
    'save_video': False,
}


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument("--ep_length", type=int, default=2048 * 10)
    parser.add_argument("--ep_updates", type=int, default=8, help="Model updates per episode. Recommended for 20480 ep_length: 8 (default)")
    parser.add_argument("--headless", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--extra_buttons", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--save_video", action="store_true", help="Save videos of the training. Makes training VERY slow!")
    parser.add_argument("--only_video", action="store_true", help="Only save videos of the model playing. No training.")
    parser.add_argument("--reward_scale", type=float, default=4)
    parser.add_argument("--explore_weight", type=float, default=3)
    parser.add_argument("--explore_type", type=lambda x: str(x).lower(), choices=["screen", "coords"], default="screen", help="Type of exploration reward: 'screen' with k-NN or 'coords'.")
    parser.add_argument("--init_state", type=str, default='../has_pokedex_nballs.state')
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--load", type=str, help="Path to the session folder to load")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the loaded config with the current command line arguments")
    return parser.parse_args()

def load_config(sess_path):
    config_path = join(sess_path, 'train_config.json')
    if exists(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)
    return {}

def save_config(sess_path, config, default_config):
    sess_path.mkdir(parents=True, exist_ok=True)
    with open(join(sess_path, 'train_config.json'), 'w') as file:
        json.dump(config, file, indent=4)
    print("Configuration saved to:", sess_path / 'train_config.json')
    for key, value in config.items():
        default = default_config.get(key, 'No default')
        print(f"'{key}': {value}" + (f" [default: {default}]" if value != default else ""))


def find_latest_checkpoint(sess_path):
    checkpoints = glob(join(sess_path, 'poke_*.zip'))
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.split('_')[-2]))
    return None

if __name__ == '__main__':
    args = parse_args()

    config = default_config.copy()

    if args.load:
        sess_path = Path(args.load)
        loaded_config = load_config(sess_path)
        config.update(loaded_config)
    else:
        sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    # Override with command line arguments
    cmd_config = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cmd_config)

    # Save the new config if --overwrite is used
    if args.overwrite:
        save_config(sess_path, config, default_config)

    env_config = {
        'headless': config['headless'],
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': config['init_state'],
        'max_steps': config['ep_length'],
        'print_rewards': True,
        'save_video': config['save_video'],
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True if config['explore_type'] == 'screen' else False,
        'reward_scale': config['reward_scale'],
        'extra_buttons': config['extra_buttons'],
        'explore_weight': config['explore_weight']
    }


    n_envs = config['n_envs']
    env = SubprocVecEnv([make_env(i, env_config) for i in range(n_envs)])
    checkpoint_callback = CheckpointCallback(save_freq=config['ep_length'], save_path=str(sess_path), name_prefix='poke')

    learn_steps = 40
    file_name = find_latest_checkpoint(sess_path) if args.load else 'session_e41c9eff/poke_38207488_steps'

    if file_name and exists(file_name):
        print('\nLoading checkpoint:', file_name)
        model = PPO.load(file_name, env=env)
        model.n_steps = config['ep_length'] // config['ep_updates']
        model.n_envs = n_envs
        model.rollout_buffer.buffer_size = config['ep_length'] // config['ep_updates']
        model.rollout_buffer.n_envs = n_envs
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=config['ep_length'] // config['ep_updates'], batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)

    only_run_and_save_video = False
    if only_run_and_save_video or args.only_video:
        for _ in range(learn_steps):
            obs = env.reset()
            done = [False] * n_envs
            _step = 0
            while not all(done):
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, done, info = env.step(action)
                _step += 1
    else:
        for i in range(learn_steps):
            model.learn(total_timesteps=config['ep_length'] * n_envs * 1000, callback=CallbackList([checkpoint_callback, TensorboardCallback()]))
