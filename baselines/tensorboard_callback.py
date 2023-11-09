from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
import numpy as np
from einops import rearrange

def merge_dicts_by_mean(dicts):
    sum_dict = {}
    count_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):  # Handle non-dictionary values
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
            elif isinstance(v, dict):  # Handle nested dictionaries
                if k not in sum_dict:
                    sum_dict[k] = {}
                    count_dict[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):  # Make sure the value is numeric
                        sum_dict[k][sub_k] = sum_dict[k].get(sub_k, 0) + sub_v
                        count_dict[k][sub_k] = count_dict[k].get(sub_k, 0) + 1

    mean_dict = {}
    for k, v in sum_dict.items():
        if isinstance(v, dict):  # Handle nested dictionaries
            mean_dict[k] = {}
            for sub_k, sub_v in v.items():
                mean_dict[k][sub_k] = sub_v / count_dict[k][sub_k]
        else:  # Handle non-dictionary values
            mean_dict[k] = v / count_dict[k]

    return mean_dict


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos = merge_dicts_by_mean(all_final_infos)

            # Iterate over each key in the dictionary
            for key, value in mean_infos.items():
                if isinstance(value, dict):  # Nested dictionary
                    for sub_key, sub_value in value.items():
                        self.logger.record(f"env_stats-{key}/{sub_key}", sub_value)
                else:  # Default category
                    self.logger.record(f"env_stats/{key}", value)

            images = self.training_env.env_method("render") # use reduce_res=False for full res screens
            images_arr = np.array(images)
            images_row = rearrange(images_arr, "b h w c -> h (b w) c")
            self.logger.record("trajectory/image", Image(images_row, "HWC"), exclude=("stdout", "log", "json", "csv"))

