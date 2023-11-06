import numpy as np

from PokemonMove import MOVE_BY_ID


def render_progress_bar(observation: np.ndarray, value: float, max_value: float, y: int, x: int, ch: int, height: int = 8, width: int = 1):
    """
    Fills a section of a (h, w, 3) uint8 image array with a vertical progress bar representation. 
    A filled ratio determined by `value` and `max_value` is rendered in a specified channel `ch` and location `(y, x)`.
    Intensity within the bar is maxed (255) for fully covered pixels and scaled for the uppermost partial pixel.

    Parameters:
    - observation (np.ndarray): Target image array for in-place progress bar rendering.
    - value (float): Current value to visualize.
    - max_value (float): Maximum value defining the bar scale.
    - y (int): Vertical start position for the bar in the array.
    - x (int): Horizontal start position for the bar in the array.
    - ch (int): Color channel index for rendering (0, 1, 2 for R, G, B).
    - height (int, optional): Vertical size of the bar (default 8).
    - width (int, optional): Horizontal thickness of the bar (default 1).
    """
    stat_ratio = value / max(1, max_value)
    full_stat_pixels = int(stat_ratio * height)
    partial_stat_pixel_intensity = int((stat_ratio * height - full_stat_pixels) * 255)

    # Set the full pixels
    if full_stat_pixels > 0:
        observation[y:y+full_stat_pixels, x:x+width, ch] = 255

    # Set the partial pixel
    if full_stat_pixels < height:
        observation[y+full_stat_pixels, x:x+width, ch] = partial_stat_pixel_intensity


def map_value_to_byte(value, max_value):
    return int((value / max_value) * 255)


def render_move_data(observation: np.ndarray, move_id: int, current_pp: int, y: int, x: int):
    # Retrieve the move data
    move_data = MOVE_BY_ID[move_id].value

    # Map and assign the move data to the observation ndarray
    observation[y, x, 0] = move_id  # Move ID
    observation[y, x, 1] = move_data.ptype.value * 8  # Type value
    observation[y, x, 2] = map_value_to_byte(move_data.max_pp, 40)  # Max PP
    observation[y+1, x, 0] = map_value_to_byte(current_pp, 40)  # Current PP
    observation[y+1, x, 1] = map_value_to_byte(move_data.power, 170)  # Power
    observation[y+1, x, 2] = map_value_to_byte(move_data.accuracy, 100)  # Accuracy
