from typing import Optional, Tuple
from collections import defaultdict
import numpy as np
from shapely.geometry import box



class MapTracker:
    decay_factor: float
    threshold: float
    presence_decay_period: int
    default_map_size: Tuple[int, int]
    _maps: defaultdict
    _map_decay_presence_stacks: np.ndarray
    _map_decay_movement_stacks: np.ndarray
    _last_registered_player_pos: Optional[Tuple[int, int]]
    _last_registered_map_id: Optional[int]


    def __init__(
            self,
            decay_factor: float = 0.996,
            threshold: float = 0.01,
            presence_decay_period: int = 16,
            default_map_size: Tuple[int, int] = (256, 256)
            ):
        self.decay_factor: float = decay_factor
        self.threshold: float = threshold
        self.presence_decay_period = presence_decay_period
        self.default_map_size: Tuple[int, int] = default_map_size
        self.reset()


    def reset(self):
        self._maps = defaultdict(lambda: np.zeros(self.default_map_size + (3,)))
        self._map_decay_presence_stacks = np.zeros((255,), dtype=np.int32)
        self._map_decay_movement_stacks = np.zeros((255,), dtype=np.int32)
        self._last_registered_player_pos = None
        self._last_registered_map_id = None
        self._registered_step_counter = 0


    def register_step_to(self, map_id: int, player_pos: Tuple[int, int]) -> None:
        """Update presence and movement data, then apply decay.

        Presence is marked on `player_pos`.
        Movement is marked on previously registered position, with direction to `player_pos`.
        """
        self._decay_movement()
        if self._registered_step_counter % self.presence_decay_period == 0:
            self._decay_presence()
        # Always decay map wit registered step to ensure step-decay integrity (no steps on map without decay inbetween)
        self._apply_stacked_decay(map_id)

        self._mark_presence_at_loc(map_id, player_pos)

        prev_map_id = self._last_registered_map_id
        prev_pos = self._last_registered_player_pos
        if prev_map_id == map_id and prev_pos is not None:
            move_vector = list([a - b for a, b in zip(player_pos, prev_pos)])
            self._mark_movement_from_loc(map_id, prev_pos, move_vector)

        self._last_registered_map_id = map_id
        self._last_registered_player_pos = player_pos
        self._registered_step_counter += 1


    def _resize_map_if_needed(self, map_id: int, player_pos: Tuple[int, int]) -> None:
        """Resizes the map to fit specified coordinates if needed."""
        x, y = player_pos            
        height, width, _ = self._maps[map_id].shape
        if y >= height or x >= width:
            new_height: int = max(height, y + 8)
            new_width: int = max(width, x + 8)
            if (new_height, new_width) != (height, width):
                self._maps[map_id] = np.pad(self._maps[map_id],
                                        ((0, new_height - height), (0, new_width - width), (0, 0)),
                                        mode='constant', constant_values=0)

    def _mark_presence_at_loc(self, map_id: int, player_pos: Tuple[int, int]) -> None:
        """Set presence data at the specified location."""
        self._resize_map_if_needed(map_id, player_pos)
        x, y = player_pos
        self._maps[map_id][y, x, 2] = 1.0


    def _mark_movement_from_loc(self, map_id: int, player_pos: Tuple[int, int], move_vector: Tuple[int, int]) -> None:
        """Set movement data at `player_pos` for movement direction `move_vector`.
        Data is only set if `move_vector` represents movement from directly adjacent position."""
        if abs(sum(move_vector)) == 0:
            self._resize_map_if_needed(map_id, player_pos)
            x, y = player_pos
            movement: np.ndarray = np.array(move_vector)
            self._maps[map_id][y, x, :2] = movement


    def _direction_to_vector(self, direction: str) -> np.ndarray:
        vectors = {'right': [1, 0], 'left': [-1, 0], 'down': [0, 1], 'up': [0, -1]}
        return np.array(vectors.get(direction, [0, 0]))


    def _decay_movement(self, decay_only_map_id: Optional[int] = None) -> None:
        """Adds one exponential decay stack to the movement channels."""
        if decay_only_map_id is None:
            for map_id, _ in self._maps.items():
                self._map_decay_movement_stacks[map_id] += 1
        else:
            self._map_decay_movement_stacks[decay_only_map_id] += 1


    def _decay_presence(self, decay_only_map_id: Optional[int] = None) -> None:
        """Adds one exponential decay stack to the presence channel."""
        if decay_only_map_id is None:
            for map_id, _ in self._maps.items():
                self._map_decay_presence_stacks[map_id] += 1
        else:
            self._map_decay_presence_stacks[decay_only_map_id] += 1


    def _apply_stacked_decay(self, map_id: int) -> None:
        """Applies stacked decay steps for `map_id`"""
        self._apply_decay_map_movement(map_id, self._map_decay_movement_stacks[map_id])
        self._map_decay_movement_stacks[map_id] = 0
        self._apply_decay_map_presence(map_id, self._map_decay_presence_stacks[map_id])
        self._map_decay_presence_stacks[map_id] = 0


    def _apply_decay_map_movement(self, map_id: int, stacks: int = 1) -> None:
        """Decays movement data for a single map."""
        if stacks > 0:
            movement = self._maps[map_id][:, :, :2]
            movement *= pow(self.decay_factor, stacks)
            movement[np.abs(movement) < self.threshold] = 0


    def _apply_decay_map_presence(self, map_id: int, stacks: int = 1) -> None:
        """Decays presence data for a single map. Non-zero presence values decay until `self.threshold` and stay at this value."""
        if stacks > 0:
            presence = self._maps[map_id][:, :, 2]
            presence[presence > self.threshold] *= pow(self.decay_factor, stacks)
            below_threshold = np.logical_and(presence > 0, presence < self.threshold)
            presence[below_threshold] = self.threshold


    @staticmethod
    def _crop_map_view_slices_to_bounds(view_dims: Tuple[int, int], view_loc: Tuple[int, int], 
                                        map_dims: Tuple[int, int], map_loc: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """Calculates the intersection of the view and map planes based on the player position on map and in view. Returns the slices of the view and map plane that overlap."""
        
        """Plan
        1. Calculate translation vector t_mv from map_loc to view_loc
        2. Calculate map_t by applying t_mv to (0, 0, map_dims)
        3. Calculate intersection i_v between (0, 0, view_dims) and map_t using shapely
        4. Calculate translation vector t_vm by reversing t_mv
        5. Calculate intersection i_m by applying t_vm to i_v
        6. Return i_v, i_m
        """
        # Calculate translation vector from map_loc to view_loc
        t_mv = (view_loc[0] - map_loc[0], view_loc[1] - map_loc[1])

        # Calculate transformed map (map_t) by applying t_mv to map dimensions
        map_t = box(t_mv[0], t_mv[1], t_mv[0] + map_dims[0], t_mv[1] + map_dims[1])

        # Calculate intersection (i_v) between view and map_t
        view = box(0, 0, view_dims[0], view_dims[1])
        i_v = view.intersection(map_t)

        # If there is no intersection, return None
        if i_v.is_empty:
            return None

        # Calculate reverse translation vector
        t_vm = (-t_mv[0], -t_mv[1])

        # Calculate intersection (i_m) by applying t_vm to i_v
        i_m = box(i_v.bounds[0] + t_vm[0], i_v.bounds[1] + t_vm[1], 
                i_v.bounds[2] + t_vm[0], i_v.bounds[3] + t_vm[1])

        # Return slices for view and map
        slice_view = (int(i_v.bounds[0]), int(i_v.bounds[1]), int(i_v.bounds[2]), int(i_v.bounds[3]))
        slice_map = (int(i_m.bounds[0]), int(i_m.bounds[1]), int(i_m.bounds[2]), int(i_m.bounds[3]))

        return (slice_view, slice_map)


    def get_view_raw(self, map_id: int, x: int, y: int, view_size: Tuple[int, int] = (40, 36), scale: int = 1) -> np.ndarray:
        """Returns a 'camera view' centered at the specified location."""

        view_width, view_height = view_size
        view_center_x, view_center_y = view_width // 2, view_height // 2

        # Step 1: Initialize a view array filled with zeros
        view = np.zeros((view_height, view_width, 3))

        # If the map_id does not exist, return the empty view
        if map_id not in self._maps.keys():
            print(f"Map ID {map_id} not found. Returning empty view.")
            return view

        # Make sure waiting decay stacks are applied
        self._apply_stacked_decay(map_id)

        # Step 2: Map Bounds
        map_max_x, map_max_y, _ = self._maps[map_id].shape
        
        
        slices = self._crop_map_view_slices_to_bounds(view_size, (view_center_x, view_center_y), (map_max_x, map_max_y), (x, y))
        if slices is None:
            return view
        else:
            ((x1_view, y1_view, x2_view, y2_view),
            (x1_map, y1_map, x2_map, y2_map)) = slices

        #print(f"Final map slice: x1_map={x1_map}, x2_map={x2_map}, y1_map={y1_map}, y2_map={y2_map}")
        #print(f"Final view slice: x1_view={x1_view}, x2_view={x2_view}, y1_view={y1_view}, y2_view={y2_view}")

        try:
            view[y1_view:y2_view, x1_view:x2_view, :] = self._maps[map_id][y1_map:y2_map, x1_map:x2_map, :]
        except ValueError as e:
            print(f"Error during slice assignment: {e}")
            print(f"Map slice shape: {self._maps[map_id][y1_map:y2_map, x1_map:x2_map, :].shape}, View slice shape: {view[y1_view:y2_view, x1_view:x2_view, :].shape}")
            #raise e

        if scale > 1:
            # Duplicate each row and column 'scale' times
            view = np.repeat(view, scale, axis=0)  # Repeat rows
            view = np.repeat(view, scale, axis=1)  # Repeat columns

        return view


    def get_view_8bit_rgb(self, map_id: int, x: int, y: int, view_size: Tuple[int, int] = (40, 36), scale: int = 1) -> np.ndarray:
        """Returns a 'camera view' centered at the specified location."""
        return self._raw_view_to_8bit_rgb(
            self.get_view_raw(map_id, x, y, view_size, scale)
        )


    @staticmethod
    def _raw_view_to_8bit_rgb(view: np.ndarray) -> np.ndarray:
        view[:, :, 0:2] = (view[:, :, 0:2] + 1.0) * 127.5  # Map movement channels
        view[:, :, 2] = view[:, :, 2] * 255.0  # Map presence channel
        view = view.astype(np.uint8)
        return view





def visualize_vertical_lines_movement_color_normalized(output_filename="visualize_map_tracker.mp4"):
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    mt = MapTracker(0.998, 0.1, (200, 200))
    map_id = 1
    initial_x, initial_y = 100, 100
    n = 24  # Number of steps up or down
    total_lines = 12  # Number of vertical lines to be drawn (each line consists of up and down movement)
    x, y = initial_x, initial_y
    frames = []
    
    counter = 0
    gap_counter = 0
    def get_gap(gap_counter):
        return 1 + (gap_counter % 3 != 0)
    
    def step_and_record(map_id, x, y, dx, dy, counter):
        mt._mark_presence_at_loc(map_id, (x, y))
        mt._mark_movement_from_loc(map_id, (x, y), (dx, dy))
        mt._decay_movement(map_id)
        counter += 1
        if counter % 16 == 0:
            mt._decay_presence(map_id)

        if counter % 2 == 0:
            # Save frame
            view = mt.get_view_raw(map_id, x, y, (48, 48))
            view[:, :, 0:2] = (view[:, :, 0:2] + 1.0) * 127.5  # Normalize movement channels
            view[:, :, 2] = view[:, :, 2] * 255.0  # Normalize presence channel
            frames.append(view.astype(np.uint8))
        
        return counter
        
    for k in range(3):
        for line in range(total_lines):
            # Move down n times
            for _ in range(n):
                y += 1
                counter = step_and_record(map_id, x, y, 0, 1, counter)


            # Move right 2 times
            gap_counter += 1
            for _ in range(get_gap(gap_counter)):
                x += 1
                counter = step_and_record(map_id, x, y, 1, 0, counter)


            # Move up n times
            for _ in range(2*n):
                y -= 1
                counter = step_and_record(map_id, x, y, 0, -1, counter)


            # Move right 2 times
            gap_counter += 1
            for _ in range(get_gap(gap_counter)):
                x += 1
                counter = step_and_record(map_id, x, y, 1, 0, counter)

            # Move down n times
            for _ in range(n):
                y += 1
                counter = step_and_record(map_id, x, y, 0, 1, counter)



        # Move down n-2 times
        for _ in range(n-2):
            y += 1
            counter = step_and_record(map_id, x, y, 0, 1, counter)

        # Move left n times
        for _ in range(n):
            x -= 1
            counter = step_and_record(map_id, x, y, -1, 0, counter)


        for line in range(total_lines):
            # Move left n times
            for _ in range(n):
                x -= 1
                counter = step_and_record(map_id, x, y, -1, 0, counter)


            # Move up 2 times
            gap_counter += 1
            for _ in range(get_gap(gap_counter)):
                y -= 1
                counter = step_and_record(map_id, x, y, 0, -1, counter)


            # Move right n times
            for _ in range(2*n):
                x += 1
                counter = step_and_record(map_id, x, y, 1, 0, counter)


            # Move up 2 times
            gap_counter += 1
            for _ in range(get_gap(gap_counter)):
                y -= 1
                counter = step_and_record(map_id, x, y, 0, -1, counter)

            # Move left n times
            for _ in range(n):
                x -= 1
                counter = step_and_record(map_id, x, y, -1, 0, counter)

        if k % 3 == 0:
            # Move left n times
            for _ in range(2 * n):
                x -= 1
                counter = step_and_record(map_id, x, y, -1, 0, counter)
        elif k % 3 == 1:
            # Move right n times
            for _ in range(2 * n):
                x += 1
                counter = step_and_record(map_id, x, y, 1, 0, counter)
        elif k % 3 == 2:
            # Move down n times
            for _ in range(n):
                y += 1
                counter = step_and_record(map_id, x, y, 0, 1, counter)


    # Save frames to video using ffmpeg
    writer = imageio.get_writer(output_filename, fps=60, format='mp4') # type: ignore
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == '__main__':
    visualize_vertical_lines_movement_color_normalized()
