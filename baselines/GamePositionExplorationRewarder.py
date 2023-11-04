class GamePositionExplorationRewarder:

    _visited_cells: set
    _cell_size: int
    _reward_per_cell: float

    _total_reward_curr: float
    _total_reward_prev: float
    _delta_reward_step: float


    def __init__(self,
                 reward_per_cell: float = 0.01,
                 cell_size: int = 2,
                 ):

        self._reward_per_cell = reward_per_cell
        self._cell_size = cell_size
        self.reset()


    def reset(self):
        self._total_reward_curr = 0
        self._total_reward_prev = 0
        self._delta_reward_step = 0
        self.reset_visited_cells()


    def reset_visited_cells(self):
        self._visited_cells = set()


    def register_step(self, map_id: int, pos: tuple[int, int]) -> bool:
        self._total_reward_prev = self._total_reward_curr
        reward = 0.0
        entry_added = False

        x, y = pos
        cell_y = y // self._cell_size
        cell_x = x // self._cell_size
        cell_entry = (map_id, cell_y, cell_x)

        if cell_entry not in self._visited_cells:
            self._visited_cells.add(cell_entry)
            reward = self._reward_per_cell
            entry_added = True

        self._delta_reward_step = reward
        self._total_reward_curr = self._total_reward_prev + reward
        return entry_added


    def get_total_reward(self) -> float:
        return self._total_reward_curr


    def get_delta_reward(self) -> float:
        return self._delta_reward_step


    def get_visited_cell_count(self) -> int:
        return len(self._visited_cells)


    @property
    def reward_per_cell(self) -> float:
        return self._reward_per_cell

    @reward_per_cell.setter
    def reward_per_cell(self, reward_per_cell: float):
        self._reward_per_cell = reward_per_cell

