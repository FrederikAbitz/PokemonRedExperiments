import numpy as np


from typing import Callable, Optional

import obs_utils
from PokemonType import PokemonType


def type_to_one_hot_8pxcol_2ch(pokemon_type: PokemonType) -> np.ndarray:
    """
    Converts a PokemonType to a one-hot encoded array.
    """
    one_hot = np.zeros((1, 8, 2), dtype=np.uint8)

    # Map the type value to the corresponding position and channel
    type_value = pokemon_type.value
    if type_value < 0x08:
        # Use the red channel for types 0x00 to 0x07
        channel = 0
        position = type_value
    else:
        # Use the green channel for types 0x08 and above
        channel = 1
        position = type_value - 0x08 if type_value < 0x10 else type_value - 0x14

    one_hot[0, position, channel] = 255
    return one_hot



class PartyPokemon:
    """Game data related to a Pokemon the player currently carries."""

    nr: int
    """Pokemon identifier."""
    level: int
    """Level."""
    exp: int
    """Experience points."""

    max_hp: int
    """Max HP."""
    hp: int
    """Current HP."""

    status_raw: int
    """Raw status byte."""
    paralyzed: bool
    """Indicates if paralyzed."""
    frozen: bool
    """Indicates if frozen."""
    burned: bool
    """Indicates if burned."""
    poisoned: bool
    """Indicates if poisoned."""
    sleep_counter: int
    """Sleep counter (0-7)."""

    type1: PokemonType
    """Primary type."""
    type2: Optional[PokemonType]
    """Secondary type."""
    moves: list[int]
    """List of move identifiers."""
    pp: list[int]
    """PP for each move."""

    trainer_id: int
    """Trainer ID."""

    hp_ev: int
    """HP Effort Value."""
    attack_ev: int
    """Attack Effort Value."""
    defense_ev: int
    """Defense Effort Value."""
    speed_ev: int
    """Speed Effort Value."""
    special_ev: int
    """Special Effort Value."""
    attack_defense_iv: int
    """Attack/Defense Individual Value."""
    speed_special_iv: int
    """Speed/Special Individual Value."""
    attack: int
    """Attack stat."""
    defense: int
    """Defense stat."""
    speed: int
    """Speed stat."""
    special: int
    """Special stat."""

    def __init__(self, pyboy_memreader: Callable[[int], int], start_addr: int):
        self.nr = pyboy_memreader(start_addr)
        self.hp = pyboy_memreader(start_addr + 0x1) + (pyboy_memreader(start_addr + 0x2) << 8)
        self.status_raw = pyboy_memreader(start_addr + 0x4)
        self.paralyzed = bool(self.status_raw & 0b01000000)  # Bit 6
        self.frozen = bool(self.status_raw & 0b00100000)  # Bit 5
        self.burned = bool(self.status_raw & 0b00010000)  # Bit 4
        self.poisoned = bool(self.status_raw & 0b00001000)  # Bit 3
        self.sleep_counter = self.status_raw & 0b00000111  # Bits 0-2
        type1_byte = pyboy_memreader(start_addr + 0x5)
        type2_byte = pyboy_memreader(start_addr + 0x6)
        self.type1 = PokemonType.from_byte_or_none(type1_byte)
        self.type2 = PokemonType.from_byte_or_none(type2_byte)
        self.moves = [
            pyboy_memreader(start_addr + 0x7),
            pyboy_memreader(start_addr + 0x8),
            pyboy_memreader(start_addr + 0x9),
            pyboy_memreader(start_addr + 0xA)
        ]
        self.trainer_id = pyboy_memreader(start_addr + 0xB) + (pyboy_memreader(start_addr + 0xC) << 8)
        self.exp = sum(pyboy_memreader(start_addr + 0xD + i) << (i * 8) for i in range(3))
        self.hp_ev = pyboy_memreader(start_addr + 0x10) + (pyboy_memreader(start_addr + 0x11) << 8)
        self.attack_ev = pyboy_memreader(start_addr + 0x12) + (pyboy_memreader(start_addr + 0x13) << 8)
        self.defense_ev = pyboy_memreader(start_addr + 0x14) + (pyboy_memreader(start_addr + 0x15) << 8)
        self.speed_ev = pyboy_memreader(start_addr + 0x16) + (pyboy_memreader(start_addr + 0x17) << 8)
        self.special_ev = pyboy_memreader(start_addr + 0x18) + (pyboy_memreader(start_addr + 0x19) << 8)
        self.attack_defense_iv = pyboy_memreader(start_addr + 0x1A)
        self.speed_special_iv = pyboy_memreader(start_addr + 0x1B)
        self.pp = [
            pyboy_memreader(start_addr + 0x1C),
            pyboy_memreader(start_addr + 0x1D),
            pyboy_memreader(start_addr + 0x1E),
            pyboy_memreader(start_addr + 0x1F)
        ]
        self.level = pyboy_memreader(start_addr + 0x20)
        self.max_hp = pyboy_memreader(start_addr + 0x21) + (pyboy_memreader(start_addr + 0x22) << 8)
        self.attack = pyboy_memreader(start_addr + 0x23) + (pyboy_memreader(start_addr + 0x24) << 8)
        self.defense = pyboy_memreader(start_addr + 0x25) + (pyboy_memreader(start_addr + 0x26) << 8)
        self.speed = pyboy_memreader(start_addr + 0x27) + (pyboy_memreader(start_addr + 0x28) << 8)
        self.special = pyboy_memreader(start_addr + 0x29) + (pyboy_memreader(start_addr + 0x2A) << 8)


    @classmethod
    @property
    def observation_size(self) -> tuple[int, int]:
        """
        Size of the observation array.
        """
        # Keep 8x10 reserved for changes in reprentation
        return (8, 10)


    def render_observation_8bit_rgb(self) -> np.ndarray:
        """
        Renders the Pokemon data into an 8-bit RGB array.
        """
        observation = np.zeros(self.observation_size + (3,), dtype=np.uint8)

        # Render Pokemon ID (Binary Encoding)
        for i in range(8):
            if self.nr & (1 << i):
                observation[i, 0, 1] = 255

        # Render Types (One-Hot Encoding)
        type_encoded = type_to_one_hot_8pxcol_2ch(self.type1)
        if self.type2 is not None:
            type2_encoded = type_to_one_hot_8pxcol_2ch(self.type2)
            type_encoded = np.bitwise_or(type_encoded, type2_encoded)
        observation[:, 1, 1:] = type_encoded

        # Render Level and Experience (Fading Progress Bars on R and G channels)
        level_max = 100  # Assuming the maximum level is 100
        exp_max = int(5 * pow(self.level + 1.0, 3) / 4)  # Assuming slow leveling group
        obs_utils.render_progress_bar(observation, self.level, level_max, 0, 2, 0)
        obs_utils.render_progress_bar(observation, self.exp, exp_max, 0, 2, 1)

        # Render HP and Max HP (Fading Progress Bars on R and B channels)
        stat_max = 160  # Assuming a stat maximum of 160, should be good for lvl <= 50
        obs_utils.render_progress_bar(observation, self.hp, self.max_hp, 0, 3, 0)
        obs_utils.render_progress_bar(observation, self.max_hp, stat_max, 0, 3, 2)

        # Render Attack and Defense (Fading Progress Bars on R and G channels)
        obs_utils.render_progress_bar(observation, self.attack, stat_max, 0, 4, 0)
        obs_utils.render_progress_bar(observation, self.defense, stat_max, 0, 4, 1)

        # Render Speed and Special (Fading Progress Bars on G and B channels)
        obs_utils.render_progress_bar(observation, self.speed, stat_max, 0, 5, 1)
        obs_utils.render_progress_bar(observation, self.special, stat_max, 0, 5, 2)

        # Render Status Effects
        if self.paralyzed:
            observation[0, 6, 1] = 255
        if self.frozen:
            observation[1, 6, 1] = 255
        if self.burned:
            observation[2, 6, 1] = 255
        if self.poisoned:
            observation[3, 6, 1] = 255
        if self.sleep_counter > 0:
            observation[4, 6, 1] = 255
        # Render Sleep counter as a progress bar in the same column
        sleep_max = 7
        obs_utils.render_progress_bar(observation, self.sleep_counter, sleep_max, 5, 6, 1, height=3)

        # Moves
        move_column = 7
        for i, (move_id, current_pp) in enumerate(zip(self.moves, self.pp)):
            if move_id == 0 or move_id > 0xa5:
                continue # Empty or illegal
            row = i * 2  # Each move takes up 2 rows
            obs_utils.render_move_data(observation, move_id, current_pp, row, move_column)

        return observation
