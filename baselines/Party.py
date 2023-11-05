from PartyPokemon import PartyPokemon


import numpy as np


from typing import Callable


class Party:
    pkmn: list[PartyPokemon]
    PARTY_SIZE_ADDR = 0xD163
    PKMN_BASE_ADDRS = (0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247)

    def __init__(self, pyboy_memreader: Callable[[int], int]):
        party_size = pyboy_memreader(self.PARTY_SIZE_ADDR)
        self.pkmn = []
        for base_addr in self.PKMN_BASE_ADDRS[:party_size]:
            self.pkmn.append(PartyPokemon(pyboy_memreader, base_addr))


    @classmethod
    @property
    def observation_size(self) -> tuple[int, int]:
        """
        Size of the observation array.
        """
        return np.multiply(PartyPokemon.observation_size, (1, 6))


    def render_observation_8bit_rgb(self) -> np.ndarray:
        """
        Renders the Pokemon data into an 8-bit RGB array.
        """
        observation = np.zeros((*self.observation_size, 3), dtype=np.uint8)
        for i, pkmn in enumerate(self.pkmn):
            observation[:, i * PartyPokemon.observation_size[1]:(i + 1) * PartyPokemon.observation_size[1], :] = pkmn.render_observation_8bit_rgb()
        return observation


    def __len__(self):
        return len(self.pkmn)

    def __getitem__(self, index):
        return self.pkmn.__getitem__(self, index)

    @property
    def size(self):
        return self.__len__()
