from enum import Enum


class PokemonType(Enum):
    NORMAL = 0x00
    FIGHTING = 0x01
    FLYING = 0x02
    POISON = 0x03
    GROUND = 0x04
    ROCK = 0x05
    BIRD = 0x06 # Only Glitch, MissingNo / M
    BUG = 0x07
    GHOST = 0x08
    FIRE = 0x14
    WATER = 0x15
    GRASS = 0x16
    ELECTRIC = 0x17
    PSYCHIC = 0x18
    ICE = 0x19
    DRAGON = 0x1A

    @classmethod
    def from_byte_or_none(cls, byte_value):
        return cls(byte_value) if byte_value in cls._value2member_map_ else None
