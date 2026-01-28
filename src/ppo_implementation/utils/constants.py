from enum import Enum


class Identifier(Enum):
    NOTHING = 0
    BOMB = -1
    UNREVEALED = -2
    OFF_BOARD = -3


NUM_TO_COLOR = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (0, 0, 0),
    6: (0, 0, 0),
    7: (0, 0, 0),
    8: (0, 0, 0),
    -1: (255, 0, 0),
}
