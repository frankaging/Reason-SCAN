from collections import namedtuple
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product

from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym_minigrid.minigrid import Circle
from gym_minigrid.minigrid import Square
from gym_minigrid.minigrid import Cylinder
from gym_minigrid.minigrid import DIR_TO_VEC
from utils import one_hot
from utils import generate_possible_object_names
from utils import numpy_array_to_image

SemType = namedtuple("SemType", "name")
Position = namedtuple("Position", "column row")
Object = namedtuple("Object", "size color shape")
PositionedObject = namedtuple("PositionedObject", "object position vector overflow overlap", defaults=(None, None, None, False, False))
Variable = namedtuple("Variable", "name sem_type")
fields = ("action", "is_transitive", "manner", "adjective_type", "noun")
Weights = namedtuple("Weights", fields, defaults=(None, ) * len(fields))

ENTITY = SemType("noun")
COLOR = SemType("color")
SIZE = SemType("size")
EVENT = SemType("verb")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
FORWARD = Direction("forward")

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0
}

INT_TO_DIR = {direction_int: direction for direction, direction_int in DIR_TO_INT.items()}

SIZE_TO_INT = {
    "small": 1,
    "average": 2,
    "big": 3
}

# TODO put somewhere different

ACTIONS_DICT = {
    "light": "push",
    "heavy": "push push"
}

DIR_STR_TO_DIR = {
    "n": NORTH,
    "e": EAST,
    "s": SOUTH,
    "w": WEST,
}

DIR_VEC_TO_DIR = {
    (1, 0): "e",
    (0, 1): "n",
    (-1, 0): "w",
    (0, -1): "s",
    (1, 1): "ne",
    (1, -1): "se",
    (-1, -1): "sw",
    (-1, 1): "nw"
}


Command = namedtuple("Command", "action event")
UNK_TOKEN = 'UNK'

# ReaSCAN supported relations
Relation = namedtuple("Relation", "name")
SAME_ROW = Relation("samerow")
SAME_COL = Relation("samecol")
SAME_COLOR = Relation("samecolor")
SAME_SHAPE = Relation("sameshape")
SAME_SIZE = Relation("samesize")
SAME_ALL = Relation("sameall")
IS_INSIDE = Relation("isinside")
IS_NEXT_TO = Relation("isnextto")
SIZE_SMALLER = Relation("sizesmaller")
SIZE_BIGGER = Relation("sizebigger")

def object_to_repr(object: Object) -> dict:
    return {
        "shape": object.shape,
        "color": object.color,
        "size": str(object.size)
    }


def position_to_repr(position: Position) -> dict:
    return {
        "row": str(position.row),
        "column": str(position.column)
    }


def positioned_object_to_repr(positioned_object: PositionedObject) -> dict:
    return {
        "vector": ''.join([str(idx) for idx in positioned_object.vector]),
        "position": position_to_repr(positioned_object.position),
        "object": object_to_repr(positioned_object.object)
    }


def parse_object_repr(object_repr: dict) -> Object:
    return Object(shape=object_repr["shape"], color=object_repr["color"], size=int(object_repr["size"]))


def parse_position_repr(position_repr: dict) -> Position:
    return Position(column=int(position_repr["column"]), row=int(position_repr["row"]))


def parse_object_vector_repr(object_vector_repr: str) -> np.ndarray:
    return np.array([int(idx) for idx in object_vector_repr])


def parse_positioned_object_repr(positioned_object_repr: dict):
    return PositionedObject(object=parse_object_repr(positioned_object_repr["object"]),
                            position=parse_position_repr(positioned_object_repr["position"]),
                            vector=parse_object_vector_repr(positioned_object_repr["vector"]))


class Situation(object):
    """
    Specification of a situation that can be used for serialization as well as initialization of a world state.
    """
    def __init__(self, grid_size: int, agent_position: Position, agent_direction: Direction,
                 target_object: PositionedObject, placed_objects: List[PositionedObject], carrying=None):
        self.grid_size = grid_size
        self.agent_pos = agent_position  # position is [col, row] (i.e. [x-axis, y-axis])
        self.agent_direction = agent_direction
        self.placed_objects = placed_objects
        self.carrying = carrying  # The object the agent is carrying
        self.target_object = target_object
        
        # TODO: some validation checks here?
        # 1. object collisions & object and agent collisions
        # 2. boundary checks
        

    @property
    def distance_to_target(self):
        """Number of grid steps to take to reach the target position from the agent position."""
        return abs(self.agent_pos.column - self.target_object.position.column) + \
               abs(self.agent_pos.row - self.target_object.position.row)

    @property
    def direction_to_target(self):
        """Direction to the target in terms of north, east, south, north-east, etc. Needed for a grounded scan split."""
        column_distance = self.target_object.position.column - self.agent_pos.column
        column_distance = min(max(-1, column_distance), 1)
        row_distance = self.agent_pos.row - self.target_object.position.row
        row_distance = min(max(-1, row_distance), 1)
        return DIR_VEC_TO_DIR[(column_distance, row_distance)]

    def to_dict(self) -> dict:
        """Represent this situation in a dictionary."""
        return {
            "agent_position": Position(column=self.agent_pos[0], row=self.agent_pos[1]),
            "agent_direction": self.agent_direction,
            "target_object": self.target_object,
            "grid_size": self.grid_size,
            "objects": self.placed_objects,
            "carrying": self.carrying
        }

    def to_representation(self) -> dict:
        """Represent this situation in serializable dict that can be written to a file."""
        return {
            "grid_size": self.grid_size,
            "agent_position": position_to_repr(self.agent_pos),
            "agent_direction": DIR_TO_INT[self.agent_direction],
            "target_object": positioned_object_to_repr(self.target_object) if self.target_object else None,
            "distance_to_target": str(self.distance_to_target) if self.target_object else None,
            "direction_to_target": self.direction_to_target if self.target_object else None,
            "placed_objects":  {str(i): positioned_object_to_repr(placed_object) for i, placed_object
                                in enumerate(self.placed_objects)},
            "carrying_object": object_to_repr(self.carrying) if self.carrying else None
        }

    @classmethod
    def from_representation(cls, situation_representation: dict):
        """Initialize this class by some situation as represented by .to_representation()."""
        target_object = situation_representation["target_object"]
        carrying_object = situation_representation["carrying_object"]
        placed_object_reps = situation_representation["placed_objects"]
        placed_objects = []
        for placed_object_rep in placed_object_reps.values():
            placed_objects.append(parse_positioned_object_repr(placed_object_rep))
        situation = cls(grid_size=situation_representation["grid_size"],
                        agent_position=parse_position_repr(situation_representation["agent_position"]),
                        agent_direction=INT_TO_DIR[situation_representation["agent_direction"]],
                        target_object=parse_positioned_object_repr(target_object) if target_object else None,
                        placed_objects=placed_objects,
                        carrying=parse_object_repr(carrying_object) if carrying_object else None)
        return situation

    def __eq__(self, other) -> bool:
        """Recursive function to compare this situation to another and determine if they are equivalent."""
        representation_other = other.to_representation()
        representation_self = self.to_representation()

        def compare_nested_dict(value_1, value_2, unequal_values):
            if len(unequal_values) > 0:
                return
            if isinstance(value_1, dict):
                for k, v_1 in value_1.items():
                    v_2 = value_2.get(k)
                    if not v_2 and v_1:
                        unequal_values.append(False)
                    compare_nested_dict(v_1, v_2, unequal_values)
            else:
                if value_1 != value_2:
                    unequal_values.append(False)
            return
        result = []
        compare_nested_dict(representation_self, representation_other, result)
        return not len(result) > 0