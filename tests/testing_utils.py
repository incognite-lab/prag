from time import time
from typing import Callable, Optional, TypeVar

import numpy as np
import pytest

from rddl import Operand, Variable


def time_function(f: Callable, *args, **kwargs):
    start = time()
    result = f(*args, **kwargs)
    end = time()
    return end - start, result


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(np.array(point_A) - np.array(point_B))


def is_holding(gripper, obj) -> bool:
    return bool(np.linalg.norm(gripper.location - obj.location) < NEAR_THRESHOLD)


NEAR_THRESHOLD = 0.1

mapping = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance,
    "near_threshold": NEAR_THRESHOLD,
}

Operand.set_mapping(mapping)

from rddl import Entity, Reward
from rddl.action import AtomicAction
from rddl.entity import Gripper, Location, ObjectEntity
from rddl.operator import NotOp
from rddl.predicate import IsHolding, Near

# CONSTANTS

# END OF CUSTOM FUNCTIONS & VARIABLES


class EnvObjectProxy:  # This should be replaced by EnvObject from myGym

    def __init__(self, **kwds):
        super().__init__()
        self._name = self.reference
        self._position = np.random.randn(3)
        self._orientation = np.random.randn(3)

    def set_position(self, position: np.ndarray):
        self._position = position

    def set_orientation(self, orientation: np.ndarray):
        self._orientation = orientation

    def get_position(self) -> np.ndarray:
        return self._position

    def get_orientation(self) -> np.ndarray:
        return self._orientation

    def get_position_and_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_position(), self.get_orientation()

    def get_name(self) -> str:
        return self._name

    def get_uuid(self) -> str:
        return self.name + "_uuid"


class EnvSimulator:  # This should be replaced by actual env from myGym
    """Simulates the simulation environment"""

    def __init__(self, list_of_objects: list[EnvObjectProxy]):
        self._objects = list_of_objects

    def reset(self):
        pass

    def step(self, action):
        # generate observation
        obs = {}
        for obj in self._objects:
            obj.set_position(np.random.randn(3))  # randomly jiggle the object about
            obj.set_orientation(np.random.randn(3))
            obs[obj.get_name()] = obj
        return obs


class Observer:  # This class serves merely as a container & memory for current observation

    def __init__(self):
        self.obs = None

    def set_observation(self, obs):
        self.obs = obs

    def get_observation(self):
        return self.obs


Entity.set_observation_getter(lambda self: self)
Location.monkey_patch(Location._get_location, lambda self: self.get_position())


class Apple(ObjectEntity, EnvObjectProxy):

    def __init__(self, reference: Optional[str] = None, kind: str = "RedDelicious"):
        super().__init__(self._get_generic_reference() if reference is None else reference, "apple")
        self._kind = kind


class TiagoGripper(Gripper, EnvObjectProxy):

    def __init__(self, reference: Optional[str] = None):
        super().__init__("gripper_tiago" if reference is None else reference)
        # self._is_holding_predicate = IsHolding(self)


class ApproachReward(Reward):
    _0_EUCLIDEAN_DISTANCE = "euclidean_distance"

    _VARIABLES = {
        "gripper": Gripper,
        "object": ObjectEntity
    }

    def __init__(self, gripper: Variable[Gripper], obj: Variable[ObjectEntity]) -> None:
        super().__init__()
        self._gripper = gripper
        self._obj = obj

    def __call__(self) -> float:
        return ApproachReward._0_EUCLIDEAN_DISTANCE(self._gripper.location, self._obj.location)


class Approach(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": ObjectEntity
    }

    # def __init__(self, g: Gripper, o: ObjectEntity):
    def __init__(self):
        super().__init__()
        self._predicate = Near(object_A=self.get_variable("gripper"), object_B=self.get_variable("object"))
        self._initial = NotOp(operand=self._predicate)
        self._reward = ApproachReward(self.get_variable("gripper"), self.get_variable("object"))


# @pytest.fixture
def create_approach_action() -> AtomicAction:
    a = Approach()

    return a


# @pytest.fixture
def create_gripper_and_apple() -> dict[str, Entity]:
    gripper_name = "tiago_gripper"
    apple_name = "apple_01"

    t_gripper = TiagoGripper(gripper_name)
    apple = Apple(apple_name)
    objects_for_approach = {
        "gripper": t_gripper,
        "object": apple
    }

    return objects_for_approach
