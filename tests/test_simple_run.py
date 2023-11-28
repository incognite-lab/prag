import numpy as np

import os
import sys
SCRIPT_DIR = os.path.dirname(__file__)
MODULE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(MODULE_DIR)

from rddl.entity import Variable, LocationType, Gripper, ObjectEntity, Entity
from rddl.predicate import Predicate
from rddl.reward import Reward
from rddl.action import AtomicAction


def euclidean_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    return np.linalg.norm(point_A - point_B)


def is_holding(gripper, obj) -> bool:
    return gripper.is_holding(obj)


functions = {
    "is_holding": is_holding,
    "euclidean_distance": euclidean_distance
}

# CONSTANTS
NEAR_THRESHOLD = 0.01

Entity.set_observation_getter(lambda : {

})

class Near(Predicate):
    _EDISTANCE_PREDICATE = functions["euclidean_distance"]

    def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def decide(self):
        print("Checking near predicate")
        return Near._EDISTANCE_PREDICATE(self.object_A.location, self.object_B.location) < NEAR_THRESHOLD

    def evaluate(self):
        print("Evaluating near predicate")
        return Near._EDISTANCE_PREDICATE(self.object_A.location, self.object_B.location) - NEAR_THRESHOLD


class NearReward(Near, Reward):
    pass


class Apple(ObjectEntity):

    def __init__(self, kind: str = "RedDelicious"):
        super().__init__(self._get_generic_reference(), "apple")
        self._kind = kind

    def _get_location(self):
        return np.random.randn(3)


class Reach(AtomicAction):

    # def __init__(self, g: Gripper, o: ObjectEntity):
    def __init__(self, g: Variable[Gripper], o: Variable[ObjectEntity]):
        self._g: Variable[Gripper] = g
        self._o: Variable[ObjectEntity] = o
        self._predicate = Near(self._g, self._o)
        self._reward = NearReward(self._g, self._o)


class TiagoGripper(Gripper):

    def __init__(self):
        super().__init__("gripper_tiago")

    def _is_holding(self, obj: ObjectEntity) -> bool:
        return obj.name == "apple"

    def _get_location(self):
        return np.random.randn(3)


if __name__ == "__main__":
    # ve = Variable("test_ve", ObjectEntity)
    g1 = Variable("gripper", Gripper)
    o1 = Variable("apple", ObjectEntity)

    r = Reach(g1, o1)

    print(f"gripper: {g1}")

    g1.bind(TiagoGripper())
    o1.bind(Apple())

    print(f"gripper: {g1}")

    print(r.decide())
    print(r.evaluate())

    # ve.bind(e)

    # print(f"ve: {ve}")
    # print(f"e: {e}")

    # print(ve.location)
