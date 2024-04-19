from rddl import Reward, Variable
from rddl.entity import Gripper, LocationType, ObjectEntity
from rddl.predicate import IsHolding


class NearReward(Reward):
    _0_EDISTANCE_PREDICATE = "euclidean_distance"
    _0_NEAR_THRESHOLD = "near_threshold"

    def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def __call__(self):
        print("Evaluating near predicate")
        return NearReward._0_EDISTANCE_PREDICATE(self.object_A.location, self.object_B.location) - NearReward._0_NEAR_THRESHOLD


class GraspReward(Reward):

    def __init__(self, g: Gripper, o: ObjectEntity) -> None:
        self._g = g
        self._o = o
        self._predicate = IsHolding(self._g)

    def __call__(self):
        print("Evaluating goal predicate")
        return 1 if self._predicate(self._o) else 0
