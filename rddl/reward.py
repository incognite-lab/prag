from rddl import Reward, Variable
from rddl.entity import LocationType


class NearReward(Reward):
    _0_EDISTANCE_PREDICATE = "euclidean_distance"
    _0_NEAR_THRESHOLD = "near_threshold"

    def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def evaluate(self):
        print("Evaluating near predicate")
        return NearReward._0_EDISTANCE_PREDICATE(self.object_A.location, self.object_B.location) - NearReward._0_NEAR_THRESHOLD
