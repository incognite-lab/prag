from typing import Any, Callable, ClassVar, Union

from rddl import Predicate, Variable
from rddl.entity import Gripper, Location, LocationType, ObjectEntity


class Near(Predicate):
    _0_EDISTANCE_DISTANCE: ClassVar[Union[Callable, str]] = "euclidean_distance"
    _0_NEAR_THRESHOLD: ClassVar[Union[float, str]] = "near_threshold"
    _VARIABLES = {"object_A": Location, "object_B": Location}

    # def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return Near._0_EDISTANCE_DISTANCE(self.object_A.location, self.object_B.location) < Near._0_NEAR_THRESHOLD


class IsHolding(Predicate):
    _0_IS_HOLDING_FUNCTION = "is_holding"
    _VARIABLES = {"gripper": Gripper, "object": ObjectEntity}

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self) -> Any:
        return IsHolding._0_IS_HOLDING_FUNCTION(self.gripper, self.object)


# class Not(Predicate):
#     _VARIABLES = {"gripper": Gripper, "object": ObjectEntity}

#     def __init__(self, predicate: Predicate) -> None:
#         self._predicate = predicate

#     def __call__(self):
#         return not self._predicate.decide()
