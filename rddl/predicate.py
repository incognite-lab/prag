from typing import Any, Callable, ClassVar, Union

from rddl import Predicate, Variable
from rddl.entity import Gripper, LocationType, ObjectEntity


class Near(Predicate):
    _0_EDISTANCE_PREDICATE: ClassVar[Union[Callable, str]] = "euclidean_distance"
    _0_NEAR_THRESHOLD: ClassVar[Union[float, str]] = "near_threshold"

    def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
        self.object_A = object_A
        self.object_B = object_B

    def __call__(self):
        return Near._0_EDISTANCE_PREDICATE(self.object_A.location, self.object_B.location) < Near._0_NEAR_THRESHOLD


class IsHolding(Predicate):
    _0_IS_HOLDING_FUNCTION = "is_holding"

    def __init__(self, gripper: Gripper) -> None:
        super().__init__()
        self._gripper = gripper

    def __call__(self, obj: ObjectEntity) -> Any:
        return IsHolding._0_IS_HOLDING_FUNCTION(self._gripper, obj)
