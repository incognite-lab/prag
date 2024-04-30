from typing import Any, Callable, ClassVar, Union

from rddl import Predicate, Variable
from rddl.entities import (GraspableObject, Gripper, Location, LocationType,
                           ObjectEntity)


class Near(Predicate):
    _0_EDISTANCE_DISTANCE: ClassVar[Union[Callable, str]] = "euclidean_distance"
    _0_NEAR_THRESHOLD: ClassVar[Union[float, str]] = "near_threshold"
    _VARIABLES = {"object_A": Location, "object_B": Location}

    # def __init__(self, object_A: Variable[LocationType], object_B: Variable[LocationType]) -> None:
    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return Near._0_EDISTANCE_DISTANCE(self.object_A.location, self.object_B.location) < Near._0_NEAR_THRESHOLD


class GripperAt(Predicate):
    _0_GRIPPER_AT_CHECK: ClassVar[Union[Callable, str]] = "gripper_at"
    _VARIABLES = {"gripper": Gripper, "object": GraspableObject}

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return GripperAt._0_GRIPPER_AT_CHECK(self.gripper, self.object)


class GripperOpen(Predicate):
    _0_GRIPPER_OPEN_CHECK: ClassVar[Union[Callable, str]] = "gripper_open"
    _VARIABLES = {"gripper": Gripper}

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return GripperOpen._0_GRIPPER_OPEN_CHECK(self.gripper)


class IsReachable(Predicate):
    _0_REACHABLE_TEST: ClassVar[Union[Callable, str]] = "is_reachable"
    _VARIABLES = {"gripper": Gripper, "location": Location}

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self):
        return IsReachable._0_REACHABLE_TEST(self.gripper, self.location)


class IsHolding(Predicate):
    _0_IS_HOLDING_FUNCTION = "is_holding"
    _VARIABLES = {"gripper": Gripper, "object": ObjectEntity}

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)

    def __call__(self) -> Any:
        return IsHolding._0_IS_HOLDING_FUNCTION(self.gripper, self.object)

