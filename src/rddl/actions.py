from rddl import AtomicAction, Reward, Variable
from rddl.entities import GraspableObject, Gripper, Location
from rddl.operators import NotOp, ParallelAndOp
from rddl.predicates import (GripperAt, GripperOpen, IsHolding, IsReachable,
                             Near, ObjectAt)


class ApproachReward(Reward):
    _0_EUCLIDEAN_DISTANCE = "euclidean_distance"

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], obj: Variable[GraspableObject]) -> None:
        super().__init__()
        self._gripper = gripper
        self._obj = obj

    def __call__(self) -> float:
        return ApproachReward._0_EUCLIDEAN_DISTANCE(self._gripper.location, self._obj.location)


class Approach(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self):
        super().__init__()
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        self._predicate = GripperAt(gripper=gripper, object=obj)
        self._initial = ParallelAndOp(
            left=IsReachable(gripper=gripper, location=obj),
            right=ParallelAndOp(left=GripperOpen(gripper=gripper), right=NotOp(operand=self._predicate))
        )
        self._reward = ApproachReward(gripper, obj)


class WithdrawReward(Reward):

    def __init__(self) -> None:
        super().__init__()


class Withdraw(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_at = GripperAt(gripper=gripper, object=obj)
        self._initial = ParallelAndOp(
            left=gripper_at,
            right=GripperOpen(gripper=gripper)
        )
        self._predicate = NotOp(operand=gripper_at)
        self._reward = NotOp(ApproachReward(gripper, obj))


class GraspReward(Reward):

    def __init__(self) -> None:
        super().__init__

    def __call__(self) -> float:
        return 1


class Grasp(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_open = GripperOpen(gripper=gripper)
        self._initial = ParallelAndOp(
            left=GripperAt(gripper=gripper, object=obj),
            right=gripper_open
        )
        self._predicate = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=NotOp(operand=gripper_open)
        )
        self._reward = GraspReward()


class DropReward(Reward):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        return 1


class Drop(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        gripper_open_and_released = ParallelAndOp(
            left=NotOp(operand=IsHolding(gripper=gripper, object=obj)),
            right=GripperOpen(gripper=gripper)
        )
        self._initial = ParallelAndOp(
            left=GripperAt(gripper=gripper, object=obj),
            right=NotOp(operand=gripper_open_and_released)
        )
        self._predicate = gripper_open_and_released
        self._reward = DropReward()


class MoveReward(Reward):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        return 1


class Move(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "location": Location
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        loc = self.get_argument("location")

        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=IsReachable(gripper=gripper, location=loc)
        )
        self._predicate = ObjectAt(object=obj, location=loc)
        self._reward = MoveReward()

