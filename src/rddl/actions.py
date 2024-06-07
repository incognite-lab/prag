from typing import Callable

from rddl import AtomicAction, Reward, Variable
from rddl.entities import AbstractRotation, GraspableObject, Gripper, Location
from rddl.operators import NotOp, ParallelAndOp
from rddl.predicates import (Exists, GripperAt, GripperOpen, IsHolding, IsReachable,
                             Near, ObjectAt, ObjectAtPose)


""" HOW TO DEFINE A REWARD

class <Some>Reward(Reward):

    _0_<FUNCTION_THIS_REWARD_NEEDS> = "name_of_function"  # "name_of_function" is a function defined externally
    ...  # possibly more functions

    _VARIABLES = {  # Variables that will be part of the observation; not necessarily all variables used by the reward
        "<variable_name>": <variable_type>,
        ...
    }

    def __init__(self, <args>, <kwargs>) -> None:  # eventually, takes arguments from _VARIABLES
        ...

    def __call__(self) -> float:
        return <self._0_<FUNCTION_THIS_REWARD_NEEDS>(<args>, <kwargs>)>  # possibly some other computations
"""

""" HOW TO DEFINE ATOMIC ACTION

class <Some>AtomicAction(AtomicAction):

    _VARIABLES = {
        "<variable_name>": <variable_type>,
        ...
    }

    def __init__(self, **kwds) -> None:  # do not take _VARIABLES in __init__
        super().__init__(**kwds)
        self._predicate: LogicalOperand  # goal condition
        self._initial: LogicalOperand  # initial condition
        self._reward: Operand/Reward  # associated reward

"""


class ApproachReward(Reward):
    _0_EUCLIDEAN_DISTANCE: Callable = "euclidean_distance"

    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, gripper: Variable[Gripper], object: Variable[GraspableObject]) -> None:
        super().__init__(gripper=gripper, object=object)
        self._gripper = gripper
        self._obj = object

    def __call__(self) -> float:
        return ApproachReward._0_EUCLIDEAN_DISTANCE(self._gripper.location, self._obj.location)


class Approach(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
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
        super().__init__()

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


class RotateReward(Reward):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        return 1


class Rotate(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": GraspableObject,
        "angle": AbstractRotation
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        angle = self.get_argument("angle")

        self._initial = ParallelAndOp(
            left=IsHolding(gripper=gripper, object=obj),
            right=Exists(entity=angle)
        )
        self._predicate = ObjectAtPose(object=obj, angle=angle)
        self._reward = RotateReward()


class TransformReward(Reward):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        return 1


class Transform(AtomicAction):
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
        self._reward = TransformReward()


class FollowReward(Reward):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        return 1


class Follow(AtomicAction):
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
        self._reward = FollowReward()
