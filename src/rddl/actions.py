from rddl import AtomicAction, Reward, Variable
from rddl.entities import Gripper, ObjectEntity
from rddl.operators import NotOp, ParallelAndOp
from rddl.predicates import IsReachable, Near


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
        gripper = self.get_argument("gripper")
        obj = self.get_argument("object")
        self._predicate = Near(object_A=gripper, object_B=obj)
        self._initial = ParallelAndOp(left=IsReachable(gripper=gripper, location=obj), right=NotOp(operand=self._predicate))
        self._reward = ApproachReward(gripper, obj)


class WithdrawReward(Reward):

    def __init__(self) -> None:
        super().__init__()


class Withdraw(AtomicAction):
    _VARIABLES = {
        "gripper": Gripper,
        "object": ObjectEntity
    }

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        self._initial = Near(object_A=self.get_argument("gripper"), object_B=self.get_argument("object"))
        self._predicate = NotOp(operand=self._initial)
        self._reward = NotOp(ApproachReward(self.get_argument("gripper"), self.get_argument("object")))
