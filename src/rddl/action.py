from rddl import AtomicAction, Reward, Variable
from rddl.entity import Gripper, ObjectEntity
from rddl.operator import NotOp
from rddl.predicate import Near


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
        self._predicate = Near(object_A=self.get_argument("gripper"), object_B=self.get_argument("object"))
        self._initial = NotOp(operand=self._predicate)
        self._reward = ApproachReward(self.get_argument("gripper"), self.get_argument("object"))


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
