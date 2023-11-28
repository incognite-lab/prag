from rddl.operand import Operand
from rddl.predicate import Predicate
from rddl.reward import Reward
from abc import ABCMeta


class AtomicAction(Operand, metaclass=ABCMeta):

    def __init__(self) -> None:
        self._predicate: Predicate
        self._reward: Reward

    def decide(self):
        print(f"Checking action {str(self.__class__)}")
        return self._predicate.decide()

    def evaluate(self):
        print(f"Evaluating action {str(self.__class__)}")
        return self._reward.evaluate()
