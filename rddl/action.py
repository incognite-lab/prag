from abc import ABCMeta

from rddl import Operand
from rddl.predicate import Predicate
from rddl.reward import Reward


class AtomicAction(Operand, metaclass=ABCMeta):
    """Container for atomic action. AA is an operand (it can be evaluated or decided)
    and combines initial condition, goal (terminating) condition and reward function.
    """

    def __init__(self) -> None:
        self._predicate: Predicate
        self._reward: Reward

    def decide(self):
        """ Decide whether goal condition is met.

        Returns:
            bool: True if goal condition is met, False otherwise.
        """
        print(f"Checking action {str(self.__class__)}")
        return self._predicate.decide()

    def evaluate(self):
        """Evaluate reward function for the action.

        Returns:
            float: The reward value.
        """
        print(f"Evaluating action {str(self.__class__)}")
        return self._reward.evaluate()

    def __repr__(self) -> str:
        return f"AA: {str(self.__class__)}"
