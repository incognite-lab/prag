from abc import ABCMeta

from rddl import Predicate, Reward


class AtomicAction(Predicate, metaclass=ABCMeta):
    """Container for atomic action. AA is an operand (it can be evaluated or decided)
    and combines initial condition, goal (terminating) condition and reward function.
    """

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        self._predicate: Predicate
        self._reward: Reward

    def __call__(self):
        """ Decide whether goal condition is met.

        Returns:
            bool: True if goal condition is met, False otherwise.
        """
        print(f"Checking action {str(self.__class__.__name__)}")
        return self._predicate.decide()

    def reward(self):
        """Evaluate reward function for the action.

        Returns:
            float: The reward value.
        """
        print(f"Evaluating action {str(self.__class__.__name__)}")
        return self._reward.evaluate()
