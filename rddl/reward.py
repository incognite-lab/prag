from rddl.predicate import Predicate


class Reward(Predicate):

    def __init__(self) -> None:
        self._predicate: Predicate

    def evaluate(self):
        return self._predicate.evaluate()
