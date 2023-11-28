from typing import ClassVar
# from entities import Entity
from abc import ABCMeta
from rddl.predicate import Predicate
from rddl.operand import Operand


class Operator(Operand, metaclass=ABCMeta):
    _ARITY: ClassVar[int]
    _SYMBOL: ClassVar[str]

    def __init_subclass__(cls) -> None:
        dd = dir(cls)
        if '_SYMBOL' not in dd:
            raise ValueError(f"Class '{cls}' does not have a '_SYMBOL' attribute! Every sub-class of 'Operator' must define a '_SYMBOL' that defines its string representation!")
        if '_ARITY' not in dd:
            raise ValueError(f"Class '{cls}' does not have a '_ARITY' attribute! Every sub-class of 'Operator' must define a '_ARITY' that defines arity of the operation!")
        return super().__init_subclass__()

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    @property
    def SYMBOL(cls) -> str:
        return cls._SYMBOL

    @classmethod
    @property
    def ARITY(cls) -> int:
        return cls._ARITY


class NullaryOperator(Operator):
    _ARITY = 0

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self._SYMBOL


class UnaryOperator(Operator):
    _ARITY = 1

    def __init__(self, operand: Operand) -> None:
        super().__init__()
        self._operand = operand

    def __repr__(self) -> str:
        return f"{self._SYMBOL}{self._operand}"


class BinaryOperator(Operator):
    _ARITY = 2

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__()
        self._left: Operand = left
        self._right: Operand = right

    def __repr__(self) -> str:
        return f"{self._left} {self._SYMBOL} {self._right}"


class SequentialAnd(BinaryOperator):
    _SYMBOL: ClassVar[str] = "&"

    def __init__(self, left: Predicate, right: Predicate) -> None:
        super().__init__(left, right)

    def _evaluate(self) -> float:
        return self._left.evaluate() and self._right.evaluate()


class AndOp(BinaryOperator):
    _SYMBOL = "&"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def decide(self):
        left_check = self._left.decide()
        right_check = self._right.decide()
        print(f"Checking and operator; left: {left_check}, right: {right_check}, result: {left_check and right_check}")
        result = left_check and right_check
        return result

    def evaluate(self):
        left_eval = self._left.evaluate()
        right_eval = self._right.evaluate()
        print(f"Evaluating and operator; left: {left_eval}, right: {right_eval}, result: {left_eval + right_eval}")
        result = left_eval + right_eval
        return result


class SequentialOp(BinaryOperator):
    _SYMBOL = "->"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def decide(self):
        first_result = self._left.decide()
        after_result = self._right.decide()
        print(f"Checking sequential operator; first: {first_result}, after: {after_result}")
        return after_result

    def evaluate(self):
        first_evaluation = self._left.evaluate()
        after_evaluation = self._right.evaluate()
        print(f"Evaluating sequential operator; first: {first_evaluation}, after: {after_evaluation}")
        return first_evaluation + after_evaluation if self._left.decide() else first_evaluation
