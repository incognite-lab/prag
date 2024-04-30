# from entities import Entity
from typing import Optional

from rddl import Operator
from rddl.core import LogicalOperand, Operand, Variable


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
        self._operand: Operand = operand
        self._append_arguments(operand)

    def __repr__(self) -> str:
        return f"{self._SYMBOL}({self._operand})"

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        return self._operand.set_symbolic_value(value, only_if_contains)

    def gather_variables(self) -> list[Variable]:
        return self._operand.gather_variables() if isinstance(self._operand, LogicalOperand) else []


class BinaryOperator(Operator):
    _ARITY = 2

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__()
        self._left: Operand = left
        self._right: Operand = right
        self._append_arguments(left, right)

    def __repr__(self) -> str:
        return f"({self._left} {self._SYMBOL} {self._right})"

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        self._left.set_symbolic_value(value, only_if_contains)
        self._right.set_symbolic_value(value, only_if_contains)

    def gather_variables(self) -> list[Variable]:
        return (self._left.gather_variables() if isinstance(self._left, LogicalOperand) else []) + (self._right.gather_variables() if isinstance(self._right, LogicalOperand) else [])


class NAryOperator(Operator):
    _ARITY = None

    def __init__(self, operands: list[Operand]) -> None:
        super().__init__()
        self._operands: list[Operand] = operands
        self._append_arguments(*operands)

    def __repr__(self) -> str:
        return f' {self._SYMBOL} '.join([str(op) for op in self._operands])

# class SequentialAndOp(BinaryOperator):
#     _SYMBOL = "&>"

#     def __init__(self, left: Predicate, right: Predicate) -> None:
#         super().__init__(left, right)

#     def __evaluate__(self) -> float:
#         return self._left.evaluate() and self._right.evaluate()


class ParallelAndOp(BinaryOperator):
    _SYMBOL = "&"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def __decide__(self):
        left_check = self._left.decide()
        right_check = self._right.decide()
        # print(f"Checking and operator; left: {left_check}, right: {right_check}, result: {left_check and right_check}")
        result = left_check and right_check
        return result

    def __evaluate__(self):
        left_eval = self._left.evaluate()
        right_eval = self._right.evaluate()
        # print(f"Evaluating and operator; left: {left_eval}, right: {right_eval}, result: {left_eval + right_eval}")
        result = left_eval + right_eval
        return result


class SequentialOp(BinaryOperator):
    _SYMBOL = "->"

    def __init__(self, left: Operand, right: Operand) -> None:
        super().__init__(left, right)

    def __decide__(self):
        first_result = self._left.decide()
        after_result = self._right.decide()
        # print(f"Checking sequential operator; first: {first_result}, after: {after_result}")
        return after_result

    def __evaluate__(self):
        first_evaluation = self._left.evaluate()
        after_evaluation = self._right.evaluate()
        # print(f"Evaluating sequential operator; first: {first_evaluation}, after: {after_evaluation}")
        return first_evaluation + after_evaluation if self._left.decide() else first_evaluation


class NotOp(UnaryOperator):
    _SYMBOL = "~"

    def __init__(self, operand: Operand) -> None:
        super().__init__(operand)

    def __decide__(self):
        return not self._operand.decide()

    def __evaluate__(self):
        return -self._operand.evaluate()

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        self._operand.set_symbolic_value(not value, only_if_contains)


class NAryAndOp(NAryOperator):
    _SYMBOL = "&"

    def __init__(self, operands: list[Operand]) -> None:
        super().__init__(operands)

    def __decide__(self):
        return all(op.decide() for op in self._operands)

    def __evaluate__(self):
        return sum(op.evaluate() for op in self._operands)

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        for op in self._operands:
            op.set_symbolic_value(value, only_if_contains)
