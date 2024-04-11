import re
from collections import deque
from typing import TypeVar

from rddl import Entity, Operand, Operator

EntityType = TypeVar("EntityType", bound=Entity)  # all subclasses of Entity
OperatorType = TypeVar("OperatorType", bound=Operator, covariant=True)  # all subclasses of Operator
PredicateType = TypeVar("PredicateType", bound=Operand, covariant=True)  # all subclasses of Predicate  #FIXME: maybe should be predicate?



class RDDLParser:
    pred_ex = re.compile(r'(?P<predicate>\w\S*)\(')
    arg_ex = re.compile(r'(?P<args>\w\S*)(?P<end>\s*[,\)]\s*)')
    var_ex = re.compile(r'(?P<var>\w\S*)\:(?P<type>\w\S*)(?P<end>\s*[,\)]\s*)')

    def __init__(self, combinator_mapping: dict[str, type[OperatorType]], predicate_mapping: dict[str, type[PredicateType]], type_definitions: dict[str, type[EntityType]]):
        self.combinator_mapping = combinator_mapping
        self.predicate_mapping = predicate_mapping
        self.type_definitions = type_definitions
        self.all_ops = '|'.join([f'({op})' for op in combinator_mapping.keys()])
        self.op_ex = re.compile(fr'\s*(?P<op>{self.all_ops})\s*')

    def _get_operand(self, name) -> type[Operand]:
        if name not in self.predicate_mapping:
            raise ValueError(f"Unknown function {name}! The mapping provided does not define such function!")
        return self.predicate_mapping[name]

    def _get_operator(self, symbol) -> type[Operator]:
        if symbol not in self.combinator_mapping:
            raise ValueError(f"Unknown operator {symbol}! The mapping provided does not define such operator!")
        return self.combinator_mapping[symbol]

    @staticmethod
    def match_and_trim(regex, text):
        match = regex.match(text)
        if match:
            text = text[match.end():]
        return match, text

    @staticmethod
    def match_predicate(text):
        predicate, args = None, None
        match, text = RDDLParser.match_and_trim(RDDLParser.pred_ex, text)
        if match:
            predicate = match.group('predicate')
            args = []
            while True:
                match, text = RDDLParser.match_and_trim(RDDLParser.arg_ex, text)
                if match:
                    args.append(match.group('args'))
                    if ")" in match.group('end'):
                        break
            print(f"{predicate}: {args}")
        return predicate, args, text

    # def create_predicate(self, predicate, args):
    #     true_args = [self.entity_bindings[arg] for arg in args]
    #     return self._get_function(predicate)(*true_args)

    def create_predicate(self, predicate, args):
        true_args = None  # [self.entity_bindings[arg] for arg in args]
        return self._get_operand(predicate)(*true_args)

    def parse_action_predicate(self, text):
        expression_stack = deque()

        while True:
            predicate, args, text = RDDLParser.match_predicate(text)
            if predicate:
                current_predicate: Operand = self.create_predicate(predicate, args)
                expression_stack.append(current_predicate)
            else:
                break

    def parse(self, text):
        expression_stack = deque()
        operator_stack = deque()

        while True:
            predicate, args, text = RDDLParser.match_predicate(text)
            if predicate:
                current_predicate = self.create_predicate(predicate, args)
                expression_stack.append(current_predicate)
            else:
                break
            match, text = RDDLParser.match_and_trim(self.op_ex, text)
            if match:
                operator: type[Operator] = self._get_operator(match.group('op'))
                operator_stack.append(operator)
            else:
                break

        while operator_stack:
            operator = operator_stack.popleft()
            operands = [expression_stack.popleft() for _ in range(operator.ARITY)]
            expression_stack.appendleft(operator(*operands))

        full_expression = expression_stack.pop()
        return full_expression
