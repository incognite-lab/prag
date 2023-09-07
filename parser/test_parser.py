import re
from test_comp import Reach, Grasp, Move, Release, Leave, AndOp, SequentialOp, Operator, IsHolding, Location, Gripper, ObjectEntity
from collections import deque


function_mapping = {
    "reach": Reach,
    "grasp": Grasp,
    "move": Move,
    "release": Release,
    "leave": Leave
}

entity_bindings = {
    "gripper": Gripper(),
    "apple": ObjectEntity(),
    "loc": Location()
}

operator_mapping = {
    "&": AndOp,
    "->": SequentialOp
}


class Parser:

    def __init__(self):
        pass

    def parse(self, text):
        pred_ex = re.compile(r'(?P<predicate>\w\S*)\(')
        arg_ex = re.compile(r'(?P<args>\w\S*)(?P<end>\s*[,\)]\s*)')
        all_ops = '|'.join([f'({op})' for op in operator_mapping.keys()])
        op_ex = re.compile(fr'\s*(?P<op>{all_ops})\s*')

        def match_and_trim(regex, text):
            match = regex.match(text)
            if match:
                text = text[match.end():]
            return match, text

        def match_predicate(text):
            predicate, args = None, None
            match, text = match_and_trim(pred_ex, text)
            if match:
                predicate = match.group('predicate')
                args = []
                while True:
                    match, text = match_and_trim(arg_ex, text)
                    if match:
                        args.append(match.group('args'))
                        if ")" in match.group('end'):
                            break
                print(f"{predicate}: {args}")
            return predicate, args, text

        def create_predicate(predicate, args):
            true_args = [entity_bindings[arg] for arg in args]
            return function_mapping[predicate](*true_args)

        expression_stack = deque()
        operator_stack = deque()
        while True:
            predicate, args, text = match_predicate(text)
            if predicate:
                current_predicate = create_predicate(predicate, args)
                expression_stack.append(current_predicate)
            else:
                break
            match, text = match_and_trim(op_ex, text)
            if match:
                operator = operator_mapping[match.group('op')]
                operator_stack.append(operator)
            else:
                break

        while operator_stack:
            operator = operator_stack.popleft()
            operands = [expression_stack.popleft() for i in range(operator.ARITY)]
            expression_stack.appendleft(operator(*operands))

        full_expression = expression_stack.pop()
        return full_expression


if __name__ == '__main__':
    action = "reach(gripper, apple) & grasp(gripper, apple) ->" \
        "move(apple, loc) -> release(gripper, apple) -> leave(gripper, apple)"

    parser = Parser()
    result = parser.parse(action)

    print(result)
    print(result.check())
    print(result.evaluate())
