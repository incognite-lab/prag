from re import A
from typing import Generator, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from testing_utils import Apple

from rddl import AtomicAction, Operand, Variable
from rddl.actions import Approach, Withdraw
from rddl.core import Entity, SymbolicCacheContainer
from rddl.entities import Gripper, ObjectEntity
from rddl.predicates import IsReachable, Near
from rddl.rddl_parser import (EntityType, OperatorType, PredicateType,
                              RDDLParser)

SEED = np.random.randint(0, 2**32)


class SymbolicEntity(Variable):

    def __init__(self, typ: type[Entity]):
        super().__init__(typ, base_name=f"entity_{typ.__name__}")


class StrictSymbolicCacheContainer(SymbolicCacheContainer):

    def __init__(self):
        super().__init__()
        self._variable_registry = {}

    def _make_key(self, func, internal_args, *args, **kwds):
        for arg in internal_args:
            if arg not in self._variable_registry:
                raise ValueError(f"Variable with id '{arg}' is not registered in the SymbolicCacheContainer! Register a variable with the correct id (global name) first.")
        return super()._make_key(func, internal_args, *args, **kwds)

    def register_variable(self, variable: Variable) -> None:
        self._variable_registry[variable.id] = variable

    def remove_variables(self, variables: list[Variable]) -> None:
        for variable in variables:
            del self._variable_registry[variable.id]

    @property
    def variables(self):
        return self._variable_registry

    def __contains__(self, variable: Variable) -> bool:
        return variable.id in self._variable_registry


def weighted_shuffle(items, weights, RNG) -> NDArray:
    order = sorted(range(len(items)), key=lambda i: RNG.random() ** (1.0 / weights[i]))
    return np.asarray([items[i] for i in order])


class RDDLWorld:

    ROBOTS: NDArray = np.asanyarray([Gripper])
    OBJECTS: NDArray = np.asanyarray([ObjectEntity])
    VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach])
    VALID_ACTIONS: NDArray = np.asanyarray([Approach, Withdraw])
    RNG = np.random.default_rng(SEED)

    INITIAL_ACTION_WEIGHT_PENALTY_COEFF = 0.9
    ACTION_WEIGHT_PENALTY_COEFF = 0.9

    STATE_IDLE = 1
    STATE_GENERATING = 2

    def __init__(self) -> None:
        self._symbolic_table = StrictSymbolicCacheContainer()
        # self._entities = {}
        self.full_reset()

    def sample_generator(self,
                         sequence_length: int = 2,
                         add_robots: bool = True,
                         retry_ad_infinitum: bool = True) -> Generator[AtomicAction, bool, bool]:
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=add_robots)
        action: AtomicAction

        def get_random_action(choices: NDArray, weights_dict) -> Generator[tuple[AtomicAction, list[Variable]], None, None]:
            action_classes = weighted_shuffle(choices, list(weights_dict.values()), self.RNG)  # shuffle action classes (so they are in varying order each time)
            n_choices = len(action_classes)
            choice_idx = 0
            condition = lambda ci, nc, ad_inf=retry_ad_infinitum: True if ad_inf else ci < nc

            while condition(choice_idx, n_choices):
                choice_idx %= n_choices
                action = action_classes[choice_idx]()  # get and instantiate next action
                a_variables = action.gather_variables()
                yield action, a_variables
                choice_idx += 1

        sample_idx = 0
        # sample random action
        while sample_idx < sequence_length:
            action_generator = get_random_action(self.VALID_INITIAL_ACTIONS, self._initial_action_weights) if sample_idx == 0 else get_random_action(self.VALID_ACTIONS, self._action_weights)
            if sample_idx == 0:
                action, a_variables = next(action_generator)
                self._lookup_and_bind_variables(a_variables)
                action.initial.set_symbolic_value(True)
            else:
                while True:
                    action, a_variables = next(action_generator)
                    added_vars = self._lookup_and_bind_variables(a_variables)
                    if action.initial.decide():
                        break
                    self._symbolic_table.remove_variables(added_vars)

            action.predicate.set_symbolic_value(True)
            response = yield action

            if response:
                continue

            if sample_idx > 0:
                self._penalize_action(action)
            else:
                self._penalize_initial_action(action)
            sample_idx += 1

        self.deactivate_symbolic_mode()
        return sample_idx == sequence_length

    def _penalize_action(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        self._action_weights[cls_name] = self.ACTION_WEIGHT_PENALTY_COEFF * self._action_weights[cls_name]

    def _penalize_initial_action(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        self._initial_action_weights[cls_name] = self.INITIAL_ACTION_WEIGHT_PENALTY_COEFF * self._initial_action_weights[cls_name]

    def sample_world(self, sequence_length: int = 2, add_robots: bool = True) -> tuple[list[AtomicAction], list[Variable]]:
        generator = self.sample_generator(sequence_length=sequence_length, add_robots=add_robots)
        action_sequence: list[AtomicAction] = list(generator)

        # Only for testing purposes:
        self.activate_symbolic_mode()
        a, b = Variable(Gripper, global_name="entity_TiagoGripper_1"), Variable(Apple, global_name="entity_Apple_2")
        n = Near(object_A=a, object_B=b)
        print(IsReachable(gripper=a, location=b).decide())
        print(n.decide())
        self.deactivate_symbolic_mode()


        # check action variables
        # add missing variables
        # make init = true
        # make goal = true

        return action_sequence, self.get_created_variables()

    def get_created_variables(self) -> list[Variable]:
        return list(self._variables.values())

    def _lookup_and_bind_variables(self, variables: list[Variable]) -> list[Variable]:
        missing_vars = []
        for v in variables:
            like_v = self._find_something_like_this(v)
            if like_v is None:
                like_v = self._get_random_variable(v.type)
                missing_vars.append(like_v)
                self._add_variable(v.name, like_v)  # FIXME: name should be somehow estimated
            v.link_to(like_v)
        return missing_vars

    def _find_something_like_this(self, variable: Variable) -> Optional[Variable]:
        for v in self._variables.values():
            if issubclass(v.type, variable.type):
                return v
        else:
            return None

    def _initialize_world(self, add_robots: bool = True):
        self.activate_symbolic_mode()
        if add_robots:
            self._add_variable("gripper", self._get_random_variable(Gripper))

    def _add_variable(self, name: str, variable: SymbolicEntity) -> None:
        self._variables[name] = variable
        self._symbolic_table.register_variable(variable)

    def _sample_subclass(self, base_type: type[Entity]) -> type[Entity]:
        options = np.asanyarray([sc for sc in base_type.list_subclasses()])
        return self.RNG.choice(options)

    def _get_random_variable(self, typ: type[Entity]) -> SymbolicEntity:
        return SymbolicEntity(self._sample_subclass(typ))

    def activate_symbolic_mode(self):
        Operand.set_cache_symbolic(self._symbolic_table)

    def deactivate_symbolic_mode(self):
        Operand.set_cache_normal()

    def reset_world(self):
        self.__state = self.STATE_IDLE
        self._variables = {}
        self._symbolic_table.reset()

    def reset_weights(self):
        self._initial_action_weights: dict[str, float] = dict(zip([action.__name__ for action in self.VALID_INITIAL_ACTIONS], np.ones(len(self.VALID_INITIAL_ACTIONS))))
        self._action_weights: dict[str, float] = dict(zip([action.__name__ for action in self.VALID_ACTIONS], np.ones(len(self.VALID_ACTIONS))))

    def full_reset(self):
        self.reset_world()
        self.reset_weights()


class RDDL:

    def __init__(self):
        self._parser = None

    def initialize_parser(self, combinator_mapping: dict[str, type[OperatorType]], predicate_mapping: dict[str, type[PredicateType]], type_definitions: dict[str, type[EntityType]]):
        self._parser = RDDLParser(combinator_mapping, predicate_mapping, type_definitions)

    def load_definitions(self, aa_definitions: dict[str, dict]):
        if self._parser is None:
            raise ValueError("Please initialize the parser first!")

        for action_name, aa_def in aa_definitions.items():
            if action_name in self._parser.predicate_mapping:
                raise ValueError(f"The name {action_name} is already defined as a function!")
            action = self._extract_action(action_name, aa_def)
            self._parser.predicate_mapping[action_name] = action

    def step(self):
        Operand.reset_cache()

    def _extract_action(self, action_name: str, action_def: dict[str, dict[str, str]]) -> type[AtomicAction]:
        if self._parser is None:
            raise ValueError("Please initialize the parser first!")

        if "predicate" not in action_def:
            raise ValueError(f"Action {action_name} does not have a predicate!")
        predicate = self._parser.parse_action_predicate(action_def["predicate"])

        return type(action_name, (AtomicAction,), {
            "_predicate": predicate,
        })


def gentoo(n=10):
    a = 33
    for i in range(n):
        a = a + i
        ret = yield a
        if ret is not None:
            print(ret)
