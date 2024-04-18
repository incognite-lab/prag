from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from rddl import AtomicAction, Operand, Variable
from rddl.action import Approach, Withdraw
from rddl.core import Entity, SymbolicCacheContainer
from rddl.entity import Gripper, ObjectEntity
from rddl.rddl_parser import (EntityType, OperatorType, PredicateType,
                              RDDLParser)

SEED = 0


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


class RDDLWorld:

    ROBOTS: ArrayLike = np.asanyarray([Gripper])
    OBJECTS: ArrayLike = np.asanyarray([ObjectEntity])
    VALID_INITIAL_ACTIONS: ArrayLike = np.asanyarray([Approach])
    VALID_ACTIONS: ArrayLike = np.asanyarray([Approach, Withdraw])
    RNG = np.random.default_rng(SEED)

    def __init__(self) -> None:
        self._symbolic_table = StrictSymbolicCacheContainer()
        self._variables = {}
        # self._entities = {}

    def sample_world(self, sequence_length: int = 2, add_robots: bool = True):
        self._initialize_world(add_robots=add_robots)
        action_class: type[AtomicAction]
        action: AtomicAction

        # sample random action
        for i in range(sequence_length):
            if i == 0:
                action_class = self.RNG.choice(self.VALID_INITIAL_ACTIONS)
                action = action_class()
                a_variables = action.gather_variables()
                self._lookup_and_bind_variables(a_variables)
                action.initial.set_symbolic_value(True)
                print(action.initial.decide())
            else:
                action_class = self.RNG.choice(self.VALID_ACTIONS)
                action = action_class()

        # check action variables
        # add missing variables
        # make init = true
        # make goal = true

    def _lookup_and_bind_variables(self, variables: list[Variable]):
        for v in variables:
            like_v = self._find_something_like_this(v)
            if like_v is None:
                like_v = self._get_random_variable(v.type)
                self._add_variable(v.name, like_v)  # FIXME: name should be somehow estimated
            v.link_to(like_v)

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
