from copy import deepcopy
from re import A
from typing import Callable, ClassVar, Generator, Iterable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from testing_utils import Apple

from rddl import AtomicAction, Operand, Variable
from rddl.actions import Approach, Drop, Grasp, Move, Withdraw
from rddl.core import Entity, LogicalOperand, Predicate, SymbolicCacheContainer
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

    def show_table(self):
        c = self._cache_decide
        for key, value in c.items():
            hashed_internal_args = key.arg_list
            try:
                args = ', '.join([self._variable_registry[id].name for id in hashed_internal_args])
            except KeyError:
                continue
            if key.other:
                args += ', ' + ', '.join(key.other)
            print(f"{key.class_name.__name__}({args}) -> {value}")

    @property
    def variables(self):
        return self._variable_registry

    def __contains__(self, variable: Variable) -> bool:
        return variable.id in self._variable_registry

    def clone(self):
        s = StrictSymbolicCacheContainer()
        s._variable_registry = deepcopy(self._variable_registry)
        s._cache_decide = self._cache_decide.clone()
        return s


class Weighter:
    INITIAL_WEIGHT_PENALTY_COEFF = 0.9
    WEIGHT_PENALTY_COEFF = 0.9
    RETRY_AD_INFINITUM: ClassVar[bool] = True
    EPS = np.finfo(float).min

    RNG: ClassVar[np.random.Generator]

    def __init__(self, items: Iterable[AtomicAction], weights: Optional[list[float]] = None, initial_weights: Optional[list[float]] = None) -> None:
        self._items = np.asarray(items)
        if weights is None:
            self._weights = {item.__name__: 1.0 for item in items}
        else:
            self._weights = {item.__name__: weight for item, weight in zip(items, weights)}
        self._initial_weights = {item.__name__: 1.0 for item in items}
        if initial_weights is None:
            self._initial_weights = {item.__name__: 1.0 for item in items}
        else:
            self._initial_weights = {item.__name__: weight for item, weight in zip(items, initial_weights)}

        self._bkp_weights = deepcopy(self._weights)
        self._bkp_initial_weights = deepcopy(self._initial_weights)

    @staticmethod
    def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction, list[Variable]], None, None]:
        n_choices = len(choices)
        choice_idx = 0

        while condition(choice_idx, n_choices):
            choice_idx %= n_choices
            action = choices[choice_idx]()  # get and instantiate next action
            a_variables = action.gather_variables()
            yield action, a_variables
            choice_idx += 1

    def get_random_generator(self) -> Generator:
        choices = self._weighted_shuffle(list(self._weights.values()))  # shuffle action classes (so they are in varying order each time)
        condition = lambda ci, nc, ad_inf=self.RETRY_AD_INFINITUM: ad_inf or ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def get_initial_generator(self) -> Generator:
        choices = self._weighted_shuffle(list(self._initial_weights.values()))  # shuffle action classes (so they are in varying order each time)
        choices = np.asarray([c for c in choices if self._initial_weights[c.__name__] > 0.0])  # cleanup improbable choices
        condition = lambda ci, nc: ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def _weighted_shuffle(self, weights) -> NDArray:
        order = sorted(range(len(self._items)), key=lambda i: self.RNG.random() ** np.exp(1.0 / (weights[i] + self.EPS)))
        return np.asarray([self._items[i] for i in order])

    def penalize_action(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        self._weights[cls_name] = self.WEIGHT_PENALTY_COEFF * self._weights[cls_name]

    def penalize_initial_action(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        self._initial_weights[cls_name] = self.INITIAL_WEIGHT_PENALTY_COEFF * self._initial_weights[cls_name]

    def reset_weights(self) -> None:
        self._weights = deepcopy(self._bkp_weights)
        self._initial_weights = deepcopy(self._bkp_initial_weights)

    @classmethod
    def set_rng(cls, rng: np.random.Generator) -> None:
        cls.RNG = rng


class RDDLWorld:

    ROBOTS: NDArray = np.asanyarray([Gripper])
    OBJECTS: NDArray = np.asanyarray([ObjectEntity])
    VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach])
    VALID_ACTIONS: NDArray = np.asanyarray([Approach, Withdraw, Grasp, Drop, Move])
    RNG = np.random.default_rng(SEED)

    STATE_IDLE = 1
    STATE_GENERATING = 2

    def __init__(self) -> None:
        self._symbolic_table = StrictSymbolicCacheContainer()
        # self._entities = {}
        self._allowed_entities = []
        Weighter.set_rng(self.RNG)
        self._weighter = Weighter(
            items=self.VALID_ACTIONS,
            initial_weights=[1.0 if a in self.VALID_INITIAL_ACTIONS else 0.0 for a in self.VALID_ACTIONS]
        )
        self.full_reset()

    def sample_generator(self,
                         sequence_length: int = 2,
                         add_robots: bool = True,
                         retry_ad_infinitum: bool = True) -> Generator[AtomicAction, bool, bool]:
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=add_robots)
        action: AtomicAction
        added_vars: list[Variable] = []

        sample_idx = 0
        self._initial_world_state: StrictSymbolicCacheContainer
        self._goal_world_state: StrictSymbolicCacheContainer

        Weighter.RETRY_AD_INFINITUM = retry_ad_infinitum

        # sample random action
        while sample_idx < sequence_length:
            action_generator = self._weighter.get_random_generator() if sample_idx > 0 else self._weighter.get_initial_generator()
            yield_accepted = False
            while not yield_accepted:
                if sample_idx == 0:
                    try:
                        action, a_variables = next(action_generator)
                    except StopIteration:
                        raise RuntimeError("Failed to sample initial action! This should never happen. Check if the initial action space is not empty.")
                    self._lookup_and_bind_variables(a_variables)
                    action.initial.set_symbolic_value(True)
                    self._initial_world_state = self._symbolic_table.clone()
                else:
                    while True:
                        action, a_variables = next(action_generator)
                        added_vars = self._lookup_and_bind_variables(a_variables)
                        if added_vars:
                            action.initial.set_symbolic_value(True, set(added_vars))
                        if action.initial.decide():
                            break
                        # TODO: cleanup if initial still not true (clone sym. table and delete)
                        self._remove_variables(added_vars)

                # if response is not None and not response:
                #     self._remove_variables(added_vars)
                # else:
                yield_accepted = True

            action.predicate.set_symbolic_value(True)
            response = yield action

            if sample_idx > 0:
                self._weighter.penalize_action(action)
            else:
                self._weighter.penalize_initial_action(action)
            sample_idx += 1

        # self._symbolic_table.show_table()
        self._goal_world_state = self._symbolic_table.clone()
        self.deactivate_symbolic_mode()
        return sample_idx == sequence_length

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

    def show_world_state(self) -> None:
        self._symbolic_table.show_table()

    def show_initial_world_state(self) -> None:
        self._initial_world_state.show_table()

    def show_goal_world_state(self) -> None:
        self._goal_world_state.show_table()

    def set_allowed_entities(self, entities: list[type]) -> None:
        self._allowed_entities = entities

    def _lookup_and_bind_variables(self, variables: list[Variable]) -> list[Variable]:
        missing_vars = []
        linked_vars = []
        for v in variables:
            like_gen = self._find_something_like_this(v)
            try:
                while (like_v := next(like_gen)) is not None:
                    if like_v not in linked_vars:
                        break
            except StopIteration:
                like_v = None

            if like_v is None:
                like_v = self._get_random_variable(v.type)
                missing_vars.append(like_v)
                self._add_variable(v.name, like_v)  # FIXME: name should be somehow estimated
            v.link_to(like_v)
            linked_vars.append(like_v)
        return missing_vars

    def _find_something_like_this(self, variable: Variable) -> Generator[Variable, None, Optional[Variable]]:
        for v in self._variables.values():
            if issubclass(v.type, variable.type):
                yield v
        return None

    def _initialize_world(self, add_robots: bool = True):
        self.activate_symbolic_mode()
        if add_robots:
            self._add_variable("gripper", self._get_random_variable(Gripper))

    def _add_variable(self, name: str, variable: SymbolicEntity) -> None:
        self._variables[name] = variable
        self._symbolic_table.register_variable(variable)

    def _remove_variables(self, variables: list[Variable]) -> None:
        self._symbolic_table.remove_variables(variables)
        for variable in variables:
            for local_name, local_var in self._variables.items():
                if local_var == variable:
                    del self._variables[local_name]
                    break

    def _sample_subclass(self, base_type: type[Entity]) -> type[Entity]:
        if self._allowed_entities:
            options = np.asanyarray([sc for sc in base_type.list_subclasses() if sc in self._allowed_entities])
        else:
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
        self._weighter.reset_weights()

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
