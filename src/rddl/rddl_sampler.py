from collections import deque
from copy import deepcopy
from threading import Lock, Thread
from typing import Callable, ClassVar, Generator, Iterable, Optional, Type, Union

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
#from testing_utils import Apple

from rddl import AtomicAction, Operand, Variable
from rddl.actions import Approach, Drop, Follow, Grasp, Move, Rotate, Withdraw, Transform
from rddl.core import Entity, LogicalOperand, Predicate, SymbolicCacheContainer
from rddl.entities import Gripper, ObjectEntity
from rddl.predicates import IsReachable, Near
from rddl.rddl_parser import (EntityType, OperatorType, PredicateType,
                              RDDLParser)
from memory_profiler import profile


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
        self._variable_registry[variable.symbolic_id] = variable

    def remove_variables(self, variables: list[Variable]) -> None:
        for variable in variables:
            del self._variable_registry[variable.symbolic_id]

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
        return variable.symbolic_id in self._variable_registry

    def clone(self):
        s = StrictSymbolicCacheContainer()
        s._cache_sentinel = self._cache_sentinel
        s._variable_registry = deepcopy(self._variable_registry)
        s._cache_decide = self._cache_decide.clone()
        return s

    def reset(self):
        super().reset()
        self.remove_variables(list(self._variable_registry.values()))


class Weighter:
    INITIAL_WEIGHT_PENALTY_COEFF = 0.9  # coefficient to multiply the initial weight when an action is selected
    WEIGHT_PENALTY_COEFF = 0.9  # coefficient to multiply the weight when an action is selected
    SEQUENCE_PENALTY_COEFF = 0.95  # coefficient to multiply the sequence weight when an action is selected, given previous actions
    RETRY_AD_INFINITUM: ClassVar[bool] = True  # whether to supply actions (from randomly shuffled sequence) for ever or terminate after single loop through the sequence
    EPS = np.finfo(float).min

    RNG: ClassVar[np.random.Generator]

    MODE_NONE: ClassVar[int] = 0
    MODE_INITIAL: ClassVar[int] = 1  # use initial weights
    MODE_WEIGHT: ClassVar[int] = 2  # use weights for individual actions
    MODE_SEQUENCE: ClassVar[int] = 4  # use sequence weights
    MODE_RANDOM: ClassVar[int] = 8  # randomize the total weight to add noise to the weights
    MODE_MAX_NOISE: ClassVar[int] = 16

    BASE_MODE: ClassVar[int] = MODE_WEIGHT | MODE_INITIAL | MODE_SEQUENCE | MODE_RANDOM | MODE_MAX_NOISE

    def __init__(self, items: Iterable[AtomicAction], weights: Optional[list[float]] = None, initial_weights: Optional[list[float]] = None) -> None:
        self.set_mode(self.BASE_MODE)

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

        # if self._mode & self.MODE_SEQUENCE:
        self._previous_item_queue = deque(maxlen=3)
        self._weight_graph = nx.MultiDiGraph()

    @staticmethod
    def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction, list[Variable]], None, None]:
        """Yields items from the list of choices.

        Args:
            choices (NDArray): list of items to loop through.
            condition (Callable): function that returns True if the loop should continue.

        Yields:
            Generator[tuple[AtomicAction, list[Variable]], None, None]: Returns (action, [list of variables]).
        """
        n_choices = len(choices)
        choice_idx = 0

        while condition(choice_idx, n_choices):
            choice_idx %= n_choices
            action = choices[choice_idx]()  # get and instantiate next action
            a_variables = action.gather_variables()
            yield action, a_variables
            choice_idx += 1

    def set_mode(self, mode: int) -> None:
        """Sets the mode of the weighter.

        Args:
            mode (int): Mode to set.
        """
        self._mode = mode
        if self._mode & self.MODE_RANDOM:
            self._sampling_key_function = lambda weights: lambda i: self.RNG.random() ** np.exp(1.0 / (weights[i] + self.EPS))
        else:
            self._sampling_key_function = lambda weights: lambda i: 1 - weights[i]

    def _get_seq_weights(self) -> list[float]:
        """Computes weights based on the sequence of previous actions and individual action weights.


        Returns:
            list[float]: List of weights.
        """
        coefs = deepcopy(self._weights)  # get a copy of the individual weights
        for item, weight in coefs.items():  # for each action / weight
            w_data = self._weight_graph.get_edge_data(self._previous_item_queue[-1], item)
            if w_data is None:
                continue
            preceding = self._previous_item_queue[-2]  # find the action preceding the previous action
            triple_w = w_data[preceding]['weight'] if preceding in w_data else 1
            coefs[item] = weight * w_data[0]['weight'] * triple_w  # individual weight * previous->current weight * preceding->previous->current weight
        return list(coefs.values())

    def _add_noise(self, weights: NDArray) -> NDArray:
        m = weights.max()
        where_max = weights == m
        noise = self.RNG.uniform(0, 0.01, np.count_nonzero(where_max))
        weights[where_max] += noise
        return weights

    def get_random_generator(self) -> Generator:
        """Yields items from the list of choices. The items are randomly shuffled based on their weights.

        Returns:
            Generator: Generator yeilding (action, [list of variables]).

        Yields:
            action, list[Variable]: Returns (action, [list of variables]).
        """
        if self._mode & self.MODE_SEQUENCE and len(self._previous_item_queue) == 3:
            weighted_weights = np.array(self._get_seq_weights())
        else:
            weighted_weights = np.array(list(self._weights.values()))

        weighted_weights = self._add_noise(weighted_weights)

        choices = self._weighted_shuffle(weighted_weights)  # shuffle action classes (so they are in varying order each time)
        condition = lambda ci, nc, ad_inf=self.RETRY_AD_INFINITUM: ad_inf or ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def get_initial_generator(self) -> Generator:
        choices = self._weighted_shuffle(list(self._initial_weights.values()))  # shuffle action classes (so they are in varying order each time)
        choices = np.asarray([c for c in choices if self._initial_weights[c.__name__] > 0.0])  # cleanup improbable choices
        condition = lambda ci, nc: ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def _weighted_shuffle(self, weights) -> NDArray:
        order = sorted(range(len(self._items)), key=self._sampling_key_function(weights))
        return np.asarray([self._items[i] for i in order])

    def _get_weight(self, from_item: str, to_item: str, preceded_by: Union[str, int] = 0) -> float:
        attrs = self._weight_graph.get_edge_data(from_item, to_item, preceded_by)
        return 1.0 if attrs is None else attrs['weight']

    def penalize(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        if self._mode & self.MODE_WEIGHT:
            self._weights[cls_name] = self.WEIGHT_PENALTY_COEFF * self._weights[cls_name]

    def penalize_initial(self, action: AtomicAction) -> None:
        cls_name = action.__class__.__name__
        if self._mode & self.MODE_INITIAL:
            self._initial_weights[cls_name] = self.INITIAL_WEIGHT_PENALTY_COEFF * self._initial_weights[cls_name]

    def penalize_bad(self, action: AtomicAction) -> None:
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            self._previous_item_queue.append(cls_name)
            if len(self._previous_item_queue) == 3:
                triple_weight = 0.98 * self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-2], self._previous_item_queue[-1], self._previous_item_queue[-3])
                double_weight = 0.98 * self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-2], self._previous_item_queue[-1])
                self._weight_graph.add_edge(self._previous_item_queue[-2], self._previous_item_queue[-1], self._previous_item_queue[-3], weight=triple_weight)
                self._weight_graph.add_edge(self._previous_item_queue[-2], self._previous_item_queue[-1], 0, weight=double_weight)

    def add_and_penalize(self, action: AtomicAction) -> None:
        self.penalize(action)
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            if len(self._previous_item_queue) == 3:
                triple_weight = self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-1], cls_name, self._previous_item_queue[-2])
                double_weight = self.SEQUENCE_PENALTY_COEFF * self._get_weight(self._previous_item_queue[-1], cls_name)
                self._weight_graph.add_edge(self._previous_item_queue[-1], cls_name, self._previous_item_queue[-2], weight=triple_weight)
                self._weight_graph.add_edge(self._previous_item_queue[-1], cls_name, 0, weight=double_weight)

    def add_and_penalize_initial(self, action: AtomicAction) -> None:
        self.penalize_initial(action)
        if self._mode & self.MODE_SEQUENCE:
            cls_name = action.__class__.__name__
            self._previous_item_queue.append(cls_name)

    def reset_weights(self) -> None:
        self._weights = deepcopy(self._bkp_weights)
        self._initial_weights = deepcopy(self._bkp_initial_weights)
        self._previous_item_queue = deque(maxlen=3)
        self._weight_graph = nx.MultiDiGraph()

    @classmethod
    def set_rng(cls, rng: np.random.Generator) -> None:
        cls.RNG = rng

    @property
    def mode(self) -> int:
        return self._mode


class RDDLWorld:

    ROBOTS: NDArray = np.asanyarray([Gripper])
    OBJECTS: NDArray = np.asanyarray([ObjectEntity])
    VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach])
    VALID_ACTIONS: NDArray = np.asanyarray([Approach, Withdraw, Grasp, Drop, Move, Rotate, Follow, Transform])
    RNG = np.random.default_rng(SEED)

    STATE_IDLE = 1
    STATE_GENERATING = 2

    def __init__(self,
                 allowed_entities: Optional[Iterable[Type[Entity]]] = None,
                 allowed_actions: Optional[Iterable[Type[AtomicAction]]] = None,
                 allowed_initial_actions: Optional[Iterable[Type[AtomicAction]]] = None
                 ) -> None:
        self._symbolic_table_stack = deque()
        self._symbolic_table_stack.append(StrictSymbolicCacheContainer())
        self.__real_cache = None

        Weighter.set_rng(self.RNG)

        self._allowed_entities: Optional[NDArray]
        self._allowed_actions: NDArray
        self._allowed_initial_actions: NDArray
        if allowed_entities is not None:
            self._allowed_entities = np.asanyarray(allowed_entities)
        else:
            self._allowed_entities = None
        if allowed_actions is None:
            self._allowed_actions = self.VALID_ACTIONS
        else:
            self._allowed_actions = np.asanyarray(allowed_actions)
        if allowed_initial_actions is None:
            self._allowed_initial_actions = self.VALID_INITIAL_ACTIONS
        else:
            self._allowed_initial_actions = np.asanyarray(allowed_initial_actions)

        self._weighter = Weighter(
            items=self._allowed_actions,
            initial_weights=[1.0 if a in self._allowed_initial_actions else 0.0 for a in self._allowed_actions]
        )
        self.full_reset()

    def assess_tree(self, sequence_length: int) -> None:
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=True)
        action: AtomicAction
        added_vars: list[Variable] = []
        sample_idx = 0

        def generate_actions():
            for action_class in self._allowed_actions:
                action = action_class()
                a_variables = action.gather_variables()
                yield action, a_variables

        results = deque()
        # for sample_idx in range(sequence_length):
        def explore_sequence(current_sequence, added_vars, result_que):
            action_generator = generate_actions()
            sample_idx = len(current_sequence)
            while True:
                if sample_idx == 0:
                    while True:
                        try:
                            action, a_variables = next(action_generator)
                        except StopIteration:
                            return
                        self._lookup_and_bind_variables(a_variables)
                        action.initial.set_symbolic_value(True)
                        if action.initial.decide():
                            self._initial_world_state = self._symbolic_table.clone()
                            break
                        self._remove_variables(added_vars)
                else:
                    try:
                        while True:
                            action, a_variables = next(action_generator)
                            added_vars = self._lookup_and_bind_variables(a_variables)
                            if added_vars:
                                action.initial.set_symbolic_value(True, set(added_vars))
                            if action.initial.decide():
                                break
                            # TODO: cleanup if initial still not true (clone sym. table and delete)
                            self._remove_variables(added_vars)
                    except StopIteration:
                        result_que.append(current_sequence)
                        return

                action.predicate.set_symbolic_value(True)

        self.deactivate_symbolic_mode()

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
                    while True:
                        try:
                            action, a_variables = next(action_generator)
                        except StopIteration:
                            raise RuntimeError("Failed to sample initial action! This should never happen. Check if the initial action space is not empty.")
                        self._lookup_and_link_variables(a_variables)
                        action.initial.set_symbolic_value(True)
                        if action.initial.decide():
                            self._initial_world_state = self._symbolic_table.clone()
                            break
                        self._weighter.penalize_initial(action)
                        self._remove_variables(added_vars)
                else:
                    while True:
                        action, a_variables = next(action_generator)
                        added_vars = self._lookup_and_link_variables(a_variables)
                        if added_vars:
                            action.initial.set_symbolic_value(True, set(added_vars))
                        if action.initial.decide():
                            break
                        # TODO: cleanup if initial still not true (clone sym. table and delete)
                        self._weighter.penalize_bad(action)
                        self._remove_variables(added_vars)

                # if response is not None and not response:
                #     self._remove_variables(added_vars)
                # else:
                yield_accepted = True

            action.predicate.set_symbolic_value(True)
            self.deactivate_symbolic_mode()
            response = yield action
            self.activate_symbolic_mode()

            if sample_idx > 0:
                self._weighter.add_and_penalize(action)
            else:
                self._weighter.add_and_penalize_initial(action)
            sample_idx += 1

        # self._symbolic_table.show_table()
        self._goal_world_state = self._symbolic_table.clone()
        self.deactivate_symbolic_mode()
        return sample_idx == sequence_length

    def _recursive_sampling(self, sample_idx, action_sequence, state_sequence):
        if sample_idx == self.__recurse_max_samples:
            # store sequences
            self.__action_stack.append(action_sequence)
            self.__state_stack.append(state_sequence)
            self.__recurse_gen_lock.release()
            self.__recurse_gen_lock.acquire()
            return

        action_generator = self._weighter.get_random_generator() if sample_idx > 0 else self._weighter.get_initial_generator()
        while True:
            try:
                action, a_variables = next(action_generator)
            except StopIteration:
                break
            except BaseException as e:
                print(e)
            current_world = self._symbolic_cache_duplicate_and_stack()
            added_vars = self._lookup_and_link_variables(a_variables)
            if added_vars:
                if sample_idx > 0:
                    action.initial.set_symbolic_value(True, set(added_vars))
                else:
                    action.initial.set_symbolic_value(True)
            if not action.initial.decide():
                self._weighter.penalize_initial(action)
                self._remove_variables(added_vars)
                self._symbolic_cache_pop()
                continue
            action.predicate.set_symbolic_value(True)
            self._recursive_sampling(sample_idx + 1, action_sequence + [action], state_sequence + [current_world])

    def recurse_generator(self,
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

        Weighter.RETRY_AD_INFINITUM = False

        self.__action_stack = deque()
        self.__state_stack = deque()
        self.__recurse_max_samples = sequence_length
        self.__recurse_gen_lock = Lock()
        self.__recurse_gen_lock.acquire()

        generator_thread = Thread(target=self._recursive_sampling, args=(0, [], []), daemon=True)
        generator_thread.start()

        # self._recursive_sampling(0, [], [])


        while True:
            self.__recurse_gen_lock.acquire()
            if len(self.__action_stack) > 0:
                action = self.__action_stack.popleft()
                state = self.__state_stack.popleft()
                self.deactivate_symbolic_mode()
                # self._goal_world_state = self._symbolic_table.clone()
                yield action
                self.activate_symbolic_mode()
                self.__recurse_gen_lock.release()
            else:
                break

        self.deactivate_symbolic_mode()

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

    def _lookup_and_link_variables(self, variables: list[Variable]) -> list[Variable]:
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
        self.reset_world()
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
        if self._allowed_entities is not None:
            options = np.asanyarray([sc for sc in base_type.list_subclasses() if sc in self._allowed_entities])
        else:
            options = np.asanyarray([sc for sc in base_type.list_subclasses()])
        return self.RNG.choice(options)

    def _get_random_variable(self, typ: type[Entity]) -> SymbolicEntity:
        return SymbolicEntity(self._sample_subclass(typ))

    @property
    def _symbolic_table(self) -> StrictSymbolicCacheContainer:
        return self._symbolic_table_stack[-1]

    def _symbolic_cache_duplicate_and_stack(self) -> StrictSymbolicCacheContainer:
        current_world = self._symbolic_table.clone()
        self._symbolic_table_stack.append(current_world)
        Operand.set_cache_symbolic(current_world)
        return current_world

    def _symbolic_cache_pop(self) -> None:
        if len(self._symbolic_table_stack) == 1:
            raise RuntimeError("Cannot pop last cache!")
        self._symbolic_table_stack.pop()

    def activate_symbolic_mode(self):
        self.__real_cache = Operand.get_cache()
        Operand.set_cache_symbolic(self._symbolic_table)

    def deactivate_symbolic_mode(self):
        if self.__real_cache is None:
            return
        Operand.set_cache_normal(self.__real_cache)

    def reset_world(self):
        self.__state = self.STATE_IDLE
        self._variables = {}
        self._symbolic_table.reset()

    def reset_weights(self, mode: Optional[int] = None):
        self._weighter.reset_weights()
        if mode is not None:
            self._weighter.set_mode(mode)

    def full_reset(self):
        self.reset_world()
        self.reset_weights()

    @property
    def weighter(self) -> Weighter:
        return self._weighter

    @classmethod
    def set_seed(cls, seed: int) -> None:
        BitGen = type(cls.RNG.bit_generator)
        cls.RNG.bit_generator.state = BitGen(seed).state


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
