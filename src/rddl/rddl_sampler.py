from collections import defaultdict, deque
from copy import deepcopy
from functools import lru_cache
from html import entities
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Callable, ClassVar, Generator, Iterable, Optional, Type, Union
from warnings import warn

from click import option
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from traitlets import default
#from testing_utils import Apple

from rddl import AtomicAction, Operand, Variable
from rddl.actions import Approach, Drop, Follow, Grasp, Move, Rotate, Withdraw, Transform
from rddl.core import CacheMode, Entity, LogicalOperand, Predicate, SymbolicCacheContainer
from rddl.entities import Gripper, ObjectEntity
from rddl.predicates import IsReachable, Near
from rddl.rddl_parser import (EntityType, OperatorType, PredicateType,
                              RDDLParser)
from memory_profiler import profile

from rddl.rule_book import RuleBook
from rddl.sampling_utils import StrictSymbolicCacheContainer, SymbolicEntity, Weighter


SEED = np.random.randint(0, 2**32)


class RDDLWorld:

    ROBOTS: NDArray = np.asanyarray([Gripper])
    OBJECTS: NDArray = np.asanyarray([ObjectEntity])
    # VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach])
    VALID_INITIAL_ACTIONS: NDArray = np.asanyarray([Approach, Grasp, Drop, Move, Rotate, Follow, Transform])
    VALID_ACTIONS: NDArray = np.asanyarray([Approach, Withdraw, Grasp, Drop, Move, Rotate, Follow, Transform])
    RNG = np.random.default_rng(SEED)

    STATE_IDLE = 1
    STATE_GENERATING = 2

    def __init__(self,
                 allowed_entities: Optional[Iterable[Type[Entity]]] = None,
                 allowed_actions: Optional[Iterable[Type[AtomicAction]]] = None,
                 allowed_initial_actions: Optional[Iterable[Type[AtomicAction]]] = None,
                 sample_single_object_per_class: bool = False,
                 action_weights: Optional[list[float]] = None,
                 object_weights: Optional[dict[Type[Entity], float]] = None
                 ) -> None:
        self._reset_symbolic_table_stack()

        self.__real_cache = None

        self._rule_book = RuleBook()

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
        self._sample_single_object_per_class = sample_single_object_per_class

        self._object_weights = defaultdict(lambda: 1.0)

        if action_weights is None:
            action_weights = [1.0] * len(self._allowed_actions)
        else:
            if len(action_weights) != len(self._allowed_actions):
                raise ValueError("Length of action weights must be equal to the number of allowed actions!")

        self._weighter = Weighter(
            items=self._allowed_actions,
            weights=action_weights,
            initial_weights=[1.0 if a in self._allowed_initial_actions else 0.0 for a in self._allowed_actions],
        )
        self.full_reset()

    @property
    def rule_book(self) -> RuleBook:
        return self._rule_book

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
                            action = next(action_generator)
                        except StopIteration:
                            raise RuntimeError("Failed to sample initial action! This should never happen. Check if the initial action space is not empty.")
                        action_variables = action.gather_variables()
                        self._symbolic_cache_duplicate_and_stack()
                        added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
                        action.initial.set_symbolic_value(True)
                        if action.initial.decide():
                            self._initial_world_state = self._symbolic_table.clone()
                            break
                        self._remove_variables(added_vars)
                        self._weighter.penalize_initial(action)
                        self._symbolic_cache_pop()
                else:
                    while True:
                        action = next(action_generator)
                        action_variables = action.gather_variables()
                        self._symbolic_cache_duplicate_and_stack()
                        added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
                        if added_vars:
                            action.initial.set_symbolic_value(True, set(added_vars))
                        if action.initial.decide():
                            break
                        # TODO: cleanup if initial still not true (clone sym. table and delete)
                        self._weighter.penalize_bad(action)
                        self._remove_variables(added_vars)
                        self._symbolic_cache_pop()

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
        self._reset_symbolic_table_stack()
        return sample_idx == sequence_length

    def _recursive_sampling(self, sample_idx, action_sequence, state_sequence, start_state=None):
        if sample_idx == self.__recurse_max_samples:
            # store sequences
            self.__action_state_stack.put((action_sequence, state_sequence, start_state))
            return

        action_generator = self._weighter.get_random_generator() if sample_idx > 0 else self._weighter.get_initial_generator()
        while True:
            try:
                action = next(action_generator)
            except StopIteration:
                break
            except BaseException as e:
                print(e)
            action_variables = action.gather_variables()
            current_world = self._symbolic_cache_duplicate_and_stack()
            added_vars = self._symbolic_table.lookup_and_link_variables(action_variables, self.sample_object_subclass)
            if added_vars:
                if sample_idx > 0:
                    action.initial.set_symbolic_value(True, set(added_vars))
                    start_state = self._symbolic_table.clone()
                else:
                    action.initial.set_symbolic_value(True)
            if not action.initial.decide():
                self._weighter.penalize_initial(action)
                self._remove_variables(added_vars)
                self._symbolic_cache_pop()
                continue
            action.predicate.set_symbolic_value(True)
            self._recursive_sampling(sample_idx + 1, action_sequence + [action], state_sequence + [current_world], start_state)

    def recurse_generator(self,
                          sequence_length: int = 2,
                          add_robots: bool = True,
                          n_samples_requested: int = np.iinfo(np.int32).max,
                          ) -> Generator['RDDLTask', bool, bool]:
        if self.__state != self.STATE_IDLE:
            raise RuntimeError("Generator is already running!")

        self._initialize_world(add_robots=add_robots)

        self._initial_world_state: StrictSymbolicCacheContainer
        self._goal_world_state: StrictSymbolicCacheContainer

        Weighter.RETRY_AD_INFINITUM = False

        self.__action_state_stack = Queue(maxsize=10)
        self.__recurse_max_samples = sequence_length

        generator_thread = Thread(target=self._recursive_sampling, args=(0, [], []), daemon=True, name="rddl_generator")
        generator_thread.start()

        # self._recursive_sampling(0, [], [])
        n_samples_out = 0

        while True:
            try:
                # output = self.__action_state_stack.get(block=True, timeout=1 * self.__recurse_max_samples)
                output = self.__action_state_stack.get(block=True)
            except Empty:
                warn("Generator timed out after {n_samples_out} samples!")
                break
            if output is None:
                break
            else:
                actions, states, start_state = output
                n_samples_out += 1

            # self._goal_world_state = self._symbolic_table.clone()
            # self.__action_state_stack.task_done()
            task = RDDLTask(actions=actions, states=states, initial_state=start_state, world_generator=self)

            # c = Operand.get_cache()
            # c.show_table()

            yield task
            if n_samples_out >= n_samples_requested:
                break

        self.deactivate_symbolic_mode()
        return True

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

    # def _lookup_and_link_variables(self, variables: list[Variable]) -> list[Variable]:
    #     missing_vars = []
    #     linked_vars = []
    #     for v in variables:
    #         like_gen = self._symbolic_table.find_variable_like_this(v)
    #         try:
    #             while (like_v := next(like_gen)) is not None:
    #                 if like_v not in linked_vars:
    #                     break
    #         except StopIteration:
    #             like_v = None

    #         if like_v is None:
    #             like_v = self._get_random_variable(v.type)
    #             missing_vars.append(like_v)
    #             self._add_variable(v.name, like_v)  # FIXME: name should be somehow estimated
    #         v.link_to(like_v)
    #         linked_vars.append(like_v)
    #     return missing_vars

    def _initialize_world(self, add_robots: bool = True):
        self.activate_symbolic_mode()
        self.reset_world()
        if add_robots:
            self._add_variable("gripper", self._symbolic_table._get_random_variable(Gripper, self.sample_object_subclass))

    def _add_variable(self, name: str, variable: SymbolicEntity) -> None:
        # self._variables[name] = variable
        self._symbolic_table.register_variable(variable)

    def _remove_variables(self, variables: list[Variable]) -> None:
        self._symbolic_table.remove_variables(variables)
        # for variable in variables:
        #     for local_name, local_var in self._variables.items():
        #         if local_var == variable:
        #             del self._variables[local_name]
        #             break

    @lru_cache(maxsize=128)
    def _find_allowed_subclasses(self, base_type: type[Entity]) -> NDArray:
        if self._allowed_entities is not None:
            options = np.asanyarray([sc for sc in base_type.list_subclasses() if sc in self._allowed_entities])
        else:
            options = np.asanyarray([sc for sc in base_type.list_subclasses()])
        return options

    def _weight_object_classes(self, subclasses: NDArray, object_weights: dict[type[Entity], float]) -> NDArray:
        weights = np.array([object_weights[sc] for sc in subclasses])
        weights /= np.sum(weights)
        return weights

    def _reduce_object_class_weight(self, object_type: type[Entity], object_weights: dict[type[Entity], float]) -> None:
        if self._sample_single_object_per_class:
            object_weights[object_type] = 0
        else:
            object_weights[object_type] *= 0.9

    def sample_object_subclass(self, base_type: type[Entity], object_weights: Optional[dict[type[Entity], float]] = None) -> type[Entity]:
        options = self._find_allowed_subclasses(base_type)
        if object_weights is not None:
            probs = self._weight_object_classes(options, object_weights)
        else:
            probs = None
        choice = self.RNG.choice(options, p=probs)
        if object_weights is not None:
            self._reduce_object_class_weight(choice, object_weights)
        return choice

    # def _get_random_variable(self, typ: type[Entity]) -> SymbolicEntity:
    #     return SymbolicEntity(self.sample_object_subclass(typ))

    @property
    def _symbolic_table(self) -> StrictSymbolicCacheContainer:
        return self._symbolic_table_stack[-1]

    @property
    def _variables(self) -> dict[str, Variable]:
        return self._symbolic_table.variables

    def _reset_symbolic_table_stack(self) -> None:
        self._symbolic_table_stack = deque()
        self._symbolic_table_stack.append(StrictSymbolicCacheContainer())

    def _symbolic_cache_duplicate_and_stack(self) -> StrictSymbolicCacheContainer:
        """
        Duplicate the current symbolic table, set it as the current world state
        (in Operand cache) and push it onto the stack.
        This creates a new "branch" in the search tree.
        """
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
        # self._variables = {}
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


class RDDLTask:

    def __init__(self, actions: list[AtomicAction], states: list[StrictSymbolicCacheContainer], initial_state: StrictSymbolicCacheContainer, world_generator: Optional[RDDLWorld] = None) -> None:
        self._actions = actions
        self._states = states
        self._world_generator = world_generator

        self._initial_state: StrictSymbolicCacheContainer = initial_state
        self._final_state: StrictSymbolicCacheContainer = states[-1]

    def gather_objects(self) -> list[Variable]:
        return list(self._final_state.variables.values())

    def current_action(self) -> Optional[AtomicAction]:
        for action in self._actions:
            if action.can_be_executed():
                return action

    def regenerate_objects(self) -> None:
        if self._world_generator is None:
            raise RuntimeError("Cannot regenerate objects without a world generator!")

        previous_cache_mode = Operand.get_cache_mode()
        previous_cache = Operand.get_cache()

        new_state = StrictSymbolicCacheContainer()
        Operand.set_cache_symbolic(new_state)
        for a_idx, (action, state) in enumerate(zip(self._actions, self._states)):
            a_vars = action.gather_variables()
            added_vars = new_state.lookup_and_link_variables(a_vars, self._world_generator.sample_object_subclass)
            if a_idx > 0:
                action.initial.set_symbolic_value(True, set(added_vars))
            else:
                action.initial.set_symbolic_value(True)
            # print(f"regenerating {action} with {new_state}")
            action.predicate.set_symbolic_value(True)

        self._final_state = new_state

        if previous_cache_mode == CacheMode.SYMBOLIC:
            Operand.set_cache_symbolic(previous_cache)
        else:
            Operand.set_cache_normal(previous_cache)

    def get_actions(self) -> Iterable[AtomicAction]:
        return iter(self._actions)

    def next_action(self) -> AtomicAction:
        raise NotImplementedError

    def current_reward(self) -> float:
        ca = self.current_action()
        if ca is None:
            return np.nan
        return ca.compute_reward()

    @property
    def initial_state(self) -> StrictSymbolicCacheContainer:
        return self._initial_state

    @property
    def final_state(self) -> StrictSymbolicCacheContainer:
        return self._final_state
