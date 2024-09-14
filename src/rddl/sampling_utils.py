from collections import defaultdict, deque
from copy import deepcopy
from typing import Callable, ClassVar, Generator, Iterable, Optional, Type, Union
from rddl import AtomicAction, Operand, Variable
from rddl.core import Entity, LogicalOperand, Predicate, SymbolicCacheContainer
import numpy as np
import networkx as nx
from numpy.typing import ArrayLike, NDArray


class SymbolicEntity(Variable):

    def __init__(self, typ: type[Entity]):
        super().__init__(typ, base_name=f"entity_{typ.__name__}")


class StrictSymbolicCacheContainer(SymbolicCacheContainer):

    def __init__(self):
        super().__init__()
        self._variable_registry = {}
        self._object_weights = defaultdict(lambda: 1.0)

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

    def get_predicates(self) -> list[tuple[Predicate, list[Variable], bool]]:
        c = self._cache_decide
        result = []
        for key, value in c.items():
            hashed_internal_args = key.arg_list
            try:
                args = [self._variable_registry[id].name for id in hashed_internal_args]
            except KeyError:
                continue
            if key.other:
                args += key.other
            result.append((key.class_name, args, value))
        return result

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
        s._object_weights = deepcopy(self._object_weights)
        return s

    def reset(self):
        super().reset()
        self.remove_variables(list(self._variable_registry.values()))

    def find_variable_like_this(self, variable: Variable) -> Generator[Variable, None, Optional[Variable]]:
        """Yields variables from the current world that are of the same type as the given variable.

        Args:
            variable: The variable to find similar variables for.

        Yields:
            Variable: A variable that is of the same type as the given variable.

        Returns:
            None: If no variable of the same type is found.
        """
        for v in self.variables.values():
            if issubclass(v.type, variable.type):
                yield v
        return None

    def _get_random_variable(self, typ: type[Entity], object_sampling_function: Callable[[type[Entity], Optional[dict[type, float]]], type[Entity]]) -> SymbolicEntity:
        return SymbolicEntity(object_sampling_function(typ, self._object_weights))

    def lookup_and_link_variables(self, variables: list[Variable], object_sampling_function: Callable[[type[Entity], Optional[dict[type, float]]], type[Entity]]) -> list[Variable]:
        missing_vars = []
        linked_vars = []
        for v in variables:
            like_gen = self.find_variable_like_this(v)
            try:
                while (like_v := next(like_gen)) is not None:
                    if like_v not in linked_vars:
                        break
            except StopIteration:
                like_v = None

            if like_v is None:
                like_v = self._get_random_variable(v.type, object_sampling_function)
                missing_vars.append(like_v)
                self.register_variable(like_v)
            v.link_to(like_v)
            linked_vars.append(like_v)
        return missing_vars


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
    def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction], None, None]:
        # def _get_random_item(choices: NDArray, condition: Callable) -> Generator[tuple[AtomicAction, list[Variable]], None, None]:
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
            # a_variables = action.gather_variables()
            yield action  # , a_variables
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

    def _add_max_noise(self, weights: NDArray) -> NDArray:
        m = weights.max()
        where_max = weights == m
        noise = self.RNG.uniform(0, 0.01, np.count_nonzero(where_max))
        weights[where_max] += noise
        return weights

    def get_random_generator(self) -> Generator:
        """Yields items from the list of choices. The items are randomly shuffled based on their weights.

        Returns:
            Generator: Generator yielding (action, [list of variables]).

        Yields:
            action, list[Variable]: Returns (action, [list of variables]).
        """
        if self._mode & self.MODE_SEQUENCE and len(self._previous_item_queue) == 3:
            weighted_weights = np.array(self._get_seq_weights())
        else:
            weighted_weights = np.array(list(self._weights.values()))

        weighted_weights = self._add_max_noise(weighted_weights)

        choices = self._weighted_shuffle(weighted_weights)  # shuffle action classes (so they are in varying order each time)
        condition = lambda ci, nc, ad_inf=self.RETRY_AD_INFINITUM: ad_inf or ci < nc  # noqa
        return self._get_random_item(choices, condition)

    def get_initial_generator(self) -> Generator:
        weights = self._add_max_noise(np.asarray(list(self._initial_weights.values())))
        choices = self._weighted_shuffle(weights)  # shuffle action classes (so they are in varying order each time)
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
