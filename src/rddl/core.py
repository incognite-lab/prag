import traceback as tb
from abc import ABCMeta, abstractmethod
from functools import _make_key
from inspect import isabstract
from typing import Any, Callable, ClassVar, Generic, Optional, TypeVar, final
from warnings import warn


class Entity(metaclass=ABCMeta):
    """Entity abstract base class. All environment entities should inherit from this class.
    """
    _observation_getter: ClassVar[Optional[Callable]] = None
    __class_instance_counter: ClassVar[int]

    def __init_subclass__(cls) -> None:
        cls.__class_instance_counter = 0

    @classmethod
    def set_observation_getter(cls, observation_getter: Callable):
        cls._observation_getter = observation_getter

    def __new__(cls, *args, **kwargs):
        cls.__class_instance_counter = cls.__class_instance_counter + 1
        return super().__new__(cls)

    def __init__(self, reference: Optional[str] = None) -> None:
        if Entity._observation_getter is None:
            raise ValueError(f"Observation getter for class {self.__class__} is not set!"
                             " You must first set observation getter callback"
                             " before instantiating any Entity class! (call Entity.set_observation_getter)")

        if reference is None:
            reference = self._get_generic_reference()
        self.__reference = reference
        super().__init__()

    def __call__(self) -> Any:
        return Entity._observation_getter(self)

    @classmethod
    def _get_observation(cls, obs_name: str) -> Any:
        return Entity._observation_getter(self)

    @classmethod
    def _get_reference_count(cls) -> int:
        return cls.__class_instance_counter

    def _get_generic_reference(self) -> str:
        return f"{str(self.__class__.__name__).lower()}_{self.__class__._get_reference_count()}"

    @property
    def reference(self):
        return self.__reference

    @classmethod
    def monkey_patch(cls, method: Callable, alternative: Callable):
        method_name = method.__name__
        if not hasattr(cls, method_name):
            raise ValueError(f"Method {method_name} is not defined in class {cls.__name__}. "
                             "Don't use monkey patch to add new methods! Use inheritance instead.")

        warn(f"Warning: Monkey patching {cls.__name__}.{method_name}. This will change behavior of all subclasses! Be careful about it!")
        setattr(cls, method_name, alternative)


variable_class = TypeVar('variable_class', bound=Entity)


class Variable(Generic[variable_class]):
    """Generic variable container. The purpose of this class is to define a placeholder
    for a variable of a specific type that is later bound to a specific value of that type.
    For example, at definition time, there is no variable with a specific value that can be provided
    directly to some function. However, it is known that the function requires a variable of a specific type.
    Thus, this class can be used to define such a variable.

    Example:
    ```
        future_variable = Variable(<variable_name>, <variable_type>)
        obj = SomeObject(future_variable)
        # using functions of obj that operate on the variable would result in an error at this time, since it is not bound, yet

        some_variable = <subclass_of_variable_type>()
        future_variable.bind(some_variable)

        obj.some_function_using_future_variable()  # works
    ```

    This class will try to directly get methods and attributes from the bound value.
    For example, from the example above, `future_variable.some_method()` will directly call `some_variable.some_method()`.
    Naturally, error will be raised if the bound value does not have such method or attribute.

    The type of the bound value must be a subclass of the type of the variable, otherwise an error will be raised.

    !!!
    The contained variable must not define methods or attributes called "bind" and "type" or rely on external used of methods
    starting with "_" (internal call of such methods is okay). These will be obscured by methods of this class.
    !!!

    Additionally, simple variables are not allowed. They are only allowed if they are defined via getter/setter functions.
    For example:
    ```
    class MyVariable:
        def __init__(self):
            self.a = 1

        @property
        def b(self):
            return self.a

    var = MyVariable()
    variable_container = Variable("var", MyVariable)
    variable_container.bind(var)

    # this will work (b is a property):
    print(variable_container.b)  # 1
    # this will raise an error (a is a simple variable):
    print(variable_container.a)  # AttributeError
    ```
    """

    def __init__(self, name: str, typ: type[variable_class]):
        """Initializes a variable container.

        Args:
            name (str): Name of the variable for internal reference.
            typ (type): Expected type of the variable. The variable must be of this class or a subclass of this class.
        """
        super().__init__()
        self._name = name
        self._type: type[variable_class] = typ
        self._value: Optional[variable_class] = None

    def bind(self, value: variable_class) -> None:
        """Binds the variable to a specific value.
        This will raise an error if the value is not a subclass of the type of the variable.

        Args:
            value (variable_class): any variable that is a subclass of the type of this container.
        """
        assert issubclass(type(value), self._type), f"Value {value} of type {type(value)} is not a subclass of {self._type}, required by variable '{self._name}'"
        self._value = value

    def type(self) -> type[variable_class]:
        """Returns the type of the variable container.

        Returns:
            type: Expected type of the variable.
        """
        return self._type

    def __getattr__(self, name: str) -> Any:
        """
        Return the value of the attribute with the given name from the bound value.

        Parameters:
            __name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            ValueError: If the variable is not bound.
            AttributeError: If the variable does not have the specified attribute.
        """
        if name.startswith("_"):  # ignore private attributes
            super().__getattribute__(name)
        if not self._is_bound():
            raise ValueError(f"Variable {self._name} was not bound, yet!")
        if not hasattr(self._type, name):
            raise AttributeError(f"Variable '{self._name}' of type {self._type} has no function or property '{name}'\nPerhaps, it is not a property (defined via getter/setter functions) or a class variable? Simple variables are not allowed.")
        return getattr(self._value, name)

    def _is_bound(self) -> bool:
        """
        Check if the value of the object is bound.

        Returns:
            bool: True if the value is bound, False otherwise.
        """
        return self._value is not None

    def __call__(self) -> variable_class:
        """
        Retrieves the value of the variable, if it was bound.

        Returns:
            variable_class: The value of the variable.

        Raises:
            ValueError: If the variable is not bound.
        """
        if not self._is_bound:
            raise ValueError(f"Variable {self._name} was not bound, yet!")
        return self._value

    def __repr__(self):
        if self._value is None:
            return f"Unbound variable {self._name}: {self._type}"
        else:
            return f"Variable {self._name}: {self._type} := {self._value})"


class _Cache():

    def __init__(self):
        self.__cache_decide: dict[Any, bool] = {}
        self.__get_decide_cache = self.__cache_decide.get
        self.__cache_eval: dict[Any, float] = {}
        self.__get_eval_cache = self.__cache_eval.get
        self.__cache_sentinel = object()

    def decide(self, compute_func: Callable, *args: Any, **kwds: Any) -> bool:
        key = _make_key(args, kwds, typed=True)
        result = self.__get_decide_cache(key, self.__cache_sentinel)
        if result is not self.__cache_sentinel:
            return result  # type: ignore
        # cls = compute_func.__self__.__class__
        result = compute_func(*args, **kwds)
        self.__cache_decide[key] = result
        return result

    def eval(self, compute_func: Callable, *args: Any, **kwds: Any) -> float:
        key = _make_key(args, kwds, typed=True)
        result = self.__get_eval_cache(key, self.__cache_sentinel)
        if result is not self.__cache_sentinel:
            return result  # type: ignore
        # result = cls.__evaluate__(*args, **kwds)
        result = compute_func(*args, **kwds)
        self.__cache_eval[key] = result
        return result

    def reset(self):
        self.__cache_decide.clear()
        self.__cache_eval.clear()


class Operand(metaclass=ABCMeta):
    """ABC for all operand types. Operand can be evaluated (returns a float) or decided
    (returns a bool). All subclasses must implement the evaluate and decide methods.

    """
    __MAPPING: ClassVar[dict[str, Any]] = {}
    __CACHE = _Cache()

    @classmethod
    def reset_cache(cls):
        cls.__CACHE.reset()

    @classmethod
    def set_mapping(cls, mapping: dict):
        cls.__MAPPING = mapping

    @classmethod
    def __register_attributes(cls, **kwargs):
        # print(cls.__name__)
        dd = vars(cls)
        for k, v in dd.items():
            if k.startswith("_0_"):
                if v in Operand.__MAPPING:
                    setattr(cls, k, Operand.__MAPPING[v])
                else:
                    raise ValueError(f"The required user defined mapping for property {k} with name {v} of class {cls.__name__} is not defined!")
                print(f"Class '{cls}' has an attribute {k}!")
        # setattr(cls, "__init__subclass__", Operand.__register_attributes)

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__register_attributes()
        # setattr(cls, "__init__subclass__", Operand.__register_attributes)
        # setattr(cls, "set_mapping", Operand.set_mapping)

    # def __new__(cls, *args, **kwargs):
    #     for k, v in vars(cls).items():
    #         if k.startswith("_0_"):
    #             if v in Operand.__MAPPING:
    #                 setattr(cls, k, Operand.__MAPPING[v])
    #             else:
    #                 raise ValueError(f"The required user defined mapping for property {k} with name {v} of class {cls.__name__} is not defined!")
    #     return super().__new__(cls)

    def __init__(self) -> None:
        super().__init__()

    @final
    def decide(self, *args: Any, **kwds: Any) -> bool:
        return self.__CACHE.decide(self.__decide__, *args, **kwds)

    @abstractmethod
    def __decide__(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError(f"'decide' method not implemented for {self.__class__} operand")

    @final
    def evaluate(self, *args: Any, **kwds: Any) -> float:
        return self.__CACHE.eval(self.__evaluate__, *args, **kwds)

    @abstractmethod
    def __evaluate__(self, *args: Any, **kwds: Any) -> float:
        raise NotImplementedError(f"'evaluate' method not implemented for {self.__class__} operand")

    # def _register_variable(self, name: str, typ: type) -> None:
    #     self.__vars[name] = value

    # def __setattr__(self, name: str, value: Any) -> None:
    #     super(Operand, self).__setattr__(name, value)
    #     if type(value) is Variable:
    #         self._register_variable(name, type(value))
    #     inspect.get_annotations(getattr(self, name))


class Operator(Operand, metaclass=ABCMeta):
    """Operator operates (executes evaluate or decide method) over zero or more operands. It is also itself an operand,
    meaning, it can be evaluated or decided.

    Operator class must define _ARITY and _SYMBOL attributes. _ARITY defines arity of the operation (number of operands).
    _SYMBOL defines string representation of the operation. It is used in parsing of definition files.
    """
    _ARITY: ClassVar[int]
    _SYMBOL: ClassVar[str]

    def __init_subclass__(cls) -> None:
        dd = dir(cls)
        if '_SYMBOL' not in dd and not isabstract(cls):
            raise ValueError(f"Class '{cls}' does not have a '_SYMBOL' attribute! Every sub-class of 'Operator' must define a '_SYMBOL' that defines its string representation!")
        if '_ARITY' not in dd:
            raise ValueError(f"Class '{cls}' does not have a '_ARITY' attribute! Every sub-class of 'Operator' must define a '_ARITY' that defines arity of the operation!")
        super().__init_subclass__()
        # return super().__init_subclass__()

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


AA = TypeVar("AA")


class Predicate(Operand, metaclass=ABCMeta):
    """Predicates are special operands that cannot be evaluated. The can only return true or false.
    """
    _VARIABLES: ClassVar[dict[str, type[Entity]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for name, typ in cls._VARIABLES.items():
            proxy = property(lambda self, name=name: self.__variables[name]())
            setattr(cls, name, proxy)

    def __init__(self, **kwds) -> None:
        super().__init__()
        self.__variables: dict[str, Variable] = {}
        for name, typ in self._VARIABLES.items():
            if name in kwds:
                variable = kwds[name]
                assert isinstance(variable, Variable), f"Variable {name} is not an instance of 'Variable'!"
                assert variable.type() == typ, f"Variable {name} is not of type {typ}!"
                self.__add_variable(name, variable)
            else:
                self.__register_variable(name, typ)

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError("call not implemented for generic predicate")

    @final
    def __decide__(self, *args: Any, **kwds: Any) -> bool:
        return self(*args, **kwds)

    @final
    def __evaluate__(self):
        warn(f"Predicates can't be evaluated! Returning None. Evaluate called for: {self.__class__.__name__} @ {tb.format_stack()}")
        return None

    def __register_variable(self, name: str, typ: type[variable_class]) -> None:
        variable = Variable(name, typ)
        self.__add_variable(name, variable)

    def __add_variable(self, name: str, variable: Variable):
        self.__variables[name] = variable

    def get_variable(self, name: str) -> Variable:
        return self.__variables[name]

    def bind(self, variables: dict[str, Entity]) -> None:
        for var_name, variable in self.__variables.items():
            if var_name in variables:
                variable.bind(variables[var_name])
            else:
                warn(f"Variable '{var_name}' of class '{self.__class__.__name__}' is not bound to any value in the provided dictionary. This may lead to errors!")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}: {v.type().__name__}' for k, v in self.__variables.items()])})"


class Reward(Operand):

    # def __init__(self) -> None:
    #     self._reward_function: Callable[[], float]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return NotImplementedError(f"Evaluate not implemented for {self.__class__} reward")

    def __evaluate__(self, *args: Any, **kwds: Any):
        return self(*args, **kwds)

    @final
    def decide(self):
        warn(f"Rewards can't be decided! Returning None. Decide called for: {self.__class__} @ {tb.format_stack()}")
        return None
