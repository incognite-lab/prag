import traceback as tb
from abc import ABCMeta, abstractmethod
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

    def __call__(self) -> Any:
        return Entity._observation_getter()[self.__reference]

    @classmethod
    def _get_observation(cls, obs_name: str) -> Any:
        return Entity._observation_getter()[obs_name]

    @classmethod
    def _get_reference_count(cls) -> int:
        return cls.__class_instance_counter

    def _get_generic_reference(self) -> str:
        return f"{str(self.__class__).lower()}_{self.__class__._get_reference_count()}"

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


class Operand(metaclass=ABCMeta):
    """ABC for all operand types. Operand can be evaluated (returns a float) or decided
    (returns a bool). All subclasses must implement the evaluate and decide methods.

    """
    __MAPPING: ClassVar[dict[str, Any]] = {}

    @classmethod
    def set_mapping(cls, mapping: dict):
        cls.__MAPPING = mapping

    @classmethod
    def __register_attributes(cls, **kwargs):
        print(cls.__name__)
        dd = vars(cls)
        for k, v in vars(cls).items():
            if '_0_' in dd:
                print(f"Class '{cls}' has an attribute {k}!")
        setattr(cls, "__init__subclass__", Operand.__register_attributes)

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__register_attributes()
        # setattr(cls, "__init__subclass__", Operand.__register_attributes)
        # setattr(cls, "set_mapping", Operand.set_mapping)

    def __new__(cls, *args, **kwargs):
        for k, v in vars(cls).items():
            if k.startswith("_0_"):
                if v in Operand.__MAPPING:
                    setattr(cls, k, Operand.__MAPPING[v])
                else:
                    raise ValueError(f"The required user defined mapping for property {k} with name {v} of class {cls.__name__} is not defined!")
        return super().__new__(cls)

    @abstractmethod
    def decide(self) -> bool:
        raise NotImplementedError(f"'decide' method not implemented for {self.__class__} operand")

    @abstractmethod
    def evaluate(self) -> float:
        raise NotImplementedError(f"'evaluate' method not implemented for {self.__class__} operand")

    # def _register_variable(self, name: str, typ: type) -> None:
    #     self.__vars[name] = value

    # def __setattr__(self, name: str, value: Any) -> None:
    #     super(Operand, self).__setattr__(name, value)
    #     if type(value) is Variable:
    #         self._register_variable(name, type(value))
    #     inspect.get_annotations(getattr(self, name))


class Predicate(Operand, metaclass=ABCMeta):
    """Predicates are special operands that cannot be evaluated. The can only return true or false.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError("call not implemented for generic predicate")

    @final
    def decide(self):
        return self()

    @final
    def evaluate(self):
        warn(f"Predicates can't be evaluated! Returning None. Evaluate called for: {self.__class__.__name__} @ {tb.format_stack()}")
        return None


class Reward(Operand):

    def __init__(self) -> None:
        self._reward_function: Callable[[], float]

    def evaluate(self):
        return self._reward_function()

    @final
    def decide(self):
        warn(f"Rewards can't be decided! Returning None. Decide called for: {self.__class__.__name__} @ {tb.format_stack()}")
        return None
