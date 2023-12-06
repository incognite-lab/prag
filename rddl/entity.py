from typing import Any, Callable, ClassVar, Optional, Iterable, TypeVar, Generic
from abc import ABCMeta, abstractmethod


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

    def __new__(cls):
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


class Location(Entity):

    def __init__(self, reference: Optional[str] = None):
        if reference is None and self.__class__ is Location:
            reference = f"loc_{Location._get_reference_count()}"
        super().__init__(reference)

    @abstractmethod
    def _get_location(self):
        raise NotImplementedError(f"_get_location is not implemented for {self.__class__}")

    @property
    def location(self) -> Iterable[float]:
        return self._get_location()


class ObjectEntity(Location):

    def __init__(self, reference: str, name: Optional[str] = None):
        super().__init__(reference)
        self._name = name if name is not None else reference

    @property
    def name(self):
        return self._name


class Gripper(Location):

    def __init__(self, reference: Optional[str] = None):
        super().__init__(reference)

    @abstractmethod
    def _is_holding(self, obj: ObjectEntity):
        raise NotImplementedError(f"_is_holding is not implemented for {self.__class__}")

    def is_holding(self, obj: ObjectEntity) -> bool:
        return self._is_holding(obj)


LocationType = TypeVar("LocationType", bound=Location)
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
