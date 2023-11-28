from typing import Any, Callable, ClassVar, Optional, Iterable, TypeVar, Generic
from abc import ABCMeta, abstractmethod


class Entity(metaclass=ABCMeta):
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
    def _get_reference_count(cls) -> int:
        return cls.__class_instance_counter

    def _get_generic_reference(self) -> str:
        return f"{str(self.__class__).lower()}_{self.__class__._get_reference_count()}"


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

    def __init__(self, name: str, typ: type[variable_class]):
        self._name = name
        self._type: type[variable_class] = typ
        self._value: Optional[variable_class] = None

    def bind(self, value: variable_class) -> None:
        self._value = value

    def get(self) -> variable_class:
        return self()

    def type(self) -> type[variable_class]:
        return self._type

    def __getattr__(self, __name: str) -> Any:
        if not self._is_bound():
            raise ValueError(f"Variable {self._name} was not bound, yet!")
        if not hasattr(self._type, __name):
            raise AttributeError(f"Variable {self._name} of type {self._type} has no attribute '{__name}'")
        return getattr(self._value, __name)

    def _is_bound(self) -> bool:
        return self._value is not None

    def __call__(self) -> variable_class:
        if not self._is_bound:
            raise ValueError(f"Variable {self._name} was not bound, yet!")
        return self._value

    def __repr__(self):
        if self._value is None:
            return f"Unbound variable {self._name}: {self._type}"
        else:
            return f"Variable {self._name}: {self._type} = {self._value})"


