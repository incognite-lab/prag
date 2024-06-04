from abc import abstractmethod
from typing import Iterable, Optional, TypeVar

from rddl import Entity, Predicate


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


class GraspableObject(ObjectEntity):

    def __init__(self, reference: str, name: Optional[str] = None):
        super().__init__(reference, name)


class Gripper(Location):

    def __init__(self, reference: Optional[str] = None):
        super().__init__(reference)
        self._is_holding_predicate: Predicate

    def is_holding(self, obj: ObjectEntity) -> bool:
        return self._is_holding_predicate(obj)


class AbstractLocation(Location):

    def __init__(self, reference: Optional[str] = None):
        super().__init__(reference)

    @property
    def value(self):
        raise NotImplementedError


class AbstractRotation(Entity):

    def __init__(self, reference: str):
        super().__init__(reference)

    @property
    def value(self):
        raise NotImplementedError


LocationType = TypeVar("LocationType", bound=Location)
