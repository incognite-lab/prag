from abc import ABCMeta, abstractmethod


class Operand(metaclass=ABCMeta):
    """ABC for all operand types. Operand can be evaluated (returns a float) or decided
    (returns a bool). All subclasses must implement the evaluate and decide methods.

    """

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