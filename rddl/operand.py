from abc import ABCMeta, abstractmethod


class Operand(metaclass=ABCMeta):

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