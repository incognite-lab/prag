import traceback as tb
from abc import ABCMeta, abstractmethod
from functools import _make_key
from inspect import isabstract
from typing import (Any, Callable, ClassVar, Generic, Optional, TypeVar, Union,
                    final)
from warnings import warn


class Entity(metaclass=ABCMeta):
    """Entity abstract base class (ABC). All environment entities should
    inherit from this class. It represents any "thing" that can exist.
    """
    _observation_getter: ClassVar[Optional[Callable]] = None
    __class_instance_counter: ClassVar[int]

    def __init_subclass__(cls) -> None:
        cls.__class_instance_counter = 0  # instance counter for all subclasses

    @classmethod
    def set_observation_getter(cls, observation_getter: Callable):
        cls._observation_getter = observation_getter

    def __new__(cls, *args, **kwargs):
        cls.__class_instance_counter = cls.__class_instance_counter + 1  # count instances
        return super().__new__(cls)

    @abstractmethod
    def __init__(self, reference: Optional[str] = None) -> None:
        if Entity._observation_getter is None:  # TODO: maybe remove this in the future, if better obs getting is implemented
            raise ValueError(f"Observation getter for class {self.__class__} is not set!"
                             " You must first set observation getter callback"
                             " before instantiating any Entity class! (call Entity.set_observation_getter)")

        if reference is None:
            reference = self._get_generic_reference()
        self.__reference = reference
        super().__init__()

    def __call__(self) -> Any:
        return Entity._observation_getter(self)  # type: ignore

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
        """Monkey patch replaces a method with another. This is to be used
        to modify functionality for all subclasses, without changing the original code
        or to dynamically change behavior of objects.
        However, care must be taken, since it changes behavior of all subclasses
        without them "knowing" about it.
        For safety, only existing methods can be replaced, i.e., this is not intended
        for addition of new methods.

        Args:
            method (Callable): the method to be replaced
            alternative (Callable): new method that replaces the original

        Raises:
            ValueError: If the method does not exist.
        """
        method_name = method.__name__
        if not hasattr(cls, method_name):
            raise ValueError(f"Method {method_name} is not defined in class {cls.__name__}. "
                             "Don't use monkey patch to add new methods! Use inheritance instead.")

        warn(f"Warning: Monkey patching {cls.__name__}.{method_name}. This will change behavior of all subclasses! Be careful about it!")
        setattr(cls, method_name, alternative)

    @classmethod
    def list_subclasses(cls) -> set[type]:
        """List all subclasses of this class. Includes the current class.

        Returns:
            set[type]: Set of all subclasses of this class.
        """
        result = [cls] if not isabstract(cls) else []
        for subcls in cls.__subclasses__():
            if not isabstract(subcls):
                result.append(subcls)
            result.extend(subcls.list_subclasses())
        return set(result)  # type: ignore


variable_class = TypeVar('variable_class', bound=Entity)


class _VarRecord:

    def __init__(self, var_type: type, value: Entity) -> None:
        self._var_type: type = var_type
        self._value: Entity = value

    @property
    def value(self) -> Entity:
        return self._value

    @property
    def type(self) -> type:
        return self._var_type


class Variable(Generic[variable_class]):
    """Generic variable container. The purpose of this class is to define a placeholder
    for a variable of a specific type that is later bound to a specific value of that type.
    For example, at definition time, there is no variable with a specific value that can be provided
    directly to some function. However, it is known that the function requires a variable of a specific type.
    Thus, this class can be used to define such a variable.

    Example:
    ```
        future_variable = Variable(<variable_type>, <variable_name>)
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
    variable_container = Variable(MyVariable, "arg_name", "global_name")
    variable_container.bind(var)  # or variable_container.bind({"global_name": var}, based_on_global_names=True)

    # this will work (b is a property):
    print(variable_container.b)  # 1
    # this will raise an error (a is a simple variable):
    print(variable_container.a)  # AttributeError
    ```
    """
    _VAR_VAL_TABLE: ClassVar[dict[int, Any]] = {}
    __VAR_LIST: ClassVar[list['Variable']] = []
    __class_instance_counter: ClassVar[int] = 0

    def __init__(self, typ: type[variable_class], *, arg_name: Optional[str] = None, global_name: Optional[str] = None, base_name: Optional[str] = None):
        """Initializes a variable container.
        Providing global name will create a variable persistent across all predicates.
        If two variables have the same global name, they will ALWAYS point to the same value.
        Meaning, binding one will result in the other being bound. Unbinding one will unbind the other.
        Only changing the global name will break the link.

        Args:
            typ (type): Expected type of the variable. The variable must be of this class or a subclass of this class.
            arg_name (str): Argument name of the variable inside the predicate where it is used for internal reference.
            global_name (str): Global name of the variable. This name persists across all predicates.
            base_name (str): Base name can be used to create a global name with reference count (e.g., base_name="apple" will result in global_name="apple_5" iff there were 4 other vars before).
        """
        super().__init__()
        self._type: type[variable_class] = typ
        if base_name is not None:
            assert global_name is None, "Only one of base_name and global_name can be provided!"
            global_name = f"{base_name}_{self._get_reference_count()}"
            while global_name in self._VAR_VAL_TABLE:
                global_name += "_"  # so that the name is unique

        self._name = global_name if global_name else f"var_{str(self._type.__name__).lower()}_{self._get_reference_count()}"
        if self.is_bound():
            assert issubclass(self._VAR_VAL_TABLE[hash(self)].type, self._type), f"Variable '{self._name}' is going to be bound to a value of type '{self._VAR_VAL_TABLE[hash(self)].type}', which is not a subclass of '{self._type}'!"
        self._arg_name = arg_name if arg_name else self._name

    def __new__(cls, *args, **kwargs):
        cls.__class_instance_counter = cls.__class_instance_counter + 1
        instance = super().__new__(cls)
        cls.__VAR_LIST.append(instance)
        return instance

    @final
    @classmethod
    def pre_bind(cls, var_name: str, entity: Entity) -> None:
        """Binds a global name to some value without creating a variable.
        If a new variable is created with the same name, it will be thus already bound
        to this value.

        Args:
            var_name (str): (global) name of the variable.
            entity (Entity): value to bind.
        """
        key = hash(var_name)
        assert key not in cls._VAR_VAL_TABLE, f"Variable '{var_name}' is already bound to {cls._VAR_VAL_TABLE[key]}!"
        cls._VAR_VAL_TABLE[key] = _VarRecord(type(entity), entity)

    @final
    def bind(self, value: variable_class) -> None:
        """Binds the variable to a specific value.
        This will raise an error if the value is not a subclass of the type of the variable.

        Args:
            value (variable_class): any variable that is a subclass of the type of this container.
        """
        assert issubclass(type(value), self._type), f"Value {value} of type {type(value)} is not a subclass of {self._type}, required by variable '{self._arg_name}'"
        assert not self.is_bound(), f"Variable {self.name} is already bound to {self._VAR_VAL_TABLE[hash(self)]}, Unbind it before binding to a new value!"
        self._VAR_VAL_TABLE[hash(self)] = _VarRecord(self._type, value)

    @final
    @property
    def type(self) -> type[variable_class]:
        """Returns the type of the variable container.

        Returns:
            type: Expected type of the variable.
        """
        return self._type

    @final
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
        if not self.is_bound():
            raise ValueError(f"Variable {self._arg_name} was not bound, yet!")
        if not hasattr(self._type, name):
            raise AttributeError(f"Variable '{self._arg_name}' of type {self._type} has no function or property '{name}'\nPerhaps, it is not a property (defined via getter/setter functions) or a class variable? Simple variables are not allowed.")
        return getattr(self.value, name)

    @final
    def is_bound(self) -> bool:
        """
        Check if the value of the object is bound.

        Returns:
            bool: True if the value is bound, False otherwise.
        """
        return self.value is not None

    @final
    def unbind(self) -> None:
        """Unbinds all variables with the same name.

        Example:
            >>> x = Variable(Apple, global_name='apple_1')
            >>> y = Variable(Apple, global_name='apple_1')
            >>> x.is_bound()  # true
            >>> y.is_bound()  # true
            >>> x.unbind()
            >>> x.is_bound()  # false
            >>> y.is_bound()  # false, since x was unbound and had the same name
        """
        assert self.is_bound(), f"Variable {self._arg_name} was not bound, yet!"
        self._VAR_VAL_TABLE[hash(self)] = None

    @final
    def __call__(self) -> variable_class:
        """
        Retrieves the value of the variable, if it was bound.

        Returns:
            variable_class: The value of the variable.

        Raises:
            ValueError: If the variable is not bound.
        """
        if not self.is_bound:
            raise ValueError(f"Variable {self._arg_name} was not bound, yet!")
        return self.value  # type: ignore

    @final
    @property
    def value(self) -> Optional[variable_class]:
        """The entity bound to this variable.
        Is set to None if the variable is not bound.

        Returns:
            Optional[variable_class]: The entity bound to this variable.
        """
        result = self._VAR_VAL_TABLE.get(hash(self), None)
        return result.value if result else None

    def __repr__(self):
        if self.is_bound():
            return f"Variable {self._name} (argument name: {self._arg_name}): {self._type} := {self.value})"
        else:
            return f"Unbound variable {self._name} (argument name: {self._arg_name}): {self._type}"

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Variable):
            return hash(self) == hash(other)
        else:
            return False

    @final
    @property
    def id(self) -> int:
        """An identifier of this variable. Two variables have the same ID iff they do or can represent
        the same value. Meaning, the two variables are the same.

        Returns:
            int: The identifier of this variable.
        """
        if self.is_bound():
            return hash(self.value)
        else:
            return hash(self)

    @final
    @property
    def arg_name(self) -> str:
        """Argument name of this variable. That is, the name as called in this specific predicate
        (see self._VARIABLES of that specific predicate).
        This should not normally be used externally. Typically, global name should be used.

        Example:
            For `Near(object_A: Location, object_B: Location)`, the argument name is `object_A` and `object_B`.
            Meaning, if Near creates variables internally, it will give them these names.


        Returns:
            str: argument name.
        """
        return self._arg_name

    @final
    @property
    def name(self) -> str:
        """Global name of this variable. This name represents a unique entity in the world.
        Even when unbound, two variables with the same global name refer to the same (would be) entity.

        Returns:
            str: global name.
        """
        return self._name

    @classmethod
    def _get_reference_count(cls) -> int:
        return cls.__class_instance_counter

    @classmethod
    def _align_variable_to_another(cls, destination_variable: 'Variable', source_variable: 'Variable') -> None:
        assert issubclass(source_variable.type, destination_variable.type), f"Cannot align variable '{destination_variable.name}' of type {destination_variable._type} to variable '{source_variable._arg_name}' of type {source_variable._type}! Types must match!"
        assert not destination_variable.is_bound(), f"Cannot align variable '{destination_variable.name}', variable is bound! Unbind it before trying to link it to another variable!"
        destination_variable._name = source_variable._name

    @classmethod
    def variable_exists(cls, name: str) -> bool:
        """Whether a variable with the given name already exists.
        Or more specifically, whether such global name is bound to any value.

        Args:
            name (str): global name of the variable.

        Returns:
            bool: True if a variable with the given name already exists, False otherwise.
        """
        return hash(name) in cls._VAR_VAL_TABLE

    @final
    def link_to(self, other: 'Variable') -> None:
        """Links this variable to another variable.
        Linked variables will have the same name, meaning they refer to the same entity.
        Even if unbound, they will represent the same thing. Operations on one are typically
        translated to the other. E.g., if one is bound, the other will be bound as well.
        Only variables of the same type can be linked.
        Linking directly changes current global name of the variable.

        Args:
            other (Variable): Variable to link to.
        """
        self._align_variable_to_another(self, other)

    @final
    def unlink(self, new_global_name: str) -> None:
        """Breaks the link between this variable and all other variables with the same global name.
        If the variable is bound, it will keep its current value but it will no longer linked to the other variable.
        For example, unbinding after unlinking will not unbind the value of the other variable.

        Args:
            new_global_name (str): New, unique global name to give to the this variable.
        """
        assert new_global_name not in self._VAR_VAL_TABLE, f"Cannot rename variable '{self._name}' to '{new_global_name}', because a variable with that name already exists!"
        if self.is_bound():
            self._VAR_VAL_TABLE[hash(new_global_name)] = self._VAR_VAL_TABLE[hash(self)]  # copy current value, delinking does not change value.
        self._name = new_global_name

    @final
    def global_rename(self, new_global_name: str) -> None:
        """Renames the global name of the variable.
        This will rename all variables with the same global name!
        That is, this is not meant to break link from variables sharing the same global name.
        Use unlink() for that.

        Args:
            new_global_name (str): New, unique global name to give to the this variable.
        """
        assert new_global_name not in self._VAR_VAL_TABLE, f"Cannot rename variable '{self._name}' to '{new_global_name}', because a variable with that name already exists!"
        current_name = self._name
        if self.is_bound():
            self._VAR_VAL_TABLE[hash(new_global_name)] = self._VAR_VAL_TABLE[hash(current_name)]
            del self._VAR_VAL_TABLE[hash(current_name)]

        for var in self.__VAR_LIST:  # rename all variables
            if var._name == current_name:
                var._name = new_global_name


cache_type = TypeVar("cache_type")


class _Cache(Generic[cache_type]):

    def __init__(self, default_value: object, typ: type) -> None:
        self.__default_value: object = default_value
        self.__type: type = typ
        self.__cache: dict[Any, cache_type] = {}
        self.__cache_get: Callable = self.__cache.get

    @property
    def type(self) -> type:
        return self.__type

    def __call__(self, key: Any) -> Union[cache_type, object]:
        return self.__cache_get(key, self.__default_value)

    def __setitem__(self, key: Any, value: cache_type) -> None:
        self.__cache[key] = value

    def clear(self) -> None:
        self.__cache.clear()


class _CacheContainer():

    def __init__(self):
        self._cache_sentinel = object()
        self._cache_decide: _Cache[bool] = _Cache(self._cache_sentinel, bool)
        self._cache_eval: _Cache[float] = _Cache(self._cache_sentinel, float)

    def _fetch_from_cache(self, cache: _Cache[cache_type], compute_func: Callable, internal_args, *args: Any, **kwds: Any) -> cache_type:
        key = self._make_key(compute_func, internal_args, *args, **kwds)
        result = cache(key)
        return self._handle_cached_result(result, key, cache, compute_func, *args, **kwds)

    def _handle_cached_result(self, result, key, cache: _Cache[cache_type], compute_func: Callable, *args: Any, **kwds: Any) -> cache_type:
        if result is not self._cache_sentinel:
            return result
        result = compute_func(*args, **kwds)
        cache[key] = result
        return result

    def _make_key(self, func, internal_args, *args, **kwds):
        return _make_key(tuple([func.__self__.__class__] + internal_args + list(args)), kwds, typed=True)

    def decide(self, compute_func: Callable, internal_args, *args: Any, **kwds: Any) -> bool:
        return self._fetch_from_cache(self._cache_decide, compute_func, internal_args, *args, **kwds)

    def eval(self, compute_func: Callable, internal_args, *args: Any, **kwds: Any) -> float:
        return self._fetch_from_cache(self._cache_eval, compute_func, internal_args, *args, **kwds)

    def reset(self):
        self._cache_decide.clear()
        self._cache_eval.clear()


class SymbolicCacheContainer(_CacheContainer):

    def _handle_cached_result(self, result, key, cache: _Cache[cache_type], compute_func: Callable[..., Any], *args: Any, **kwds: Any) -> cache_type:
        if result is not self._cache_sentinel:
            return result
        result = False if cache.type is bool else 0.0
        cache[key] = result
        return result

    def set_value(self, value, compute_func: Callable[..., Any], internal_args, *args: Any, **kwds: Any) -> None:
        key = self._make_key(compute_func, internal_args, *args, **kwds)
        self._cache_decide[key] = value

    def eval(self, compute_func: Callable[..., Any], internal_args, *args: Any, **kwds: Any) -> float:
        raise NotImplementedError("Evaluation cannot be called in symbolic mode!")


class Operand(metaclass=ABCMeta):
    """ABC for all operand types. Operand can be evaluated (returns a float) or decided
    (returns a bool). All subclasses must implement the evaluate and decide methods.

    """
    __MAPPING: ClassVar[dict[str, Any]] = {}
    __CACHE: ClassVar[_CacheContainer] = _CacheContainer()
    _USE_CACHE: ClassVar[bool] = True

    @classmethod
    def set_cache_normal(cls, cache: Optional[_CacheContainer] = None):
        """Sets the cache mode to 'normal' mode.
        Cache is used to store previously computed results. Normal mode is used
        during "normal" operation, i.e., when predicates actually needs to be evaluated (computed).

        Args:
            cache (Optional[_CacheContainer], optional): Cache to use. Defaults to None,
            in which case a new cache object is creates.

        Raises:
            ValueError: If cache is set to symbolic cache.
        """
        if cache is not None:
            if isinstance(cache, SymbolicCacheContainer):
                raise ValueError("Cannot set normal cache to symbolic cache value!")
            cls.__CACHE = cache
        else:
            warn("Setting cache to normal mode. This resets the cache!")
            cls.__CACHE = _CacheContainer()

    @classmethod
    def set_cache_symbolic(cls, cache: Optional[SymbolicCacheContainer] = None):
        """Sets the cache mode to 'symbolic' mode. In symbolic mode,
        predicates are not computed and can only be 'decided' based on truth table
        stored in the cache. Values in the table are modified using `set_symbolic_value`
        method of predicates.

        Args:
            cache (Optional[SymbolicCacheContainer], optional): Optionally provide a cache. Defaults to None,
            in which case a new cache object is created.
        """
        if cache is not None:
            cls.__CACHE = cache
        else:
            warn("Setting cache to symbolic mode. This resets the cache!")
            cls.__CACHE = SymbolicCacheContainer()

    @classmethod
    def reset_cache(cls):
        """Resets cache. For normal cache, it means that all predicates will need to be
        re-computed. For symbolic cache, it means that all values will be reset to false.
        """
        cls.__CACHE.reset()

    @classmethod
    def set_mapping(cls, mapping: dict):
        """Sets the mapping for the functions required by the operands.
        Use `get_required_mappings` to list all required mappings.

        Args:
            mapping (dict): _description_
        """
        cls.__MAPPING = mapping

    @classmethod
    def __register_attributes(cls, **kwargs):
        dd = vars(cls)
        for k, v in dd.items():
            if k.startswith("_0_"):
                # TODO: add check type of v compared to v in __MAPPING
                if v in Operand.__MAPPING:
                    setattr(cls, k, Operand.__MAPPING[v])
                else:
                    raise ValueError(f"The required user defined mapping for property {k} with name {v} of class {cls.__name__} is not defined!"
                                     "Use `Operand.list_required_mappings` (before anything else) to list all required mappings.")
                print(f"Class '{cls}' has an attribute {k}, which is mapped to {v}.")

    @final
    @classmethod
    def list_required_mappings(cls) -> set[str]:
        subclasses = cls._list_subclasses()
        attrs = set()
        for subcls in subclasses:
            dd = vars(subcls)
            for k, v in dd.items():
                if k.startswith("_0_"):
                    attrs.add(v)
        return attrs

    @final
    @classmethod
    def print_required_mappings(cls) -> None:
        pretty = '\n'.join([str(v) for v in Operand.list_required_mappings()])
        print(f"Required mappings: {pretty}")

    @classmethod
    def _list_subclasses(cls) -> set[type]:
        """List all subclasses of this class. Includes the current class.

        Returns:
            set[type]: Set of all subclasses of this class.
        """
        result = [] if cls == Operand else [cls]
        for subcls in cls.__subclasses__():
            result.append(subcls)
            result.extend(subcls._list_subclasses())
        return set(result)  # type: ignore

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__register_attributes()

    def __init__(self) -> None:
        super().__init__()
        self.__args = []

    def _append_arguments(self, *args: Any) -> None:
        self.__args.extend(args)

    def _prepare_args_for_key(self) -> list[Any]:
        return self.__args

    @final
    def decide(self, *args: Any, **kwds: Any) -> bool:
        if self._USE_CACHE:
            return self.__CACHE.decide(self.__decide__, self._prepare_args_for_key(), *args, **kwds)
        else:
            return self.__decide__(*args, **kwds)

    @abstractmethod
    def __decide__(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError(f"'decide' method not implemented for {self.__class__} operand")

    @final
    def evaluate(self, *args: Any, **kwds: Any) -> float:
        if self._USE_CACHE:
            return self.__CACHE.eval(self.__evaluate__, self._prepare_args_for_key(), *args, **kwds)
        else:
            return self.__evaluate__(*args, **kwds)

    @abstractmethod
    def __evaluate__(self, *args: Any, **kwds: Any) -> float:
        raise NotImplementedError(f"'evaluate' method not implemented for {self.__class__} operand")

    def set_symbolic_value(self, value: bool, only_if_contains: Optional[set[Variable]] = None) -> None:
        assert isinstance(self.__CACHE, SymbolicCacheContainer), "Setting symbolic value only works in symbolic mode!"
        if only_if_contains is not None:
            for v in only_if_contains:
                if v not in self.__args:
                    return
        self.__CACHE.set_value(value, self.__decide__, self._prepare_args_for_key())


class LogicalOperand(Operand, metaclass=ABCMeta):
    """LogicalOperand operates (executes decide method) over zero or more operands. It is also itself an operand,
    meaning, it can be evaluated or decided.
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def gather_variables(self) -> list[Variable]:
        raise NotImplementedError(f"gather_variables not implemented for {self.__class__} operand")


class Operator(LogicalOperand, metaclass=ABCMeta):
    """Operator operates (executes evaluate or decide method) over zero or more operands. It is also itself an operand,
    meaning, it can be evaluated or decided.

    Operator class must define _ARITY and _SYMBOL attributes. _ARITY defines arity of the operation (number of operands).
    _SYMBOL defines string representation of the operation. It is used in parsing of definition files.
    """
    _ARITY: ClassVar[int]
    _SYMBOL: ClassVar[str]
    _USE_CACHE = False

    def __init_subclass__(cls) -> None:
        dd = dir(cls)
        if '_SYMBOL' not in dd and not isabstract(cls):
            raise ValueError(f"Class '{cls}' does not have a '_SYMBOL' attribute! Every sub-class of 'Operator' must define a '_SYMBOL' that defines its string representation!")
        if '_ARITY' not in dd:
            raise ValueError(f"Class '{cls}' does not have a '_ARITY' attribute! Every sub-class of 'Operator' must define a '_ARITY' that defines arity of the operation!")
        super().__init_subclass__()

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.decide(*args, **kwds)

    @classmethod
    @property
    def SYMBOL(cls) -> str:
        return cls._SYMBOL

    @classmethod
    @property
    def ARITY(cls) -> int:
        return cls._ARITY


AA = TypeVar("AA")


class Predicate(LogicalOperand, metaclass=ABCMeta):
    """Predicates are special operands that cannot be evaluated. The can only return true or false.
    It is meant as a function that is computed externally.
    """
    _VARIABLES: ClassVar[dict[str, type[Entity]]] = {}
    __class_instance_counter = 0

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
                assert isinstance(variable, Variable), f"Externally provided variable {name} is not an instance of 'Variable'!"
                assert issubclass(variable.type, typ), f"Externally provided variable {name} is not of the required type {typ}!"
                self.__add_variable(name, variable)
                del kwds[name]
            else:
                self.__register_variable(name, typ)
        if len(kwds) > 0:
            raise ValueError(f"Unknown argument {list(kwds.keys())} provided to {self.__class__.__name__}!")

    def __new__(cls, *args, **kwargs):
        cls.__class_instance_counter = cls.__class_instance_counter + 1
        return super().__new__(cls)

    @classmethod
    def _get_class_instance_counter(cls) -> int:
        return cls.__class_instance_counter

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> bool:
        raise NotImplementedError("call not implemented for generic predicate")

    def _prepare_args_for_key(self) -> list[Any]:
        return [v.id for v in self.variables.values()]

    @final
    def __decide__(self, *args: Any, **kwds: Any) -> bool:
        return self(*args, **kwds)

    @final
    def __evaluate__(self):
        warn(f"Predicates can't be evaluated! Returning None. Evaluate called for: {self.__class__.__name__} @ {tb.format_stack()}")
        return None

    def __register_variable(self, name: str, typ: type[variable_class]) -> None:
        variable = Variable(typ, base_name=f"{self.__class__.__name__}_{self._get_class_instance_counter()}_{name}")
        self.__add_variable(name, variable)

    def __add_variable(self, name: str, variable: Variable):
        self._append_arguments(variable)
        self.__variables[name] = variable

    def get_argument(self, arg_name: str) -> Variable:
        """Returns a variable according to its argument name.
        That is, the name as called in this specific predicate (see self._VARIABLES).
        Example: For a predicate `gimme(a: Apple)` you can get the variable `a` by calling `self.get_argument('a')`.

        Args:
            arg_name (str): Argument name of the variable

        Returns:
            Variable: Variable with the given argument name.
        """
        return self.__variables[arg_name]

    def get_variable(self, global_name: str) -> Variable:
        """Returns a variable according to its global name.
        That is, the name given globally to variable (used across all predicates).
        Example: For a predicate `gimme(a: Apple)` and if `a` has a global name `apple_007`
        you can get the variable `a` by calling `self.get_variable('apple_007')`.
        In this example, `a` was declared as `Variable(typ=Apple, arg_name='a', global_name='apple_007')`.

        Args:
            global_name (str): Global name of the variable.

        Raises:
            ValueError: Raises a ValueError if the variable with the given global name is not found.

        Returns:
            Variable: Variable with the given global name.
        """
        try:
            return next((v for v in self.__variables.values() if v.name == global_name))
        except StopIteration:
            raise ValueError(f"Variable {global_name} not found in {self.__class__.__name__}!")

    @property
    def variables(self) -> dict[str, Variable]:
        return self.__variables

    def gather_variables(self) -> list[Variable]:
        return list(self.__variables.values())

    def bind(self, variables: Union[list[Entity], dict[str, Entity]], based_on_global_names: bool = False) -> None:
        """Bind predicate variables to specific values. These should be subclasses of the type 'Entity'.
        Input can be a list of values, in which case they must be provided in the same order as they are
        written in the cls._VARIABLES dictionary. Otherwise, a dictionary using variable names
        (as defined in the cls._VARIABLES dictionary) as keys is required.

        Args:
            variables (Union[list[Entity], dict[str, Entity]]): List of concrete values or dictionary of variable names and concrete values
            based_on_global_names
        """
        if based_on_global_names:
            if isinstance(variables, list):
                raise ValueError("Cannot bind list of values without names when global names are to be used! Use dictionary specifying global name as the key.")
            internal_vars = ((v.name, v) for v in self.__variables.values())
        else:
            if isinstance(variables, list):
                variables = dict(zip(self.__variables, variables))
            internal_vars = self.__variables.items()
        for var_name, variable in internal_vars:
            if var_name is None:
                if based_on_global_names:
                    warn(f"Variable '{variable.arg_name}' for predicate '{self.__class__.__name__}' does not have a global name! Cannot bind according to the provided dictionary!")
                else:
                    raise ValueError("Something went wrong and a variable does not have a name!")
            if var_name in variables:
                variable.bind(variables[var_name])
            else:
                warn(f"Variable '{var_name}' for predicate '{self.__class__.__name__}' is not bound to any value in the provided dictionary. This may lead to errors!")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}: {v.type.__name__}' for k, v in self.__variables.items()])})"


class Reward(Operand):

    def __init__(self) -> None:
        super().__init__()
        self._reward_function: Callable[[], float]

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return NotImplementedError(f"Evaluate not implemented for {self.__class__} reward")

    def __evaluate__(self, *args: Any, **kwds: Any):
        return self(*args, **kwds)

    @final
    def __decide__(self):
        warn(f"Rewards can't be decided! Returning None. Decide called for: {self.__class__} @ {tb.format_stack()}")
        return None


class AtomicAction(Predicate, metaclass=ABCMeta):
    """Container for atomic action. AA is an operand (it can be evaluated or decided)
    and combines initial condition, goal (terminating) condition and reward function.
    """
    _USE_CACHE = False

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        self._predicate: LogicalOperand
        self._initial: LogicalOperand
        self._reward: Operand

    def __call__(self):
        """ Decide whether goal condition is met.

        Returns:
            bool: True if goal condition is met, False otherwise.
        """
        print(f"Checking action {str(self.__class__.__name__)}")
        return self._predicate.decide()

    def can_be_executed(self) -> bool:
        return self._initial.decide()

    def reward(self):
        """Evaluate reward function for the action.

        Returns:
            float: The reward value.
        """
        print(f"Evaluating action {str(self.__class__.__name__)}")
        return self._reward.evaluate()

    @property
    def predicate(self) -> Union[Predicate, Operator]:
        return self._predicate

    @property
    def initial(self) -> Union[Predicate, Operator]:
        return self._initial
