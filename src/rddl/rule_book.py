from typing import Any, Optional, Union
from sympy import Predicate

from rddl.core import LogicalOperand, Operand, Variable
from rddl.sampling_utils import StrictSymbolicCacheContainer


class Rule:

    def __init__(self, predicate_classes) -> None:
        self.__predicate_classes = predicate_classes

    def __contains__(self, predicate: Union[LogicalOperand, type[LogicalOperand]]) -> bool:
        if isinstance(predicate, LogicalOperand):
            predicate = predicate.__class__
        return predicate in self.__predicate_classes


class ExclusivityRule(Rule):
    """
    Exclusive rule
    All predicates in this rule are mutually exclusive. When one is True, all other must be False.
    """

    def __init__(self, *predicates: type[LogicalOperand]):
        super().__init__(predicates)
        self._exclusive_predicates = predicates
        self._n_predicates = len(predicates)

    def check(self) -> bool:
        for i in range(self._n_predicates):
            for j in range(i + 1, self._n_predicates):
                if self._exclusive_predicates[i].decide() and self._exclusive_predicates[j].decide():
                    return False
        return True

    def list_breaking_predicates(self, exemplar: LogicalOperand, variables: Optional[list[Variable]] = None) -> list[LogicalOperand]:
        if variables is None:
            variables = exemplar.gather_variables()
        for predicate in self._exclusive_predicates:
            pass


class Consequent(Rule):
    """
    Consequent rule
    When the exemplar predicate is true, the consequent must be true but not vice versa.
    """

    def __init__(self, exemplar: type[LogicalOperand], *consequences: type[LogicalOperand]):
        super().__init__(consequences + (exemplar,))
        self._exemplar = exemplar
        self._consequences = consequences

    def apply(self) -> None:
        if self._exemplar.decide():
            for predicate in self._consequences:
                predicate.set_symbolic_value(True)


class RuleBook:

    def __init__(self) -> None:
        self._exclusivity_rules = []
        self._consequential_rules = []

    def add_rule(self, rule: Rule) -> None:
        if isinstance(rule, ExclusivityRule):
            self._exclusivity_rules.append(rule)
        elif isinstance(rule, Consequent):
            self._consequential_rules.append(rule)
        else:
            raise ValueError("Unknown rule type")

    def _get_current_predicate_set(self) -> list[tuple[Predicate, list[Variable], bool]]:
        c = Operand.get_cache()
        if not isinstance(c, StrictSymbolicCacheContainer):
            raise ValueError("Current cache is not a 'StrictSymbolicCacheContainer'! Rule book cannot be used!")
        return c.get_predicates()

    def _construct_exclusivity_predicates(self) -> None:
        current_predicates = self._get_current_predicate_set()
        exclusivity_predicates = []
        for predicate, variables, value in current_predicates:
            if not value:
                continue
            for rule in self._exclusivity_rules:
                if predicate not in rule or not value:  # skip if predicate not in the rule or not true
                    continue


    def check_if_breaks_consistency(self, *predicates: LogicalOperand) -> bool:
        current_predicates = self._get_current_predicate_set()
        for cpredicate, cvariables, cvalue in current_predicates:
            for predicate in predicates:
                variables = predicate.gather_variables()


    def check_consistency(self) -> bool:
        for rule in self._exclusivity_rules:
            if not rule.check():
                return False
        return True

    def apply_rules(self) -> None:
        for rule in self._consequential_rules:
            rule.apply()
